from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.q_networks import QuantileQNetwork
from model_free.common.policies.base_head import QLearningHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_qrdqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a :class:`QRDQNHead` instance on a Ray worker (CPU-only).

    Ray typically reconstructs policy modules inside remote worker processes using:
      1) a module-level entrypoint (importable / pickle-friendly), and
      2) a JSON-safe kwargs payload.

    This factory enforces worker-side invariants:

    - **CPU-only** execution on rollout workers (avoid GPU contention).
    - **Activation resolution**: serialized activation identifiers (e.g., ``"ReLU"``)
      are converted back into a torch activation class via ``_resolve_activation_fn``.
    - **Inference mode**: the created head is switched to evaluation/inference
      behavior via ``set_training(False)``.

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments intended for :class:`QRDQNHead`.
        The payload should be JSON/pickle-safe. In particular, ``activation_fn``
        is expected to be serialized (string/None) and is resolved here.

    Returns
    -------
    nn.Module
        A CPU-allocated :class:`QRDQNHead` set to inference mode.
    """
    cfg = dict(kwargs)  # defensive copy (Ray may reuse dict objects)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = QRDQNHead(**cfg).to("cpu")
    head.set_training(False)
    return head


class QRDQNHead(QLearningHead):
    """
    Quantile Regression DQN (QR-DQN) head (online + target quantile Q-networks).

    QR-DQN learns a **distributional** action-value function by approximating the
    return distribution ``Z(s,a)`` with a fixed set of quantiles.

    Representation
    --------------
    For each state ``s`` and action ``a``, the network outputs ``N`` quantile values.
    With a batch of size ``B`` and ``A`` actions, the quantile tensor has shape:

    - ``(B, N, A)``

    This head exposes both:
    - **Quantiles**: distributional outputs for distributional RL updates, and
    - **Expected Q-values**: mean over quantiles for epsilon-greedy action selection
      and compatibility with classic Q-learning interfaces.

    Inherited contract (from QLearningHead)
    --------------------------------------
    This class assumes :class:`~model_free.common.policies.base_head.QLearningHead`
    provides utilities such as:

    - ``self.device`` device resolution/storage
    - ``_to_tensor_batched(obs)`` conversion helper for ``obs`` to (B, obs_dim)
    - ``set_training(training: bool)``
    - epsilon-greedy ``act(obs, epsilon=..., deterministic=...)`` built on ``q_values()``
    - target net helpers: ``hard_update``, ``soft_update``, ``freeze_target``

    Notes
    -----
    - The target network is conventionally frozen (no gradients) for stability.
    - This head is “thin”: it owns networks and serialization metadata; the learning
      rule lives in the core.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 200,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension (flattened state vector size).
        n_actions : int
            Number of discrete actions (``A``).
        n_quantiles : int, default=200
            Number of quantiles per action (``N``). Typical values: 50–200.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer sizes for the quantile Q-network MLP.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function **class** (e.g., ``nn.ReLU``).
            This may be serialized for Ray and reconstructed with
            ``_resolve_activation_fn``.
        dueling_mode : bool, default=False
            If True, enable a dueling architecture (value + advantage streams)
            inside :class:`~model_free.common.networks.q_networks.QuantileQNetwork`
            if supported by your implementation.
        init_type : str, default="orthogonal"
            Initialization strategy string (passed to network builder).
        gain : float, default=1.0
            Gain scaling for initialization (depends on ``init_type``).
        bias : float, default=0.0
            Bias initialization constant.
        device : Union[str, torch.device], default=("cuda" if available else "cpu")
            Device for allocating the networks.

        Raises
        ------
        ValueError
            If ``n_actions`` or ``n_quantiles`` are not positive.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)

        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got: {self.n_actions}")
        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got: {self.n_quantiles}")

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.dueling_mode = bool(dueling_mode)

        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---------------------------------------------------------------------
        # Online / target quantile networks
        #
        # Both networks return quantiles with shape: (B, N, A)
        # ---------------------------------------------------------------------
        self.q = QuantileQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            n_quantiles=self.n_quantiles,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.q_target = QuantileQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            n_quantiles=self.n_quantiles,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target parameters from online parameters, then freeze target.
        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    # =============================================================================
    # Quantiles API
    # =============================================================================
    def quantiles(self, obs: Any) -> th.Tensor:
        """
        Compute online quantiles ``Z(s, a)``.

        Parameters
        ----------
        obs : Any
            Observation input. Accepts:
            - single observation of shape ``(obs_dim,)``
            - batch of observations of shape ``(B, obs_dim)``
            - torch tensors / numpy arrays / lists (as supported by your base head)

        Returns
        -------
        quantiles : torch.Tensor
            Quantile outputs with shape ``(B, N, A)`` where:
            - ``B`` is the batch size,
            - ``N`` is the number of quantiles,
            - ``A`` is the number of actions.
        """
        s = self._to_tensor_batched(obs)  # (B, obs_dim)
        return self.q(s)                  # (B, N, A)

    @th.no_grad()
    def quantiles_target(self, obs: Any) -> th.Tensor:
        """
        Compute target quantiles ``Z_target(s, a)``.

        Parameters
        ----------
        obs : Any
            Observation input, same conventions as :meth:`quantiles`.

        Returns
        -------
        quantiles : torch.Tensor
            Target quantile outputs with shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        return self.q_target(s)

    @staticmethod
    def q_mean_from_quantiles(quantiles: th.Tensor) -> th.Tensor:
        """
        Convert quantiles to expected Q-values by averaging over quantile dimension.

        Parameters
        ----------
        quantiles : torch.Tensor
            Quantile tensor with shape ``(B, N, A)``.

        Returns
        -------
        q_values : torch.Tensor
            Expected Q-values with shape ``(B, A)`` computed as ``mean(dim=1)``.

        Raises
        ------
        ValueError
            If ``quantiles`` is not a rank-3 tensor of shape ``(B, N, A)``.
        """
        if quantiles.dim() != 3:
            raise ValueError(f"Expected quantiles shape (B,N,A), got: {tuple(quantiles.shape)}")
        return quantiles.mean(dim=1)

    # =============================================================================
    # Expected Q API (QLearningHead compatibility)
    # =============================================================================
    def q_values(self, obs: Any) -> th.Tensor:
        """
        Compute expected Q-values from the **online** quantiles.

        This is the quantity used for epsilon-greedy action selection and for
        Double DQN action selection (argmax) in the core.

        Parameters
        ----------
        obs : Any
            Observation input, same conventions as :meth:`quantiles`.

        Returns
        -------
        q_values : torch.Tensor
            Expected Q-values with shape ``(B, A)``.
        """
        return self.q_mean_from_quantiles(self.quantiles(obs))

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        """
        Compute expected Q-values from the **target** quantiles.

        Parameters
        ----------
        obs : Any
            Observation input, same conventions as :meth:`quantiles_target`.

        Returns
        -------
        q_values : torch.Tensor
            Expected target Q-values with shape ``(B, A)``.
        """
        return self.q_mean_from_quantiles(self.quantiles_target(obs))

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe form.

        Uses
        ----
        - Checkpoint metadata (reconstruction/debugging)
        - Ray worker instantiation via :class:`PolicyFactorySpec`

        Notes
        -----
        - ``activation_fn`` is converted to a stable identifier using
          ``self._activation_to_name`` because classes/functions are not
          JSON-serializable.
        - ``device`` is included for convenience; Ray worker factory overrides it to CPU.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable kwargs payload.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "n_quantiles": int(self.n_quantiles),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save a QR-DQN head checkpoint.

        Parameters
        ----------
        path : str
            Output path. Appends ``.pt`` if missing.

        Notes
        -----
        The checkpoint contains:
        - ``kwargs``   : JSON-safe constructor arguments
        - ``q``        : online network state_dict
        - ``q_target`` : target network state_dict

        This makes the checkpoint self-contained for reconstruction and debugging.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load a QR-DQN head checkpoint into the current instance.

        Parameters
        ----------
        path : str
            Path to checkpoint saved by :meth:`save`. Appends ``.pt`` if missing.

        Notes
        -----
        - Loads weights only; does not reconstruct a new object.
        - Restores target weights if present; otherwise synchronizes target from online.
        - Re-applies target freezing/eval invariants after load.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.q.load_state_dict(ckpt["q"])

        if ckpt.get("q_target", None) is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
            self.freeze_target(self.q_target)
            self.q_target.eval()
        else:
            self.hard_update(self.q_target, self.q)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly factory specification for this head.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint`` : module-level factory function (pickle/import friendly)
            - ``kwargs``     : JSON-safe constructor kwargs used by the factory
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_qrdqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
