from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.q_networks import QuantileQNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import QLearningHead


# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_qrdqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build QRDQNHead on CPU.

    Purpose
    -------
    When using Ray remote workers, the policy object must be constructible
    from a module-level function (so that it is pickleable/importable).

    Notes
    -----
    - `device` is overridden to "cpu" because Ray workers typically keep policies on CPU.
    - `activation_fn` is serialized as a string or None; it is resolved here back to
      the actual torch.nn.Module class/function.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = QRDQNHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


class QRDQNHead(QLearningHead):
    """
    QR-DQN Head: (Online Quantile Q) + (Target Quantile Q)

    High-level behavior
    -------------------
    QR-DQN learns a distributional Q-function, represented by N quantiles per action:
        Z(s, a) ~ distribution over returns
    The network outputs quantiles:
        quantiles(s) -> (B, N, A)

    This head provides both:
      - distributional outputs (quantiles)
      - expected Q-values (mean over quantiles) for compatibility with
        classic Q-learning interfaces.

    Inherited from QLearningHead
    ----------------------------
    This class assumes QLearningHead provides:
      - self.device
      - _to_tensor_batched(obs): convert numpy/list -> torch tensor (B, obs_dim)
      - set_training(training): toggles train/eval mode
      - act(obs, epsilon=..., deterministic=...):
            epsilon-greedy wrapper built on top of q_values(obs)
      - hard_update(target, source), soft_update(target, source, tau), freeze_target(module)

    QRDQN overrides / additions
    ---------------------------
    - quantiles(obs) -> (B, N, A)
    - quantiles_target(obs) -> (B, N, A)
    - q_values(obs) -> (B, A)            (expected Q; mean over N)
    - q_values_target(obs) -> (B, A)     (expected Q; mean over N)
    - save/load: includes ctor kwargs + both nets (online/target)
    - get_ray_policy_factory_spec(): ray-compatible factory spec
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
            Number of discrete actions (A).
        n_quantiles : int
            Number of quantiles per action (N). Typical values: 50~200.
        hidden_sizes : Sequence[int]
            MLP hidden layer sizes for the quantile Q-network.
        activation_fn : Any
            Activation function class, e.g., nn.ReLU, nn.SiLU, nn.Tanh.
        dueling_mode : bool
            If True, use a dueling architecture (value + advantage streams)
            inside QuantileQNetwork (if supported).
        init_type : str
            Initialization strategy string (passed through to QuantileQNetwork).
        gain : float
            Gain scaling for initialization (depends on init_type).
        bias : float
            Bias init constant.
        device : Union[str, torch.device]
            Torch device for the networks.
        """
        super().__init__(device=device)

        # --- basic config ---
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
        # Online / Target networks
        #
        # Both networks output quantiles:
        #   (B, N, A)
        #
        # Convention:
        #   - Online net: trained by gradient descent
        #   - Target net: updated periodically via hard/soft updates and frozen
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

        # Copy online -> target, then freeze target (no gradients) for stability
        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    # =============================================================================
    # Quantiles
    # =============================================================================
    def quantiles(self, obs: Any) -> th.Tensor:
        """
        Compute online quantiles Z(s, a).

        Parameters
        ----------
        obs : Any
            Single observation (obs_dim,) or batch (B, obs_dim).
            Accepts numpy arrays, lists, torch tensors, etc.

        Returns
        -------
        quantiles : torch.Tensor
            Quantile samples with shape (B, N, A):
              - B : batch size
              - N : number of quantiles
              - A : number of discrete actions
        """
        s = self._to_tensor_batched(obs)  # (B, obs_dim)
        return self.q(s)                  # (B, N, A)

    @th.no_grad()
    def quantiles_target(self, obs: Any) -> th.Tensor:
        """
        Compute target quantiles Z_target(s, a).

        Returns
        -------
        quantiles : torch.Tensor
            Shape (B, N, A)
        """
        s = self._to_tensor_batched(obs)
        return self.q_target(s)

    @staticmethod
    def q_mean_from_quantiles(quantiles: th.Tensor) -> th.Tensor:
        """
        Convert distributional quantiles into expected Q-values.

        Parameters
        ----------
        quantiles : torch.Tensor
            Shape (B, N, A)

        Returns
        -------
        q_values : torch.Tensor
            Expected Q-values, shape (B, A), computed as mean over quantiles.
        """
        if quantiles.dim() != 3:
            raise ValueError(f"Expected quantiles shape (B,N,A), got: {tuple(quantiles.shape)}")
        return quantiles.mean(dim=1)

    # =============================================================================
    # Expected Q (override for QLearningHead compatibility)
    # =============================================================================
    def q_values(self, obs: Any) -> th.Tensor:
        """
        Expected Q-values from online quantiles.

        This is used by epsilon-greedy action selection in QLearningHead.act().
        """
        return self.q_mean_from_quantiles(self.quantiles(obs))  # (B, A)

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        """
        Expected Q-values from target quantiles.

        Used by QRDQN core when constructing bootstrapped targets.
        """
        return self.q_mean_from_quantiles(self.quantiles_target(obs))  # (B, A)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export ctor kwargs in JSON-safe form.

        Notes
        -----
        - `activation_fn` is stored as a string name so it can be reconstructed on Ray workers.
        - `device` is included for convenience, but Ray worker factory will override to CPU.
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
        Save head checkpoint.

        Saves:
          - ctor kwargs (json-safe)
          - online network weights
          - target network weights

        This makes the checkpoint self-contained and robust to restarts.
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
        Load head checkpoint.

        Behavior
        --------
        - Always restores online weights.
        - Restores target weights if present.
          Otherwise, rebuilds target as a copy of online.
        - Ensures target stays frozen + eval mode.
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
        Return Ray policy factory specification.

        The spec includes:
          - entrypoint: module-level factory function
          - kwargs: JSON-safe ctor kwargs for rebuilding the head on workers
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_qrdqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),  # NOTE: ensure this method exists (typo risk)
        )
