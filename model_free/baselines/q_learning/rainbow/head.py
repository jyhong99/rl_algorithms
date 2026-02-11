from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.q_networks import RainbowQNetwork
from model_free.common.policies.base_head import QLearningHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_rainbow_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Construct a :class:`RainbowHead` instance on a Ray rollout worker (CPU-only).

    Ray typically reconstructs policy modules in remote worker processes using an
    importable (module-level) entrypoint and a JSON/pickle-safe ``kwargs`` payload.
    This factory is designed to be:

    - **pickle-friendly**: defined at module scope (importable symbol)
    - **portable**: forces ``device="cpu"`` to avoid GPU contention on workers
    - **robust**: resolves serialized ``activation_fn`` back into a torch activation

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments intended for :class:`RainbowHead`.

        Notes
        -----
        - Values must be JSON/pickle-safe.
        - ``activation_fn`` is expected to be a serialized representation (e.g., a
          string like ``"ReLU"`` or ``None``). It is resolved via
          :func:`model_free.common.utils.ray_utils._resolve_activation_fn`.

    Returns
    -------
    nn.Module
        A :class:`RainbowHead` allocated on CPU and placed into inference mode via
        ``set_training(False)`` (and typical ``eval()`` behavior inherited from
        the base head).
    """
    cfg = dict(kwargs)  # defensive copy (Ray may reuse kwargs payload)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = RainbowHead(**cfg).to("cpu")
    head.set_training(False)  # rollout workers should be inference-only
    return head


class RainbowHead(QLearningHead):
    """
    Rainbow head for discrete-action Q-learning (C51 + NoisyNet).

    This "head" owns the neural networks and immutable distributional support
    (C51 atoms) required by Rainbow-style algorithms.

    Components
    ----------
    - **Online network** ``q``:
        A categorical distributional Q-network returning a probability mass
        function (PMF) over fixed support atoms for each action.
    - **Target network** ``q_target``:
        A frozen copy of the online network, updated periodically by the core
        (hard update) or Polyak averaging (soft update).
    - **Support buffer** ``support``:
        Fixed atom values for C51, registered as a buffer to move with device
        and remain non-trainable.

    Rainbow building blocks covered here
    ------------------------------------
    - **C51 distributional Q-values**: categorical return distribution over atoms
    - **NoisyNet exploration**: noise is reset before acting to induce stochasticity

    Expected usage by Q-learning algorithms/cores
    ---------------------------------------------
    The surrounding algorithm/core is expected to call:

    - :meth:`q_values` / :meth:`q_values_target`:
        Return expected Q-values (mean of the categorical distribution).
    - :meth:`dist` / :meth:`dist_target` (implemented by underlying networks):
        Return PMFs of shape (B, A, K) where K is ``atom_size``.
    - :meth:`act`:
        Epsilon-greedy action selection based on expected Q-values with Noisy reset.
    - Target update utilities inherited from :class:`QLearningHead`:
        ``hard_update``, ``soft_update``, ``freeze_target``

    Notes
    -----
    - ``support`` is registered as a buffer so it:
        * moves with ``.to(device)``
        * is included in ``state_dict`` if needed
        * is not trainable (no gradients)
    - The target network is frozen via ``freeze_target`` to prevent accidental
      gradient flow and to enforce deterministic evaluation behavior.
    - The head does **not** implement the distributional Bellman projection;
      that belongs in the core/update engine.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        # -----------------------------
        # C51 distributional settings
        # -----------------------------
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # -----------------------------
        # MLP trunk
        # -----------------------------
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        # -----------------------------
        # Initialization
        # -----------------------------
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        # -----------------------------
        # NoisyNet
        # -----------------------------
        noisy_std_init: float = 0.5,
        # -----------------------------
        # Device
        # -----------------------------
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Flattened observation/state dimension.
        n_actions : int
            Number of discrete actions.
        atom_size : int, default=51
            Number of support atoms (C51). Must be >= 2.
        v_min : float, default=-10.0
            Minimum value of the support (lower bound of return distribution).
        v_max : float, default=10.0
            Maximum value of the support (upper bound of return distribution).
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer sizes for the Rainbow Q-network MLP trunk.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function **class** used by the network (e.g., ``nn.ReLU``).
            For Ray/checkpointing, this is typically serialized to/from a string name.
        init_type : str, default="orthogonal"
            Initialization scheme identifier passed through to the network builder.
        gain : float, default=1.0
            Optional initialization gain multiplier.
        bias : float, default=0.0
            Optional bias initialization constant.
        noisy_std_init : float, default=0.5
            Initial standard deviation used by NoisyNet layers.
        device : Union[str, torch.device], default="cuda" if available else "cpu"
            Torch device for allocating network parameters and buffers.

        Raises
        ------
        ValueError
            If dimensions or distributional support configuration is invalid
            (e.g., non-positive action count, ``atom_size < 2``, or ``v_min >= v_max``).
        """
        super().__init__(device=device)

        # -----------------------------
        # Basic dimensions / config
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        self.atom_size = int(atom_size)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn

        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.noisy_std_init = float(noisy_std_init)

        # -----------------------------
        # Sanity checks (fail fast)
        # -----------------------------
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")
        if self.atom_size <= 1:
            raise ValueError(f"atom_size must be >= 2, got {self.atom_size}")
        if not (self.v_min < self.v_max):
            raise ValueError(f"Require v_min < v_max, got v_min={self.v_min}, v_max={self.v_max}")

        # ---------------------------------------------------------------------
        # Fixed C51 support (atoms)
        #
        # - Register as buffer so it moves with device and is not trainable.
        # - Clone to ensure no shared-storage surprises.
        # ---------------------------------------------------------------------
        support = th.linspace(self.v_min, self.v_max, self.atom_size, dtype=th.float32)
        self.register_buffer("support", support.detach().clone())  # (K,)

        # ---------------------------------------------------------------------
        # IMPORTANT: do not share the same tensor object across modules if there
        # is any chance of in-place modification (even accidental).
        #
        # Give each network its own detached+cloned support tensor.
        # ---------------------------------------------------------------------
        support_q = self.support.detach().clone()
        support_t = self.support.detach().clone()

        # -----------------------------
        # Online network
        # -----------------------------
        self.q = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_q,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        # -----------------------------
        # Target network
        # -----------------------------
        self.q_target = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_t,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        # Initialize target = online, then freeze to enforce invariants.
        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    # =============================================================================
    # NoisyNet utilities
    # =============================================================================
    def reset_noise(self) -> None:
        """
        Resample NoisyNet noise parameters (best-effort).

        Many Rainbow implementations reset the online NoisyNet noise at each action
        selection to provide stochastic exploration even when epsilon is small (or 0).

        Notes
        -----
        This method is best-effort:
        - It is only executed if the underlying networks expose ``reset_noise()``.
        """
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()
        if hasattr(self.q_target, "reset_noise"):
            self.q_target.reset_noise()

    # =============================================================================
    # Acting (inject Noisy reset)
    # =============================================================================
    @th.no_grad()
    def act(self, obs: Any, *, epsilon: float = 0.0, deterministic: bool = True) -> th.Tensor:
        """
        Select an action using epsilon-greedy policy on expected Q-values.

        This overrides the base :meth:`QLearningHead.act` behavior to refresh NoisyNet
        noise before computing action values.

        Parameters
        ----------
        obs : Any
            Observation (obs_dim,) or batch of observations (B, obs_dim). Accepts
            numpy arrays, lists, or torch tensors (the base head normalizes inputs).
        epsilon : float, default=0.0
            Probability of selecting a uniformly random action (epsilon-greedy).
        deterministic : bool, default=True
            If True, select greedy actions (subject to epsilon if epsilon > 0).
            If False, allow stochasticity via epsilon-greedy (and NoisyNet noise).

        Returns
        -------
        action : torch.Tensor
            Discrete action indices. Shape is typically (B,) for batched inputs.
        """
        # Online NoisyNet reset at act-time is the standard pattern.
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()

        return super().act(obs, epsilon=epsilon, deterministic=deterministic)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments in a JSON-safe dict.

        This payload is suitable for:
        - checkpoint metadata (reconstruction/debugging)
        - Ray reconstruction via :class:`PolicyFactorySpec`

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor kwargs. ``activation_fn`` is converted to a stable
            string identifier via the base head helper.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "atom_size": int(self.atom_size),
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "noisy_std_init": float(self.noisy_std_init),
            # Kept for convenience; Ray worker factory overrides to CPU.
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save a Rainbow head checkpoint (weights + metadata).

        Parameters
        ----------
        path : str
            Output file path. ``.pt`` is appended if missing.

        Notes
        -----
        The checkpoint contains:
        - ``kwargs``   : JSON-safe constructor arguments
        - ``support``  : C51 support atoms tensor (saved on CPU)
        - ``q``        : online network state_dict
        - ``q_target`` : target network state_dict
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "support": self.support.detach().cpu(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the existing instance.

        Parameters
        ----------
        path : str
            Path to checkpoint saved by :meth:`save`. ``.pt`` is appended if missing.

        Raises
        ------
        ValueError
            If the checkpoint format is not recognized.

        Notes
        -----
        - This loads weights only and does not reconstruct the object.
        - The current instance must be compatible in architecture and shapes.
        - Target network is re-frozen and set to eval mode to enforce invariants.
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

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
            self.freeze_target(self.q_target)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly construction spec for this head.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint`` : module-level worker factory (pickle-friendly)
            - ``kwargs``     : JSON-safe constructor args (portable across workers)
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_rainbow_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
