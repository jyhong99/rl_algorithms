from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DeterministicPolicyNetwork
from model_free.common.networks.value_networks import StateActionValueNetwork
from model_free.common.policies.base_head import DeterministicActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_ddpg_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a :class:`DDPGHead` instance on a Ray worker (CPU-only).

    Ray often reconstructs policies in remote worker processes from a serialized
    "factory spec" consisting of:

    - an importable *module-level* entrypoint (this function)
    - a JSON-serializable ``kwargs`` payload

    This function enforces worker-safe defaults and repairs JSON artifacts
    (e.g., lists instead of numpy arrays).

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`DDPGHead`.

        Notes
        -----
        - ``device`` is forcibly overridden to ``"cpu"`` to prevent GPU allocation
          on rollout workers.
        - ``activation_fn`` may arrive as a string name; it is resolved into an
          actual activation constructor via :func:`_resolve_activation_fn`.
        - ``action_low`` and ``action_high`` may arrive as Python lists; they are
          converted into ``np.ndarray`` for stable downstream shape checks.

    Returns
    -------
    torch.nn.Module
        Constructed :class:`DDPGHead` placed on CPU and set to inference mode via
        ``set_training(False)`` (best-effort; depends on base class implementation).

    See Also
    --------
    DDPGHead.get_ray_policy_factory_spec :
        Produces the factory spec that references this entrypoint.
    """
    kwargs = dict(kwargs)

    # Force CPU on Ray worker side (avoid accidental GPU allocation).
    kwargs["device"] = "cpu"

    # Resolve activation function identifier (e.g., "relu") into nn.ReLU, etc.
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))

    # Convert bounds into numpy arrays (common when kwargs come from JSON).
    if kwargs.get("action_low", None) is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    if kwargs.get("action_high", None) is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)

    head = DDPGHead(**kwargs).to("cpu")

    # Rollout workers typically require inference-only behavior.
    head.set_training(False)
    return head


# =============================================================================
# DDPGHead
# =============================================================================
class DDPGHead(DeterministicActorCriticHead):
    """
    DDPG head: deterministic actor + critic + target actor/critic.

    This head wires the neural networks used by deterministic off-policy methods
    such as DDPG (and, structurally, also TD3/SAC-style variants with different
    cores). It focuses on *architecture construction* and *persistence*, while
    the parent :class:`~model_free.common.policies.base_head.DeterministicActorCriticHead`
    is expected to implement the behavior-facing utilities (acting, target updates,
    device helpers, etc.).

    Components
    ----------
    - **Actor**:
      Deterministic policy :math:`a = \\pi(s)`.

    - **Critic**:
      State-action value function :math:`Q(s,a)` returning a scalar.

    - **Target networks**:
      Slow-moving copies :math:`\\pi_{\\text{targ}}` and :math:`Q_{\\text{targ}}`
      used to compute stable bootstrap targets.

    Parameters
    ----------
    obs_dim : int
        Observation (state) vector dimension.
    action_dim : int
        Action vector dimension.
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer widths used for both actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation constructor (e.g., ``nn.ReLU``) used in MLP blocks.

        Notes
        -----
        - When reconstructed via Ray, this may be provided as a string name and
          should be resolved by the worker factory.
    init_type : str, default="orthogonal"
        Initialization scheme string forwarded to the network constructors.
    gain : float, default=1.0
        Initialization gain forwarded to the network constructors.
    bias : float, default=0.0
        Bias initialization forwarded to the network constructors.
    device : str or torch.device, default=("cuda" if available else "cpu")
        Device where the online networks are placed.

    action_low : np.ndarray or Sequence[float], optional
        Lower action bound. If provided, must be shape ``(action_dim,)``.
        Passed to the policy network to enforce output squashing/clipping.
    action_high : np.ndarray or Sequence[float], optional
        Upper action bound. If provided, must be shape ``(action_dim,)``.

    noise : Any, optional
        Optional exploration noise object. This head stores it, but the application
        of noise is expected to happen in the parent head's ``act()`` logic or in
        the algorithm wrapper.
    noise_clip : float, optional
        Optional clipping threshold for action noise (if used by your act logic).

    Attributes
    ----------
    actor : DeterministicPolicyNetwork
        Online actor network.
    critic : StateActionValueNetwork
        Online critic network.
    actor_target : DeterministicPolicyNetwork
        Target actor network (frozen; updated by hard/soft update).
    critic_target : StateActionValueNetwork
        Target critic network (frozen; updated by hard/soft update).
    action_low : np.ndarray or None
        Stored lower bound as a 1D float32 numpy array.
    action_high : np.ndarray or None
        Stored upper bound as a 1D float32 numpy array.

    Notes
    -----
    Target networks are initialized via a hard copy from the online networks and
    then frozen to ensure they are updated only via explicit target-update rules.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
        action_low: Optional[Union[np.ndarray, Sequence[float]]] = None,
        action_high: Optional[Union[np.ndarray, Sequence[float]]] = None,
        noise: Optional[Any] = None,
        noise_clip: Optional[float] = None,
    ) -> None:
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Store constructor args (useful for introspection / export)
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # Store bounds as 1D float32 numpy arrays for stable serialization.
        self.action_low = (
            None if action_low is None else np.asarray(action_low, dtype=np.float32).reshape(-1)
        )
        self.action_high = (
            None if action_high is None else np.asarray(action_high, dtype=np.float32).reshape(-1)
        )

        # If one bound is provided, the other must also be provided.
        if (self.action_low is None) ^ (self.action_high is None):
            raise ValueError("action_low and action_high must be provided together, or both be None.")

        # Validate bound shapes against action_dim.
        if self.action_low is not None:
            if self.action_low.shape[0] != self.action_dim or self.action_high.shape[0] != self.action_dim:
                raise ValueError(
                    f"action_low/high must have shape ({self.action_dim},), "
                    f"got {self.action_low.shape}, {self.action_high.shape}"
                )

        # Optional exploration noise object (used by act() if your base head supports it).
        self.noise = noise
        self.noise_clip = None if noise_clip is None else float(noise_clip)

        # ---------------------------------------------------------------------
        # Actor: deterministic policy π(s) -> a
        # ---------------------------------------------------------------------
        self.actor = DeterministicPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            action_low=self.action_low,
            action_high=self.action_high,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: Q(s,a) -> scalar
        # ---------------------------------------------------------------------
        self.critic = StateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Target networks: slow-moving copies used to stabilize learning
        # ---------------------------------------------------------------------
        self.actor_target = DeterministicPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            action_low=self.action_low,
            action_high=self.action_high,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.critic_target = StateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize targets to match online networks, then freeze them.
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable format.

        The exported mapping is intended for:
        - checkpoint metadata (debugging / reconstruction reference)
        - Ray worker policy reconstruction (kwargs serialization)

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor arguments.

            Included keys
            -------------
            - obs_dim, action_dim, hidden_sizes
            - activation_fn (as a stable string name)
            - init_type, gain, bias
            - device (stored as string; Ray workers override to CPU)
            - action_low, action_high (as Python lists or None)

        Notes
        -----
        - ``noise`` and ``noise_clip`` are intentionally excluded by default since
          they are often runtime-only and may not be serializable. If you need to
          persist them, extend this payload explicitly.
        """
        low = None if self.action_low is None else [float(x) for x in self.action_low.reshape(-1)]
        high = None if self.action_high is None else [float(x) for x in self.action_high.reshape(-1)]
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "action_low": low,
            "action_high": high,
        }

    def save(self, path: str) -> None:
        """
        Save head parameters and JSON-safe metadata to a ``.pt`` checkpoint.

        Parameters
        ----------
        path : str
            Output checkpoint path. If the path does not end with ``.pt``,
            the suffix is appended.

        Stored Payload
        --------------
        The checkpoint is stored via ``torch.save`` as a dict with keys:

        - ``"kwargs"`` : JSON-safe constructor metadata (see ``_export_kwargs_json_safe``)
        - ``"actor"`` : ``actor.state_dict()``
        - ``"critic"`` : ``critic.state_dict()``
        - ``"actor_target"`` : ``actor_target.state_dict()``
        - ``"critic_target"`` : ``critic_target.state_dict()``

        Notes
        -----
        Optimizer state is not stored here; that is typically the responsibility of
        the algorithm/core checkpointing layer.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load head parameters from a ``.pt`` checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path produced by :meth:`save`. If the path does not end
            with ``.pt``, the suffix is appended.

        Raises
        ------
        ValueError
            If the checkpoint format is not recognized.

        Notes
        -----
        - Loads tensors onto ``self.device`` using ``map_location=self.device``.
        - If target network weights are missing, targets are reconstructed via a
          hard copy from online networks.
        - Target networks are frozen after loading to ensure they are not updated
          by optimizers.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized DDPGHead checkpoint format at: {path}")

        # Restore online networks.
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Restore targets if available; otherwise reconstruct from online nets.
        if ckpt.get("actor_target", None) is not None:
            self.actor_target.load_state_dict(ckpt["actor_target"])
        else:
            self.hard_update(self.actor_target, self.actor)

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        # Ensure targets are frozen.
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-friendly policy factory specification.

        Returns
        -------
        PolicyFactorySpec
            Factory spec containing:

            - ``entrypoint`` : module-level function used by Ray workers to rebuild the head
            - ``kwargs`` : JSON-safe kwargs required for reconstruction

        Notes
        -----
        - Ray requires the entrypoint to be importable by name from the worker process,
          hence it must be defined at module scope.
        - The worker-side entrypoint overrides ``device`` to CPU.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_ddpg_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
