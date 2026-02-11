from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================


def build_acktr_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build an :class:`ACKTRHead` inside a Ray rollout worker (CPU-only).

    Ray pattern and rationale
    -------------------------
    Ray typically reconstructs policy modules through a pickled "entrypoint"
    callable plus JSON-serializable kwargs. The most robust pattern is:

    - Keep the factory at **module scope** (pickle-friendly).
    - Pass **JSON-safe kwargs** (strings, numbers, lists, dicts).
    - Force the resulting module onto **CPU** for rollout workers, so remote
      inference does not depend on GPUs or CUDA drivers.

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments for :class:`ACKTRHead`. In Ray settings,
        these kwargs are usually produced by :meth:`ACKTRHead._export_kwargs_json_safe`.

        Notes
        -----
        - ``activation_fn`` is stored as a string name (or ``None``) and is resolved
          here via :func:`_resolve_activation_fn`.
        - ``device`` is overwritten to ``"cpu"`` unconditionally.

    Returns
    -------
    torch.nn.Module
        An :class:`ACKTRHead` instance on CPU with training disabled via
        :meth:`set_training(False)`.

    See Also
    --------
    ACKTRHead.get_ray_policy_factory_spec
        Returns the entrypoint+kwargs pair used by Ray for worker reconstruction.
    """
    cfg = dict(kwargs)

    # Always build worker-side policy on CPU for portability.
    cfg["device"] = "cpu"

    # Resolve activation function from serialized representation (e.g., "relu").
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = ACKTRHead(**cfg).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# ACKTRHead (standard on-policy actor-critic container)
# =============================================================================


class ACKTRHead(OnPolicyContinuousActorCriticHead):
    """
    ACKTR network head for **continuous** action spaces (Gaussian actor + V(s) critic).

    This module is a *network container* (actor + critic) intended to be used by an
    ACKTR-style algorithm. Importantly, ACKTR's distinguishing feature (K-FAC /
    natural-gradient approximation) lives in the **optimizer/core**, not in the
    network architecture itself.

    Architecture
    ------------
    Actor
        Diagonal Gaussian policy :math:`\\pi(a\\mid s)` implemented by
        :class:`ContinuousPolicyNetwork`.

        Notes
        -----
        - Uses ``squash=False`` to produce an **unsquashed** Gaussian.
        - Action bounding (tanh/clipping) can be handled by the environment or a wrapper.

    Critic
        State-value baseline :math:`V(s)` implemented by :class:`StateValueNetwork`.

    Inherited interface
    -------------------
    The parent class :class:`OnPolicyContinuousActorCriticHead` is expected to provide
    (or require) the following API:

    - ``set_training(training: bool) -> None``
    - ``act(obs, deterministic=False) -> torch.Tensor``
    - ``evaluate_actions(obs, action, as_scalar=False) -> Dict[str, Any]``
    - ``value_only(obs) -> torch.Tensor`` with shape ``(B, 1)``
    - persistence hooks (save/load) and Ray factory spec methods

    Notes
    -----
    Returned tensors generally follow the base head conventions:

    - value    : ``(B, 1)``
    - log_prob : ``(B, 1)`` or ``(B, action_dim)`` depending on distribution impl
    - entropy  : ``(B, 1)`` or ``(B, action_dim)``
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of the flattened observation vector.
        action_dim : int
            Dimension of the continuous action vector.
        hidden_sizes : Sequence[int], default=(64, 64)
            Hidden layer sizes used for both actor and critic MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class (not an instance), e.g. ``nn.ReLU``.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier understood by your network builders.
        gain : float, default=1.0
            Optional initialization gain passed to network builders.
        bias : float, default=0.0
            Optional initialization bias passed to network builders.
        device : str | torch.device, default="cpu"
            Torch device on which this head will live (e.g., ``"cpu"``, ``"cuda:0"``).
        log_std_mode : str, default="param"
            Log-standard-deviation parameterization mode for the Gaussian actor.

            Common options (implementation-dependent)
            - ``"param"``: trainable, state-independent log-std vector.
            - other modes may exist in your codebase.
        log_std_init : float, default=-0.5
            Initial log-std value used by the Gaussian actor when applicable.

        Notes
        -----
        The constructor stores a JSON-safe subset of arguments for checkpoint metadata
        and Ray reconstruction. Non-JSON values like ``activation_fn`` are serialized
        by name.
        """
        super().__init__(device=device)

        # ---- Store configuration for checkpoint/Ray reconstruction
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---------------------------------------------------------------------
        # Actor: unsquashed diagonal Gaussian policy π(a|s)
        # ---------------------------------------------------------------------
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,  # ACKTR/A2C/PPO-style commonly uses an unsquashed Gaussian
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: state-value function V(s)
        # ---------------------------------------------------------------------
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Persistence (PPOHead-style checkpointing)
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        This payload is intended for:
        - checkpoint metadata (reproducibility/debugging)
        - Ray worker reconstruction (kwargs must be JSON-safe)

        Returns
        -------
        Dict[str, Any]
            JSON-safe configuration dictionary.

        Notes
        -----
        - ``activation_fn`` is exported by name using ``_activation_to_name`` and must
          be resolved back to a callable via :func:`_resolve_activation_fn` when
          reconstructing.
        - ``device`` is stored as a string for portability.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "log_std_mode": str(self.log_std_mode),
            "log_std_init": float(self.log_std_init),
        }

    def _state_payload(self) -> Dict[str, Any]:
        """
        Build the checkpoint payload for :func:`torch.save`.

        Returns
        -------
        Dict[str, Any]
            Checkpoint payload containing:

            kwargs : dict
                JSON-safe constructor configuration.
            actor : dict
                ``state_dict`` for the actor network.
            critic : dict
                ``state_dict`` for the critic network.

        Notes
        -----
        This method exists to keep save/load logic centralized and consistent with
        other on-policy heads (e.g., PPO-style checkpointing).
        """
        return {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def _load_state_payload(self, ckpt: Dict[str, Any]) -> None:
        """
        Load weights from an in-memory checkpoint payload into this instance.

        Parameters
        ----------
        ckpt : Dict[str, Any]
            A checkpoint payload as produced by :meth:`_state_payload`.

        Raises
        ------
        ValueError
            If required keys are missing.

        Notes
        -----
        - This does **not** reconstruct the module; it only loads parameters.
        - The existing instance must have compatible shapes.
        """
        if "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError("Unrecognized ACKTRHead checkpoint payload (missing actor/critic).")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def save(self, path: str) -> None:
        """
        Save a checkpoint to disk.

        Parameters
        ----------
        path : str
            Output file path. If ``.pt`` is missing, it is appended automatically.

        Saved contents
        --------------
        - ``kwargs`` : JSON-safe constructor config (for reconstruction/debugging)
        - ``actor``  : actor ``state_dict``
        - ``critic`` : critic ``state_dict``

        Notes
        -----
        The checkpoint contains weights and metadata only. It does not store an
        optimizer state (that belongs to the core/algorithm).
        """
        out = path if path.endswith(".pt") else (path + ".pt")
        th.save(self._state_payload(), out)

    def load(self, path: str) -> None:
        """
        Load a checkpoint from disk into the *existing* instance.

        Parameters
        ----------
        path : str
            Checkpoint path. If ``.pt`` is missing, it is appended automatically.

        Raises
        ------
        ValueError
            If the checkpoint is not a dictionary payload.

        Notes
        -----
        - Loads weights only; does not reconstruct the object.
        - Uses ``map_location=self.device`` for CPU/GPU portability.
        """
        ckpt_path = path if path.endswith(".pt") else (path + ".pt")
        ckpt = th.load(ckpt_path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

        self._load_state_payload(ckpt)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-safe construction spec (entrypoint + JSON-safe kwargs).

        Returns
        -------
        PolicyFactorySpec
            A lightweight spec containing:
            - ``entrypoint``: picklable reference to :func:`build_acktr_head_worker_policy`
            - ``kwargs``: JSON-safe constructor args for worker-side reconstruction

        Notes
        -----
        The worker factory forces CPU and resolves the activation function name.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_acktr_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
