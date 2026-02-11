from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.policies.base_head import OnPolicyDiscreteActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================


def build_ppo_discrete_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a :class:`PPODiscreteHead` inside a Ray rollout worker (CPU-only).

    Ray pattern and rationale
    -------------------------
    Ray frequently reconstructs policy modules using:
    - a picklable *module-level* entrypoint callable, and
    - a JSON-serializable kwargs payload.

    This helper exists to make that reconstruction deterministic and portable.

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments for :class:`PPODiscreteHead`. In Ray usage,
        these kwargs are typically produced by :meth:`PPODiscreteHead._export_kwargs_json_safe`.

        Notes
        -----
        - ``device`` is overwritten to ``"cpu"`` unconditionally for worker safety.
        - ``activation_fn`` is expected to be a string name (or ``None``) and is
          resolved back to a callable via :func:`_resolve_activation_fn`.

    Returns
    -------
    torch.nn.Module
        A :class:`PPODiscreteHead` instance placed on CPU and set to evaluation-like
        mode via :meth:`set_training(False)`.

    See Also
    --------
    PPODiscreteHead.get_ray_policy_factory_spec
        Produces the Ray spec (entrypoint + kwargs) that points to this function.
    """
    cfg = dict(kwargs)

    # Rollout workers typically only require inference; keep them CPU-only.
    cfg["device"] = "cpu"

    # activation_fn is serialized (string/None) -> resolve to callable class (e.g., nn.ReLU)
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = PPODiscreteHead(**cfg).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# PPODiscreteHead
# =============================================================================


class PPODiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    PPO head for **discrete** action spaces (categorical actor + V(s) critic).

    This module is a *network container* used by a PPO-style update core.
    PPO's defining mechanics (ratio clipping, KL penalties, value clipping, etc.)
    belong to the **core**, not to this head.

    Architecture
    ------------
    Actor
        Categorical policy :math:`\\pi(a\\mid s)` implemented by
        :class:`DiscretePolicyNetwork`. The network is expected to provide
        ``get_dist(obs)`` returning a ``torch.distributions.Categorical``-like
        distribution over ``n_actions``.

    Critic
        State-value baseline :math:`V(s)` implemented by :class:`StateValueNetwork`.

    Inherited interface
    -------------------
    The parent class :class:`OnPolicyDiscreteActorCriticHead` is expected to provide
    (or require) an API similar to:

    - ``set_training(training: bool) -> None``
    - ``act(obs, deterministic=False) -> torch.Tensor`` (typically action indices)
    - ``evaluate_actions(obs, action, as_scalar=False) -> Dict[str, Any]``
    - ``value_only(obs) -> torch.Tensor`` (typically shape ``(B, 1)``)
    - persistence hooks (save/load)
    - Ray factory spec via :meth:`get_ray_policy_factory_spec`

    Shape conventions
    -----------------
    observations
        ``(obs_dim,)`` or ``(B, obs_dim)``.
    actions
        Discrete indices; typically ``LongTensor`` of shape ``(B,)`` (or a scalar).

    Notes
    -----
    - This head normalizes and stores ``device`` as a :class:`torch.device`.
      Prefer calling ``super().__init__(device=...)`` if your base class supports it.
      If your base class does **not** accept a device arg, this implementation is fine.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of the flattened observation vector.
        n_actions : int
            Number of discrete actions (size of the categorical distribution).
        hidden_sizes : Sequence[int], default=(64, 64)
            Hidden layer sizes used for both actor and critic MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class for MLP layers.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier understood by your network builders.
        gain : float, default=1.0
            Optional initialization gain passed to the network builders.
        bias : float, default=0.0
            Optional initialization bias passed to the network builders.
        device : str | torch.device, default="cpu"
            Target device for the networks and computation.

        Raises
        ------
        ValueError
            If ``n_actions <= 0``.
        """
        # Prefer: super().__init__(device=device) if your base supports it.
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # Normalize device
        self.device = th.device(device)

        # ---------------------------------------------------------------------
        # Actor: categorical policy π(a|s)
        # ---------------------------------------------------------------------
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: state-value baseline V(s)
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
    # Serialization helpers
    # =============================================================================
    def _activation_to_name_safe(self, activation_fn: Any) -> Optional[str]:
        """
        Convert an activation callable/class into a stable string identifier.

        Parameters
        ----------
        activation_fn : Any
            Activation function class (e.g., ``nn.ReLU``) or None.

        Returns
        -------
        Optional[str]
            Best-effort name for serialization. Returns None if activation_fn is None.

        Notes
        -----
        - JSON-safe export prefers a stable name so Ray workers can resolve it back
          using :func:`_resolve_activation_fn`.
        - If the activation is not a simple class with ``__name__``, we fall back
          to ``str(activation_fn)``.
        """
        if activation_fn is None:
            return None
        return getattr(activation_fn, "__name__", None) or str(activation_fn)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        This payload is intended for:
        - checkpoint metadata (reproducibility/debugging)
        - Ray worker reconstruction (PolicyFactorySpec kwargs)

        Returns
        -------
        Dict[str, Any]
            JSON-safe configuration dictionary.

        Notes
        -----
        ``activation_fn`` is exported as a string because callables are not reliably
        JSON-serializable and may not be stable across processes/machines.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name_safe(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def save(self, path: str) -> None:
        """
        Save a head checkpoint to disk.

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
        This checkpoint stores weights + metadata only. Optimizer/scheduler states
        belong to the core/algorithm.
        """
        out = path if path.endswith(".pt") else (path + ".pt")
        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            out,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the *existing* instance.

        Parameters
        ----------
        path : str
            Checkpoint path. If ``.pt`` is missing, it is appended automatically.

        Raises
        ------
        ValueError
            If the checkpoint does not contain required keys.

        Notes
        -----
        - Loads weights only; does not reconstruct the object.
        - Assumes the current instance was created with compatible shapes.
        - Uses ``map_location=self.device`` for CPU/GPU portability.
        """
        ckpt_path = path if path.endswith(".pt") else (path + ".pt")
        ckpt = th.load(ckpt_path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized PPODiscreteHead checkpoint format: {ckpt_path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-safe construction spec (entrypoint + JSON-safe kwargs).

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint``: picklable reference to
              :func:`build_ppo_discrete_head_worker_policy`
            - ``kwargs``: JSON-safe constructor arguments

        Notes
        -----
        The worker factory forces CPU and resolves the activation function name.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_ppo_discrete_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
