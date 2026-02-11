from __future__ import annotations

from typing import Any, Dict, Sequence, Union

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
# Ray worker factory (MUST be module-level)
# =============================================================================


def build_a2c_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build an :class:`A2CDiscreteHead` inside a Ray rollout worker (CPU-only).

    Ray pattern and rationale
    -------------------------
    Ray commonly reconstructs policies through a pickled entrypoint callable plus
    JSON-serializable kwargs. To make this robust across different machines and
    cluster setups:

    - Keep this factory at **module scope** (pickle-friendly).
    - Expect kwargs to be **JSON-safe** (strings/numbers/lists/dicts only).
    - Force the policy module onto **CPU** for rollout workers (GPU should remain
      optional and typically reserved for the learner).

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments for :class:`A2CDiscreteHead`. In Ray use,
        this dict is usually produced by :meth:`A2CDiscreteHead._export_kwargs_json_safe`.

        Notes
        -----
        - ``activation_fn`` is serialized as a string name (or ``None``) and must
          be resolved back to a callable activation class via :func:`_resolve_activation_fn`.
        - ``device`` is overwritten to ``"cpu"`` unconditionally.

    Returns
    -------
    torch.nn.Module
        An :class:`A2CDiscreteHead` instance on CPU with training disabled via
        :meth:`set_training(False)`.

    See Also
    --------
    A2CDiscreteHead.get_ray_policy_factory_spec
        Produces the (entrypoint, kwargs) pair used by Ray for worker reconstruction.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = A2CDiscreteHead(**cfg).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# A2C head (DISCRETE)
# =============================================================================


class A2CDiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    A2C network head for **discrete** action spaces.

    This module is a *network container* (actor + critic) used by on-policy
    algorithms such as A2C/PPO variants configured for categorical actions.

    Architecture
    ------------
    Actor
        Categorical policy :math:`\\pi(a\\mid s)` implemented by
        :class:`DiscretePolicyNetwork`, producing logits over ``n_actions``.

    Critic
        State-value baseline :math:`V(s)` implemented by :class:`StateValueNetwork`.

    Inherited interface
    -------------------
    The parent class :class:`OnPolicyDiscreteActorCriticHead` is expected to provide
    (or require) the following common API:

    - ``act(obs, deterministic=False)`` -> action indices, shape ``(B,)``
    - ``value_only(obs)`` -> value tensor, shape ``(B, 1)``
    - ``evaluate_actions(obs, action)`` -> mapping with keys:
        - ``"value"``    : ``(B, 1)``
        - ``"log_prob"`` : ``(B, 1)``
        - ``"entropy"``  : ``(B, 1)``

    Builder compatibility note
    --------------------------
    Many codebases use a unified builder that forwards a shared kwargs set for both
    continuous and discrete policies. To keep checkpoint/Ray reconstruction robust,
    this class is conservative about what it exports (JSON-safe only) and uses
    stable keys.

    Notes
    -----
    - This head stores enough constructor configuration to support:
        - checkpoint metadata
        - Ray worker reconstruction
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
            Hidden layer sizes for both actor and critic MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class (not an instance), e.g. ``nn.ReLU``.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier understood by your network builders.
        gain : float, default=1.0
            Optional initialization gain passed to network builders.
        bias : float, default=0.0
            Optional initialization bias passed to network builders.
        device : str | torch.device, default="cpu"
            Torch device on which this head will live.

        Raises
        ------
        ValueError
            If ``n_actions <= 0``.

        Notes
        -----
        The actor is expected to implement:

        - ``actor.get_dist(obs)`` -> Categorical-like distribution
        - ``dist.log_prob(action_idx)`` -> ``(B,)`` (or broadcastable to it)
        - ``dist.entropy()`` -> ``(B,)``
        """
        super().__init__(device=device)

        # -----------------------------
        # Store configuration
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.action_space = "discrete"

        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # -----------------------------
        # Actor: categorical policy π(a|s)
        # -----------------------------
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Critic: V(s)
        # -----------------------------
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Persistence / Ray
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
        Key naming
            This implementation exports:

            - ``n_actions`` under the key ``action_dim``

            If your broader codebase standardizes discrete sizing under ``n_actions``,
            consider changing this to export ``"n_actions": ...`` instead, and updating
            the builder/reconstruction path accordingly.

        Activation serialization
            ``activation_fn`` is exported by name using ``_activation_to_name`` and is
            resolved back to a callable via :func:`_resolve_activation_fn` in the worker
            factory.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_space": "discrete",
            # If your codebase standardizes `n_actions`, consider exporting "n_actions".
            "action_dim": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save a head checkpoint to disk.

        Parameters
        ----------
        path : str
            Output path. If ``.pt`` is missing, it is appended automatically.

        Saved contents
        --------------
        kwargs : dict
            JSON-safe constructor config (for reconstruction/debugging).
        actor : dict
            ``state_dict`` for the actor network.
        critic : dict
            ``state_dict`` for the critic network.

        Notes
        -----
        - This method stores weights + metadata.
        - Object reconstruction is not performed here; create a compatible instance
          and then call :meth:`load` to restore weights.
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
            If the checkpoint format does not match expectations.

        Notes
        -----
        - This loads weights only; it does not reconstruct the object.
        - Assumes the current instance was built with compatible shapes.
        - Uses ``map_location=self.device`` to support CPU/GPU portability.
        """
        ckpt_path = path if path.endswith(".pt") else (path + ".pt")
        ckpt = th.load(ckpt_path, map_location=self.device)

        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-safe construction spec (entrypoint + JSON-safe kwargs).

        Returns
        -------
        PolicyFactorySpec
            A lightweight spec containing:
            - ``entrypoint``: picklable reference to :func:`build_a2c_head_worker_policy`
            - ``kwargs``: JSON-safe constructor arguments for worker-side reconstruction

        Notes
        -----
        Worker-side reconstruction forces CPU and resolves ``activation_fn`` from name.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_a2c_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
