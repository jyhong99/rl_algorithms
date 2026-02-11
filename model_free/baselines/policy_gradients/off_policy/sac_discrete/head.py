from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.q_networks import DoubleQNetwork
from model_free.common.policies.base_head import OffPolicyDiscreteActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_discrete_sac_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
   Build a Discrete SAC head on a Ray rollout worker (CPU-only).

    Ray reconstructs policies in remote worker processes from a serialized
    (entrypoint, kwargs) specification. The entrypoint must be a module-level
    function so Ray can import it by dotted path.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments used to construct :class:`SACDiscreteHead`.
        These usually come from a JSON/pickle payload (e.g., ``PolicyFactorySpec``).

        Common keys include:
        - ``obs_dim`` : int
        - ``n_actions`` : int
        - ``actor_hidden_sizes`` : Sequence[int]
        - ``critic_hidden_sizes`` : Sequence[int]
        - ``activation_fn`` : str or callable
        - ``dueling_mode`` : bool
        - ``init_type`` : str
        - ``gain`` : float
        - ``bias`` : float
        - ``device`` : str or torch.device (ignored/overridden on workers)

    Returns
    -------
    nn.Module
        A fully constructed :class:`SACDiscreteHead` placed on CPU, with
        eval-like behavior enabled via ``set_training(False)``.

    Notes
    -----
    - Workers typically run inference/rollouts only, so forcing CPU avoids
      accidental GPU allocation and contention.
    - ``activation_fn`` may be serialized as a string; it is resolved here to
      a torch activation class (e.g., ``"relu" -> nn.ReLU``).
    """
    cfg = dict(kwargs)

    # Force CPU on worker side (rollout/inference).
    cfg["device"] = "cpu"

    # Resolve activation identifier (string/name) into nn.Module class.
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = SACDiscreteHead(**cfg).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# SACDiscreteHead
# =============================================================================
class SACDiscreteHead(OffPolicyDiscreteActorCriticHead):
    """
   Discrete Soft Actor-Critic head.

    This head encapsulates all neural modules required by a discrete-action SAC
    variant:

    - **Actor**: categorical policy :math:`\\pi(a\\mid s)` over ``n_actions``.
    - **Critic**: twin Q networks :math:`Q_1(s, \\cdot), Q_2(s, \\cdot)` producing
      action-value vectors of shape ``(B, n_actions)``.
    - **Target critic**: a lagged (Polyak/EMA) copy of the twin critics used to
      construct stable bootstrap targets in the core update.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    n_actions : int
        Number of discrete actions.
    actor_hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for the actor MLP.
    critic_hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for the critic MLP(s).
    activation_fn : Any, default=nn.ReLU
        Activation function class (e.g., ``nn.ReLU``). If you serialize this as a
        string, use :func:`model_free.common.utils.ray_utils._resolve_activation_fn`
        on reconstruction.
    dueling_mode : bool, default=False
        Whether the critic network uses a dueling architecture (if supported by
        your ``DoubleQNetwork`` implementation).
    init_type : str, default="orthogonal"
        Initialization scheme passed through to network modules.
    gain : float, default=1.0
        Initialization gain multiplier passed through to network modules.
    bias : float, default=0.0
        Initialization bias constant passed through to network modules.
    device : Union[str, torch.device], default="cpu"
        Device where the online networks are allocated.

    Attributes
    ----------
    actor : nn.Module
        Discrete policy network producing logits/probabilities over actions.
    critic : nn.Module
        Twin Q network that outputs (q1, q2) each shaped ``(B, n_actions)``.
    critic_target : nn.Module
        Frozen target copy of ``critic`` updated only via hard/soft updates.

    Notes
    -----
    - This head focuses on wiring networks, checkpoint I/O, and Ray policy factory
      integration. The SAC update rules (entropy temperature, Bellman targets,
      actor/critic losses) belong in the corresponding *core* module.
    - Target parameters are frozen (``requires_grad=False``) to prevent optimizers
      from accidentally updating them. They are still updated by explicit Polyak
      or hard-copy utilities.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__(device=device)

        # -----------------------------
        # Problem dimensions / config
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        self.actor_hidden_sizes = tuple(int(x) for x in actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(int(x) for x in critic_hidden_sizes)
        self.activation_fn = activation_fn

        self.dueling_mode = bool(dueling_mode)
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # -----------------------------
        # Actor: categorical policy π(a|s)
        # -----------------------------
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.actor_hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Critic: twin Q(s,·) heads
        # -----------------------------
        # For discrete SAC, critics typically map state -> Q-values over all actions:
        #   q1, q2 = critic(s) where each is (B, n_actions)
        self.critic = DoubleQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.critic_hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Target critic: frozen copy
        # -----------------------------
        self.critic_target = DoubleQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.critic_hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target weights from online critic and freeze target params.
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
       Export constructor kwargs in a JSON-safe representation.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable constructor arguments sufficient to reconstruct the
            head architecture. This is used for:
            - checkpoints (store config alongside weights)
            - Ray worker reconstruction (kwargs payload for PolicyFactorySpec)

        Notes
        -----
        - ``activation_fn`` is exported as a stable string name using the base-head
          helper ``_activation_to_name`` and resolved on load/worker construction.
        - ``device`` is stored as a string. Ray workers override device to CPU.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "actor_hidden_sizes": [int(x) for x in self.actor_hidden_sizes],
            "critic_hidden_sizes": [int(x) for x in self.critic_hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
       Save head weights and minimal configuration to a ``.pt`` checkpoint.

        Parameters
        ----------
        path : str
            Output checkpoint path. If it does not end with ``.pt``, the suffix is
            appended automatically.

        Notes
        -----
        The checkpoint payload is a ``dict`` containing:
        - ``kwargs``: JSON-safe constructor args (for reconstruction/debugging)
        - ``actor``: actor ``state_dict()``
        - ``critic``: critic ``state_dict()``
        - ``critic_target``: target critic ``state_dict()``
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
       Load head weights from a ``.pt`` checkpoint created by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to checkpoint file. If it does not end with ``.pt``, the suffix is
            appended automatically.

        Raises
        ------
        ValueError
            If the checkpoint payload does not match the expected format.

        Notes
        -----
        - Weights are loaded onto ``self.device`` using ``map_location``.
        - If ``critic_target`` weights are missing (older checkpoints), the target
          critic is reconstructed by hard-copying from the online critic.
        - The target critic is frozen after loading to prevent optimizer updates.
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized SACDiscreteHead checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        self.freeze_target(self.critic_target)
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
       Return a Ray-serializable policy factory specification.

        Returns
        -------
        PolicyFactorySpec
            A spec containing:
            - ``entrypoint``: importable module-level factory function
            - ``kwargs``: JSON-safe constructor arguments

        Notes
        -----
        Ray workers reconstruct the head by importing the entrypoint and calling it
        with the provided kwargs. On the worker side, the factory forces CPU and
        resolves ``activation_fn`` if it was serialized as a string.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_discrete_sac_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
