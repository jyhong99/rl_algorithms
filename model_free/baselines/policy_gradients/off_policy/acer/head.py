from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.q_networks import QNetwork, DoubleQNetwork
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)
from model_free.common.policies.base_head import OffPolicyDiscreteActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for entrypoint resolver)
# =============================================================================
def build_acer_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build an :class:`ACERHead` instance on the Ray worker side (CPU-only).

    This function exists to support Ray-style policy reconstruction, where remote
    workers rebuild a policy from a serialized "factory spec" consisting of:

    - an importable module-level entrypoint (this function)
    - a JSON-serializable ``kwargs`` payload

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`ACERHead`.

        Notes
        -----
        - ``device`` is forcibly overridden to ``"cpu"`` to avoid GPU contention
          or accidental device placement on worker processes.
        - ``activation_fn`` may arrive as ``None`` or a string name; it is resolved
          into a PyTorch activation constructor via ``_resolve_activation_fn``.

    Returns
    -------
    torch.nn.Module
        The constructed head on CPU. A best-effort call to
        ``head.set_training(False)`` is made to put the module into inference
        behavior (depending on the base head implementation).

    See Also
    --------
    ACERHead.get_ray_policy_factory_spec :
        Produces the corresponding Ray-friendly factory spec.
    """
    kwargs = dict(kwargs)

    # Force CPU on Ray worker side (avoid accidental GPU allocation).
    kwargs["device"] = "cpu"

    # activation_fn can be a name/string; resolve to actual nn.Module constructor.
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))

    head = ACERHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# ACERHead (config-free)
# =============================================================================
class ACERHead(OffPolicyDiscreteActorCriticHead):
    """
    Discrete ACER head: stochastic actor + Q critic + target critic.

    This head bundles the network components typically required for
    off-policy discrete actor-critic methods (including ACER-style training):

    - **Actor**: a categorical policy :math:`\\pi(a\\mid s)`
    - **Critic**: an action-value function :math:`Q(s,a)`
    - **Target critic**: a delayed copy :math:`Q_{\\text{targ}}(s,a)` used to
      stabilize bootstrapped targets

    The head is designed to be "algorithm-agnostic" at the network level:
    ACER-specific training (importance sampling, retrace, truncated IS, etc.)
    is expected to live in your algorithm/core update logic, while the head
    provides the neural building blocks and utilities.

    Parameters
    ----------
    obs_dim : int
        Observation (state) vector dimension :math:`d_s`.
    n_actions : int
        Number of discrete actions :math:`|\\mathcal{A}|`.
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer widths for both actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation constructor (e.g., ``nn.ReLU``) used in MLP blocks.

        Notes
        -----
        - This value may also be provided as a string name when reconstructed
          by Ray; in that case it should be resolved before instantiation via
          ``_resolve_activation_fn``.
    dueling_mode : bool, default=False
        If ``True`` and supported by your critic implementation, enables
        dueling Q architecture:

        .. math::
            Q(s,a) = V(s) + A(s,a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a'} A(s,a')

    double_q : bool, default=True
        If ``True``, uses a double Q critic (two independent Q estimators) which
        is commonly used to reduce overestimation bias in TD targets.
    init_type : str, default="orthogonal"
        Weight initialization strategy string forwarded to the underlying
        network constructors (actor/critic). The accepted values depend on
        your network implementations.
    gain : float, default=1.0
        Initialization gain forwarded to network constructors.
    bias : float, default=0.0
        Bias initialization value forwarded to network constructors.
    device : str or torch.device, default=("cuda" if available else "cpu")
        Device used for the online networks. The Ray worker factory will override
        this to CPU regardless of what is saved in a checkpoint.

    Attributes
    ----------
    actor : DiscretePolicyNetwork
        Categorical policy network producing action logits or probabilities
        depending on your implementation.
    critic : QNetwork or DoubleQNetwork
        Online critic network :math:`Q(s,a)`.
    critic_target : QNetwork or DoubleQNetwork
        Target critic network :math:`Q_{\\text{targ}}(s,a)`, synchronized from the
        online critic at initialization and then updated via hard/soft update.

    Notes
    -----
    Inherited utilities from ``OffPolicyDiscreteActorCriticHead`` (names may vary
    by your base implementation):

    - device management and tensor conversion helpers (e.g., batching)
    - ``set_training(training: bool)`` to toggle train/eval behavior
    - target-network helpers such as:
      ``hard_update(dst, src)``, ``freeze_target(module)``,
      and optionally ``hard_update_target()`` / ``soft_update_target(tau)``

    See Also
    --------
    build_acer_head_worker_policy :
        Ray worker-side entrypoint for reconstruction.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        double_q: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Store constructor args for introspection / persistence
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn

        self.dueling_mode = bool(dueling_mode)
        self.double_q = bool(double_q)

        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---------------------------------------------------------------------
        # Actor: discrete policy network π(a|s)
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
        # Critic: Q(s,a) + target Q_target(s,a)
        # ---------------------------------------------------------------------
        CriticCls = DoubleQNetwork if self.double_q else QNetwork

        self.critic = CriticCls(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.critic_target = CriticCls(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Initialize target critic from online critic, then freeze it.
        # ---------------------------------------------------------------------
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        This payload is intended to be safe for:
        - saving alongside checkpoints
        - sending across processes (e.g., Ray workers)

        Returns
        -------
        dict
            JSON-safe kwargs sufficient to reconstruct this head. Notably:

            - ``activation_fn`` is converted into a stable string name via the
              base helper ``_activation_to_name``.
            - ``device`` is stored as a string, but Ray worker reconstruction
              will override it to CPU.

        Notes
        -----
        The base head is expected to provide ``_activation_to_name``. If it does
        not, you should implement a stable mapping in this class.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "double_q": bool(self.double_q),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),  # overridden to "cpu" on Ray worker anyway
        }

    def save(self, path: str) -> None:
        """
        Save actor/critic/target parameters and minimal reconstruction kwargs.

        Parameters
        ----------
        path : str
            Output path. If ``.pt`` is not present, it is appended.

        Notes
        -----
        The checkpoint is stored via ``torch.save`` as a dict with keys:

        - ``"kwargs"``: JSON-safe constructor args (for reconstruction)
        - ``"actor"``: actor ``state_dict()``
        - ``"critic"``: critic ``state_dict()``
        - ``"critic_target"``: target critic ``state_dict()``

        This format is intentionally minimal and does not include optimizer state.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load actor/critic/target parameters from a checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path produced by :meth:`save`. If ``.pt`` is not present,
            it is appended.

        Raises
        ------
        ValueError
            If the checkpoint payload format is unrecognized.

        Notes
        -----
        - Loads onto ``self.device`` via ``map_location=self.device``.
        - If ``critic_target`` is absent in the checkpoint, the target network is
          refreshed from the online critic via ``hard_update_target()``.
        - After loading, the target critic is frozen and placed in eval mode to
          prevent optimizer updates and stabilize target computation.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.freeze_target(self.critic_target)
        else:
            # Base head is expected to implement this convenience method.
            self.hard_update_target()

        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-friendly factory spec for policy reconstruction.

        Returns
        -------
        PolicyFactorySpec
            A factory specification containing:

            - ``entrypoint``: importable module-level entrypoint for Ray workers
            - ``kwargs``: JSON-safe constructor args (see ``_export_kwargs_json_safe``)

        Notes
        -----
        - The entrypoint **must** be module-level for Ray to import it in a
          separate process.
        - The kwargs must be JSON-safe because Ray typically serializes this
          payload for worker creation.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_acer_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
