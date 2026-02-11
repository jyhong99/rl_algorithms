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
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_trpo_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Construct a :class:`TRPOHead` instance on a Ray worker (CPU-only).

    Ray commonly reconstructs policy modules inside remote worker processes using:
    - a **pickleable entrypoint** (must be a module-level symbol), and
    - a **JSON/pickle-safe kwargs payload**.

    This factory enforces the typical rollout-worker constraints:
    - **CPU-only** (workers should not depend on GPU availability),
    - **deterministic reconstruction** (kwargs are copied defensively),
    - **activation resolution** (serialized activation identifiers -> callable/class).

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments intended for :class:`TRPOHead`.

        Notes on serialization
        ----------------------
        - This payload should be JSON/pickle-safe.
        - In particular, ``activation_fn`` is expected to be a serialized
          representation (e.g., string name or ``None``), which is resolved via
          :func:`_resolve_activation_fn`.

    Returns
    -------
    nn.Module
        A :class:`TRPOHead` allocated on CPU and placed into inference mode via
        ``set_training(False)``.

    Notes
    -----
    - Rollout workers typically do **inference only** (action sampling). Training
      state (e.g., dropout, batchnorm) should be disabled.
    - The returned module is moved to CPU explicitly even if the driver uses GPU.
    """
    # Defensive copy to avoid mutating caller state (important for Ray object reuse).
    cfg = dict(kwargs)

    # Rollout workers should not depend on GPU availability.
    cfg["device"] = "cpu"

    # activation_fn may be serialized as string/None -> resolve back to callable/class.
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = TRPOHead(**cfg).to("cpu")

    # Workers usually perform inference only; disable training-specific behavior.
    head.set_training(False)
    return head


# =============================================================================
# TRPOHead (continuous-only)
# =============================================================================
class TRPOHead(OnPolicyContinuousActorCriticHead):
    """
    Actor-critic network container for TRPO (continuous action spaces).

    This class bundles the neural networks required by TRPO-style on-policy methods:

    - **Actor**: diagonal Gaussian policy :math:`\\pi(a\\mid s)` (unsquashed)
    - **Critic**: state-value baseline :math:`V(s)`

    The TRPO optimization logic (conjugate gradient, Fisher-vector products, KL
    constraint, and line search) is intentionally **not** implemented here and
    should live in the corresponding TRPO core/update engine.

    Design goals
    ------------
    - **Head-only** responsibility: define and host networks, device placement,
      and persistence metadata.
    - **Ray-friendly** reconstruction: export JSON-safe kwargs and provide a
      module-level worker factory for pickled entrypoints.

    Notes
    -----
    - The actor uses an **unsquashed** Gaussian policy (``squash=False``). This is
      the standard TRPO formulation. If you apply squashing (e.g., tanh), you must
      handle the induced distribution transformation (and log-prob corrections).
    - This head exports JSON-safe constructor kwargs for:
        * checkpoint metadata / reproducibility
        * Ray rollout-worker reconstruction via :class:`PolicyFactorySpec`

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation/state vector.
    action_dim : int
        Dimension of the continuous action vector.
    hidden_sizes : Sequence[int], default=(64, 64)
        Hidden layer sizes used by both actor and critic MLP builders.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function **class** (e.g., ``nn.ReLU``). For Ray/checkpointing,
        this is typically serialized/deserialized via a stable string identifier.
    init_type : str, default="orthogonal"
        Weight initialization scheme identifier (interpreted by your network builders).
    gain : float, default=1.0
        Optional initialization gain passed to network builders.
    bias : float, default=0.0
        Optional bias initialization constant passed to network builders.
    device : str or torch.device, default="cpu"
        Device on which the head's parameters are allocated.
    log_std_mode : str, default="param"
        Gaussian log-standard-deviation parameterization mode for the actor.
        Common choices:
          - ``"param"``: a learnable parameter vector (state-independent)
          - other modes depend on :class:`ContinuousPolicyNetwork`
    log_std_init : float, default=-0.5
        Initial value for the Gaussian log-std parameters when applicable.

    Raises
    ------
    ValueError
        If ``obs_dim`` or ``action_dim`` is not positive.
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
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Basic shape validation / configuration
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---------------------------------------------------------------------
        # Actor: diagonal Gaussian policy π(a|s) (unsquashed)
        # ---------------------------------------------------------------------
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: baseline state-value function V(s)
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
    # Persistence / JSON-safe kwargs export
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments as a JSON-safe dictionary.

        This payload is suitable for:
        - checkpoint metadata (reconstruction/debugging)
        - Ray reconstruction via :class:`PolicyFactorySpec`

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor kwargs. ``activation_fn`` is converted to a stable
            string identifier via ``_activation_to_name``.

        Notes
        -----
        - Keeping this method stable is important for backward/forward compatibility
          of checkpoints and Ray worker reconstruction.
        - ``device`` is included for transparency; Ray workers override it to "cpu".
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

    def save(self, path: str) -> None:
        """
        Save head weights and metadata to disk.

        Parameters
        ----------
        path : str
            Output path. The suffix ``.pt`` is appended if missing.

        Notes
        -----
        The checkpoint contains:
        - ``kwargs`` : JSON-safe constructor arguments
        - ``actor``  : actor ``state_dict``
        - ``critic`` : critic ``state_dict``

        This is a **head-only** checkpoint (weights + reconstruction metadata).
        Optimizer/scheduler state belongs to the algorithm/core and is intentionally
        excluded here.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the existing instance.

        Parameters
        ----------
        path : str
            Path to checkpoint saved by :meth:`save`. The suffix ``.pt`` is appended
            if missing.

        Raises
        ------
        ValueError
            If the checkpoint structure is not recognized.

        Notes
        -----
        - Loads weights only; does not reconstruct/resize the module.
        - The current instance must be architecture-compatible with the checkpoint.
        - Uses ``map_location=self.device`` for CPU/GPU portability.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-safe construction spec for this head.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint`` : module-level worker factory (pickle-friendly)
            - ``kwargs``     : JSON-safe constructor args (portable across workers)

        Notes
        -----
        - Ray requires the entrypoint to be a module-level symbol so it can be pickled.
        - Worker policies are typically CPU-only; the worker factory enforces that.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_trpo_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
