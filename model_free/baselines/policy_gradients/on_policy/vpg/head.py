from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_vpg_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: construct a VPGHead instance on CPU.

    This function MUST be defined at module scope because Ray (and your entrypoint
    resolver) typically require an importable top-level function.

    Notes
    -----
    - `device` is forced to "cpu" on workers:
        Rollout workers should generally run inference only, and keeping them on
        CPU avoids GPU contention and reduces serialization/transfer overhead.
    - `activation_fn` is expected to be serialized (e.g., "ReLU") and is resolved
      back into an actual torch.nn.Module class here.
    """
    # Copy to avoid mutating the caller's dictionary
    kwargs = dict(kwargs)

    # Force CPU for worker-side policy instantiation
    kwargs["device"] = "cpu"

    # Convert serialized activation function name/string -> torch module/class
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Build head and ensure it is in inference mode on workers
    head = VPGHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


class VPGHead(OnPolicyContinuousActorCriticHead):
    """
    VPG head for continuous actions (Gaussian policy + optional value baseline).

    This head provides the neural components for:
      - Actor  : Gaussian policy π(a|s) (unsquashed)
      - Critic : Optional baseline V(s) for variance reduction

    Modes
    -----
    - use_baseline=False:
        REINFORCE-style training (no critic).
        Advantages are typically computed directly from returns (A = R).
    - use_baseline=True:
        Vanilla Policy Gradient with a value baseline (recommended).
        Advantages are computed as A = R - V(s) (or via GAE upstream).

    API Assumptions
    ---------------
    This class inherits from OnPolicyContinuousActorCriticHead, which is expected
    to provide:
      - act(obs, deterministic=False)
      - evaluate_actions(obs, act)
      - value_only(obs)
    `value_only()` is only meaningful when use_baseline=True.
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
        # Gaussian policy parameters
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
        # Baseline toggle
        use_baseline: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation/state dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int]
            Hidden layer sizes for actor (and critic if enabled).
        activation_fn : Any
            Activation function class (e.g., nn.ReLU). May be serialized for Ray.
        init_type : str
            Weight initialization method.
        gain : float
            Initialization gain multiplier.
        bias : float
            Bias initialization value.
        device : Union[str, torch.device]
            Torch device for actor/critic.
        log_std_mode : str
            Gaussian log-std parameterization mode (e.g., "param").
        log_std_init : float
            Initial log standard deviation.
        use_baseline : bool
            Whether to build a value network baseline V(s).
        """
        # Base class stores device and training/inference helpers
        super().__init__(device=device)

        # ----- core dimensions -----
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        # ----- shared MLP config -----
        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ----- Gaussian policy std config -----
        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ----- baseline toggle -----
        self.use_baseline = bool(use_baseline)

        # ---------------------------------------------------------------------
        # Actor: unsquashed diagonal Gaussian policy
        # ---------------------------------------------------------------------
        # VPG/TRPO often uses an unsquashed Gaussian policy for continuous control.
        # (If squashing is used, log_prob correction must be handled carefully.)
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,  # IMPORTANT: unsquashed Gaussian in action space
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Optional critic: state-value baseline V(s)
        # ---------------------------------------------------------------------
        # Baseline reduces variance in the policy gradient estimate.
        # If disabled, downstream code must not expect critic/value outputs.
        self.critic: Optional[StateValueNetwork] = None
        if self.use_baseline:
            self.critic = StateValueNetwork(
                state_dim=self.obs_dim,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

    # ------------------------------------------------------------------
    # Persistence + Ray spec
    # ------------------------------------------------------------------
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments in a JSON-safe form.

        This is used for:
        - Checkpoint metadata (reconstruction/debugging)
        - Ray worker instantiation (kwargs must be serializable)
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
            "use_baseline": bool(self.use_baseline),
        }

    def save(self, path: str) -> None:
        """
        Save checkpoint to disk.

        Stored payload
        --------------
        - kwargs : JSON-safe config used to build this head (for reproducibility/debugging)
        - actor  : actor state_dict
        - critic : critic state_dict or None (if baseline disabled)

        Notes
        -----
        - Optimizer state belongs to the core/algorithm, not the head.
        - Adds ".pt" suffix if missing.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": None if self.critic is None else self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint from disk into this instance.

        Notes
        -----
        - Loads weights only; does not reconstruct a new object.
        - Uses map_location=self.device for CPU/GPU portability.
        - Enforces baseline compatibility:
            * baseline OFF instance cannot load baseline ON checkpoint (critic mismatch)
            * baseline ON  instance cannot load baseline OFF checkpoint
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        if "actor" not in ckpt:
            raise ValueError("Unrecognized VPGDiscreteHead checkpoint payload (missing actor).")

        # Always load actor weights
        self.actor.load_state_dict(ckpt["actor"])

        ckpt_critic = ckpt.get("critic", None)

        # Baseline compatibility checks
        if self.critic is None:
            # current instance baseline OFF
            if ckpt_critic is not None:
                raise ValueError("Checkpoint contains critic weights but this VPGDiscreteHead has use_baseline=False.")
        else:
            # current instance baseline ON
            if ckpt_critic is None:
                raise ValueError("Checkpoint has no critic weights but this VPGDiscreteHead has use_baseline=True.")
            self.critic.load_state_dict(ckpt_critic)

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly factory specification for constructing this head on workers.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_vpg_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
