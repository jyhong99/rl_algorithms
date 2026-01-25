from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_trpo_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build a TRPOHead instance on CPU.

    This function must be defined at module scope (not inside a class)
    because Ray (and your entrypoint resolver) typically require a globally
    importable symbol.

    Notes
    -----
    - `device` is forced to "cpu" on workers to avoid GPU contention and
      reduce overhead in remote rollout processes.
    - `activation_fn` is expected to be serialized (e.g., as "ReLU") and is
      resolved back to a torch.nn.Module class here.
    """
    # Copy kwargs to avoid mutating the caller's dictionary
    kwargs = dict(kwargs)

    # Force CPU in Ray worker processes
    kwargs["device"] = "cpu"

    # Resolve activation function name/string -> actual torch module/class
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Construct policy head on CPU
    head = TRPOHead(**kwargs).to("cpu")

    # Ray workers usually do inference/rollout only (no gradient updates)
    head.set_training(False)
    return head


# =============================================================================
# TRPOHead (refactored: BaseHead + On-policy continuous actor-critic)
# =============================================================================
class TRPOHead(OnPolicyContinuousActorCriticHead):
    """
    TRPO Network Container (Actor + Critic)
    ======================================
    Lightweight, config-free TRPO head for continuous control.

    This module is responsible for the *neural networks only*:
      - Actor  : Gaussian policy π(a|s) (unsquashed)
      - Critic : Value function V(s)

    Important
    ---------
    TRPO-specific optimization logic (e.g., conjugate gradient, KL constraints,
    line search) belongs to the *core/trainer*, not this head.
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
            Dimension of observation/state vector.
        action_dim : int
            Dimension of continuous action vector.
        hidden_sizes : Sequence[int]
            Hidden layer sizes for both actor and critic networks.
        activation_fn : Any
            Activation function class (e.g., nn.ReLU). Can be serialized by name for Ray.
        init_type : str
            Weight initialization policy (e.g., "orthogonal").
        gain : float
            Initialization gain multiplier.
        bias : float
            Bias initialization value.
        device : Union[str, torch.device]
            Torch device used by the head (actor + critic).
        log_std_mode : str
            Standard deviation parameterization mode for Gaussian policy
            (e.g., "param" for global learnable log_std).
        log_std_init : float
            Initial log standard deviation value for Gaussian policy.
        """
        # Base class stores device / training-mode utilities
        super().__init__(device=device)

        # ----- store architecture config -----
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        # ----- store common MLP init config -----
        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ----- Gaussian policy std config -----
        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---------------------------------------------------------------------
        # Actor network: Gaussian (unsquashed)
        # ---------------------------------------------------------------------
        # TRPO typically uses a Gaussian policy in action space without tanh squash.
        # Squashing changes the distribution and requires correction terms; TRPO
        # is usually implemented with unsquashed Gaussian for simplicity.
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,  # IMPORTANT: unsquashed Gaussian for TRPO-style policy
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic network: State value function V(s)
        # ---------------------------------------------------------------------
        # TRPO usually uses a separate baseline value function to reduce variance.
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Persistence utilities (PPOHead-style)
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments in a JSON-safe form.

        This enables:
        - Reconstructing the object later (e.g., from a checkpoint)
        - Sending head configs to Ray workers (must be serializable)
        - Debugging and experiment tracking (human-readable configs)
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
        Save a TRPOHead checkpoint to disk.

        Stored fields
        -------------
        - kwargs : JSON-safe config for reconstruction/debugging
        - actor  : actor state_dict
        - critic : critic state_dict

        Notes
        -----
        - This saves only the network weights and reconstruction metadata.
        - Optimizer state (TRPO core) is intentionally excluded.
        """
        # Enforce .pt extension for consistency
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
        Load checkpoint weights into the existing TRPOHead instance.

        Notes
        -----
        - Loads weights only (no object reconstruction).
        - Assumes this instance was created with compatible network shapes.
        - Uses map_location=self.device to support CPU/GPU portability.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)

        # Validate checkpoint format early for clearer error messages
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray PolicyFactorySpec so remote workers can reconstruct this head.

        Returns
        -------
        PolicyFactorySpec
            Includes:
            - entrypoint: module-level factory function reference
            - kwargs     : JSON-safe constructor arguments
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_trpo_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
