from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    make_entrypoint,
    resolve_activation_fn,
)
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_a2c_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build `A2CHead` on CPU.

    Why this exists
    ---------------
    Ray workers typically construct policies via pickled entrypoints. The safest
    pattern is:
      - keep the factory function at module scope (pickle-friendly)
      - pass JSON-safe kwargs
      - force worker policies onto CPU for portability

    Behavior
    --------
    - Forces `device="cpu"` to avoid GPU dependencies inside rollout workers.
    - `activation_fn` is serialized as a string (or None), so we resolve it here.
    - Sets the returned head to evaluation mode via `set_training(False)`.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = A2CHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# A2C (CONTINUOUS ONLY)
# =============================================================================
class A2CHead(OnPolicyContinuousActorCriticHead):
    """
    A2C network container for CONTINUOUS action spaces.

    Components
    ----------
    Actor
      - `ContinuousPolicyNetwork`: Gaussian policy (diagonal covariance).
      - Uses `squash=False` to produce an *unsquashed* Gaussian policy, which is the
        most common setup for A2C/PPO-style methods (action bounding can be handled
        by environment wrappers if needed).

    Critic
      - `StateValueNetwork`: state-value baseline V(s).

    Inherited contract
    ------------------
    - act(obs, deterministic=False) -> action tensor
    - value_only(obs) -> value tensor (B, 1)
    - evaluate_actions(obs, action) -> dict with:
        value    : (B, 1)
        log_prob : (B, 1) or (B, action_dim) depending on distribution implementation
        entropy  : (B, 1) or (B, action_dim)

    Notes
    -----
    This class only defines networks and basic persistence. The actual update rule
    (A2C loss, GAE, batching, etc.) is handled by the algorithm/core.
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
        # Gaussian std parameterization
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int]
            MLP hidden layer sizes for both actor and critic.
        activation_fn : Any
            Activation function class (e.g., nn.ReLU).
        init_type : str
            Weight initialization scheme name (as supported by your network builders).
        gain : float
            Optional initialization gain.
        bias : float
            Optional initialization bias.
        device : str | torch.device
            Torch device for this head.
        log_std_mode : str
            Log-std parameterization mode for the Gaussian policy (e.g., "param").
        log_std_init : float
            Initial log-std value.
        """
        super().__init__(device=device)

        # Store configuration (useful for save/load and Ray reconstruction)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # -----------------------------
        # Actor: Gaussian (unsquashed)
        # -----------------------------
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,  # unsquashed Gaussian policy
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
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
        Export constructor arguments in a JSON-safe format.

        Used for
        --------
        - Reproducible checkpoint metadata (save/load)
        - Ray worker reconstruction (kwargs must be serializable)
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_space": "continuous",
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
        Save head checkpoint to disk.

        Stored fields
        -------------
        - kwargs : JSON-safe constructor config (for reconstruction/debugging)
        - actor  : actor state_dict
        - critic : critic state_dict
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
        Load checkpoint weights into the *existing* instance.

        Notes
        -----
        - This only loads weights; it does NOT reconstruct the object.
        - Assumes the current instance was created with compatible shapes.
        """
        if not path.endswith(".pt"):
            path += ".pt"
        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-safe construction spec (entrypoint + JSON-safe kwargs).
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_a2c_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
