from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DeterministicPolicyNetwork
from model_free.common.networks.value_networks import StateActionValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import DeterministicActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_ddpg_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build DDPGHead on CPU.

    Notes
    -----
    - Ray entrypoint resolvers typically require the factory function to be defined
      at module scope (not inside a class).
    - Forces device="cpu" to avoid GPU allocation on rollout workers.
    - Resolves activation_fn from a string/name into an actual nn.Module class.
    - Converts action_low/action_high from list -> numpy for safety/consistency.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"

    # Resolve activation function identifier (e.g., "relu") into nn.ReLU, etc.
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Convert bounds into numpy arrays (common when kwargs come from JSON).
    if kwargs.get("action_low", None) is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    if kwargs.get("action_high", None) is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)

    head = DDPGHead(**kwargs).to("cpu")

    # Workers are used only for inference/rollout, so keep it in eval mode.
    head.set_training(False)
    return head


# =============================================================================
# DDPGHead
# =============================================================================
class DDPGHead(DeterministicActorCriticHead):
    """
    DDPG Head (Actor + Critic + Target Actor/Critic)

    Responsibilities
    ---------------
    - Constructs the actor/critic networks and their target copies.
    - Handles model persistence (save/load).
    - Provides Ray integration via a PolicyFactorySpec.

    Notes
    -----
    - The *behavioral logic* (act(), q_value(), q_value_target(), target updates, etc.)
      is expected to be implemented in the parent class DeterministicActorCriticHead.
    - This class focuses on architecture wiring and I/O.
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
        # Base head stores device, common utilities, and update helpers.
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # Store bounds as numpy arrays for stable serialization and shape checking.
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32).reshape(-1)

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

        # Optional exploration noise object (may be used by head.act()).
        self.noise = noise
        self.noise_clip = None if noise_clip is None else float(noise_clip)

        # ---------------------------------------------------------------------
        # Actor network: deterministic policy π(s) -> a
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
        # Critic network: Q(s, a) -> scalar
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

        # Initialize targets to match online networks (hard copy), then freeze them.
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-friendly format.

        Notes
        -----
        - activation_fn is exported as a string (via _activation_to_name) for portability.
        - action bounds are exported as Python lists.
        - noise/noise_clip are intentionally excluded by default, because they are often
          runtime-only and may not be serializable.
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
            # NOTE: noise/noise_clip are intentionally NOT serialized by default.
        }

    def save(self, path: str) -> None:
        """
        Save head weights and minimal config into a .pt file.

        Stored payload
        --------------
        - kwargs: JSON-safe constructor kwargs (for reference/debugging)
        - actor, critic, actor_target, critic_target: state_dicts
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
        Load head weights from a saved .pt checkpoint.

        Behavior
        --------
        - Restores actor/critic weights.
        - Restores target weights if present, otherwise hard-copies from online nets.
        - Freezes targets after loading.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized DDPGHead checkpoint format at: {path}")

        # Restore main networks.
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Restore targets if available, else reconstruct from online nets.
        if ckpt.get("actor_target", None) is not None:
            self.actor_target.load_state_dict(ckpt["actor_target"])
        else:
            self.hard_update(self.actor_target, self.actor)

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        # Ensure targets are frozen (no gradients).
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-serializable spec to build this head on workers.

        Notes
        -----
        - entrypoint must be a module-level function.
        - kwargs must be JSON-safe.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_ddpg_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
