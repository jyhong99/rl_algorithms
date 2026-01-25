# head.py
from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.q_networks import DoubleQNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OffPolicyDiscreteActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_discrete_sac_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build a Discrete SAC head on CPU.

    Why this exists
    ---------------
    In your Ray setup, workers reconstruct the policy from a (entrypoint, kwargs)
    spec. The entrypoint must be a module-level function so it can be resolved by
    import path (pickle-safe).

    Behavior
    --------
    - Forces `device="cpu"` for workers. (Typical: inference/rollout on CPU.)
    - Resolves `activation_fn` from a JSON-safe representation (e.g., "ReLU")
      into an actual torch activation class/function via `resolve_activation_fn`.
    - Returns a head with `training=False` to ensure deterministic eval-mode
      behavior for rollout collection.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = SACDiscreteHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


class SACDiscreteHead(OffPolicyDiscreteActorCriticHead):
    """
    Discrete SAC Head (Actor + Twin Critic + Target Twin Critic)

    High-level role
    ---------------
    This "head" owns the neural networks used by Discrete SAC:
      - Actor: categorical policy π(a|s) over `n_actions`
      - Critic: twin Q-networks Q1(s,·), Q2(s,·) producing (B, n_actions)
      - Target critic: Polyak/EMA copy of the critic used for stable targets

    OffPolicyAlgorithm contract (duck-typed)
    ----------------------------------------
    Expected attributes/methods used by your OffPolicyAlgorithm/Core:
      - device: torch.device
      - set_training(training): toggles train/eval mode appropriately
      - act(obs, deterministic=False) -> action tensor shaped (B,) or (B,1)
      - q_values(obs) -> (q1, q2), each of shape (B, n_actions)
      - q_values_target(obs) -> (q1t, q2t), each of shape (B, n_actions)
      - hard_update_target(), soft_update_target(tau)  (via BaseHead utilities)
      - save(path), load(path)
      - get_ray_policy_factory_spec()

    Notes
    -----
    - This file only defines networks + persistence + Ray factory spec.
      The actual SAC update logic (entropy, Bellman targets, etc.) belongs in Core.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        # Actor and critic can use different MLP widths by design.
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        # Critic option: dueling architecture for Q(s,a)
        dueling_mode: bool = False,
        # Initialization knobs (kept consistent with your other heads)
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        # BaseHead typically provides:
        # - self.device (torch.device)
        # - hard_update / soft_update utilities
        # - freeze_target utility (sets requires_grad=False)
        super().__init__(device=device)

        # Basic problem dimensions
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        # Store config (useful for logging / checkpoint kwargs)
        self.actor_hidden_sizes = tuple(int(x) for x in actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(int(x) for x in critic_hidden_sizes)
        self.activation_fn = activation_fn

        self.dueling_mode = bool(dueling_mode)
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---------------------------------------------------------------------
        # Actor: categorical policy π(a|s)
        # ---------------------------------------------------------------------
        # DiscretePolicyNetwork is expected to expose something like:
        # - forward(obs) -> logits or probs
        # - act(obs, deterministic=False) -> sampled/argmax actions
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.actor_hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: twin Q networks producing Q(s,·) for all actions
        # ---------------------------------------------------------------------
        # Important: for discrete SAC, critics typically take state only and output
        # (B, n_actions), i.e. Q(s,a) for all a in one forward.
        #
        # In your codebase, DoubleQNetwork is expected to output (q1, q2) each
        # shaped (B, n_actions).
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

        # ---------------------------------------------------------------------
        # Target critic: EMA/Polyak copy used for stable bootstrap targets
        # ---------------------------------------------------------------------
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

        # Initialize target weights = online weights, then freeze target params
        # (core/head will still update them via hard/soft update utilities).
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe format.

        Purpose
        -------
        - Used by save/load to store config alongside weights.
        - Used by Ray policy factory spec, so workers can reconstruct the head.

        Note
        ----
        activation_fn is exported as a name string and later resolved with
        `resolve_activation_fn` in the worker factory.
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
        Save model weights + config to a single .pt checkpoint.

        Filesystem behavior
        -------------------
        If `path` does not end with ".pt", it is appended automatically.
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
        Load model weights from a .pt checkpoint created by `save()`.

        Safety / compatibility
        ----------------------
        - Validates minimal expected keys exist.
        - Restores target critic if available; otherwise reconstructs it from critic.
        - Freezes target critic parameters after loading to keep optimizer clean.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized DiscreteSACHead checkpoint format at: {path}")

        # Restore online networks
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Restore or rebuild target
        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            # Backward compatibility: if old checkpoints don't store target explicitly
            self.hard_update(self.critic_target, self.critic)

        # Ensure target is non-trainable and in eval mode
        self.freeze_target(self.critic_target)
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly factory spec to reconstruct this head on workers.

        The returned spec includes:
          - entrypoint: module-level function import path
          - kwargs: JSON-safe hyperparameters needed to rebuild the head
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_discrete_sac_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
