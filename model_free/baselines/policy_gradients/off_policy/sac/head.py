from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import DoubleStateActionValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_sac_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build SACHead on CPU.

    Notes
    -----
    - Must be module-level so Ray can import it by entrypoint string.
    - Forces device="cpu" because Ray rollout workers typically run on CPU.
    - Resolves activation_fn from a string/name into torch.nn module class.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = SACHead(**kwargs).to("cpu")
    # Worker policies are usually inference-only; keep them in eval mode.
    head.set_training(False)
    return head


# =============================================================================
# SACHead
# =============================================================================
class SACHead(OffPolicyContinuousActorCriticHead):
    """
    SAC Head: stochastic actor + twin critics + twin target critics.

    This "Head" encapsulates all neural networks needed by SAC:
      - actor: squashed Gaussian policy (ContinuousPolicyNetwork)
      - critic: double Q network (DoubleStateActionValueNetwork)
      - critic_target: target copy of critic for stable TD targets

    Contract (for OffPolicyAlgorithm)
    --------------------------------
    - device
    - set_training(training)
    - act(obs, deterministic=False)
    - sample_action_and_logp(obs)
    - q_values(obs, action) -> (q1, q2)
    - q_values_target(obs, action) -> (q1t, q2t)
    - hard_update_target()  (inherited helper)
    - save(path), load(path)
    - get_ray_policy_factory_spec()

    Design notes
    ------------
    - Targets are initialized from online critics, then frozen (no grad).
    - `OffPolicyActorCriticHead` provides common utilities:
        * action sampling + log-prob (SAC-style)
        * q_values / q_values_target wrappers
        * target update helpers (hard/soft) and freezing helpers
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
        # Policy distribution / log-std parameterization
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
    ) -> None:
        # Base class stores device + provides common SAC-like sampling helpers.
        super().__init__(device=device)

        # -----------------------------
        # Store configuration (for save/export)
        # -----------------------------
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
        # Actor: squashed Gaussian policy
        # -----------------------------
        # ContinuousPolicyNetwork typically outputs a distribution for actions.
        # `squash=True` means tanh squashing (actions in [-1, 1]).
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=True,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Critic: twin Q networks (Q1, Q2)
        # -----------------------------
        # DoubleStateActionValueNetwork should implement:
        #   critic(s, a) -> (q1, q2) each shaped (B, 1)
        self.critic = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Target critic: frozen copy of online critic
        # -----------------------------
        # Used to compute TD targets:
        #   y = r + gamma * (min(Q1_t, Q2_t) - alpha * log pi(a'|s'))
        self.critic_target = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target weights from online critic, then freeze target parameters.
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Serialization helpers
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe format.

        Notes
        -----
        - activation_fn is converted to a string name so Ray / JSON can carry it.
        - device is stringified because torch.device is not JSON-serializable.
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
        Save SACHead checkpoint.

        Stores:
          - kwargs: constructor configuration (for reproducibility)
          - actor: actor parameters
          - critic: online critic parameters
          - critic_target: target critic parameters
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
        Load SACHead checkpoint.

        Behavior
        --------
        - Restores actor + critic weights.
        - If target weights exist, restore and re-freeze.
        - If target weights are missing, hard-copy online critic -> target critic.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized SACHead checkpoint format at: {path}")

        # Restore online networks
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Restore or rebuild target
        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            # Ensure target stays frozen even if checkpoint was created differently.
            self.freeze_target(self.critic_target)
        else:
            # Backward-compat: checkpoint without target -> derive target from online.
            self.hard_update(self.critic_target, self.critic)

        # Targets are always eval mode.
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray PolicyFactorySpec so rollout workers can reconstruct this head.

        Notes
        -----
        - `entrypoint` is a string that Ray can import (must point to a module-level function).
        - `kwargs` are JSON-safe constructor args used on the worker side.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_sac_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
