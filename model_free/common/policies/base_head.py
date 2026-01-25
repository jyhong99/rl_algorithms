from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Tuple

import torch as th
import torch.nn as nn
import numpy as np

from ..networks.distributions import EPS
from ..utils.common_utils import to_tensor, to_scalar, to_column, reduce_joint
from ..utils.policy_utils import hard_update, soft_update, freeze_target
from ..utils.network_utils import TanhBijector


# =============================================================================
# Base Head
# =============================================================================
class BaseHead(nn.Module, ABC):
    """
    Base class for all policy/value heads.

    Responsibilities
    ----------------
    - Owns `device`
    - Provides tensor conversion helpers
    - Exposes target-network utilities (thin wrappers)

    Non-responsibilities
    --------------------
    - No optimization logic
    - No gradient clipping
    - No scheduler/target-update timing logic
    """

    device: th.device

    def __init__(self, *, device: str | th.device = "cpu") -> None:
        super().__init__()
        self.device = device if isinstance(device, th.device) else th.device(str(device))

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    def _to_tensor_batched(self, x: Any) -> th.Tensor:
        """
        Convert input to a torch.Tensor on self.device and ensure batch dimension.

        Rules
        -----
        - scalar / (D,)  -> (1, D)
        - already batched -> unchanged
        """
        t = to_tensor(x, self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    @staticmethod
    def _activation_to_name(act: Any) -> str | None:
        """Utility for logging / config dumps."""
        if act is None:
            return None
        return getattr(act, "__name__", None) or str(act)

    # ------------------------------------------------------------------
    # Target-network utilities (thin wrappers)
    # ------------------------------------------------------------------
    @staticmethod
    def hard_update(target: nn.Module, source: nn.Module) -> None:
        hard_update(target, source)

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        soft_update(target, source, tau)

    @staticmethod
    def freeze_target(module: nn.Module) -> None:
        freeze_target(module)


# =============================================================================
# 1) On-policy Actor-Critic Head
# =============================================================================
class OnPolicyContinuousActorCriticHead(BaseHead, ABC):
    """
    Head base for PPO / A2C / TRPO-style algorithms.

    Required attributes (duck-typed)
    --------------------------------
    - actor: nn.Module with:
        * act(obs, deterministic) -> (action, info)
        * get_dist(obs) -> distribution
    - critic: nn.Module with:
        * critic(obs) -> V(s)
    """

    def set_training(self, training: bool) -> None:
        self.actor.train(training)
        critic = getattr(self, "critic", None)
        if critic is not None:
            critic.train(training)

    # ------------------------------------------------------------------
    # Acting / evaluation
    # ------------------------------------------------------------------
    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False, return_info=False) -> th.Tensor:
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=deterministic)
        if return_info:
            return action, {}
        return action

    @th.no_grad()
    def value_only(self, obs: Any) -> th.Tensor:
        """
        Return V(s) with shape (B,1).

        Baseline compatibility
        ----------------------
        Some heads (e.g., VPGHead with use_baseline=False) do not have a critic.
        In that case, this returns a zero baseline tensor with shape (B,1) so that
        algorithms that expect a value estimate can still run safely.
        """
        obs_t = self._to_tensor_batched(obs)

        critic = getattr(self, "critic", None)
        if critic is None:
            return th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)

        return to_column(critic(obs_t))
    
    def evaluate_actions(
        self,
        obs: Any,
        action: Any,
        *,
        as_scalar: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate policy distribution + value at (s,a).

        Conventions
        -----------
        - as_scalar=False:
            value    : (B,1)
            log_prob : (B,1) or (B,A)  (per-dimension if distribution returns per-dim)
            entropy  : (B,1) or (B,A)
        - as_scalar=True (requires B=1):
            value/log_prob/entropy returned as Python scalar-like via to_scalar(),
            with log_prob/entropy reduced to joint scalar by summing over action dims.

        Baseline compatibility
        ----------------------
        Some heads (e.g., VPGHead with use_baseline=False) may not have a critic.
        In that case, we return a zero baseline:
        - value tensor: zeros((B,1))
        - value scalar: 0.0

        This keeps the OnPolicyAlgorithm contract stable (it can always log/store `value`).
        """
        obs_t = self._to_tensor_batched(obs)
        act_t = self._to_tensor_batched(action)

        dist = self.actor.get_dist(obs_t)

        # ------------------------------------------------------------------
        # Value (baseline): critic may be absent for REINFORCE-style heads
        # ------------------------------------------------------------------
        critic = getattr(self, "critic", None)
        if critic is None:
            # Preserve dtype/device consistency; prefer float32 if distribution returns non-float
            # but typically logp is float tensor, so use its dtype.
            logp_tmp = dist.log_prob(act_t)
            dtype = logp_tmp.dtype if th.is_tensor(logp_tmp) else th.float32
            value = th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=dtype)
            logp = to_column(logp_tmp)
        else:
            value = to_column(critic(obs_t))
            logp = to_column(dist.log_prob(act_t))

        ent = to_column(dist.entropy())

        if not as_scalar:
            return {"value": value, "log_prob": logp, "entropy": ent}

        if obs_t.shape[0] != 1:
            raise ValueError("as_scalar=True requires batch size B=1.")

        v_s = to_scalar(value.squeeze(0).squeeze(-1))
        lp_s = to_scalar(reduce_joint(logp).squeeze(0))
        ent_s = to_scalar(reduce_joint(ent).squeeze(0))

        return {"value": v_s, "log_prob": lp_s, "entropy": ent_s}



# =============================================================================
# 1) On-policy Actor-Critic Head (DISCRETE)
# =============================================================================
class OnPolicyDiscreteActorCriticHead(BaseHead):
    """
    Head base for PPO / A2C / TRPO-style algorithms (DISCRETE actions).

    Assumptions / Contracts
    -----------------------
    - Action space is discrete; actions are integer indices in [0, n_actions).
    - actor.get_dist(obs) returns a categorical-like distribution (e.g., Categorical)
      where:
        * dist.log_prob(action) returns (B,) for action shape (B,) (typical)
        * dist.entropy() returns (B,) (typical)

    Required attributes (duck-typed)
    --------------------------------
    - actor: nn.Module with:
        * act(obs_t, deterministic=...) -> (action, info)
        * get_dist(obs_t) -> distribution
    - critic: nn.Module with:
        * critic(obs_t) -> V(s)  (B,1) or (B,)
    """

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    def set_training(self, training: bool) -> None:
        self.actor.train(training)
        critic = getattr(self, "critic", None)
        if critic is not None:
            critic.train(training)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_discrete_action(self, act_t: th.Tensor) -> th.Tensor:
        """
        Normalize action tensor for discrete distributions.

        Accepts shapes:
          - (B,)       (preferred)
          - (B,1)      (common from some actor impls)

        Returns
        -------
        act_idx : torch.Tensor
            Long tensor of shape (B,).
        """
        if act_t.dim() == 2 and act_t.shape[-1] == 1:
            act_t = act_t.squeeze(-1)
        if act_t.dim() != 1:
            raise ValueError(f"Discrete action must be shape (B,) or (B,1), got {tuple(act_t.shape)}")
        return act_t.long()

    # ------------------------------------------------------------------
    # Acting / evaluation
    # ------------------------------------------------------------------
    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False, return_info: bool = False):
        """
        Sample/choose a discrete action.

        Returns
        -------
        action : torch.Tensor
            Discrete action indices, shape (B,).
        info : dict (optional)
            If return_info=True.
        """
        obs_t = self._to_tensor_batched(obs)
        action, info = self.actor.act(obs_t, deterministic=deterministic)

        # Standardize to (B,) long for discrete envs.
        if th.is_tensor(action):
            if action.dim() == 2 and action.shape[-1] == 1:
                action = action.squeeze(-1)
            action = action.long()

        if return_info:
            return action, (info if isinstance(info, dict) else {})
        return action

    @th.no_grad()
    def value_only(self, obs: Any) -> th.Tensor:
        """
        Return V(s) with shape (B,1).

        Baseline compatibility
        ----------------------
        Some heads may not have a critic (baseline disabled). In that case, return
        a zero baseline tensor with shape (B,1) to keep algorithm contracts stable.
        """
        obs_t = self._to_tensor_batched(obs)

        critic = getattr(self, "critic", None)
        if critic is None:
            return th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)

        return to_column(critic(obs_t))


    def evaluate_actions(
        self,
        obs: Any,
        action: Any,
        *,
        as_scalar: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate policy distribution + value at (s,a) for DISCRETE actions.

        Returns (as_scalar=False)
        -------------------------
        value    : (B,1)          (zero if baseline disabled)
        log_prob : (B,1)
        entropy  : (B,1)

        Returns (as_scalar=True, requires B=1)
        --------------------------------------
        value/log_prob/entropy as Python scalar-like.

        Baseline compatibility
        ----------------------
        If `self.critic` is None, `value` is returned as zero baseline.
        """
        
        obs_t = self._to_tensor_batched(obs)
        act_t = self._to_tensor_batched(action)
        if act_t.dim() == 0:
            act_t = act_t.view(1)
        if act_t.dim() == 2 and act_t.shape[-1] == 1:
            act_t = act_t.squeeze(-1)
        act_t = act_t.long()

        dist = self.actor.get_dist(obs_t)

        # ------------------------------------------------------------
        # Value: critic may be absent (baseline off)
        # ------------------------------------------------------------
        critic = getattr(self, "critic", None)
        if critic is None:
            value = th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)
        else:
            value = to_column(critic(obs_t))

        # ------------------------------------------------------------
        # log_prob / entropy: categorical -> enforce (B,1)
        # ------------------------------------------------------------
        logp = dist.log_prob(act_t)
        if th.is_tensor(logp) and logp.dim() > 1:
            # If some distribution returns per-dimension, reduce to joint.
            logp = reduce_joint(logp)
        logp = to_column(logp)

        ent = dist.entropy()
        if th.is_tensor(ent) and ent.dim() > 1:
            ent = reduce_joint(ent)
        ent = to_column(ent)

        if not as_scalar:
            return {"value": value, "log_prob": logp, "entropy": ent}

        if obs_t.shape[0] != 1:
            raise ValueError("as_scalar=True requires batch size B=1.")

        v_s = to_scalar(value.squeeze(0).squeeze(-1))
        lp_s = to_scalar(logp.squeeze(0).squeeze(-1))
        ent_s = to_scalar(ent.squeeze(0).squeeze(-1))
        return {"value": v_s, "log_prob": lp_s, "entropy": ent_s}

    

# =============================================================================
# 2) Off-policy Stochastic Actor-Critic Head (SAC / TQC)
# =============================================================================
class OffPolicyContinuousActorCriticHead(BaseHead, ABC):
    """
    Head base for stochastic off-policy actor-critic algorithms (SAC/TQC).

    Required attributes
    -------------------
    - actor: nn.Module with get_dist()
    - critic: nn.Module
    Optional:
    - critic_target: nn.Module
    """

    def set_training(self, training: bool) -> None:
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> th.Tensor:
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=deterministic)
        if action.dim() == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        return action

    # ------------------------------------------------------------------
    # Q interfaces
    # ------------------------------------------------------------------
    def q_values(self, obs: Any, action: Any) -> Tuple[th.Tensor, th.Tensor]:
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def q_values_target(self, obs: Any, action: Any) -> Tuple[th.Tensor, th.Tensor]:
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_action_and_logp(self, obs: Any) -> Tuple[th.Tensor, th.Tensor]:
        obs_t = self._to_tensor_batched(obs)
        dist = self.actor.get_dist(obs_t)

        z = dist.rsample()  # (B, A)
        bij = getattr(self, "tanh_bijector", None)
        if bij is None:
            self.tanh_bijector = TanhBijector(epsilon=EPS)
            bij = self.tanh_bijector

        action = bij.forward(z)

        logp_z = dist.log_prob(z)
        if logp_z.dim() > 1:
            logp_z = logp_z.sum(dim=-1)  # (B,)

        corr = bij.log_prob_correction(z).sum(dim=-1)  # (B,)
        logp = logp_z - corr  # (B,)

        # --- FIX: enforce column shape for downstream cores/tests ---
        if logp.dim() == 1:
            logp = logp.unsqueeze(-1)  # (B,1)

        return action, logp
    

class OffPolicyDiscreteActorCriticHead(BaseHead, ABC):
    """
    Unified discrete off-policy actor-critic head base.

    Supported critic styles
    -----------------------
    1) ACER-style
       - critic(s) -> (B,A)
       - OR critic(s) -> (q1(B,A), q2(B,A))

    2) Discrete SAC-style
       - critic(s) -> (q1(B,A), q2(B,A))  (common)
       - (optional) critic(s,a) -> (B,1) or (q1(B,1), q2(B,1))  [not used here directly]

    Public API contract (recommended)
    ---------------------------------
    - q_values(obs, reduce="min") -> (B,A)
    - q_values_target(obs, reduce="min") -> (B,A)
    - q_values_pair(obs) -> (q1, q2) each (B,A)   (if only single critic, returns (q, q))
    - q_values_target_pair(obs) -> (q1, q2) each (B,A)

    Notes
    -----
    - Default reduction is conservative "min" (standard in SAC family).
    - If you want SAC-discrete "min" behaviour everywhere, leave defaults.
    """

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    def set_training(self, training: bool) -> None:
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    # ------------------------------------------------------------------
    # Policy distribution helpers
    # ------------------------------------------------------------------
    def dist(self, obs: Any) -> Any:
        """Return π(.|s) distribution object."""
        s = self._to_tensor_batched(obs)
        return self.actor.get_dist(s)

    def logp(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute log π(a|s), standardized to shape (B,1).
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        d = self.actor.get_dist(s)
        lp = d.log_prob(a)
        if lp.dim() == 1:
            lp = lp.unsqueeze(-1)
        return lp

    def probs(self, obs: Any) -> th.Tensor:
        """
        Return π(a|s), shape (B,A).
        """
        d = self.dist(obs)
        p = getattr(d, "probs", None)
        if p is not None:
            return p

        logits = getattr(d, "logits", None)
        if logits is None:
            s = self._to_tensor_batched(obs)
            logits = self.actor(s)
        return th.softmax(logits, dim=-1)

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> th.Tensor:
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=deterministic)
        if action.dim() == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        return action

    # ------------------------------------------------------------------
    # Critic output normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _as_pair(out: Any) -> tuple[th.Tensor, th.Tensor]:
        """
        Normalize critic output to a (q1, q2) pair.

        Accepts:
          - Tensor q: returns (q, q)
          - (q1, q2) tuple/list: returns (q1, q2)
        """
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError(f"critic output tuple/list must have len=2, got len={len(out)}")
            q1, q2 = out
            return q1, q2
        if th.is_tensor(out):
            return out, out
        raise TypeError(f"critic output must be Tensor or (Tensor,Tensor), got {type(out)}")

    @staticmethod
    def _reduce_pair(q1: th.Tensor, q2: th.Tensor, mode: str = "min") -> th.Tensor:
        """
        Reduce twin critics to a single tensor.
        Default: conservative min.
        """
        if mode == "min":
            return th.min(q1, q2)
        if mode == "mean":
            return 0.5 * (q1 + q2)
        if mode == "q1":
            return q1
        if mode == "q2":
            return q2
        raise ValueError(f"Unknown reduce mode: {mode}")

    # ------------------------------------------------------------------
    # Q interfaces (pair + reduced)
    # ------------------------------------------------------------------
    def q_values_pair(self, obs: Any) -> tuple[th.Tensor, th.Tensor]:
        """
        Return (q1, q2) for all actions, each shape (B,A).
        If critic is single, returns (q, q).
        """
        s = self._to_tensor_batched(obs)
        out = self.critic(s)
        q1, q2 = self._as_pair(out)

        # shape sanity: expect (B,A)
        if q1.dim() != 2 or q2.dim() != 2:
            raise ValueError(f"q_values_pair expects q1,q2 as (B,A); got {tuple(q1.shape)}, {tuple(q2.shape)}")
        return q1, q2

    @th.no_grad()
    def q_values_target_pair(self, obs: Any) -> tuple[th.Tensor, th.Tensor]:
        """
        Return (q1, q2) for all actions from target critic, each shape (B,A).
        If target critic is single, returns (q, q).
        """
        if not hasattr(self, "critic_target"):
            raise AttributeError("This head has no critic_target.")
        s = self._to_tensor_batched(obs)
        out = self.critic_target(s)
        q1, q2 = self._as_pair(out)

        if q1.dim() != 2 or q2.dim() != 2:
            raise ValueError(
                f"q_values_target_pair expects q1,q2 as (B,A); got {tuple(q1.shape)}, {tuple(q2.shape)}"
            )
        return q1, q2

    def q_values(self, obs: Any, reduce: str = "min") -> th.Tensor:
        """
        Reduced Q(s,·) for all actions, shape (B,A). Grad ENABLED.
        """
        q1, q2 = self.q_values_pair(obs)
        return self._reduce_pair(q1, q2, mode=reduce)

    @th.no_grad()
    def q_values_target(self, obs: Any, reduce: str = "min") -> th.Tensor:
        """
        Reduced Q'(s,·) for all actions, shape (B,A). No-grad.
        """
        q1, q2 = self.q_values_target_pair(obs)
        return self._reduce_pair(q1, q2, mode=reduce)

    
# =============================================================================
# 3) Deterministic Actor-Critic Head (DDPG / TD3)
# =============================================================================
class DeterministicActorCriticHead(BaseHead, ABC):
    """
    Deterministic Policy Gradient Actor-Critic Head Base.

    Required:
      - actor, critic
    Optional:
      - actor_target, critic_target
      - action_low / action_high
      - noise (+ noise_clip)
    """

    def set_training(self, training: bool) -> None:
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "actor_target"):
            self.actor_target.eval()
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = True) -> th.Tensor:
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=bool(deterministic))

        if (not deterministic) and hasattr(self, "noise") and self.noise is not None:
            try:
                eps = self.noise.sample(action)
            except TypeError:
                eps = self.noise.sample()

            if th.is_tensor(eps):
                if eps.dim() == 1 and action.dim() == 2:
                    eps = eps.unsqueeze(0).expand_as(action)
                eps = eps.to(action.device, action.dtype)

                if getattr(self, "noise_clip", None):
                    c = float(self.noise_clip)
                    if c > 0:
                        eps = eps.clamp(-c, c)

                action = action + eps

        return self._clamp_action(action)

    @th.no_grad()
    def _clamp_action(self, a: th.Tensor) -> th.Tensor:
        if getattr(self.actor, "_has_bounds", False):
            bias = self.actor.action_bias
            scale = self.actor.action_scale
            return a.clamp(bias - scale, bias + scale)

        low = getattr(self, "action_low", None)
        high = getattr(self, "action_high", None)
        if low is not None and high is not None:
            low_t = th.as_tensor(np.asarray(low, np.float32), device=a.device, dtype=a.dtype)
            high_t = th.as_tensor(np.asarray(high, np.float32), device=a.device, dtype=a.dtype)
            return a.clamp(low_t, high_t)

        return a.clamp(-1.0, 1.0)

    def reset_exploration_noise(self) -> None:
        noise = getattr(self, "noise", None)
        if noise is not None:
            try:
                noise.reset()
            except Exception:
                pass

    def q_values(self, obs: Any, action: Any):
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def q_values_target(self, obs: Any, action: Any):
        if not hasattr(self, "critic_target"):
            raise AttributeError("This head has no critic_target.")
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)


# =============================================================================
# 4) Discrete Q-learning Head
# =============================================================================
class QLearningHead(BaseHead, ABC):
    """
    Base head for discrete Q-learning (DQN family).

    Design rule
    -----------
    - act() MUST use q_values(), never raw q-network outputs.
    - Distributional variants override q_values() to return (B,A).
    """

    def set_training(self, training: bool) -> None:
        self.q.train(training)
        if hasattr(self, "q_target"):
            self.q_target.eval()

    # ------------------------------------------------------------------
    # Distributional helpers (optional)
    # ------------------------------------------------------------------
    def dist(self, obs: Any) -> th.Tensor:
        s = self._to_tensor_batched(obs)
        return self.q.dist(s)
    
    @th.no_grad()
    def dist_target(self, obs: Any) -> th.Tensor:
        s = self._to_tensor_batched(obs)
        return self.q_target.dist(s)

    # ------------------------------------------------------------------
    # Expected Q interfaces
    # ------------------------------------------------------------------
    def q_values(self, obs: Any) -> th.Tensor:
        s = self._to_tensor_batched(obs)
        q = self.q(s)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        if q.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__}.q_values() expects (B,A), "
                f"got {tuple(q.shape)}. Override q_values() for distributional heads."
            )
        return q

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        if not hasattr(self, "q_target"):
            raise AttributeError("This head has no q_target.")
        s = self._to_tensor_batched(obs)
        q = self.q_target(s)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        if q.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__}.q_values_target() expects (B,A), "
                f"got {tuple(q.shape)}. Override for distributional heads."
            )
        return q

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @th.no_grad()
    def act(
        self,
        obs: Any,
        *,
        epsilon: float = 0.0,
        deterministic: bool = True,
    ) -> th.Tensor:
        """
        Epsilon-greedy action selection using expected Q-values.
        """
        q = self.q_values(obs)                 # (B,A)
        greedy = th.argmax(q, dim=-1)          # (B,)

        if deterministic or float(epsilon) <= 0.0:
            return greedy.long()

        B = q.shape[0]
        rand = th.randint(0, int(self.n_actions), (B,), device=self.device)
        mask = th.rand(B, device=self.device) < float(epsilon)
        return th.where(mask, rand, greedy).long()