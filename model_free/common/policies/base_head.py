from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
import torch as th
import torch.nn as nn

from ..networks.distributions import EPS
from ..utils.common_utils import _reduce_joint, _to_column, _to_scalar, _to_tensor
from ..utils.network_utils import TanhBijector
from ..utils.policy_utils import _freeze_target, _hard_update, _soft_update


# =============================================================================
# Base Head
# =============================================================================
class BaseHead(nn.Module, ABC):
    """
    Base class for all policy/value "heads".

    A "head" is a lightweight container that owns one or more neural networks
    (actor/critic/q, optional target networks), and exposes a stable interface
    for:
      - action selection (`act`)
      - value / Q evaluation (`value_only`, `q_values`, etc.)
      - policy evaluation (`evaluate_actions`, `logp`, `probs`, etc.)
      - target-network utilities (hard/soft updates + freezing)

    Responsibilities
    ----------------
    - Owns a normalized `device` attribute.
    - Provides tensor conversion helper that standardizes batch dimension.
    - Exposes target-network utilities as thin wrappers around project utilities.

    Non-responsibilities
    --------------------
    - No optimization logic (optimizers/schedulers live in cores).
    - No gradient clipping.
    - No timing logic for target updates (interval handling lives in cores).

    Parameters
    ----------
    device : str or torch.device, default="cpu"
        Device used for tensor conversions and action sampling.
        Converted to a `torch.device` and stored in `self.device`.
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
        Convert input to a torch.Tensor on `self.device` and ensure batch dimension.

        Parameters
        ----------
        x : Any
            Input object convertible by your project `_to_tensor` utility. Common
            inputs include numpy arrays, Python scalars, lists, or torch tensors.

        Returns
        -------
        t : torch.Tensor
            Tensor on `self.device`. Ensures a leading batch dimension:
            - scalar or (D,) -> (1, D)
            - already batched (B, D, ...) -> unchanged

        Notes
        -----
        This helper standardizes "single observation" inputs so downstream code
        can assume batched tensors.
        """
        t = _to_tensor(x, self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    @staticmethod
    def _activation_to_name(act: Any) -> str | None:
        """
        Convert an activation function/object to a name string.

        Parameters
        ----------
        act : Any
            An activation callable/class (e.g., torch.nn.ReLU, torch.relu),
            or None.

        Returns
        -------
        name : Optional[str]
            Name for logging/config dumps, or None if `act` is None.
        """
        if act is None:
            return None
        return getattr(act, "__name__", None) or str(act)

    # ------------------------------------------------------------------
    # Target-network utilities (thin wrappers)
    # ------------------------------------------------------------------
    @staticmethod
    def hard_update(target: nn.Module, source: nn.Module) -> None:
        """
        Copy parameters from `source` into `target`.

        Parameters
        ----------
        target : nn.Module
            Target network updated in-place.
        source : nn.Module
            Source/online network.
        """
        _hard_update(target, source)

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        """
        Polyak average parameters from `source` into `target`.

        Parameters
        ----------
        target : nn.Module
            Target network updated in-place.
        source : nn.Module
            Source/online network.
        tau : float
            Polyak factor in (0, 1]. Larger values update targets more aggressively.
        """
        _soft_update(target, source, tau)

    @staticmethod
    def freeze_target(module: nn.Module) -> None:
        """
        Freeze a target module (disable grads, set eval mode).

        Parameters
        ----------
        module : nn.Module
            Target module to freeze.
        """
        _freeze_target(module)


# =============================================================================
# 1) On-policy Actor-Critic Head (continuous)
# =============================================================================
class OnPolicyContinuousActorCriticHead(BaseHead, ABC):
    """
    Head base for PPO / A2C / TRPO-style on-policy algorithms (continuous actions).

    Required attributes (duck-typed)
    --------------------------------
    actor : nn.Module
        Must provide:
        - act(obs, deterministic) -> (action, info)
        - get_dist(obs) -> distribution with log_prob() and entropy()
    critic : Optional[nn.Module]
        If present, must provide:
        - critic(obs) -> V(s) as (B,1) or (B,)

    Notes
    -----
    - Critic may be absent for REINFORCE/VPG-style heads with baseline disabled.
      In that case `value_only` and `evaluate_actions` return a zero baseline.
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode on actor and (if present) critic.

        Parameters
        ----------
        training : bool
            True for train mode, False for eval mode.
        """
        self.actor.train(training)
        critic = getattr(self, "critic", None)
        if critic is not None:
            critic.train(training)

    # ------------------------------------------------------------------
    # Acting / evaluation
    # ------------------------------------------------------------------
    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False, return_info: bool = False):
        """
        Sample or select an action from the policy.

        Parameters
        ----------
        obs : Any
            Observation(s). Converted to batched tensor.
        deterministic : bool, default=False
            If True, the actor should return a deterministic action (e.g., mean).
        return_info : bool, default=False
            If True, returns (action, info_dict). Otherwise returns action only.

        Returns
        -------
        action : torch.Tensor
            Action tensor (batched). Shape depends on actor implementation.
        info : dict, optional
            Empty dict for compatibility (can be extended by concrete actors).
        """
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=deterministic)
        if return_info:
            return action, {}
        return action

    @th.no_grad()
    def value_only(self, obs: Any) -> th.Tensor:
        """
        Return V(s) as a column tensor with shape (B, 1).

        Parameters
        ----------
        obs : Any
            Observation(s). Converted to batched tensor.

        Returns
        -------
        value : torch.Tensor
            Value estimate with shape (B, 1). If the critic is absent, returns
            a zero baseline tensor.

        Notes
        -----
        Baseline compatibility:
        If `self.critic` does not exist, we return zeros((B,1)) so downstream
        algorithms can always store/log a value tensor.
        """
        obs_t = self._to_tensor_batched(obs)
        critic = getattr(self, "critic", None)

        if critic is None:
            return th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)

        return _to_column(critic(obs_t))

    def evaluate_actions(
        self,
        obs: Any,
        action: Any,
        *,
        as_scalar: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate value/log-prob/entropy at a batch of (s, a).

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Action batch.
        as_scalar : bool, default=False
            If True, requires batch size B=1 and returns Python scalar-like values.
            log_prob/entropy are reduced to a joint scalar (sum over action dims).

        Returns
        -------
        out : Dict[str, Any]
            If as_scalar is False:
              - "value"    : torch.Tensor of shape (B,1)
              - "log_prob" : torch.Tensor of shape (B,1) (or reduced from per-dim)
              - "entropy"  : torch.Tensor of shape (B,1) (or reduced from per-dim)
            If as_scalar is True (B must be 1):
              - "value"    : float-like
              - "log_prob" : float-like (joint)
              - "entropy"  : float-like (joint)

        Notes
        -----
        Baseline compatibility:
        If critic is absent, `value` is a zero baseline and dtype is chosen to
        match the distribution's log_prob dtype when possible.
        """
        obs_t = self._to_tensor_batched(obs)
        act_t = self._to_tensor_batched(action)

        dist = self.actor.get_dist(obs_t)

        critic = getattr(self, "critic", None)
        if critic is None:
            logp_tmp = dist.log_prob(act_t)
            dtype = logp_tmp.dtype if th.is_tensor(logp_tmp) else th.float32
            value = th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=dtype)
            logp = _to_column(logp_tmp)
        else:
            value = _to_column(critic(obs_t))
            logp = _to_column(dist.log_prob(act_t))

        ent = _to_column(dist.entropy())

        if not as_scalar:
            return {"value": value, "log_prob": logp, "entropy": ent}

        if obs_t.shape[0] != 1:
            raise ValueError("as_scalar=True requires batch size B=1.")

        v_s = _to_scalar(value.squeeze(0).squeeze(-1))
        lp_s = _to_scalar(_reduce_joint(logp).squeeze(0))
        ent_s = _to_scalar(_reduce_joint(ent).squeeze(0))
        return {"value": v_s, "log_prob": lp_s, "entropy": ent_s}


# =============================================================================
# 1b) On-policy Actor-Critic Head (discrete)
# =============================================================================
class OnPolicyDiscreteActorCriticHead(BaseHead):
    """
    Head base for PPO / A2C / TRPO-style on-policy algorithms (discrete actions).

    Assumptions / contracts
    -----------------------
    - Action space is discrete; actions are integer indices in [0, n_actions).
    - actor.get_dist(obs) returns a categorical-like distribution where:
        - dist.log_prob(action) returns (B,) for action shape (B,)
        - dist.entropy() returns (B,)

    Required attributes (duck-typed)
    --------------------------------
    actor : nn.Module
        Must provide:
        - act(obs_t, deterministic=...) -> (action, info)
        - get_dist(obs_t) -> distribution
    critic : Optional[nn.Module]
        If present, must provide:
        - critic(obs_t) -> V(s) as (B,1) or (B,)

    Notes
    -----
    - If critic is missing, this head returns a zero baseline value tensor.
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode on actor and (if present) critic.
        """
        self.actor.train(training)
        critic = getattr(self, "critic", None)
        if critic is not None:
            critic.train(training)

    def _normalize_discrete_action(self, act_t: th.Tensor) -> th.Tensor:
        """
        Normalize discrete action tensor to long indices of shape (B,).

        Parameters
        ----------
        act_t : torch.Tensor
            Candidate action tensor. Supported shapes:
            - (B,)
            - (B, 1)

        Returns
        -------
        act_idx : torch.Tensor
            Long tensor of shape (B,).

        Raises
        ------
        ValueError
            If the action has an unsupported shape.
        """
        if act_t.dim() == 2 and act_t.shape[-1] == 1:
            act_t = act_t.squeeze(-1)
        if act_t.dim() != 1:
            raise ValueError(f"Discrete action must be shape (B,) or (B,1), got {tuple(act_t.shape)}")
        return act_t.long()

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False, return_info: bool = False):
        """
        Sample/choose a discrete action.

        Parameters
        ----------
        obs : Any
            Observation(s), converted to batched tensor.
        deterministic : bool, default=False
            If True, actor should choose mode/argmax action.
        return_info : bool, default=False
            If True, returns (action, info_dict), else returns action only.

        Returns
        -------
        action : torch.Tensor
            Action indices of shape (B,) and dtype long.
        info : dict, optional
            Actor-provided info dict if any, else {}.
        """
        obs_t = self._to_tensor_batched(obs)
        action, info = self.actor.act(obs_t, deterministic=deterministic)

        if th.is_tensor(action):
            action = self._normalize_discrete_action(action)

        if return_info:
            return action, (info if isinstance(info, dict) else {})
        return action

    @th.no_grad()
    def value_only(self, obs: Any) -> th.Tensor:
        """
        Return V(s) with shape (B, 1).

        If critic is absent, returns zeros((B,1)) for baseline compatibility.
        """
        obs_t = self._to_tensor_batched(obs)
        critic = getattr(self, "critic", None)

        if critic is None:
            return th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)

        return _to_column(critic(obs_t))

    def evaluate_actions(
        self,
        obs: Any,
        action: Any,
        *,
        as_scalar: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate value/log-prob/entropy at (s,a) for discrete actions.

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Discrete actions. Accepts shapes (B,), (B,1), or scalar for B=1.
        as_scalar : bool, default=False
            If True, requires B=1 and returns Python scalar-like values.

        Returns
        -------
        out : Dict[str, Any]
            If as_scalar is False:
              - "value"    : (B,1) (zero if critic absent)
              - "log_prob" : (B,1)
              - "entropy"  : (B,1)
            If as_scalar is True (B=1):
              - scalar-like "value", "log_prob", "entropy"

        Notes
        -----
        - If a distribution returns per-dimension values (unusual for categorical),
          we reduce to joint via `_reduce_joint`.
        """
        obs_t = self._to_tensor_batched(obs)
        act_t = self._to_tensor_batched(action)

        # Accept scalar action for single-state evaluation
        if act_t.dim() == 0:
            act_t = act_t.view(1)

        act_idx = self._normalize_discrete_action(act_t)

        dist = self.actor.get_dist(obs_t)

        critic = getattr(self, "critic", None)
        if critic is None:
            value = th.zeros((obs_t.shape[0], 1), device=obs_t.device, dtype=obs_t.dtype)
        else:
            value = _to_column(critic(obs_t))

        logp = dist.log_prob(act_idx)
        if th.is_tensor(logp) and logp.dim() > 1:
            logp = _reduce_joint(logp)
        logp = _to_column(logp)

        ent = dist.entropy()
        if th.is_tensor(ent) and ent.dim() > 1:
            ent = _reduce_joint(ent)
        ent = _to_column(ent)

        if not as_scalar:
            return {"value": value, "log_prob": logp, "entropy": ent}

        if obs_t.shape[0] != 1:
            raise ValueError("as_scalar=True requires batch size B=1.")

        v_s = _to_scalar(value.squeeze(0).squeeze(-1))
        lp_s = _to_scalar(logp.squeeze(0).squeeze(-1))
        ent_s = _to_scalar(ent.squeeze(0).squeeze(-1))
        return {"value": v_s, "log_prob": lp_s, "entropy": ent_s}


# =============================================================================
# 2) Off-policy Stochastic Actor-Critic Head (SAC / TQC, continuous)
# =============================================================================
class OffPolicyContinuousActorCriticHead(BaseHead, ABC):
    """
    Head base for stochastic off-policy actor-critic algorithms (SAC/TQC), continuous actions.

    Required attributes
    -------------------
    actor : nn.Module
        Must provide:
        - act(obs, deterministic) -> (action, info)
        - get_dist(obs) -> distribution supporting rsample/log_prob/entropy
    critic : nn.Module
        Must provide:
        - critic(obs, action) -> (q1, q2) (common twin-critic contract)

    Optional attributes
    -------------------
    critic_target : nn.Module, optional
        Target critic network used for bootstrapping.

    Notes
    -----
    This head includes `sample_action_and_logp` which implements the standard
    squashed-Gaussian correction:
      log π(a|s) = log π(z|s) - Σ log(1 - tanh(z)^2)
    where a = tanh(z).
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for actor/critic and (if present) critic_target.
        """
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> th.Tensor:
        """
        Sample/choose an action from the actor.

        Returns
        -------
        action : torch.Tensor
            Action tensor (B,A) typically. If actor returns (B,1) for A=1, it is
            squeezed to (B,).
        """
        obs_t = self._to_tensor_batched(obs)
        action, _ = self.actor.act(obs_t, deterministic=deterministic)
        if action.dim() == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        return action

    # ------------------------------------------------------------------
    # Q interfaces
    # ------------------------------------------------------------------
    def q_values(self, obs: Any, action: Any) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute Q-values from the online critic.

        Returns
        -------
        q1, q2 : torch.Tensor
            Critic outputs. Shapes depend on critic contract (commonly (B,1)).
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def q_values_target(self, obs: Any, action: Any) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute Q-values from the target critic (no-grad).

        Raises
        ------
        AttributeError
            If `critic_target` is not present.
        """
        if not hasattr(self, "critic_target"):
            raise AttributeError("This head has no critic_target.")
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)

    # ------------------------------------------------------------------
    # Sampling (squashed Gaussian)
    # ------------------------------------------------------------------
    def sample_action_and_logp(self, obs: Any) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample an action using a reparameterized policy and compute log π(a|s).

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        action : torch.Tensor
            Squashed action a = tanh(z), shape (B, A).
        logp : torch.Tensor
            Log-probability of the squashed action, shape (B, 1).

        Notes
        -----
        This method:
        - draws z ~ π(z|s) using rsample() for pathwise gradients
        - applies tanh bijection to obtain bounded actions
        - applies the log-det-Jacobian correction term

        The returned `logp` is standardized to column shape (B,1) for downstream
        cores/tests.
        """
        obs_t = self._to_tensor_batched(obs)
        dist = self.actor.get_dist(obs_t)

        z = dist.rsample()  # (B,A)
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

        if logp.dim() == 1:
            logp = logp.unsqueeze(-1)  # (B,1)

        return action, logp


# =============================================================================
# 2b) Off-policy Actor-Critic Head (discrete)
# =============================================================================
class OffPolicyDiscreteActorCriticHead(BaseHead, ABC):
    """
    Unified discrete off-policy actor-critic head base.

    This head is designed to support multiple discrete off-policy families by
    normalizing critic outputs into a consistent interface.

    Supported critic styles
    -----------------------
    1) Single critic returning Q(s,·):
       - critic(s) -> (B, A)

    2) Twin critics returning Q1(s,·), Q2(s,·):
       - critic(s) -> (q1(B, A), q2(B, A))

    3) Some implementations may return (B, A) logits or other encodings; in that
       case override the pair/reduction methods to match your critic contract.

    Public API contract (recommended)
    ---------------------------------
    q_values_pair(obs) -> (q1, q2) each (B, A)
    q_values(obs, reduce="min") -> (B, A)

    q_values_target_pair(obs) -> (q1, q2) each (B, A)
    q_values_target(obs, reduce="min") -> (B, A)

    Notes
    -----
    - Default reduction is conservative "min" (standard SAC family choice).
    - If only a single critic exists, (q, q) is returned as a pair.
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for actor/critic and (if present) critic_target.
        """
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    # ------------------------------------------------------------------
    # Policy distribution helpers
    # ------------------------------------------------------------------
    def dist(self, obs: Any) -> Any:
        """
        Return the policy distribution π(.|s).

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        dist : Any
            Distribution object returned by `actor.get_dist`.
        """
        s = self._to_tensor_batched(obs)
        return self.actor.get_dist(s)

    def logp(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute log π(a|s), standardized to shape (B,1).

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Discrete actions.

        Returns
        -------
        logp : torch.Tensor
            Log-probabilities as column tensor (B,1).
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
        Return action probabilities π(a|s), shape (B, A).

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        probs : torch.Tensor
            Probability tensor (B, A).
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
        """
        Sample/choose a discrete action.

        Returns
        -------
        action : torch.Tensor
            Action indices, shape (B,) or (B,1) depending on actor. Squeezed to (B,)
            when the last dim is singleton.
        """
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

        Parameters
        ----------
        out : Any
            Critic output. Supported:
            - Tensor q: treated as single-critic, returns (q, q)
            - (q1, q2) tuple/list: returns (q1, q2)

        Returns
        -------
        q1, q2 : torch.Tensor
            Twin-critic tensors.

        Raises
        ------
        TypeError
            If output is neither Tensor nor tuple/list.
        ValueError
            If tuple/list is not length 2.
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
        Reduce twin critics into a single Q tensor.

        Parameters
        ----------
        q1, q2 : torch.Tensor
            Twin critics (B, A).
        mode : str, default="min"
            Reduction strategy:
            - "min"  : elementwise min (conservative, SAC default)
            - "mean" : elementwise mean
            - "q1"   : return q1
            - "q2"   : return q2

        Returns
        -------
        q : torch.Tensor
            Reduced Q tensor (B, A).

        Raises
        ------
        ValueError
            If mode is unknown.
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
        Return (q1, q2) for all actions, each shape (B, A). Grad enabled.

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        q1, q2 : torch.Tensor
            Twin Q tensors (B, A). If critic is single, returns (q, q).

        Raises
        ------
        ValueError
            If critic output shapes are not (B, A).
        """
        s = self._to_tensor_batched(obs)
        out = self.critic(s)
        q1, q2 = self._as_pair(out)

        if q1.dim() != 2 or q2.dim() != 2:
            raise ValueError(f"q_values_pair expects q1,q2 as (B,A); got {tuple(q1.shape)}, {tuple(q2.shape)}")
        return q1, q2

    @th.no_grad()
    def q_values_target_pair(self, obs: Any) -> tuple[th.Tensor, th.Tensor]:
        """
        Return (q1, q2) for all actions from target critic, each shape (B, A). No-grad.

        Raises
        ------
        AttributeError
            If `critic_target` is not present.
        ValueError
            If target critic output shapes are not (B, A).
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
        Reduced Q(s,·) for all actions, shape (B, A). Grad enabled.

        Parameters
        ----------
        obs : Any
            Observation batch.
        reduce : str, default="min"
            Reduction mode passed to `_reduce_pair`.

        Returns
        -------
        q : torch.Tensor
            Reduced Q tensor (B, A).
        """
        q1, q2 = self.q_values_pair(obs)
        return self._reduce_pair(q1, q2, mode=reduce)

    @th.no_grad()
    def q_values_target(self, obs: Any, reduce: str = "min") -> th.Tensor:
        """
        Reduced Q'(s,·) for all actions from target critic, shape (B, A). No-grad.
        """
        q1, q2 = self.q_values_target_pair(obs)
        return self._reduce_pair(q1, q2, mode=reduce)


# =============================================================================
# 3) Deterministic Actor-Critic Head (DDPG / TD3)
# =============================================================================
class DeterministicActorCriticHead(BaseHead, ABC):
    """
    Deterministic Policy Gradient actor-critic head base (DDPG/TD3 family).

    Required attributes
    -------------------
    actor : nn.Module
        Must provide `act(obs, deterministic=...) -> (action, info)`.

    critic : nn.Module
        Must provide `critic(obs, action) -> Q(s,a)` (or twin outputs).

    Optional attributes
    -------------------
    actor_target : nn.Module, optional
    critic_target : nn.Module, optional
    action_low / action_high : array-like, optional
        Used to clamp actions. If absent, clamps to [-1, 1] by default.
    noise : object, optional
        Exploration noise process with .sample() (and optionally .reset()).
    noise_clip : float, optional
        Clamps sampled noise.

    Notes
    -----
    - If actor advertises `_has_bounds`, this method clamps using actor-provided
      bias/scale fields to respect its internal action normalization.
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for actor/critic and (if present) targets.
        """
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, "actor_target"):
            self.actor_target.eval()
        if hasattr(self, "critic_target"):
            self.critic_target.eval()

    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = True) -> th.Tensor:
        """
        Select an action using the deterministic actor.

        Parameters
        ----------
        obs : Any
            Observation batch.
        deterministic : bool, default=True
            If False and `self.noise` exists, adds exploration noise.

        Returns
        -------
        action : torch.Tensor
            Clamped action tensor.
        """
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

                c = float(getattr(self, "noise_clip", 0.0) or 0.0)
                if c > 0.0:
                    eps = eps.clamp(-c, c)

                action = action + eps

        return self._clamp_action(action)

    @th.no_grad()
    def _clamp_action(self, a: th.Tensor) -> th.Tensor:
        """
        Clamp actions to valid bounds.

        Priority
        --------
        1) actor internal bounds via `_has_bounds` (bias/scale)
        2) head attributes `action_low` / `action_high` if provided
        3) fallback to [-1, 1]
        """
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
        """
        Reset exploration noise process if it provides .reset().
        """
        noise = getattr(self, "noise", None)
        if noise is not None:
            try:
                noise.reset()
            except Exception:
                pass

    def q_values(self, obs: Any, action: Any):
        """
        Compute Q(s,a) from the online critic (grad enabled).
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def q_values_target(self, obs: Any, action: Any):
        """
        Compute Q'(s,a) from the target critic (no-grad).
        """
        if not hasattr(self, "critic_target"):
            raise AttributeError("This head has no critic_target.")
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)


# =============================================================================
# 4) Discrete Q-learning Head (DQN family)
# =============================================================================
class QLearningHead(BaseHead, ABC):
    """
    Base head for discrete Q-learning (DQN family).

    Design rule
    -----------
    - `act()` MUST use `q_values()`, not raw q-network outputs.
    - Distributional variants should override `q_values()` to return expected
      Q-values as (B, A).

    Required attributes (duck-typed)
    --------------------------------
    q : nn.Module
        Q-network. Must return (B,A) for standard heads.
    n_actions : int
        Number of discrete actions.

    Optional attributes
    -------------------
    q_target : nn.Module, optional
        Target Q-network.
    """

    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for q and (if present) q_target.
        """
        self.q.train(training)
        if hasattr(self, "q_target"):
            self.q_target.eval()

    # ------------------------------------------------------------------
    # Distributional helpers (optional)
    # ------------------------------------------------------------------
    def dist(self, obs: Any) -> th.Tensor:
        """
        Return distributional output from q-network (if supported by q).

        Notes
        -----
        Distributional heads typically implement `q.dist(obs)` and derive expected
        Q-values by integrating over atoms. Standard heads may not implement this.
        """
        s = self._to_tensor_batched(obs)
        return self.q.dist(s)

    @th.no_grad()
    def dist_target(self, obs: Any) -> th.Tensor:
        """
        Return distributional output from target q-network (if present).
        """
        if not hasattr(self, "q_target"):
            raise AttributeError("This head has no q_target.")
        s = self._to_tensor_batched(obs)
        return self.q_target.dist(s)

    # ------------------------------------------------------------------
    # Expected Q interfaces
    # ------------------------------------------------------------------
    def q_values(self, obs: Any) -> th.Tensor:
        """
        Return expected Q-values Q(s,·), shape (B, A). Grad enabled.

        Raises
        ------
        ValueError
            If the q-network output is not 2D (B,A). Distributional heads should
            override this method.
        """
        s = self._to_tensor_batched(obs)
        q = self.q(s)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        if q.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__}.q_values() expects (B,A), got {tuple(q.shape)}. "
                "Override q_values() for distributional heads."
            )
        return q

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        """
        Return expected target Q-values Q'(s,·), shape (B, A). No-grad.

        Raises
        ------
        AttributeError
            If `q_target` is not present.
        ValueError
            If target q-network output is not 2D (B,A).
        """
        if not hasattr(self, "q_target"):
            raise AttributeError("This head has no q_target.")
        s = self._to_tensor_batched(obs)
        q = self.q_target(s)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        if q.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__}.q_values_target() expects (B,A), got {tuple(q.shape)}. "
                "Override for distributional heads."
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

        Parameters
        ----------
        obs : Any
            Observation batch.
        epsilon : float, default=0.0
            Exploration probability. With probability epsilon, choose a random
            action uniformly from [0, n_actions).
        deterministic : bool, default=True
            If True, always take greedy action (ignores epsilon).

        Returns
        -------
        action : torch.Tensor
            Action indices of shape (B,) and dtype long.
        """
        q = self.q_values(obs)        # (B,A)
        greedy = th.argmax(q, dim=-1) # (B,)

        if deterministic or float(epsilon) <= 0.0:
            return greedy.long()

        B = q.shape[0]
        rand = th.randint(0, int(self.n_actions), (B,), device=self.device)
        mask = th.rand(B, device=self.device) < float(epsilon)
        return th.where(mask, rand, greedy).long()
