from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union, Tuple

import numpy as np
import torch as th

from ..buffers import RolloutBuffer
from ..utils.common_utils import (
    to_numpy,
    to_scalar,
    is_scalar_like,
    require_mapping,
    require_scalar_like,
    infer_shape,
    to_action_np,
)
from .base_policy import BasePolicyAlgorithm


class OnPolicyAlgorithm(BasePolicyAlgorithm):
    """
    On-policy algorithm driver (rollout + PPO/A2C-style update loop).

    Responsibilities
    ----------------
    - Owns a RolloutBuffer and fills it with env transitions
    - Bootstraps last value for GAE
    - Runs multiple epochs of minibatch updates via `core.update_from_batch(batch)`
    - Aggregates scalar metrics for logging
    - Resets rollout state after update

    Required head interface (duck-typed)
    ------------------------------------
    - act(obs, deterministic=False) -> action
    - evaluate_actions(obs, action) -> Mapping with keys:
        * 'value'    : scalar-like (or tensor containing scalar for B=1)
        * 'log_prob' : scalar-like (or tensor containing scalar for B=1)
      Optional:
        * value_only(obs) -> V(s) (preferred for bootstrap)

    Required core interface
    -----------------------
    - update_from_batch(batch) -> Mapping[str, Any]
      (scalar-like values will be logged)

    Transition contract (on_env_step)
    ---------------------------------
    transition must contain:
      - 'obs'
      - 'action'
      - 'reward' : scalar-like
      - 'done'   : scalar-like/bool
    optional:
      - 'next_obs'
      - 'value'    (scalar-like)
      - 'log_prob' (scalar-like)
    """

    is_off_policy: bool = False

    _EARLY_STOP_KEY = "train/early_stop"

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        rollout_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: Optional[int] = 64,
        device: Optional[Union[str, th.device]] = None,
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
        normalize_advantages: bool = False,
        adv_eps: float = 1e-8,
    ) -> None:
        super().__init__(head=head, core=core, device=device)

        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = None if minibatch_size is None else int(minibatch_size)

        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act

        self.normalize_advantages = bool(normalize_advantages)
        self.adv_eps = float(adv_eps)

        self.rollout: Optional[RolloutBuffer] = None
        self._last_obs: Optional[Any] = None
        self._last_done: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        """
        Initialize RolloutBuffer from env spaces.

        Notes
        -----
        Uses project utility `infer_shape(space)` to extract tensor shapes.
        """
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got: {self.rollout_steps}")

        obs_shape = infer_shape(env.observation_space, name="observation_space")
        action_shape = infer_shape(env.action_space, name="action_space")

        self.rollout = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
            dtype_obs=self.dtype_obs,
            dtype_act=self.dtype_act,
            normalize_advantages=self.normalize_advantages,
            adv_eps=self.adv_eps,
        )

    def remaining_rollout_steps(self) -> int:
        """
        For Ray on-policy collection scheduling.

        Returns how many more transitions are needed to fill the rollout buffer.
        When <= 0, the algorithm is ready to update (or should update before collecting more).
        """
        if self.rollout is None:
            return int(self.rollout_steps)

        # RolloutBuffer에 현재 저장된 step 수를 나타내는 속성명이 구현마다 다를 수 있으니
        # 가장 흔한 후보들을 순차로 시도 (fallback 포함)
        for attr in ("pos", "ptr", "idx", "t", "step", "n", "size"):
            if hasattr(self.rollout, attr):
                try:
                    cur = int(getattr(self.rollout, attr))
                    return max(0, int(self.rollout_steps) - cur)
                except Exception:
                    pass

        # 마지막 fallback: full이면 0, 아니면 rollout_steps(보수적으로)
        return 0 if bool(getattr(self.rollout, "full", False)) else int(self.rollout_steps)


    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Add one transition to the rollout buffer.

        Required keys
        -------------
        - obs, action, reward, done

        Optional keys
        -------------
        - next_obs, value, log_prob
        If value/log_prob are missing, they are computed via head.evaluate_actions().
        In that case, log_prob is reduced to a joint scalar by summing action dims.
        """
        if self.rollout is None:
            raise RuntimeError("RolloutBuffer not initialized. Call setup(env) first.")

        self._env_steps += 1

        obs_raw = transition["obs"]
        act_raw = transition["action"]
        rew = require_scalar_like(transition["reward"], name="transition['reward']")
        done = bool(require_scalar_like(transition["done"], name="transition['done']"))

        obs_np = to_numpy(obs_raw).astype(self.dtype_obs, copy=False)
        act_np = np.asarray(to_action_np(act_raw), dtype=self.dtype_act)
        value_any = transition.get("value", None)
        logp_any = transition.get("log_prob", None)

        value, log_prob = self._resolve_value_and_logp(
            obs_raw=obs_raw,
            act_raw=act_raw,
            value_any=value_any,
            logp_any=logp_any,
        )

        self.rollout.add(
            obs=obs_np,
            action=act_np,
            reward=rew,
            done=done,
            value=value,
            log_prob=log_prob,
        )

        self._last_done = done
        self._last_obs = transition.get("next_obs", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_value_and_logp(
        self,
        *,
        obs_raw: Any,
        act_raw: Any,
        value_any: Any,
        logp_any: Any,
    ) -> Tuple[float, float]:
        """
        Return (value, log_prob) as Python floats.

        Policy
        ------
        - If transition provides both value/log_prob: use them as-is (scalar-like).
        - Otherwise: compute via head.evaluate_actions(obs, action).
        log_prob may be per-dim (B,A); we reduce to joint by summing over non-batch dims.
        Then require scalar-like (typically B==1).
        """
        if (value_any is not None) and (logp_any is not None):
            value = require_scalar_like(value_any, name="transition['value']")
            log_prob = require_scalar_like(logp_any, name="transition['log_prob']")
            return float(value), float(log_prob)

        eval_out = require_mapping(
            self.head.evaluate_actions(obs_raw, act_raw),
            name="head.evaluate_actions(obs, action)",
        )

        value = require_scalar_like(eval_out.get("value", None), name="evaluate_actions()['value']")

        logp = eval_out.get("log_prob", None)
        if logp is None:
            raise ValueError("evaluate_actions() must return 'log_prob'.")

        log_prob = self._joint_logp_scalar(logp, name="evaluate_actions()['log_prob'] (joint summed)")
        return float(value), float(log_prob)


    def _joint_logp_scalar(self, logp_any: Any, *, name: str) -> float:
        """
        Convert log_prob output to a scalar-like joint log_prob.

        Accepts:
        - torch.Tensor: (B,), (B,1), (B,A), (B,...) -> reduce to (B,) by summing dims != batch
        - array-like: similar reduction semantics

        Then enforces scalar-like via require_scalar_like (typically B==1).
        """
        if th.is_tensor(logp_any):
            lp = logp_any
            if lp.dim() >= 2:
                lp = lp.flatten(start_dim=1).sum(dim=1)  # (B,)
            return float(require_scalar_like(lp, name=name))

        arr = np.asarray(logp_any)
        if arr.ndim >= 2:
            arr = arr.reshape(arr.shape[0], -1).sum(axis=1)  # (B,)
        return float(require_scalar_like(arr, name=name))


    def ready_to_update(self) -> bool:
        """True if rollout buffer is full."""
        return (self.rollout is not None) and bool(self.rollout.full)

    # ------------------------------------------------------------------
    # Bootstrap value for GAE
    # ------------------------------------------------------------------
    def _bootstrap_last_value(self) -> float:
        """
        Compute V(s_T) for GAE bootstrapping.

        Rules
        -----
        - If episode terminated at last step: return 0.0
        - Else:
            prefer head.value_only(next_obs)
            fallback: evaluate_actions(next_obs, act(next_obs, deterministic=True))
        """
        if self._last_done or (self._last_obs is None):
            return 0.0

        value_only = getattr(self.head, "value_only", None)
        if callable(value_only):
            v = value_only(self._last_obs)
            return float(require_scalar_like(v, name="head.value_only(next_obs)"))

        aT = self.head.act(self._last_obs, deterministic=True)
        outT = require_mapping(
            self.head.evaluate_actions(self._last_obs, aT),
            name="head.evaluate_actions(next_obs, aT) [fallback]",
        )
        vT = require_scalar_like(outT.get("value", None), name="evaluate_actions(fallback)['value']")
        return float(vT)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """
        Run PPO/A2C-style updates from the filled RolloutBuffer.

        Behavior
        --------
        - Bootstraps last value (if needed) and computes returns/advantages
        - Iterates update_epochs and minibatches
        - Aggregates scalar metrics returned by core
        - Supports early stop if metrics contain `train/early_stop` > 0

        Returns
        -------
        metrics : Dict[str, float]
            Mean-aggregated metrics plus bookkeeping under 'onpolicy/*'.
        """
        if self.rollout is None:
            raise RuntimeError("RolloutBuffer not initialized. Call setup(env) first.")
        if self.update_epochs <= 0:
            raise ValueError(f"update_epochs must be > 0, got: {self.update_epochs}")
        if self.minibatch_size is not None and self.minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be None or > 0, got: {self.minibatch_size}")

        # 1) bootstrap + compute GAE returns/advantages
        last_value = self._bootstrap_last_value()
        self.rollout.compute_returns_and_advantage(last_value=last_value, last_done=self._last_done)

        # 2) sampling config
        rollout_size = int(self.rollout_steps)
        if self.minibatch_size is None:
            batch_size, shuffle = rollout_size, False
        else:
            batch_size, shuffle = min(self.minibatch_size, rollout_size), True

        # 3) update loop with metric aggregation
        sums: Dict[str, float] = {}
        num_minibatches = 0

        early_stop = False
        early_stop_epoch = -1

        for ep in range(self.update_epochs):
            for batch in self.rollout.sample(batch_size=batch_size, shuffle=shuffle):
                out_any = self.core.update_from_batch(batch)
                metrics_any = dict(out_any) if isinstance(out_any, Mapping) else {}

                # scalar aggregation
                for k, v in metrics_any.items():
                    if is_scalar_like(v):
                        sv = to_scalar(v)
                        if sv is not None:
                            key = str(k)
                            sums[key] = sums.get(key, 0.0) + float(sv)

                num_minibatches += 1

                # early stop (PPO target_kl, etc.)
                es = to_scalar(metrics_any.get(self._EARLY_STOP_KEY, 0.0))
                if es is not None and float(es) > 0.0:
                    early_stop = True
                    early_stop_epoch = int(ep)
                    break

            if early_stop:
                break

        means: Dict[str, float] = {}
        if num_minibatches > 0:
            inv = 1.0 / float(num_minibatches)
            for k, v in sums.items():
                means[k] = v * inv

        # bookkeeping metrics
        means["onpolicy/rollout_steps"] = float(self.rollout_steps)
        means["onpolicy/num_minibatches"] = float(num_minibatches)
        means["onpolicy/env_steps"] = float(self._env_steps)
        means["onpolicy/early_stop"] = 1.0 if early_stop else 0.0
        means["onpolicy/early_stop_epoch"] = float(early_stop_epoch if early_stop else -1)

        # NEW: learner update accounting
        # PPO/A2C style: one learner update == one minibatch update_from_batch call
        means["sys/num_updates"] = float(num_minibatches)

        # 4) reset rollout state
        self.rollout.reset()
        self._last_obs = None
        self._last_done = False

        return means