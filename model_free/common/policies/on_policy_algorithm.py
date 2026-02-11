from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch as th

from ..buffers import RolloutBuffer
from ..utils.common_utils import (
    _is_scalar_like,
    _infer_shape,
    _require_mapping,
    _require_scalar_like,
    _to_action_np,
    _to_numpy,
    _to_scalar,
)
from .base_policy import BasePolicyAlgorithm


class OnPolicyAlgorithm(BasePolicyAlgorithm):
    """
    On-policy algorithm driver (rollout + PPO/A2C/TRPO-style update loop).

    This driver owns a :class:`~RolloutBuffer` and implements the standard
    on-policy lifecycle:

    1) Collect `rollout_steps` transitions into the rollout buffer.
    2) Bootstrap the last-state value (for GAE) if the rollout ended mid-episode.
    3) Compute returns/advantages in the buffer.
    4) Run multiple epochs of minibatch SGD updates via `core.update_from_batch(batch)`.
    5) Aggregate scalar-like metrics and reset rollout state.

    Parameters and interfaces are intentionally duck-typed to allow multiple head/core
    implementations.

    Required interfaces (duck-typed)
    --------------------------------
    head
        - act(obs, deterministic=False) -> action
        - evaluate_actions(obs, action) -> Mapping[str, Any] with at least:
            * "value"    : scalar-like or tensor/array containing a scalar (typically B==1)
            * "log_prob" : scalar-like or tensor/array containing a scalar (typically B==1)
        Optional:
        - value_only(obs) -> V(s) (preferred for bootstrap)

    core
        - update_from_batch(batch) -> Mapping[str, Any]
          Scalar-like values will be aggregated into logs.

    Transition contract (on_env_step)
    ---------------------------------
    transition must contain:
        - "obs"
        - "action"
        - "reward" : scalar-like
        - "done"   : scalar-like / bool
    optional:
        - "next_obs"
        - "value"    : scalar-like (if already computed during acting)
        - "log_prob" : scalar-like (if already computed during acting)

    Notes
    -----
    - When value/log_prob are not provided in the transition, they are computed
      via `head.evaluate_actions(obs, action)`.
    - Log-probabilities may be per-dimension (e.g., (B,A)). In that case, this
      driver reduces them to a joint scalar by summing across non-batch dims.
    """

    is_off_policy: bool = False

    # A convention key a core can emit to request early stop (e.g., PPO target_kl)
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
        """
        Parameters
        ----------
        head : Any
            Policy/value head. See class docstring for required methods.
        core : Any
            Learner core that updates parameters from a sampled rollout batch.
        rollout_steps : int, default=2048
            Number of env transitions to collect before an update.
        gamma : float, default=0.99
            Discount factor used inside the rollout buffer (returns/GAE).
        gae_lambda : float, default=0.95
            GAE parameter λ.
        update_epochs : int, default=10
            Number of passes over the rollout data per update.
        minibatch_size : Optional[int], default=64
            Minibatch size for sampling from the rollout buffer during update.
            If None, uses full-batch updates without shuffling.
        device : Optional[Union[str, torch.device]], optional
            Device used by the rollout buffer (and potentially head/core).
        dtype_obs : Any, default=np.float32
            Storage dtype for observations in the rollout buffer.
        dtype_act : Any, default=np.float32
            Storage dtype for actions in the rollout buffer.
        normalize_advantages : bool, default=False
            If True, normalize advantages inside the rollout buffer.
        adv_eps : float, default=1e-8
            Numerical epsilon used in advantage normalization.
        """
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
        Initialize :class:`~RolloutBuffer` from environment spaces.

        Parameters
        ----------
        env : Any
            Environment with `.observation_space` and `.action_space`.

        Raises
        ------
        ValueError
            If `rollout_steps` is invalid.

        Notes
        -----
        Uses project utility :func:`infer_shape` to extract shapes from spaces.
        """
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got: {self.rollout_steps}")

        obs_shape = _infer_shape(env.observation_space, name="observation_space")
        action_shape = _infer_shape(env.action_space, name="action_space")

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
        Return how many more transitions are needed to fill the rollout buffer.

        This is useful for Ray-style rollout scheduling.

        Returns
        -------
        remaining : int
            Remaining transitions to collect. When <= 0, the algorithm is ready
            to update (or should update before collecting more).

        Notes
        -----
        RolloutBuffer implementations may store the write pointer under different
        attribute names; this method tries common candidates and falls back to:
        - 0 if `rollout.full` is True
        - rollout_steps otherwise (conservative)
        """
        if self.rollout is None:
            return int(self.rollout_steps)

        for attr in ("pos", "ptr", "idx", "t", "step", "n", "size"):
            if hasattr(self.rollout, attr):
                try:
                    cur = int(getattr(self.rollout, attr))
                    return max(0, int(self.rollout_steps) - cur)
                except Exception:
                    pass

        return 0 if bool(getattr(self.rollout, "full", False)) else int(self.rollout_steps)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Add one transition to the rollout buffer.

        Parameters
        ----------
        transition : Dict[str, Any]
            Required keys:
            - "obs"
            - "action"
            - "reward" : scalar-like
            - "done"   : scalar-like/bool
            Optional keys:
            - "next_obs"
            - "value"
            - "log_prob"

        Notes
        -----
        - If "value" and "log_prob" are absent, they are computed via
          `head.evaluate_actions(obs, action)`.
        - `log_prob` may be per-dimension; it will be reduced to a joint scalar by
          summing non-batch dimensions, then validated as scalar-like (usually B==1).
        """
        if self.rollout is None:
            raise RuntimeError("RolloutBuffer not initialized. Call setup(env) first.")

        self._env_steps += 1

        obs_raw = transition["obs"]
        act_raw = transition["action"]
        rew = _require_scalar_like(transition["reward"], name="transition['reward']")
        done = bool(_require_scalar_like(transition["done"], name="transition['done']"))

        obs_np = _to_numpy(obs_raw).astype(self.dtype_obs, copy=False)
        act_np = np.asarray(_to_action_np(act_raw), dtype=self.dtype_act)

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

        # Cache last step termination + last observation for bootstrapping.
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
        Resolve (value, log_prob) for the current transition.

        Parameters
        ----------
        obs_raw : Any
            Raw observation passed to head for evaluation if needed.
        act_raw : Any
            Raw action passed to head for evaluation if needed.
        value_any : Any
            Transition-provided value baseline estimate (optional).
        logp_any : Any
            Transition-provided log-probability (optional).

        Returns
        -------
        value : float
            Baseline value estimate V(s) as a Python float.
        log_prob : float
            Joint log-probability log π(a|s) as a Python float.

        Notes
        -----
        Policy:
        - If transition provides both `value` and `log_prob`, they are used
          as-is after scalar validation.
        - Otherwise, compute via `head.evaluate_actions(obs, action)` and extract:
          - "value" (scalar-like)
          - "log_prob" (possibly per-dim; reduced to a joint scalar)
        """
        if (value_any is not None) and (logp_any is not None):
            value = _require_scalar_like(value_any, name="transition['value']")
            log_prob = _require_scalar_like(logp_any, name="transition['log_prob']")
            return float(value), float(log_prob)

        eval_out = _require_mapping(
            self.head.evaluate_actions(obs_raw, act_raw),
            name="head.evaluate_actions(obs, action)",
        )

        value = _require_scalar_like(eval_out.get("value", None), name="evaluate_actions()['value']")

        logp = eval_out.get("log_prob", None)
        if logp is None:
            raise ValueError("evaluate_actions() must return 'log_prob'.")

        log_prob = self._joint_logp_scalar(logp, name="evaluate_actions()['log_prob'] (joint summed)")
        return float(value), float(log_prob)

    def _joint_logp_scalar(self, logp_any: Any, *, name: str) -> float:
        """
        Convert a log_prob output into a scalar-like joint log_prob.

        Parameters
        ----------
        logp_any : Any
            Log-probability output which may be:
            - torch.Tensor of shape (B,), (B,1), (B,A), (B,...)  (common)
            - array-like with similar semantics
        name : str
            Name used in error messages.

        Returns
        -------
        log_prob : float
            Joint log-probability as a Python float.

        Notes
        -----
        Reduction rule:
        - If logp has ndim >= 2, it is reduced to (B,) by flattening from dim=1
          and summing across the remaining dims.
        - The result is then validated as scalar-like, which typically implies
          B == 1 at collection time (common for synchronous env stepping).
        """
        if th.is_tensor(logp_any):
            lp = logp_any
            if lp.dim() >= 2:
                lp = lp.flatten(start_dim=1).sum(dim=1)  # (B,)
            return float(_require_scalar_like(lp, name=name))

        arr = np.asarray(logp_any)
        if arr.ndim >= 2:
            arr = arr.reshape(arr.shape[0], -1).sum(axis=1)  # (B,)
        return float(_require_scalar_like(arr, name=name))

    # ------------------------------------------------------------------
    # Update readiness
    # ------------------------------------------------------------------
    def ready_to_update(self) -> bool:
        """
        Return True if the rollout buffer is full.

        Returns
        -------
        ready : bool
            True if `self.rollout.full` is True.
        """
        return (self.rollout is not None) and bool(self.rollout.full)

    # ------------------------------------------------------------------
    # Bootstrap value for GAE
    # ------------------------------------------------------------------
    def _bootstrap_last_value(self) -> float:
        """
        Compute V(s_T) for GAE bootstrapping at the end of a rollout.

        Returns
        -------
        last_value : float
            Bootstrap value used by GAE/returns computation.

        Notes
        -----
        Rules:
        - If the last transition ended an episode (`done=True`) or `next_obs` is missing,
          returns 0.0 (no bootstrap across terminal).
        - Otherwise, prefers `head.value_only(next_obs)` (cheap and explicit).
        - Fallback: compute `a_T = head.act(next_obs, deterministic=True)` and call
          `head.evaluate_actions(next_obs, a_T)` to extract "value".
        """
        if self._last_done or (self._last_obs is None):
            return 0.0

        value_only = getattr(self.head, "value_only", None)
        if callable(value_only):
            v = value_only(self._last_obs)
            return float(_require_scalar_like(v, name="head.value_only(next_obs)"))

        aT = self.head.act(self._last_obs, deterministic=True)
        outT = _require_mapping(
            self.head.evaluate_actions(self._last_obs, aT),
            name="head.evaluate_actions(next_obs, aT) [fallback]",
        )
        vT = _require_scalar_like(outT.get("value", None), name="evaluate_actions(fallback)['value']")
        return float(vT)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """
        Run PPO/A2C-style learner updates from a filled rollout buffer.

        Returns
        -------
        metrics : Dict[str, float]
            Mean-aggregated scalar metrics plus bookkeeping metrics under
            'onpolicy/*' and 'sys/num_updates'.

        Raises
        ------
        RuntimeError
            If setup(env) was not called.
        ValueError
            If update hyperparameters are invalid.

        Notes
        -----
        Workflow:
        1) Bootstrap last value and compute returns/advantages in the rollout buffer.
        2) Iterate `update_epochs` and sample minibatches (optionally shuffled).
        3) Call `core.update_from_batch(batch)` and aggregate scalar-like metrics.
        4) Support early stop if `metrics["train/early_stop"] > 0`.
        5) Reset rollout state after finishing updates.
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

        # 2) minibatch sampling config
        rollout_size = int(self.rollout_steps)
        if self.minibatch_size is None:
            batch_size, shuffle = rollout_size, False
        else:
            batch_size, shuffle = min(self.minibatch_size, rollout_size), True

        # 3) update loop + metric aggregation
        sums: Dict[str, float] = {}
        num_minibatches = 0

        early_stop = False
        early_stop_epoch = -1

        for ep in range(self.update_epochs):
            for batch in self.rollout.sample(batch_size=batch_size, shuffle=shuffle):
                out_any = self.core.update_from_batch(batch)
                metrics_any = dict(out_any) if isinstance(out_any, Mapping) else {}

                for k, v in metrics_any.items():
                    if not _is_scalar_like(v):
                        continue
                    sv = _to_scalar(v)
                    if sv is None:
                        continue
                    key = str(k)
                    sums[key] = sums.get(key, 0.0) + float(sv)

                num_minibatches += 1

                # Early stop (e.g., PPO target KL)
                es = _to_scalar(metrics_any.get(self._EARLY_STOP_KEY, 0.0))
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

        # Bookkeeping metrics
        means["onpolicy/rollout_steps"] = float(self.rollout_steps)
        means["onpolicy/num_minibatches"] = float(num_minibatches)
        means["onpolicy/env_steps"] = float(self._env_steps)
        means["onpolicy/early_stop"] = 1.0 if early_stop else 0.0
        means["onpolicy/early_stop_epoch"] = float(early_stop_epoch if early_stop else -1)

        # Learner update accounting:
        # PPO/A2C style: one learner update == one minibatch `update_from_batch` call
        means["sys/num_updates"] = float(num_minibatches)

        # 4) reset rollout state
        self.rollout.reset()
        self._last_obs = None
        self._last_done = False

        return means
