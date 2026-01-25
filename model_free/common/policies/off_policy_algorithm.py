from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import inspect
import numpy as np
import torch as th

from ..buffers import ReplayBuffer, PrioritizedReplayBuffer
from ..utils.common_utils import (
    to_numpy,
    to_flat_np,
    require_scalar_like,
    infer_shape,
    to_action_np,
)
from .base_policy import BasePolicyAlgorithm
from ..utils.policy_utils import infer_n_actions_from_env


class OffPolicyAlgorithm(BasePolicyAlgorithm):
    """
    Off-policy algorithm driver composing:
      - head: action selection (exploration/noise lives in head)
      - core: gradient updates from replay batches
      - buffer: replay buffer (uniform or PER)

    Schedule semantics
    ------------------
    warmup_steps:
      - for env_steps < warmup_steps, actions come from random sampling.

    update_after:
      - updates are disabled until env_steps >= update_after.

    update_every:
      - updates are only allowed on steps where env_steps % update_every == 0
        (if update_every > 1).

    utd ("updates-to-data"):
      - fractional update budget accumulated per env step:
          update_budget += utd * n_new_steps
      - When update_budget >= 1.0, we can perform updates.
      - Each "update unit" consumes 1.0 budget and corresponds to:
          gradient_steps times calling update_once() and averaging metrics.

    gradient_steps:
      - number of gradient steps per update unit (averaged in logs).
      - If you want classic behavior (1 update per unit), set gradient_steps=1.

    Notes
    -----
    - This driver does NOT own exploration noise.
      If deterministic=False is passed to head.act(), head decides how to inject noise.
    - Behavior policy statistics can optionally be stored in buffer:
        * behavior_logp (scalar)
        * behavior_probs (A,) for discrete policies
    """

    is_off_policy: bool = True
    _PRIORITIES_KEY = "per/priorities"
    _TD_ERRORS_KEY = "per/td_errors"

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        # replay + schedule
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 10_000,
        update_after: int = 1_000,
        update_every: int = 1,
        utd: float = 1.0,
        gradient_steps: int = 1,
        max_updates_per_call: int = 1_000,
        # storage
        device: Optional[Union[str, th.device]] = None,
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
        # PER
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        per_beta_final: float = 1.0,
        per_beta_anneal_steps: int = 200_000,
        # behavior stats storage
        store_behavior_logp: bool = False,
        store_behavior_probs: bool = False,
        # n-step
        n_step: int = 1,
        # head-side noise reset policy
        reset_noise_on_done: bool = False,
        # -----------------------------
        # NEW: epsilon-greedy exploration (for DQN/Q-learning heads)
        # -----------------------------
        exploration_eps: float = 0.1,
        exploration_eps_final: float = 0.05,
        exploration_eps_anneal_steps: int = 200_000,
        exploration_eval_eps: float = 0.0,
    ) -> None:
        super().__init__(head=head, core=core, device=device)

        # schedule / sizing
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.utd = float(utd)
        self.gradient_steps = int(gradient_steps)
        self.max_updates_per_call = int(max_updates_per_call)

        # storage dtypes
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act

        # PER hyperparams
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        self.per_beta_final = float(per_beta_final)
        self.per_beta_anneal_steps = int(per_beta_anneal_steps)

        # optional behavior storage
        self.store_behavior_logp = bool(store_behavior_logp)
        self.store_behavior_probs = bool(store_behavior_probs)
        self.n_actions: Optional[int] = None  # inferred in setup if needed

        # n-step
        self.n_step = int(n_step)
        self.n_step_gamma = float(getattr(self.core, "gamma", 0.99))
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got: {self.n_step}")

        # head-side noise reset policy
        self.reset_noise_on_done = bool(reset_noise_on_done)

        # -----------------------------
        # NEW: epsilon exploration config
        # -----------------------------
        self.exploration_eps = float(exploration_eps)
        self.exploration_eps_final = float(exploration_eps_final)
        self.exploration_eps_anneal_steps = int(exploration_eps_anneal_steps)
        self.exploration_eval_eps = float(exploration_eval_eps)

        if self.exploration_eps < 0.0:
            raise ValueError(f"exploration_eps must be >= 0, got {self.exploration_eps}")
        if self.exploration_eps_final < 0.0:
            raise ValueError(f"exploration_eps_final must be >= 0, got {self.exploration_eps_final}")
        if self.exploration_eps_anneal_steps < 0:
            raise ValueError(f"exploration_eps_anneal_steps must be >= 0, got {self.exploration_eps_anneal_steps}")

        # Cache whether head.act supports epsilon to avoid repeated introspection
        self._head_act_accepts_epsilon = False
        try:
            sig = inspect.signature(self.head.act)
            self._head_act_accepts_epsilon = ("epsilon" in sig.parameters)
        except Exception:
            self._head_act_accepts_epsilon = False

        # runtime state
        self.buffer: Optional[Union[ReplayBuffer, PrioritizedReplayBuffer]] = None
        self._action_space: Optional[Any] = None
        self._action_shape: Optional[tuple[int, ...]] = None

        # Warmup sampling mode: if env expects actions in policy space (e.g., [-1,1])
        self._warmup_policy_action_space: bool = False

        # fractional updates budget
        self._update_budget: float = 0.0

        self._validate_hparams()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_hparams(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got: {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {self.batch_size}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got: {self.warmup_steps}")
        if self.update_after < 0:
            raise ValueError(f"update_after must be >= 0, got: {self.update_after}")
        if self.update_every <= 0:
            raise ValueError(f"update_every must be > 0, got: {self.update_every}")
        if self.gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be > 0, got: {self.gradient_steps}")
        if self.max_updates_per_call <= 0:
            raise ValueError(f"max_updates_per_call must be > 0, got: {self.max_updates_per_call}")
        if self.utd < 0.0:
            raise ValueError(f"utd must be >= 0, got: {self.utd}")

        if self.use_per:
            if self.per_alpha < 0.0:
                raise ValueError(f"per_alpha must be >= 0, got: {self.per_alpha}")
            if self.per_eps <= 0.0:
                raise ValueError(f"per_eps must be > 0, got: {self.per_eps}")
            if self.per_beta_anneal_steps < 0:
                raise ValueError(f"per_beta_anneal_steps must be >= 0, got: {self.per_beta_anneal_steps}")

    # =============================================================================
    # Epsilon schedule (DQN-style)
    # =============================================================================
    def _current_exploration_eps(self) -> float:
        """
        Linear annealing from exploration_eps -> exploration_eps_final over
        exploration_eps_anneal_steps environment steps.
        """
        if self.exploration_eps_anneal_steps <= 0:
            return float(self.exploration_eps)

        t = min(max(int(self._env_steps), 0), int(self.exploration_eps_anneal_steps))
        frac = float(t) / float(self.exploration_eps_anneal_steps)
        eps = self.exploration_eps + frac * (self.exploration_eps_final - self.exploration_eps)
        return float(max(eps, 0.0))
    
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        obs_shape = infer_shape(env.observation_space, name="observation_space")
        action_shape = infer_shape(env.action_space, name="action_space")

        self._action_space = env.action_space
        self._action_shape = tuple(int(x) for x in action_shape)

        # If env has an "action_rescale" flag, we interpret warmup sampling space accordingly.
        self._warmup_policy_action_space = bool(getattr(env, "action_rescale", False))

        # Infer n_actions only if we need to store discrete behavior probabilities
        if self.store_behavior_probs:
            self.n_actions = int(infer_n_actions_from_env(env))

        nstep_kwargs: Dict[str, Any] = {}
        if self.n_step > 1:
            nstep_kwargs = {"n_step": int(self.n_step), "gamma": float(self.n_step_gamma)}

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=obs_shape,
                action_shape=action_shape,
                alpha=self.per_alpha,
                beta=self.per_beta,
                eps=self.per_eps,
                device=self.device,
                dtype_obs=self.dtype_obs,
                dtype_act=self.dtype_act,
                store_behavior_logp=self.store_behavior_logp,
                store_behavior_probs=self.store_behavior_probs,
                n_actions=self.n_actions,
                **nstep_kwargs,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=obs_shape,
                action_shape=action_shape,
                device=self.device,
                dtype_obs=self.dtype_obs,
                dtype_act=self.dtype_act,
                store_behavior_logp=self.store_behavior_logp,
                store_behavior_probs=self.store_behavior_probs,
                n_actions=self.n_actions,
                **nstep_kwargs,
            )

        # Optional head-side noise reset at initialization
        reset_fn = getattr(self.head, "reset_exploration_noise", None)
        if callable(reset_fn):
            reset_fn()

        self._env_steps = 0
        self._update_budget = 0.0

    # =============================================================================
    # Acting
    # =============================================================================
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """
        Action selection.

        - Warmup: random action.
        - After warmup:
            * If head.act supports epsilon: pass epsilon-greedy parameter.
            * Else: call head.act with deterministic only (continuous policies, etc.)
        """
        if self.warmup_steps > 0 and self._env_steps < self.warmup_steps:
            return self._sample_random_action()

        # Evaluation / deterministic mode: epsilon typically should be 0.
        if deterministic:
            eps = float(self.exploration_eval_eps)
        else:
            eps = float(self._current_exploration_eps())

        if self._head_act_accepts_epsilon:
            # DQN/Q-learning heads
            return self.head.act(obs, epsilon=eps, deterministic=deterministic)

        # Actor-critic heads (SAC/TD3/DDPG etc.)
        return self.head.act(obs, deterministic=deterministic)

    # =============================================================================
    # Data ingestion
    # =============================================================================
    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Ingest one transition into replay buffer.

        Required keys
        -------------
        - obs, action, reward, next_obs, done

        Optional keys (if enabled)
        --------------------------
        - behavior_logp  (store_behavior_logp=True)
        - behavior_probs (store_behavior_probs=True), shape (A,)
        - priority (PER only): initial priority override
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        self._env_steps += 1

        obs_np = np.asarray(to_flat_np(transition["obs"]), dtype=self.dtype_obs)
        next_obs_np = np.asarray(to_flat_np(transition["next_obs"]), dtype=self.dtype_obs)

        reward = require_scalar_like(transition["reward"], name="transition['reward']")
        done = bool(require_scalar_like(transition["done"], name="transition['done']"))

        action_np = to_action_np(transition["action"], action_shape=self._action_shape)
        action_np = np.asarray(action_np, dtype=self.dtype_act)

        beh_logp: Optional[float] = None
        beh_probs: Optional[np.ndarray] = None

        if self.store_behavior_logp:
            if "behavior_logp" not in transition:
                raise ValueError("store_behavior_logp=True requires transition['behavior_logp'].")
            beh_logp = require_scalar_like(transition["behavior_logp"], name="transition['behavior_logp']")

        if self.store_behavior_probs:
            if "behavior_probs" not in transition:
                raise ValueError("store_behavior_probs=True requires transition['behavior_probs'].")
            if self.n_actions is None:
                raise RuntimeError("n_actions is not set. Did you call setup(env)?")
            bp = np.asarray(to_numpy(transition["behavior_probs"]), dtype=np.float32).reshape(-1)
            if bp.shape[0] != int(self.n_actions):
                raise ValueError(f"behavior_probs must have shape (A,), got {bp.shape} (A={self.n_actions})")
            beh_probs = bp

        priority: Optional[float] = None
        if self.use_per and ("priority" in transition):
            priority = require_scalar_like(transition["priority"], name="transition['priority']")

        # Add to buffer
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.add(
                obs=obs_np,
                action=action_np,
                reward=reward,
                next_obs=next_obs_np,
                done=done,
                priority=priority,
                behavior_logp=beh_logp,
                behavior_probs=beh_probs,
            )
        else:
            self.buffer.add(
                obs=obs_np,
                action=action_np,
                reward=reward,
                next_obs=next_obs_np,
                done=done,
                behavior_logp=beh_logp,
                behavior_probs=beh_probs,
            )

        # Accumulate budget for updates
        self._accumulate_update_budget(n_new_steps=1)

        # Episode boundary: reset head-side stateful noise (e.g., OU noise)
        if done and self.reset_noise_on_done:
            reset_fn = getattr(self.head, "reset_exploration_noise", None)
            if callable(reset_fn):
                reset_fn()

    def _accumulate_update_budget(self, n_new_steps: int) -> None:
        if self.utd <= 0.0:
            return
        self._update_budget += self.utd * float(int(n_new_steps))

    # =============================================================================
    # Update scheduling
    # =============================================================================
    def ready_to_update(self) -> bool:
        if self.buffer is None:
            return False
        if self._env_steps < self.update_after:
            return False
        if self.buffer.size < self.batch_size:
            return False
        if self._update_budget < 1.0:
            return False
        if self.update_every > 1 and (self._env_steps % self.update_every) != 0:
            return False
        return True

    # =============================================================================
    # PER helpers
    # =============================================================================
    def _current_per_beta(self) -> float:
        """
        Linear annealing from per_beta -> per_beta_final over per_beta_anneal_steps env steps.
        """
        if self.per_beta_anneal_steps <= 0:
            return float(self.per_beta)

        t = min(max(int(self._env_steps), 0), int(self.per_beta_anneal_steps))
        frac = float(t) / float(self.per_beta_anneal_steps)
        beta = self.per_beta + frac * (self.per_beta_final - self.per_beta)
        return float(beta)

    @staticmethod
    def _get_batch_indices(batch: Any) -> Optional[np.ndarray]:
        idx = getattr(batch, "indices", None)
        if idx is None:
            return None
        if th.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        return np.asarray(idx, dtype=np.int64).reshape(-1)

    def _maybe_update_per_priorities(self, *, batch: Any, metrics: Dict[str, Any]) -> None:
        """
        Update PER priorities if metrics include:
          - 'per/priorities' : (B,)
          - 'per/td_errors'  : (B,)  (abs is taken)
        """
        if not isinstance(self.buffer, PrioritizedReplayBuffer):
            return

        indices = self._get_batch_indices(batch)
        if indices is None or indices.size == 0:
            return

        pr = metrics.pop(self._PRIORITIES_KEY, None)
        td = metrics.pop(self._TD_ERRORS_KEY, None)
        if pr is None and td is None:
            return

        arr = pr if pr is not None else td
        pr_np = np.asarray(to_flat_np(arr), dtype=np.float32).reshape(-1)

        if pr is None and td is not None:
            pr_np = np.abs(pr_np)

        if pr_np.shape[0] != indices.shape[0]:
            # Mismatch: refuse update rather than corrupt buffer priorities.
            return

        self.buffer.update_priorities(indices, pr_np)

    # =============================================================================
    # Update primitives
    # =============================================================================
    def update_once(self) -> Dict[str, float]:
        """
        Sample one batch and run one core update.

        Returns scalar-only metrics.
        For PER:
          - updates priorities if metrics provide priorities/td_errors
          - returns 'per/beta' and 'per/enabled'
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            beta = self._current_per_beta()
            batch = self.buffer.sample(self.batch_size, beta=beta)

            metrics_any = self.core.update_from_batch(batch)
            metrics = dict(metrics_any) if isinstance(metrics_any, Mapping) else {}

            self._maybe_update_per_priorities(batch=batch, metrics=metrics)

            out = self._filter_scalar_metrics(metrics)
            out["per/beta"] = float(beta)
            out["per/enabled"] = 1.0
            return out

        batch = self.buffer.sample(self.batch_size)
        metrics_any = self.core.update_from_batch(batch)
        return self._filter_scalar_metrics(metrics_any)

    def update(self) -> Dict[str, float]:
        """
        Run as many updates as allowed by update_budget, capped by max_updates_per_call.

        Returns
        -------
        metrics : Dict[str, float]
            Mean-aggregated scalar metrics plus offpolicy/* bookkeeping.
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        owed = int(self._update_budget)
        if owed <= 0:
            out: Dict[str, float] = {
                "offpolicy/buffer_size": float(self.buffer.size),
                "offpolicy/env_steps": float(self._env_steps),
                "offpolicy/update_budget": float(self._update_budget),
            }
            if self.use_per:
                out["per/enabled"] = 1.0
            return out

        n_updates = min(owed, self.max_updates_per_call)
        gs = int(self.gradient_steps)

        agg: Dict[str, float] = {}

        for _ in range(n_updates):
            inner: Dict[str, float] = {}
            for _g in range(gs):
                m = self.update_once()
                for k, v in m.items():
                    inner[k] = inner.get(k, 0.0) + float(v)

            inv_gs = 1.0 / float(gs)
            for k in list(inner.keys()):
                inner[k] *= inv_gs

            for k, v in inner.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        inv_n = 1.0 / float(n_updates)
        for k in list(agg.keys()):
            agg[k] *= inv_n

        # Consume budget (one per update unit)
        self._update_budget -= float(n_updates)

        # -----------------------
        # Bookkeeping (existing)
        # -----------------------
        agg["offpolicy/buffer_size"] = float(self.buffer.size)
        agg["offpolicy/env_steps"] = float(self._env_steps)
        agg["offpolicy/updates_ran"] = float(n_updates)  # update units
        agg["offpolicy/update_budget"] = float(self._update_budget)
        if self.use_per:
            agg["per/enabled"] = 1.0

        # -----------------------
        # NEW: learner update accounting
        # -----------------------
        # "actual learner updates" == number of gradient steps executed
        agg["sys/num_updates"] = float(n_updates * gs)        # <-- 핵심
        agg["offpolicy/grad_steps"] = float(gs)               # 디버깅/가시성용(선택)
        agg["offpolicy/update_units"] = float(n_updates)      # 명시적 이름(선택)
        agg["exploration/epsilon"] = float(self._current_exploration_eps())
        
        return agg

    # =============================================================================
    # Random action sampling (warmup)
    # =============================================================================
    def _sample_random_action(self) -> Any:
        """
        Sample a random action for warmup.

        Policy-space warmup
        -------------------
        If env exposes action_rescale=True, we assume the policy emits actions in [-1,1]
        (common in continuous control), so we sample uniformly in [-1,1] for Box spaces.

        Otherwise:
        - use action_space.sample() if available
        - fallback to common patterns (Discrete, Box)
        """
        if self._action_space is None or self._action_shape is None:
            raise RuntimeError("action_space/action_shape not cached. Did you call setup(env)?")

        space = self._action_space

        # Prefer gym-style sample() if present
        if callable(getattr(space, "sample", None)):
            a = space.sample()

            # If we do NOT want policy-space warmup, keep the env sample.
            if not self._warmup_policy_action_space:
                return a

            # If we *do* want policy-space warmup, only override for Box-like spaces.
            # For discrete spaces, env sample is already correct.
            if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
                shape = tuple(int(x) for x in space.shape)
                return np.random.uniform(-1.0, 1.0, size=shape).astype(self.dtype_act)

            return a  # fallback

        # Discrete-like fallback
        if hasattr(space, "n"):
            return int(np.random.randint(0, int(space.n)))

        # Box-like fallback (no sample())
        if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            shape = tuple(int(x) for x in space.shape)

            low = np.broadcast_to(low, shape) if low.shape != shape else low
            high = np.broadcast_to(high, shape) if high.shape != shape else high

            if self._warmup_policy_action_space:
                return np.random.uniform(-1.0, 1.0, size=shape).astype(self.dtype_act)

            return np.random.uniform(low=low, high=high, size=shape).astype(self.dtype_act)

        raise ValueError(f"Unsupported action_space for random sampling: {space}")