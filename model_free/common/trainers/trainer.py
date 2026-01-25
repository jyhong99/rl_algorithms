from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import os
import numpy as np

try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Ray is an optional dependency; keep runtime imports Ray-free.
    from ..utils.ray_utils import PolicyFactorySpec
else:
    PolicyFactorySpec = Any

from ..utils.train_utils import (
    env_reset,
    make_pbar,
    maybe_call,
    wrap_make_env_with_normalize,
)

from ..wrappers.normalize_wrapper import NormalizeWrapper
from ..callbacks.episode_stats_callback import EpisodeStatsCallback

from .train_checkpoint import save_checkpoint, load_checkpoint
from .train_eval import run_evaluation
from .train_loop import train_single_env
from .train_ray import train_ray


class Trainer:
    """
    Unified trainer for on-policy and off-policy algorithms (single-env and Ray).

    This class is intentionally a thin orchestrator that delegates heavy logic to:
      - train_loop.py        (single-env loop)
      - train_ray.py         (ray multi-worker loop)
      - train_eval.py        (evaluation)
      - train_checkpoint.py  (checkpoint save/load)

    Required algorithm interface (duck-typed)
    ----------------------------------------
    algo must provide:
      - setup(env) -> None
      - set_training(training: bool) -> None
      - act(obs, deterministic: bool = False) -> Any
      - on_env_step(transition: Dict[str, Any]) -> None
      - ready_to_update() -> bool
      - update() -> Mapping[str, Any]

    Optional algorithm features
    ---------------------------
      - on_rollout(rollout) -> None
      - is_off_policy: bool
      - save(path: str) -> None
      - load(path: str) -> None
      - get_ray_policy_factory_spec() -> PolicyFactorySpec
      - remaining_rollout_steps() -> int  (on-policy ray path)

    Callback contract (best-effort)
    -------------------------------
    callbacks may provide:
      - on_train_start(trainer) -> bool
      - on_step(trainer, transition=...) -> bool
      - on_update(trainer, metrics=...) -> bool
      - on_eval_end(trainer, metrics=...) -> bool
      - on_checkpoint(trainer, path=...) -> bool
      - on_train_end(trainer) -> bool

    Notes
    -----
    - If NormalizeWrapper is enabled, both train_env and eval_env are wrapped in-place.
    - Episode indexing is owned by Trainer unless EpisodeStatsCallback is present.
    - Ray dependencies are isolated to train_ray/train_ray_utils modules.
    """

    def __init__(
        self,
        *,
        train_env: Any,
        eval_env: Any,
        algo: Any,
        # run control
        total_env_steps: int = 1_000_000,
        seed: int = 0,
        deterministic: bool = True,
        max_episode_steps: Optional[int] = None,
        log_every_steps: int = 5_000,
        # directories
        run_dir: str = "./runs/exp",
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: str = "ckpt",
        # parallel rollout
        n_envs: int = 1,
        rollout_steps_per_env: int = 256,
        sync_weights_every_updates: int = 10,
        utd: float = 1.0,
        # optional components
        logger: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        # ray injection (only used when n_envs > 1)
        ray_env_make_fn: Optional[Callable[[], Any]] = None,
        ray_policy_spec: Optional[PolicyFactorySpec] = None,
        ray_want_onpolicy_extras: Optional[bool] = None,
        # normalization (wrap envs if True)
        normalize: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        norm_obs: bool = True,
        norm_reward: bool = False,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        norm_gamma: float = 0.99,
        norm_epsilon: float = 1e-8,
        action_rescale: bool = False,
        clip_action: float = 0.0,
        reset_return_on_done: bool = True,
        reset_return_on_trunc: bool = True,
        obs_dtype: Any = np.float32,
        # step unpacking
        flatten_obs: bool = False,
        strict_checkpoint: bool = False,
        seed_envs: bool = True,
    ) -> None:
        # ---- references ----
        self.algo = algo
        self.logger = logger
        self.evaluator = evaluator
        self.callbacks = callbacks

        # ---- run control ----
        self.total_env_steps = int(total_env_steps)
        if self.total_env_steps < 1:
            raise ValueError(f"total_env_steps must be >= 1, got {self.total_env_steps}")

        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)

        self.log_every_steps = int(log_every_steps)

        # Step unpacking policy
        self.flatten_obs = bool(flatten_obs)
        self.step_obs_dtype = obs_dtype

        self.strict_checkpoint = bool(strict_checkpoint)
        self.seed_envs = bool(seed_envs)

        # ---- rollout params ----
        self.n_envs = int(n_envs)
        self.rollout_steps_per_env = int(rollout_steps_per_env)
        self.sync_weights_every_updates = int(sync_weights_every_updates)

        self.utd = float(utd)
        if self.utd < 0.0:
            raise ValueError(f"utd must be >= 0, got {self.utd}")

        # ---- directories ----
        self.run_dir = self._resolve_run_dir(str(run_dir))
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_dir = str(checkpoint_dir)
        self.checkpoint_prefix = str(checkpoint_prefix)

        self.ckpt_dir = os.path.join(self.run_dir, self.checkpoint_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ---- ray plumbing ----
        self.ray_env_make_fn = ray_env_make_fn
        self.ray_policy_spec = ray_policy_spec
        self.ray_want_onpolicy_extras = ray_want_onpolicy_extras

        # ---- counters ----
        self.global_env_step: int = 0
        self.global_update_step: int = 0
        self.episode_idx: int = 0
        self._ep_return: float = 0.0
        self._ep_len: int = 0

        # ---- one-time warning guards ----
        self._warned_norm_sync: bool = False
        self._warned_tqdm_missing: bool = False
        self._warned_checkpoint: bool = False
        self._stop_training: bool = False
        # ---- best-effort library-level seeding ----
        self._seed_libraries_best_effort()

        # ---- optional normalization wrapping ----
        self._normalize_enabled = bool(normalize)
        if self._normalize_enabled:
            train_env, eval_env = self._wrap_envs_with_normalize(
                train_env=train_env,
                eval_env=eval_env,
                obs_shape=obs_shape,
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                norm_gamma=norm_gamma,
                norm_epsilon=norm_epsilon,
                action_rescale=action_rescale,
                clip_action=clip_action,
                reset_return_on_done=reset_return_on_done,
                reset_return_on_trunc=reset_return_on_trunc,
                obs_dtype=obs_dtype,
            )

        self.train_env = train_env
        self.eval_env = eval_env

        # ---- best-effort env seeding (after wrapping) ----
        if self.seed_envs:
            self._seed_envs_best_effort()

        # ---- algo setup sees wrapped train_env ----
        self.algo.setup(self.train_env)

        # ---- allow logger to bind to trainer ----
        self._bind_logger_best_effort()

        # ---- detect EpisodeStatsCallback ownership ----
        self._episode_stats_enabled = self._detect_episode_stats_callback()

    # ---------------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------------
    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        # No resource ownership here (Ray lifecycle handled elsewhere).
        return None

    def request_stop(self) -> None:
        self._stop_training = True

    # ---------------------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------------------
    def _warn(self, message: str) -> None:
        """
        Best-effort warning emitter (stderr).
        """
        try:
            import sys

            print(f"[Trainer][WARN] {message}", file=sys.stderr)
        except Exception:
            pass

    # =============================================================================
    # Evaluation (delegated)
    # =============================================================================
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Evaluate current policy using trainer.evaluator if provided.

        Returns
        -------
        metrics : Dict[str, Any]
            Empty dict if no evaluator.
        """
        return run_evaluation(self)

    # =============================================================================
    # Training entrypoint
    # =============================================================================
    def train(self) -> None:
        """
        Train until global_env_step reaches total_env_steps or a callback requests stop.
        """
        if self.callbacks is not None:
            cont = maybe_call(self.callbacks, "on_train_start", self)
            if cont is False:
                return

        self.algo.set_training(True)
        maybe_call(self.train_env, "set_training", True)

        # 메시지 라인(상단): 진행바 없이 텍스트만
        msg_pbar = make_pbar(
            total=1,                 # <-- 0 말고 1
            initial=0,
            position=0,
            leave=True,
            bar_format="{desc}",     # 텍스트만 표시 (정상)
            dynamic_ncols=True,
        )
        msg_pbar.set_description_str(" ", refresh=True)  # 첫 렌더 강제

        # 진행바(하단)
        pbar = make_pbar(
            total=self.total_env_steps,
            initial=self.global_env_step,
            desc="Training",
            unit="step",
            position=1,
            leave=True,
            dynamic_ncols=True,
        )

        self._pbar = pbar
        self._msg_pbar = msg_pbar

        if (tqdm is None) and (not self._warned_tqdm_missing):
            self._warned_tqdm_missing = True
            self._warn("tqdm is not installed; progress bar is disabled.")

        try:
            if self.n_envs <= 1:
                train_single_env(self, self._pbar, self._msg_pbar)
            else:
                train_ray(self, self._pbar, self._msg_pbar)
        finally:
            try:
                pbar.close()
            except Exception:
                pass

            try:
                msg_pbar.close()
            except Exception:
                pass

            self._pbar = None
            self._msg_pbar = None
            self.algo.set_training(False)
            maybe_call(self.train_env, "set_training", False)
            maybe_call(self.callbacks, "on_train_end", self)

    # =============================================================================
    # Checkpointing (delegated)
    # =============================================================================
    def save_checkpoint(self, path: Optional[str] = None) -> Optional[str]:
        """
        Save a trainer checkpoint and notify callbacks.

        Returns
        -------
        ckpt_path : Optional[str]
            Path to checkpoint file, or None on failure.
        """
        ckpt_path = save_checkpoint(self, path=path)
        if ckpt_path is not None and self.callbacks is not None:
            try:
                maybe_call(self.callbacks, "on_checkpoint", self, path=ckpt_path)
            except Exception:
                pass
        return ckpt_path

    def load_checkpoint(self, path: str) -> None:
        """
        Load a trainer checkpoint from path.
        """
        load_checkpoint(self, path=path)

    # =============================================================================
    # Internal helpers
    # =============================================================================
    def _resolve_run_dir(self, default_run_dir: str) -> str:
        """
        Resolve trainer run_dir, preferring logger.run_dir if present.
        """
        run_dir = default_run_dir
        if self.logger is not None:
            lr = getattr(self.logger, "run_dir", None)
            if isinstance(lr, str) and lr:
                run_dir = lr
        return str(run_dir)

    def _seed_libraries_best_effort(self) -> None:
        """
        Best-effort library-level RNG seeding.

        Notes
        -----
        - Avoid hard dependency on a specific module path; uses try/except.
        - You already import `set_random_seed` elsewhere; if you want single-source-of-truth,
          consider using that here too.
        """
        try:
            # Prefer a project-level seed helper if available.
            from train.utils import set_random_seed as _set_seed  # type: ignore

            _set_seed(self.seed, deterministic=self.deterministic, verbose=False)
            return
        except Exception:
            pass

        # Fallback: do nothing (trainer remains robust even without global seeding).
        return

    def _seed_envs_best_effort(self) -> None:
        """
        Best-effort environment seeding via env_reset(..., seed=...).
        """
        try:
            _ = env_reset(self.train_env, seed=int(self.seed))
        except Exception:
            pass
        try:
            _ = env_reset(self.eval_env, seed=int(self.seed) + 1)
        except Exception:
            pass

    def _bind_logger_best_effort(self) -> None:
        """
        Bind logger to trainer if logger exposes bind_trainer(trainer).
        """
        if self.logger is None:
            return
        bind_fn = getattr(self.logger, "bind_trainer", None)
        if callable(bind_fn):
            try:
                bind_fn(self)
            except Exception:
                pass

    def _detect_episode_stats_callback(self) -> bool:
        """
        Detect whether EpisodeStatsCallback is installed (and thus owns episode counting).

        Returns
        -------
        enabled : bool
            True if callbacks contains an EpisodeStatsCallback instance.
        """
        try:
            cbs = getattr(self.callbacks, "callbacks", None)
            if isinstance(cbs, list):
                return any(isinstance(cb, EpisodeStatsCallback) for cb in cbs)
        except Exception:
            pass
        return False

    def _wrap_envs_with_normalize(
        self,
        *,
        train_env: Any,
        eval_env: Any,
        obs_shape: Optional[Tuple[int, ...]],
        norm_obs: bool,
        norm_reward: bool,
        clip_obs: float,
        clip_reward: float,
        norm_gamma: float,
        norm_epsilon: float,
        action_rescale: bool,
        clip_action: float,
        reset_return_on_done: bool,
        reset_return_on_trunc: bool,
        obs_dtype: Any,
    ) -> Tuple[Any, Any]:
        """
        Wrap train and eval environments with NormalizeWrapper.

        Also wraps ray_env_make_fn when:
          - n_envs > 1
          - ray_env_make_fn is provided

        Returns
        -------
        (train_env_wrapped, eval_env_wrapped) : Tuple[Any, Any]
        """
        if obs_shape is None:
            try:
                obs_shape = tuple(int(x) for x in train_env.observation_space.shape)
            except Exception as e:
                raise ValueError("normalize=True requires obs_shape or env.observation_space.shape.") from e

        obs_shape = tuple(obs_shape)

        train_env_wrapped = NormalizeWrapper(
            train_env,
            obs_shape=obs_shape,
            norm_obs=bool(norm_obs),
            norm_reward=bool(norm_reward),
            clip_obs=float(clip_obs),
            clip_reward=float(clip_reward),
            gamma=float(norm_gamma),
            epsilon=float(norm_epsilon),
            training=True,
            max_episode_steps=self.max_episode_steps,
            action_rescale=bool(action_rescale),
            clip_action=float(clip_action),
            reset_return_on_done=bool(reset_return_on_done),
            reset_return_on_trunc=bool(reset_return_on_trunc),
            obs_dtype=obs_dtype,
        )

        eval_env_wrapped = NormalizeWrapper(
            eval_env,
            obs_shape=obs_shape,
            norm_obs=bool(norm_obs),
            norm_reward=bool(norm_reward),
            clip_obs=float(clip_obs),
            clip_reward=float(clip_reward),
            gamma=float(norm_gamma),
            epsilon=float(norm_epsilon),
            training=False,
            max_episode_steps=self.max_episode_steps,
            action_rescale=bool(action_rescale),
            clip_action=float(clip_action),
            reset_return_on_done=bool(reset_return_on_done),
            reset_return_on_trunc=bool(reset_return_on_trunc),
            obs_dtype=obs_dtype,
        )

        # Ensure ray workers also wrap envs the same way (training=True on workers).
        if self.n_envs > 1 and self.ray_env_make_fn is not None:
            self.ray_env_make_fn = wrap_make_env_with_normalize(
                self.ray_env_make_fn,
                obs_shape=obs_shape,
                norm_obs=bool(norm_obs),
                norm_reward=bool(norm_reward),
                clip_obs=float(clip_obs),
                clip_reward=float(clip_reward),
                gamma=float(norm_gamma),
                epsilon=float(norm_epsilon),
                training=True,
                max_episode_steps=self.max_episode_steps,
                action_rescale=bool(action_rescale),
                clip_action=float(clip_action),
                reset_return_on_done=bool(reset_return_on_done),
                reset_return_on_trunc=bool(reset_return_on_trunc),
                obs_dtype=obs_dtype,
            )

        return train_env_wrapped, eval_env_wrapped