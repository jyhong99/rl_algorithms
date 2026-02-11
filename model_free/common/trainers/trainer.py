from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    # Ray is an optional dependency; keep runtime imports Ray-free.
    from ..utils.ray_utils import PolicyFactorySpec
else:
    PolicyFactorySpec = Any

from ..utils.train_utils import (
    _env_reset,
    _make_pbar,
    _maybe_call,
    _wrap_make_env_with_normalize,
)

from ..wrappers.normalize_wrapper import NormalizeWrapper
from ..callbacks.episode_stats_callback import EpisodeStatsCallback

from .train_checkpoint import load_checkpoint, save_checkpoint
from .train_eval import run_evaluation
from .train_loop import train_single_env
from .train_ray import train_ray


class Trainer:
    """
    Unified trainer/orchestrator for single-env and Ray multi-worker training.

    This class is intentionally a *thin orchestrator* that delegates heavy logic to:

    - ``train_loop.py``        : single-environment rollout/update loop
    - ``train_ray.py``         : Ray multi-worker rollout/update loop
    - ``train_eval.py``        : evaluation entrypoint
    - ``train_checkpoint.py``  : checkpoint save/load

    The goal is to keep Trainer focused on:
      - wiring components together (envs, algo, logger, callbacks, evaluator)
      - lifecycle (train/eval/checkpoint entrypoints)
      - shared bookkeeping (counters, run_dir/ckpt_dir)
      - optional NormalizeWrapper injection

    Parameters
    ----------
    train_env : Any
        Training environment instance. Must support a Gym/Gymnasium-like API:

        - ``reset(...)`` and ``step(action)``
        - optionally ``set_training(bool)`` (e.g., wrappers that freeze stats)
        - optionally ``observation_space.shape`` (for normalization auto-shape)

    eval_env : Any
        Evaluation environment instance (distinct from ``train_env`` recommended).

    algo : Any
        Algorithm / agent object (duck-typed). Required interface:

        - ``setup(env) -> None``
        - ``set_training(training: bool) -> None``
        - ``act(obs, deterministic: bool = False) -> Any``
        - ``on_env_step(transition: Dict[str, Any]) -> None``
        - ``ready_to_update() -> bool``
        - ``update() -> Mapping[str, Any]``

        Optional features recognized by delegated loops/utilities:

        - ``on_rollout(rollout) -> None``
        - ``is_off_policy: bool``
        - ``save(path: str) -> None`` / ``load(path: str) -> None``
        - ``get_ray_policy_factory_spec() -> PolicyFactorySpec``
        - ``remaining_rollout_steps() -> int`` (Ray on-policy path)

    total_env_steps : int, default=1_000_000
        Training budget expressed in environment transitions (trainer-side counter).

    seed : int, default=0
        Base seed for (best-effort) library/env seeding.

    deterministic : bool, default=True
        Determinism hint used by seeding helper and potentially by algorithms.

    max_episode_steps : int, optional
        Episode length cap. If normalization wrapper is used, the wrapper may own the
        time-limit behavior; otherwise loops may apply a trainer-side fallback.

    log_every_steps : int, default=5000
        Cadence for system-level logging inside loops (delegated).

    run_dir : str, default="./runs/exp"
        Root directory for run artifacts. If logger exposes ``run_dir``, Trainer may
        prefer that directory (see Notes).

    checkpoint_dir : str, default="checkpoints"
        Checkpoint subdirectory (created under ``run_dir``).

    checkpoint_prefix : str, default="ckpt"
        Prefix used by checkpoint naming utilities.

    n_envs : int, default=1
        Number of rollout workers. If ``n_envs <= 1`` Trainer uses single-env loop;
        otherwise uses Ray loop.

    rollout_steps_per_env : int, default=256
        Ray: number of steps each worker collects per iteration.

    sync_weights_every_updates : int, default=10
        Ray: weight synchronization cadence (in update steps).

    utd : float, default=1.0
        Update-To-Data ratio hint. Off-policy loop may use this to scale updates.

    logger : Any, optional
        Logger object. If provided, Trainer will best-effort call ``logger.bind_trainer(self)``
        and may prefer ``logger.run_dir`` as the effective run directory.

    evaluator : Any, optional
        Evaluator object. Expected interface:
        ``evaluate(agent) -> Mapping[str, Any]`` (see ``train_eval.py``).

    callbacks : Any, optional
        Callback container (best-effort). Supported hooks:
        ``on_train_start``, ``on_step``, ``on_update``, ``on_eval_end``,
        ``on_checkpoint``, ``on_train_end``.

    ray_env_make_fn : Callable[[], Any], optional
        Ray: factory that constructs a fresh environment on workers.

    ray_policy_spec : PolicyFactorySpec, optional
        Ray: serializable policy construction spec (optional dependency).

    ray_want_onpolicy_extras : bool, optional
        Ray: whether workers should request extras (e.g., log_prob/value).

    normalize : bool, default=False
        If True, wrap both ``train_env`` and ``eval_env`` with ``NormalizeWrapper``.
        Additionally, if Ray is used and ``ray_env_make_fn`` exists, wrap it with the same
        normalization parameters (workers run with ``training=True``).

    obs_shape : tuple[int, ...], optional
        Observation shape required for normalization. If None, Trainer tries
        ``train_env.observation_space.shape``.

    norm_obs, norm_reward : bool
        Whether to normalize observations and/or rewards.

    clip_obs, clip_reward : float
        Clipping magnitudes for normalized observations/rewards.

    norm_gamma, norm_epsilon : float
        Running-stat hyperparameters in normalization wrapper.

    action_rescale : bool
        Wrapper-dependent action rescaling toggle.

    clip_action : float
        Action clipping magnitude (if enabled in wrapper).

    reset_return_on_done, reset_return_on_trunc : bool
        Return accumulator reset behavior in wrapper.

    obs_dtype : Any, default=np.float32
        Observation dtype used by wrapper and step-unpacking policy.

    flatten_obs : bool, default=False
        Step unpacking option passed into delegated loops/utilities.

    strict_checkpoint : bool, default=False
        If True, checkpoint errors may be treated as fatal depending on IO helpers.

    seed_envs : bool, default=True
        If True, best-effort seed ``train_env`` and ``eval_env`` after wrapping.

    Notes
    -----
    - **Run directory policy**: Trainer resolves ``self.run_dir`` using the provided ``run_dir``.
      If a logger is attached and exposes ``logger.run_dir``, Trainer prefers it.
    - **Episode indexing ownership**:
      If callbacks include ``EpisodeStatsCallback``, that callback is assumed to own episode
      counting/logging and Trainer will avoid incrementing episode counters redundantly.
    - **Optional dependencies**:
      - tqdm is optional; if missing, progress bars degrade gracefully.
      - Ray is optional; Trainer avoids importing Ray at runtime.
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
        # normalization
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
        # ---- external components ----
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

        # ---- step unpacking policy ----
        self.flatten_obs = bool(flatten_obs)
        self.step_obs_dtype = obs_dtype

        # ---- checkpoint policy ----
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

        # ---- Ray plumbing (pure data holders; actual Ray imports are elsewhere) ----
        self.ray_env_make_fn = ray_env_make_fn
        self.ray_policy_spec = ray_policy_spec
        self.ray_want_onpolicy_extras = ray_want_onpolicy_extras

        # ---- counters ----
        self.global_env_step: int = 0
        self.global_update_step: int = 0
        self.episode_idx: int = 0
        self._ep_return: float = 0.0
        self._ep_len: int = 0

        # ---- one-time warning guards / control flags ----
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

        # ---- bind env references ----
        self.train_env = train_env
        self.eval_env = eval_env

        # ---- best-effort env seeding (after wrapping) ----
        if self.seed_envs:
            self._seed_envs_best_effort()

        # ---- algorithm setup sees the *wrapped* train_env ----
        self.algo.setup(self.train_env)

        # ---- allow logger to bind to trainer ----
        self._bind_logger_best_effort()

        # ---- detect EpisodeStatsCallback ownership ----
        self._episode_stats_enabled = self._detect_episode_stats_callback()

        # ---- pbar handles (assigned during train()) ----
        self._pbar = None
        self._msg_pbar = None

    # ---------------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------------
    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        # No resource ownership here (Ray lifecycle handled elsewhere).
        return None

    # ---------------------------------------------------------------------
    # Control
    # ---------------------------------------------------------------------
    def request_stop(self) -> None:
        """
        Request cooperative training termination.

        Notes
        -----
        - Delegated loops periodically check ``trainer._stop_training``.
        - Callbacks may also set this flag (e.g., early stopping).
        """
        self._stop_training = True

    # ---------------------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------------------
    def _warn(self, message: str) -> None:
        """
        Best-effort warning emitter (stderr).

        Parameters
        ----------
        message : str
            Warning message to print.
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
        Evaluate current policy using ``trainer.evaluator`` if provided.

        Returns
        -------
        metrics : Dict[str, Any]
            Evaluation metrics. Returns an empty dict when no evaluator is attached.
        """
        return run_evaluation(self)

    # =============================================================================
    # Training entrypoint
    # =============================================================================
    def train(self) -> None:
        """
        Train until one of the following termination conditions holds:

        - ``global_env_step >= total_env_steps``
        - a callback requests stop (returns False)
        - ``trainer.request_stop()`` is called (sets ``_stop_training``)
        """
        if self.callbacks is not None:
            cont = _maybe_call(self.callbacks, "on_train_start", self)
            if cont is False:
                return

        # Put algo/env in training mode (best-effort for env).
        self.algo.set_training(True)
        _maybe_call(self.train_env, "set_training", True)

        # A top "message-only" line (desc-only tqdm bar).
        msg_pbar = _make_pbar(
            total=1,
            initial=0,
            position=0,
            leave=True,
            bar_format="{desc}",
            dynamic_ncols=True,
        )
        msg_pbar.set_description_str(" ", refresh=True)

        # The main progress bar.
        pbar = _make_pbar(
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
            # Always attempt to close progress bars.
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

            # Return algo/env to eval-ish mode.
            self.algo.set_training(False)
            _maybe_call(self.train_env, "set_training", False)

            # Notify callbacks.
            _maybe_call(self.callbacks, "on_train_end", self)

    # =============================================================================
    # Checkpointing (delegated)
    # =============================================================================
    def save_checkpoint(self, path: Optional[str] = None) -> Optional[str]:
        """
        Save a trainer checkpoint and notify callbacks.

        Parameters
        ----------
        path : str, optional
            Target checkpoint file path. See ``train_checkpoint.save_checkpoint`` for rules.

        Returns
        -------
        ckpt_path : Optional[str]
            Absolute/normalized checkpoint path on success; None on failure.
        """
        ckpt_path = save_checkpoint(self, path=path)
        if ckpt_path is not None and self.callbacks is not None:
            try:
                _maybe_call(self.callbacks, "on_checkpoint", self, path=ckpt_path)
            except Exception:
                pass
        return ckpt_path

    def load_checkpoint(self, path: str) -> None:
        """
        Load a trainer checkpoint from path.

        Parameters
        ----------
        path : str
            Path to a torch checkpoint file created by ``save_checkpoint``.
        """
        load_checkpoint(self, path=path)

    # =============================================================================
    # Internal helpers
    # =============================================================================
    def _resolve_run_dir(self, default_run_dir: str) -> str:
        """
        Resolve effective trainer run directory.

        Parameters
        ----------
        default_run_dir : str
            The run directory requested by the caller.

        Returns
        -------
        run_dir : str
            Effective run directory. If logger exposes a non-empty ``logger.run_dir``,
            it takes precedence.
        """
        run_dir = str(default_run_dir)
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
        - This function is intentionally permissive: failures do not abort training.
        - If your project already has a canonical seeding utility (e.g., `_set_random_seed`
          in train_utils), it is cleaner to call that here and remove the fallback import.
        """
        try:
            # Prefer a project-level seed helper if available.
            from train.utils import set_random_seed as _set_seed  # type: ignore

            _set_seed(self.seed, deterministic=self.deterministic, verbose=False)
            return
        except Exception:
            pass

        return

    def _seed_envs_best_effort(self) -> None:
        """
        Best-effort environment seeding using ``_env_reset(..., seed=...)``.

        Notes
        -----
        - Runs after normalization wrapping so wrappers see seeded resets.
        - Uses two different seeds for train/eval by convention.
        """
        try:
            _ = _env_reset(self.train_env, seed=int(self.seed))
        except Exception:
            pass
        try:
            _ = _env_reset(self.eval_env, seed=int(self.seed) + 1)
        except Exception:
            pass

    def _bind_logger_best_effort(self) -> None:
        """
        Bind logger to trainer if logger exposes ``bind_trainer(trainer)``.
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
        Detect whether ``EpisodeStatsCallback`` is installed.

        Returns
        -------
        enabled : bool
            True if ``callbacks.callbacks`` contains an EpisodeStatsCallback instance.
            When True, episode indexing/logging is assumed to be owned elsewhere.
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
        Wrap train and eval environments with ``NormalizeWrapper``.

        Parameters
        ----------
        train_env, eval_env : Any
            Raw environments to wrap.
        obs_shape : tuple[int, ...], optional
            Required observation shape for normalization. If None, attempts to infer from
            ``train_env.observation_space.shape``.
        norm_obs, norm_reward : bool
            Whether to normalize observations and/or rewards.
        clip_obs, clip_reward : float
            Clipping magnitudes for normalized observations/rewards.
        norm_gamma, norm_epsilon : float
            Running-stat hyperparameters.
        action_rescale : bool
            Wrapper-dependent action rescale toggle.
        clip_action : float
            Action clipping magnitude (if enabled).
        reset_return_on_done, reset_return_on_trunc : bool
            Return accumulator reset behavior.
        obs_dtype : Any
            Observation dtype used by wrapper.

        Returns
        -------
        train_env_wrapped, eval_env_wrapped : Tuple[Any, Any]
            Wrapped environments.

        Notes
        -----
        - Training env wrapper uses ``training=True``.
        - Eval env wrapper uses ``training=False`` (frozen stats).
        - If Ray is used (``n_envs > 1``) and ``ray_env_make_fn`` exists, this method also
          wraps that maker so workers match the same normalization semantics (training=True).
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

        # Ensure Ray workers also wrap envs the same way (training=True on workers).
        if self.n_envs > 1 and self.ray_env_make_fn is not None:
            self.ray_env_make_fn = _wrap_make_env_with_normalize(
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
