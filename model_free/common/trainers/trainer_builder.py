from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, List

import numpy as np

from .trainer import Trainer
from .evaluator import Evaluator
from ..loggers.logger_builder import build_logger
from ..callbacks.callback_builder import build_callbacks


# =============================================================================
# Builder
# =============================================================================
def build_trainer(
    *,
    # ---- env ----
    make_train_env: Callable[[], Any],
    make_eval_env: Optional[Callable[[], Any]] = None,
    algo: Any,
    # ---- run control ----
    total_env_steps: int = 1_000_000,
    seed: int = 0,
    deterministic: bool = True,
    max_episode_steps: Optional[int] = None,
    log_every_steps: int = 5_000,
    # ---- parallel rollout ----
    n_envs: int = 1,
    rollout_steps_per_env: int = 256,
    sync_weights_every_updates: int = 10,
    utd: float = 1.0,
    # ---- ray injection (only used when n_envs > 1) ----
    ray_env_make_fn: Optional[Callable[[], Any]] = None,
    ray_policy_spec: Optional[Any] = None,  # PolicyFactorySpec (kept Any to avoid hard Ray import here)
    ray_want_onpolicy_extras: Optional[bool] = None,
    # ---- trainer directories ----
    trainer_run_dir: Optional[str] = None,
    trainer_checkpoint_dir: str = "checkpoints",
    trainer_checkpoint_prefix: str = "ckpt",
    trainer_strict_checkpoint: bool = False,
    # ---- config dump ----
    dump_config: bool = True,
    config_filename: str = "config.json",
    # ---- normalization (Trainer will wrap) ----
    enable_norm: bool = False,
    norm_obs_shape: Optional[Tuple[int, ...]] = None,
    norm_obs: bool = True,
    norm_reward: bool = False,
    norm_clip_obs: float = 10.0,
    norm_clip_reward: float = 10.0,
    norm_gamma: float = 0.99,
    norm_epsilon: float = 1e-8,
    norm_action_rescale: bool = False,
    norm_clip_action: float = 0.0,
    norm_reset_return_on_done: bool = True,
    norm_reset_return_on_trunc: bool = True,
    norm_obs_dtype: Any = np.float32,
    # ---- logger (factory args) ----
    enable_logger: bool = True,
    enable_tensorboard: bool = True,
    enable_csv: bool = True,
    enable_csv_long: bool = True,
    enable_jsonl: bool = True,
    enable_wandb: bool = False,
    logger_log_dir: str = "./runs",
    logger_exp_name: str = "exp",
    logger_run_id: Optional[str] = None,
    logger_run_name: Optional[str] = None,
    logger_overwrite: bool = False,
    logger_resume: bool = False,
    logger_wandb_project: Optional[str] = None,
    logger_wandb_entity: Optional[str] = None,
    logger_wandb_group: Optional[str] = None,
    logger_wandb_tags: Optional[Sequence[str]] = None,
    logger_wandb_run_name: Optional[str] = None,
    logger_wandb_mode: Optional[str] = None,
    logger_wandb_resume: Optional[str] = None,
    logger_console_every: int = 1000,
    logger_flush_every: int = 200,
    logger_drop_non_finite: bool = False,
    logger_strict: bool = False,
    # ---- callbacks toggles ----
    enable_eval: bool = True,
    enable_ckpt: bool = True,
    enable_best_model: bool = False,
    enable_early_stop: bool = False,
    enable_nan_guard: bool = True,
    enable_config_env_info: bool = True,
    enable_episode_stats: bool = True,
    enable_timing: bool = True,
    enable_lr_logging: bool = True,
    enable_grad_param_norm: bool = False,
    enable_ray_report: bool = False,
    enable_ray_tune_ckpt: bool = False,
    cb_episodes: int = 10,
    cb_deterministic: bool = True,
    cb_show_progress: bool = True,
    cb_eval_every_steps: int = 50_000,
    cb_save_every_steps: int = 100_000,
    cb_keep_last_checkpoints: int = 5,
    cb_best_metric_key: str = "eval/return_mean",
    cb_best_save_path: str = "best.pt",
    cb_early_stop_metric_key: str = "eval/return_mean",
    cb_early_stop_patience: int = 10,
    cb_early_stop_min_delta: float = 0.0,
    cb_early_stop_mode: str = "max",  # "max" or "min"
    cb_nan_guard_keys: Optional[Sequence[str]] = None,
    cb_episode_window: int = 100,
    cb_episode_log_every: int = 10,
    cb_timing_log_every_steps: int = 5_000,
    cb_timing_log_every_updates: int = 200,
    cb_lr_log_every_updates: int = 200,
    cb_norm_log_every_updates: int = 200,
    cb_norm_per_module: bool = False,
    cb_norm_include_param: bool = True,
    cb_norm_include_grad: bool = True,
    cb_norm_type: float = 2.0,
    cb_ray_report_every_updates: int = 1,
    cb_ray_report_on_eval: bool = True,
    cb_ray_report_keep_last_eval: bool = True,
    cb_ray_tune_report_every_saves: int = 1,
    cb_ray_tune_metric_key: Optional[str] = "eval/return_mean",
    cb_extra_callbacks: Optional[Sequence[Any]] = None,
    cb_strict_callbacks: bool = False,
) -> Trainer:
    """
    Construct environments, logger, evaluator, callbacks, and finally a `Trainer`.

    This is a high-level factory that wires together the major runtime components
    of your training stack.

    Parameters
    ----------
    make_train_env : Callable[[], Any]
        Factory that returns a fresh training environment instance.
        Must create a *new* instance on each call (no shared state unless intended).
    make_eval_env : Callable[[], Any], optional
        Factory that returns a fresh evaluation environment instance.
        If None, evaluation env is created by calling `make_train_env()` again.
    algo : Any
        Algorithm / agent object to be attached to Trainer.

    total_env_steps : int, default=1_000_000
        Total number of environment transitions to ingest (trainer-side counter).
    seed : int, default=0
        Base seed passed into Trainer (and optionally used by Evaluator).
    deterministic : bool, default=True
        Trainer-level deterministic flag (algorithm may interpret this for evaluation
        or action selection defaults).
    max_episode_steps : int, optional
        Episode length cap; passed into Trainer and Evaluator. Whether it is applied
        depends on your env wrappers and trainer-side fallback rules.
    log_every_steps : int, default=5000
        Cadence for system-level logging (e.g., counters) inside the trainer loops.

    n_envs : int, default=1
        Number of rollout workers/environments. When `n_envs > 1`, the Ray path may be used.
    rollout_steps_per_env : int, default=256
        Per-worker rollout chunk length for Ray collection.
    sync_weights_every_updates : int, default=10
        Broadcast cadence for policy weights to workers (in updates).
    utd : float, default=1.0
        Update-To-Data ratio hint. Off-policy drivers may use this to scale update frequency.

    ray_env_make_fn : Callable[[], Any], optional
        Explicit environment factory for Ray workers (only used when `n_envs > 1`).
        If None, the trainer may auto-infer one (elsewhere) from train_env spec id.
    ray_policy_spec : Any, optional
        Serializable policy construction spec for Ray workers. Kept as Any here to
        avoid importing Ray types at import time.
    ray_want_onpolicy_extras : bool, optional
        Whether workers should return on-policy extras (e.g., log_prob, value).

    trainer_run_dir : str, optional
        Root run directory for trainer artifacts. If None, follows `logger.run_dir`
        when logger exists; otherwise falls back to a safe default derived from
        `logger_log_dir/logger_exp_name`.
    trainer_checkpoint_dir : str, default="checkpoints"
        Relative or absolute checkpoint directory for trainer.
    trainer_checkpoint_prefix : str, default="ckpt"
        Prefix for checkpoint filenames.
    trainer_strict_checkpoint : bool, default=False
        If True, checkpoint failures may be treated as fatal depending on Trainer logic.

    dump_config : bool, default=True
        If True and logger supports it, dump a JSON-serializable config summary.
    config_filename : str, default="config.json"
        Filename to use for dumped config under the run directory.

    enable_norm : bool, default=False
        Whether Trainer should wrap envs with NormalizeWrapper (Trainer-owned).
    norm_obs_shape : tuple[int, ...], optional
        Observation shape required for normalization wrapper.
    norm_obs, norm_reward : bool
        Whether to normalize observations and/or rewards.
    norm_clip_obs, norm_clip_reward : float
        Clipping ranges for normalized observations/rewards.
    norm_gamma, norm_epsilon : float
        Discount factor and numerical epsilon for running-stat updates.
    norm_action_rescale : bool
        Whether to rescale actions before clipping (wrapper-dependent).
    norm_clip_action : float
        Action clipping value when action rescale/clipping is enabled.
    norm_reset_return_on_done, norm_reset_return_on_trunc : bool
        Return accumulator reset behavior in wrappers.
    norm_obs_dtype : Any
        Observation dtype used by normalization wrapper.

    enable_logger : bool
        If True, create a logger via `build_logger`.
    ... (logger args)
        Passed through to logger builder backends (tensorboard/csv/jsonl/wandb) and behavior knobs.

    enable_eval, enable_ckpt, ...
        Callback toggles that are mapped into `build_callbacks` flags/kwargs.

    Returns
    -------
    trainer : Trainer
        Fully constructed Trainer instance.

    Notes
    -----
    - Trainer is responsible for optional NormalizeWrapper wrapping when `enable_norm=True`.
    - When `make_eval_env` is None, evaluation env is created as a distinct instance by
      calling `make_train_env()` again.
    - If `dump_config` is True and logger provides `dump_config`, a config snapshot is written.
    """
    # ------------------------------------------------------------------
    # 1) Environments (distinct instances)
    # ------------------------------------------------------------------
    train_env = make_train_env()
    eval_env = make_train_env() if make_eval_env is None else make_eval_env()

    # ------------------------------------------------------------------
    # 2) Logger (optional)
    # ------------------------------------------------------------------
    logger = None
    if bool(enable_logger):
        logger = build_logger(
            log_dir=str(logger_log_dir),
            exp_name=str(logger_exp_name),
            run_id=logger_run_id,
            run_name=logger_run_name,
            overwrite=bool(logger_overwrite),
            resume=bool(logger_resume),
            # backends
            use_tensorboard=bool(enable_tensorboard),
            use_csv=bool(enable_csv),
            use_jsonl=bool(enable_jsonl),
            use_wandb=bool(enable_wandb),
            # backend kwargs
            csv_kwargs=dict(wide=True, long=bool(enable_csv_long)),
            jsonl_kwargs=dict(),
            tensorboard_kwargs=dict(),
            wandb_kwargs=dict(
                project=logger_wandb_project,
                entity=logger_wandb_entity,
                group=logger_wandb_group,
                tags=logger_wandb_tags,
                name=logger_wandb_run_name,
                mode=logger_wandb_mode,
                resume=logger_wandb_resume,
            ),
            # logger behavior
            console_every=int(logger_console_every),
            flush_every=int(logger_flush_every),
            drop_non_finite=bool(logger_drop_non_finite),
            strict=bool(logger_strict),
        )

    # ------------------------------------------------------------------
    # 3) Trainer run_dir policy
    # ------------------------------------------------------------------
    if trainer_run_dir is not None:
        run_dir = str(trainer_run_dir)
    else:
        # Prefer logger.run_dir if available; else use a deterministic fallback.
        if logger is not None:
            run_dir = str(getattr(logger, "run_dir", f"{logger_log_dir}/{logger_exp_name}"))
        else:
            run_dir = str(f"{logger_log_dir}/{logger_exp_name}")

    # ------------------------------------------------------------------
    # 4) Evaluator (optional)
    # ------------------------------------------------------------------
    evaluator: Optional[Evaluator] = None
    if bool(enable_eval):
        evaluator = Evaluator(
            env=eval_env,
            episodes=int(cb_episodes),
            deterministic=bool(cb_deterministic),
            show_progress=bool(cb_show_progress),
            max_episode_steps=max_episode_steps,
            base_seed=int(seed),
            seed_increment=1,
        )

    # ------------------------------------------------------------------
    # 5) Callbacks (via factory)
    # ------------------------------------------------------------------
    callbacks = build_callbacks(
        # switches
        use_eval=bool(enable_eval and cb_eval_every_steps > 0),
        use_checkpoint=bool(enable_ckpt and cb_save_every_steps > 0),
        use_best_model=bool(enable_best_model),
        use_early_stop=bool(enable_early_stop),
        use_nan_guard=bool(enable_nan_guard),
        use_timing=bool(enable_timing),
        use_episode_stats=bool(enable_episode_stats),
        use_config_env_info=bool(enable_config_env_info),
        use_lr_logging=bool(enable_lr_logging),
        use_grad_param_norm=bool(enable_grad_param_norm),
        # ray
        use_ray_report=bool(enable_ray_report),
        use_ray_tune_checkpoint=bool(enable_ray_tune_ckpt),
        # kwargs per callback
        eval_kwargs=dict(eval_every=int(cb_eval_every_steps)),
        checkpoint_kwargs=dict(save_every=int(cb_save_every_steps), keep_last=int(cb_keep_last_checkpoints)),
        best_model_kwargs=dict(metric_key=str(cb_best_metric_key), save_path=str(cb_best_save_path)),
        early_stop_kwargs=dict(
            metric_key=str(cb_early_stop_metric_key),
            patience=int(cb_early_stop_patience),
            min_delta=float(cb_early_stop_min_delta),
            mode=("min" if str(cb_early_stop_mode).lower() == "min" else "max"),
        ),
        nan_guard_kwargs=dict(keys=cb_nan_guard_keys),
        timing_kwargs=dict(
            log_every_steps=int(cb_timing_log_every_steps),
            log_every_updates=int(cb_timing_log_every_updates),
            log_prefix="perf/",
        ),
        episode_stats_kwargs=dict(
            window=int(cb_episode_window),
            log_every_episodes=int(cb_episode_log_every),
            log_prefix="rollout/",
        ),
        config_env_info_kwargs=dict(log_prefix="sys/"),
        lr_logging_kwargs=dict(log_every_updates=int(cb_lr_log_every_updates), log_prefix="train/"),
        grad_param_norm_kwargs=dict(
            log_every_updates=int(cb_norm_log_every_updates),
            log_prefix="debug/",
            include_param_norm=bool(cb_norm_include_param),
            include_grad_norm=bool(cb_norm_include_grad),
            norm_type=float(cb_norm_type),
            per_module=bool(cb_norm_per_module),
        ),
        ray_report_kwargs=dict(
            report_every_updates=int(cb_ray_report_every_updates),
            report_on_eval=bool(cb_ray_report_on_eval),
            keep_last_eval=bool(cb_ray_report_keep_last_eval),
        ),
        ray_tune_checkpoint_kwargs=dict(
            report_every_saves=int(cb_ray_tune_report_every_saves),
            metric_key=cb_ray_tune_metric_key,
        ),
        # injection + strictness
        extra_callbacks=cb_extra_callbacks,
        strict_callbacks=bool(cb_strict_callbacks),
    )

    # ------------------------------------------------------------------
    # 6) Build Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        train_env=train_env,
        eval_env=eval_env,
        algo=algo,
        total_env_steps=int(total_env_steps),
        seed=int(seed),
        deterministic=bool(deterministic),
        max_episode_steps=max_episode_steps,
        log_every_steps=int(log_every_steps),
        run_dir=str(run_dir),
        checkpoint_dir=str(trainer_checkpoint_dir),
        checkpoint_prefix=str(trainer_checkpoint_prefix),
        n_envs=int(n_envs),
        rollout_steps_per_env=int(rollout_steps_per_env),
        sync_weights_every_updates=int(sync_weights_every_updates),
        utd=float(utd),
        logger=logger,
        evaluator=evaluator,
        callbacks=callbacks,
        ray_env_make_fn=ray_env_make_fn,
        ray_policy_spec=ray_policy_spec,
        ray_want_onpolicy_extras=ray_want_onpolicy_extras,
        # ---- normalization (keep Trainer's expected names) ----
        normalize=bool(enable_norm),
        obs_shape=norm_obs_shape,
        norm_obs=bool(norm_obs),
        norm_reward=bool(norm_reward),
        clip_obs=float(norm_clip_obs),
        clip_reward=float(norm_clip_reward),
        norm_gamma=float(norm_gamma),
        norm_epsilon=float(norm_epsilon),
        action_rescale=bool(norm_action_rescale),
        clip_action=float(norm_clip_action),
        reset_return_on_done=bool(norm_reset_return_on_done),
        reset_return_on_trunc=bool(norm_reset_return_on_trunc),
        obs_dtype=norm_obs_dtype,
        # misc
        flatten_obs=False,  # keep explicit; expose as arg if you want
        strict_checkpoint=bool(trainer_strict_checkpoint),
        seed_envs=True,
    )

    # ------------------------------------------------------------------
    # 7) Dump config (via logger, optional)
    # ------------------------------------------------------------------
    if dump_config and logger is not None and callable(getattr(logger, "dump_config", None)):
        cfg = _build_config_dump(
            seed=seed,
            deterministic=deterministic,
            total_env_steps=total_env_steps,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
            rollout_steps_per_env=rollout_steps_per_env,
            sync_weights_every_updates=sync_weights_every_updates,
            utd=utd,
            # normalization (keep key name "normalize" for backward compatibility)
            normalize=enable_norm,
            obs_shape=norm_obs_shape,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=norm_clip_obs,
            clip_reward=norm_clip_reward,
            norm_gamma=norm_gamma,
            norm_epsilon=norm_epsilon,
            action_rescale=norm_action_rescale,
            clip_action=norm_clip_action,
            reset_return_on_done=norm_reset_return_on_done,
            reset_return_on_trunc=norm_reset_return_on_trunc,
            obs_dtype=norm_obs_dtype,
            enable_evaluator=enable_eval,
            eval_episodes=cb_episodes,
            eval_deterministic=cb_deterministic,
            run_dir=run_dir,
            trainer_checkpoint_dir=trainer_checkpoint_dir,
            trainer_checkpoint_prefix=trainer_checkpoint_prefix,
            logger_run_dir=str(getattr(logger, "run_dir", "")),
            callbacks=callbacks,
            algo=algo,
        )
        logger.dump_config(cfg, filename=str(config_filename))

    return trainer


# =============================================================================
# Config dump helpers
# =============================================================================
def _build_config_dump(
    *,
    seed: int,
    deterministic: bool,
    total_env_steps: int,
    max_episode_steps: Optional[int],
    n_envs: int,
    rollout_steps_per_env: int,
    sync_weights_every_updates: int,
    utd: float,
    normalize: bool,
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
    enable_evaluator: bool,
    eval_episodes: int,
    eval_deterministic: bool,
    run_dir: str,
    trainer_checkpoint_dir: str,
    trainer_checkpoint_prefix: str,
    logger_run_dir: str,
    callbacks: Any,
    algo: Any,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable configuration summary for logging.

    Parameters
    ----------
    seed : int
        Random seed used by trainer/evaluator.
    deterministic : bool
        Trainer-level deterministic flag.
    total_env_steps : int
        Total number of environment transitions to ingest.
    max_episode_steps : int, optional
        Episode length cap passed into Trainer/Evaluator.
    n_envs : int
        Number of rollout workers (Ray when > 1).
    rollout_steps_per_env : int
        Per-worker rollout chunk length.
    sync_weights_every_updates : int
        Weight broadcast cadence (in updates).
    utd : float
        Update-to-data ratio hint.
    normalize : bool
        Whether normalization wrapper is enabled (key name kept for backward compatibility).
    obs_shape : tuple[int, ...], optional
        Observation shape for normalization.
    norm_obs, norm_reward : bool
        Observation/reward normalization toggles.
    clip_obs, clip_reward : float
        Clipping values for normalized obs/reward.
    norm_gamma, norm_epsilon : float
        Wrapper hyperparameters for running-stat computation.
    action_rescale : bool
        Whether wrapper rescales actions before clipping.
    clip_action : float
        Action clip magnitude.
    reset_return_on_done, reset_return_on_trunc : bool
        Return accumulator reset behavior in wrapper.
    obs_dtype : Any
        Observation dtype used in normalization.
    enable_evaluator : bool
        Whether evaluator was enabled.
    eval_episodes : int
        Number of evaluation episodes per eval event.
    eval_deterministic : bool
        Whether evaluation runs deterministically.
    run_dir : str
        Trainer run directory.
    trainer_checkpoint_dir : str
        Checkpoint directory (relative to run_dir or absolute depending on Trainer).
    trainer_checkpoint_prefix : str
        Checkpoint filename prefix.
    logger_run_dir : str
        Logger run directory (may match run_dir depending on configuration).
    callbacks : Any
        Callback container returned by `build_callbacks`.
    algo : Any
        Algorithm object (used to extract algo.cfg best-effort).

    Returns
    -------
    cfg : Dict[str, Any]
        JSON-serializable config dictionary.

    Notes
    -----
    - Callback names are extracted from `callbacks.callbacks` if present (CallbackList-style).
    - Algorithm configuration is extracted from `algo.cfg` if present:
        - dataclass -> `asdict(cfg)`
        - mapping-like -> `dict(cfg)`
        - otherwise -> stringified type name
    """
    cfg: Dict[str, Any] = dict(
        seed=int(seed),
        deterministic=bool(deterministic),
        total_env_steps=int(total_env_steps),
        max_episode_steps=None if max_episode_steps is None else int(max_episode_steps),
        n_envs=int(n_envs),
        rollout_steps_per_env=int(rollout_steps_per_env),
        sync_weights_every_updates=int(sync_weights_every_updates),
        utd=float(utd),
        normalize=bool(normalize),
        obs_shape=None if obs_shape is None else tuple(int(x) for x in obs_shape),
        norm_obs=bool(norm_obs),
        norm_reward=bool(norm_reward),
        clip_obs=float(clip_obs),
        clip_reward=float(clip_reward),
        norm_gamma=float(norm_gamma),
        norm_epsilon=float(norm_epsilon),
        action_rescale=bool(action_rescale),
        clip_action=float(clip_action),
        reset_return_on_done=bool(reset_return_on_done),
        reset_return_on_trunc=bool(reset_return_on_trunc),
        obs_dtype=str(obs_dtype),
        eval_enabled=bool(enable_evaluator),
        eval_episodes=int(eval_episodes),
        eval_deterministic=bool(eval_deterministic),
        trainer_run_dir=str(run_dir),
        trainer_checkpoint_dir=str(trainer_checkpoint_dir),
        trainer_checkpoint_prefix=str(trainer_checkpoint_prefix),
        logger_run_dir=str(logger_run_dir),
    )

    # callback names (CallbackList-like)
    cfg["callbacks"] = _extract_callback_names(callbacks)

    # algo cfg (best-effort)
    cfg["algo"] = _extract_algo_cfg(algo)

    return cfg


def _extract_callback_names(callbacks: Any) -> List[str]:
    """
    Extract callback class names from a callback container.

    Parameters
    ----------
    callbacks : Any
        Expected to possibly have attribute `callbacks` which is a list of callback objects.

    Returns
    -------
    names : List[str]
        Callback class names; empty list if unavailable.
    """
    try:
        cbs = getattr(callbacks, "callbacks", None)
        if isinstance(cbs, list):
            return [cb.__class__.__name__ for cb in cbs]
    except Exception:
        pass
    return []


def _extract_algo_cfg(algo: Any) -> Any:
    """
    Best-effort extraction of algorithm configuration for config dumps.

    Parameters
    ----------
    algo : Any
        Algorithm object that may expose `algo.cfg`.

    Returns
    -------
    cfg : Any
        JSON-serializable object if possible; otherwise a fallback string.
    """
    algo_cfg = getattr(algo, "cfg", None)
    if algo_cfg is None:
        return None

    if is_dataclass(algo_cfg):
        try:
            return asdict(algo_cfg)
        except Exception:
            return str(type(algo_cfg))

    try:
        return dict(algo_cfg)  # type: ignore[arg-type]
    except Exception:
        return str(type(algo_cfg))
