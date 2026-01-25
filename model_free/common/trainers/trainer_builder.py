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
    # ---- evaluation ----
    enable_evaluator: bool = True,
    eval_episodes: int = 10,
    eval_deterministic: bool = True,
    eval_show_progress: bool = True,
    # ---- normalization (Trainer will wrap) ----
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
    # ---- ray injection (only used when n_envs > 1) ----
    ray_env_make_fn: Optional[Callable[[], Any]] = None,
    ray_policy_spec: Optional[Any] = None,  # PolicyFactorySpec (kept Any to avoid hard Ray import here)
    ray_want_onpolicy_extras: Optional[bool] = None,
    # ---- logger (factory args) ----
    enable_logger: bool = True,
    logger_log_dir: str = "./runs",
    logger_exp_name: str = "exp",
    logger_run_id: Optional[str] = None,
    logger_run_name: Optional[str] = None,
    logger_overwrite: bool = False,
    logger_resume: bool = False,
    logger_use_tensorboard: bool = True,
    logger_use_csv: bool = True,
    logger_use_csv_long: bool = True,
    logger_use_jsonl: bool = True,
    logger_use_wandb: bool = False,
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
    # ---- trainer directories (default: follow logger.run_dir if logger enabled) ----
    trainer_run_dir: Optional[str] = None,
    trainer_checkpoint_dir: str = "checkpoints",
    trainer_checkpoint_prefix: str = "ckpt",
    trainer_strict_checkpoint: bool = False,
    # ---- config dump ----
    dump_config: bool = True,
    config_filename: str = "config.json",
    # ---- callbacks injection ----
    extra_callbacks: Optional[Sequence[Any]] = None,
    strict_callbacks: bool = False,
    # ---- callbacks: standard toggles (mapped to build_callbacks) ----
    enable_eval_callback: bool = True,
    eval_every_steps: int = 50_000,
    enable_ckpt_callback: bool = True,
    save_every_steps: int = 100_000,
    keep_last_checkpoints: int = 5,
    enable_best_model: bool = False,
    best_metric_key: str = "eval/return_mean",
    best_save_path: str = "best.pt",
    enable_early_stop: bool = False,
    early_stop_metric_key: str = "eval/return_mean",
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.0,
    early_stop_mode: str = "max",  # "max" or "min"
    enable_nan_guard: bool = True,
    nan_guard_keys: Optional[Sequence[str]] = None,
    # ---- callbacks: extra (mapped to build_callbacks) ----
    enable_config_env_info: bool = True,
    enable_episode_stats: bool = True,
    episode_window: int = 100,
    episode_log_every: int = 10,
    enable_timing: bool = True,
    timing_log_every_steps: int = 5_000,
    timing_log_every_updates: int = 200,
    enable_lr_logging: bool = True,
    lr_log_every_updates: int = 200,
    enable_grad_param_norm: bool = False,
    norm_log_every_updates: int = 200,
    norm_per_module: bool = False,
    norm_include_param: bool = True,
    norm_include_grad: bool = True,
    norm_type: float = 2.0,
    # ---- ray callbacks (optional; build_callbacks will ignore if not importable) ----
    enable_ray_report: bool = False,
    ray_report_every_updates: int = 1,
    ray_report_on_eval: bool = True,
    ray_report_keep_last_eval: bool = True,
    enable_ray_tune_ckpt: bool = False,
    ray_tune_report_every_saves: int = 1,
    ray_tune_metric_key: Optional[str] = "eval/return_mean",
) -> Trainer:
    """
    Factory that builds environments, logger, evaluator, callbacks, and Trainer.

    Notes
    -----
    - Trainer is responsible for optional NormalizeWrapper wrapping when normalize=True.
    - If make_eval_env is None, eval_env is created by calling make_train_env() again.
    - If dump_config is True and logger supports dump_config, a summary config is written.
    """
    # ------------------------------------------------------------------
    # 1) Environments (distinct instances)
    # ------------------------------------------------------------------
    train_env = make_train_env()
    eval_env = make_train_env() if make_eval_env is None else make_eval_env()

    # ------------------------------------------------------------------
    # 2) Logger (via factory)
    # ------------------------------------------------------------------
    logger = None
    if enable_logger:
        logger = build_logger(
            log_dir=str(logger_log_dir),
            exp_name=str(logger_exp_name),
            run_id=logger_run_id,
            run_name=logger_run_name,
            overwrite=bool(logger_overwrite),
            resume=bool(logger_resume),
            use_tensorboard=bool(logger_use_tensorboard),
            use_csv=bool(logger_use_csv),
            use_csv_long=bool(logger_use_csv_long),
            use_jsonl=bool(logger_use_jsonl),
            use_wandb=bool(logger_use_wandb),
            wandb_project=logger_wandb_project,
            wandb_entity=logger_wandb_entity,
            wandb_group=logger_wandb_group,
            wandb_tags=logger_wandb_tags,
            wandb_run_name=logger_wandb_run_name,
            wandb_mode=logger_wandb_mode,
            wandb_resume=logger_wandb_resume,
            console_every=int(logger_console_every),
            flush_every=int(logger_flush_every),
            drop_non_finite=bool(logger_drop_non_finite),
            strict=bool(logger_strict),
        )

    # Trainer run_dir policy: follow logger.run_dir if available, unless user overrides
    run_dir = str(trainer_run_dir) if trainer_run_dir is not None else str(getattr(logger, "run_dir", "./runs/exp"))

    # ------------------------------------------------------------------
    # 3) Evaluator (optional)
    # ------------------------------------------------------------------
    evaluator: Optional[Evaluator] = None
    if enable_evaluator:
        evaluator = Evaluator(
            env=eval_env,
            episodes=int(eval_episodes),
            deterministic=bool(eval_deterministic),
            show_progress=bool(eval_show_progress),
            max_episode_steps=max_episode_steps,
            base_seed=int(seed),
            seed_increment=1,
        )

    # ------------------------------------------------------------------
    # 4) Callbacks (via factory)
    # ------------------------------------------------------------------
    callbacks = build_callbacks(
        # switches
        use_eval=bool(enable_eval_callback and eval_every_steps > 0),
        use_checkpoint=bool(enable_ckpt_callback and save_every_steps > 0),
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
        # kwargs per callback (match your callback __init__ signatures)
        eval_kwargs=dict(eval_every=int(eval_every_steps)),
        checkpoint_kwargs=dict(save_every=int(save_every_steps), keep_last=int(keep_last_checkpoints)),
        best_model_kwargs=dict(metric_key=str(best_metric_key), save_path=str(best_save_path)),
        early_stop_kwargs=dict(
            metric_key=str(early_stop_metric_key),
            patience=int(early_stop_patience),
            min_delta=float(early_stop_min_delta),
            mode=("min" if str(early_stop_mode).lower() == "min" else "max"),
        ),
        nan_guard_kwargs=dict(keys=nan_guard_keys),
        timing_kwargs=dict(
            log_every_steps=int(timing_log_every_steps),
            log_every_updates=int(timing_log_every_updates),
            log_prefix="perf/",
        ),
        episode_stats_kwargs=dict(
            window=int(episode_window),
            log_every_episodes=int(episode_log_every),
            log_prefix="rollout/",
        ),
        config_env_info_kwargs=dict(log_prefix="sys/"),
        lr_logging_kwargs=dict(log_every_updates=int(lr_log_every_updates), log_prefix="train/"),
        grad_param_norm_kwargs=dict(
            log_every_updates=int(norm_log_every_updates),
            log_prefix="debug/",
            include_param_norm=bool(norm_include_param),
            include_grad_norm=bool(norm_include_grad),
            norm_type=float(norm_type),
            per_module=bool(norm_per_module),
        ),
        ray_report_kwargs=dict(
            report_every_updates=int(ray_report_every_updates),
            report_on_eval=bool(ray_report_on_eval),
            keep_last_eval=bool(ray_report_keep_last_eval),
        ),
        ray_tune_checkpoint_kwargs=dict(
            report_every_saves=int(ray_tune_report_every_saves),
            metric_key=ray_tune_metric_key,
        ),
        extra_callbacks=extra_callbacks,
        strict_callbacks=bool(strict_callbacks),
    )

    # ------------------------------------------------------------------
    # 5) Build Trainer
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
        normalize=bool(normalize),
        obs_shape=obs_shape,
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
        obs_dtype=obs_dtype,
        flatten_obs=False,  # keep explicit; expose as arg if you want
        strict_checkpoint=bool(trainer_strict_checkpoint),
        seed_envs=True,
    )

    # ------------------------------------------------------------------
    # 6) Dump config (via logger)
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
            normalize=normalize,
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
            enable_evaluator=enable_evaluator,
            eval_episodes=eval_episodes,
            eval_deterministic=eval_deterministic,
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

    Notes
    -----
    - Captures callback class names when possible.
    - Captures algo.cfg if present (dataclass or mapping).
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
        eval_enabled=bool(enable_evaluator),
        eval_episodes=int(eval_episodes),
        eval_deterministic=bool(eval_deterministic),
        trainer_run_dir=str(run_dir),
        trainer_checkpoint_dir=str(trainer_checkpoint_dir),
        trainer_checkpoint_prefix=str(trainer_checkpoint_prefix),
        logger_run_dir=str(logger_run_dir),
    )

    # callback names (CallbackList or list)
    cb_names: List[str] = []
    try:
        cbs = getattr(callbacks, "callbacks", None)
        if isinstance(cbs, list):
            cb_names = [cb.__class__.__name__ for cb in cbs]
    except Exception:
        cb_names = []
    cfg["callbacks"] = cb_names

    # algo cfg (best-effort)
    algo_cfg = getattr(algo, "cfg", None)
    if algo_cfg is not None:
        if is_dataclass(algo_cfg):
            cfg["algo"] = asdict(algo_cfg)
        else:
            try:
                cfg["algo"] = dict(algo_cfg)  # type: ignore[arg-type]
            except Exception:
                cfg["algo"] = str(type(algo_cfg))

    return cfg