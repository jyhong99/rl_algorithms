from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

# ---- required callbacks ----
from .base_callback import BaseCallback, CallbackList
from .best_model_callback import BestModelCallback
from .checkpoint_callback import CheckpointCallback
from .eval_callback import EvalCallback
from .early_stop_callback import EarlyStopCallback
from .nan_guard_callback import NaNGuardCallback
from .timing_callback import TimingCallback
from .episode_stats_callback import EpisodeStatsCallback
from .config_and_env_info_callback import ConfigAndEnvInfoCallback
from .lr_logging_callback import LRLoggingCallback
from .grad_param_norm_callback import GradParamNormCallback

# ---- optional Ray callbacks ----
try:
    from .ray_report_callback import RayReportCallback  # type: ignore
except Exception:  # pragma: no cover
    RayReportCallback = None  # type: ignore

try:
    from .ray_tune_checkpoint_callback import RayTuneCheckpointCallback  # type: ignore
except Exception:  # pragma: no cover
    RayTuneCheckpointCallback = None  # type: ignore


def _instantiate_callback(
    cls: Any,
    kwargs: Optional[Dict[str, Any]],
    *,
    strict: bool,
) -> Optional[BaseCallback]:
    """
    Instantiate a callback with kwargs, optionally filtering unknown kwargs.

    Parameters
    ----------
    cls : Any
        Callback class (must be callable). If None, returns None.
    kwargs : Optional[Dict[str, Any]]
        Keyword arguments to pass to the constructor.
    strict : bool
        If True:
          - raise if unknown kwargs are provided (when __init__ doesn't accept **kwargs)
          - raise if instantiation fails
        If False:
          - drop unknown kwargs and return None on instantiation failure

    Returns
    -------
    Optional[BaseCallback]
        Instantiated callback, or None if cls is None or instantiation failed (strict=False).
    """
    if cls is None:
        return None

    kw: Dict[str, Any] = dict(kwargs or {})

    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # If constructor has **kwargs, pass as-is
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            obj = cls(**kw)
            return obj

        # Filter only accepted names (excluding self)
        accepted = {name for name in params.keys() if name != "self"}
        filtered = {k: v for k, v in kw.items() if k in accepted}

        if strict:
            unknown = sorted(set(kw.keys()) - accepted)
            if unknown:
                raise TypeError(f"{cls.__name__} got unexpected kwargs: {unknown}")

        obj = cls(**filtered)
        return obj

    except Exception as e:
        if strict:
            raise
        # Non-strict mode: silently skip broken callback instantiation
        return None


# =============================================================================
# Factory
# =============================================================================
def build_callbacks(
    *,
    # switches
    use_eval: bool = True,
    use_checkpoint: bool = True,
    use_best_model: bool = True,
    use_early_stop: bool = False,
    use_nan_guard: bool = True,
    use_timing: bool = True,
    use_episode_stats: bool = True,
    use_config_env_info: bool = True,
    use_lr_logging: bool = True,
    use_grad_param_norm: bool = False,
    # ray
    use_ray_report: bool = False,
    use_ray_tune_checkpoint: bool = False,
    # kwargs per callback
    eval_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_kwargs: Optional[Dict[str, Any]] = None,
    best_model_kwargs: Optional[Dict[str, Any]] = None,
    early_stop_kwargs: Optional[Dict[str, Any]] = None,
    nan_guard_kwargs: Optional[Dict[str, Any]] = None,
    timing_kwargs: Optional[Dict[str, Any]] = None,
    episode_stats_kwargs: Optional[Dict[str, Any]] = None,
    config_env_info_kwargs: Optional[Dict[str, Any]] = None,
    lr_logging_kwargs: Optional[Dict[str, Any]] = None,
    grad_param_norm_kwargs: Optional[Dict[str, Any]] = None,
    ray_report_kwargs: Optional[Dict[str, Any]] = None,
    ray_tune_checkpoint_kwargs: Optional[Dict[str, Any]] = None,
    # extra
    extra_callbacks: Optional[Sequence[Optional[BaseCallback]]] = None,
    strict_callbacks: bool = False,
) -> CallbackList:
    """
    Build a standard CallbackList for an RL Trainer.

    Notes
    -----
    - Each *_kwargs dict is filtered against the callback __init__ signature
      unless strict_callbacks=True.
    - Ray callbacks are included only if importable.
    - extra_callbacks are appended as-is (None entries ignored).

    Returns
    -------
    CallbackList
        Ordered callback dispatcher.
    """
    cbs: List[BaseCallback] = []

    # --- non-terminal / informational callbacks first ---
    if use_config_env_info:
        cb = _instantiate_callback(ConfigAndEnvInfoCallback, config_env_info_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_episode_stats:
        cb = _instantiate_callback(EpisodeStatsCallback, episode_stats_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_timing:
        cb = _instantiate_callback(TimingCallback, timing_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_lr_logging:
        cb = _instantiate_callback(LRLoggingCallback, lr_logging_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_grad_param_norm:
        cb = _instantiate_callback(GradParamNormCallback, grad_param_norm_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_nan_guard:
        cb = _instantiate_callback(NaNGuardCallback, nan_guard_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    # --- evaluation / checkpointing family ---
    if use_eval:
        cb = _instantiate_callback(EvalCallback, eval_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_checkpoint:
        cb = _instantiate_callback(CheckpointCallback, checkpoint_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_best_model:
        cb = _instantiate_callback(BestModelCallback, best_model_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_early_stop:
        cb = _instantiate_callback(EarlyStopCallback, early_stop_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    # --- optional Ray callbacks ---
    if use_ray_report and RayReportCallback is not None:
        cb = _instantiate_callback(RayReportCallback, ray_report_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    if use_ray_tune_checkpoint and RayTuneCheckpointCallback is not None:
        cb = _instantiate_callback(RayTuneCheckpointCallback, ray_tune_checkpoint_kwargs, strict=strict_callbacks)
        if cb is not None:
            cbs.append(cb)

    # --- user-provided callbacks last ---
    if extra_callbacks:
        for cb in extra_callbacks:
            if cb is None:
                continue
            cbs.append(cb)

    return CallbackList(cbs)