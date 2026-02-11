from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

# ---- required callbacks ----
from .base_callback import BaseCallback, CallbackList
from .best_model_callback import BestModelCallback
from .checkpoint_callback import CheckpointCallback
from .config_and_env_info_callback import ConfigAndEnvInfoCallback
from .early_stop_callback import EarlyStopCallback
from .episode_stats_callback import EpisodeStatsCallback
from .eval_callback import EvalCallback
from .grad_param_norm_callback import GradParamNormCallback
from .lr_logging_callback import LRLoggingCallback
from .nan_guard_callback import NaNGuardCallback
from .timing_callback import TimingCallback

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
    Instantiate a callback class with optional keyword arguments.

    This helper supports two modes:

    - **strict=True**:
        - Raise if unknown kwargs are provided (when ``__init__`` has no ``**kwargs``).
        - Raise if instantiation fails for any reason.

    - **strict=False**:
        - Drop unknown kwargs (when ``__init__`` has no ``**kwargs``).
        - Return None if instantiation fails.

    Parameters
    ----------
    cls:
        Callback class (callable). If None, this function returns None.
    kwargs:
        Keyword arguments intended for the callback constructor. May include extra
        keys; they are filtered depending on the callback signature and ``strict``.
    strict:
        Controls error handling and unknown-kwargs policy (see above).

    Returns
    -------
    Optional[BaseCallback]
        Instantiated callback object, or None if:
        - ``cls`` is None, or
        - instantiation failed and ``strict=False``.

    Notes
    -----
    Signature handling
    ------------------
    - If the constructor accepts ``**kwargs``, all provided kwargs are passed through.
    - Otherwise, only parameter names present in ``__init__`` (excluding ``self``)
      are forwarded.
    """
    if cls is None:
        return None

    kw: Dict[str, Any] = dict(kwargs or {})

    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # If constructor accepts **kwargs, pass through unchanged.
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            return cls(**kw)

        # Otherwise, only forward accepted keyword names (excluding self).
        accepted = {name for name in params.keys() if name != "self"}
        filtered = {k: v for k, v in kw.items() if k in accepted}

        if strict:
            unknown = sorted(set(kw.keys()) - accepted)
            if unknown:
                raise TypeError(f"{cls.__name__} got unexpected kwargs: {unknown}")

        return cls(**filtered)

    except Exception:
        if strict:
            raise
        # Non-strict mode: silently skip broken callback instantiation.
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
    # ray switches
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
    Build a standard :class:`CallbackList` for an RL trainer.

    The factory constructs an ordered set of callbacks with optional inclusion
    switches and per-callback kwargs. It also supports optional Ray-integrations
    if those callbacks are importable.

    Parameters
    ----------
    use_eval, use_checkpoint, use_best_model, use_early_stop, use_nan_guard, use_timing, \
    use_episode_stats, use_config_env_info, use_lr_logging, use_grad_param_norm:
        Feature switches controlling whether each corresponding callback is added.
    use_ray_report, use_ray_tune_checkpoint:
        Feature switches controlling whether Ray-related callbacks are added
        (only if importable).
    eval_kwargs, checkpoint_kwargs, best_model_kwargs, early_stop_kwargs, nan_guard_kwargs, \
    timing_kwargs, episode_stats_kwargs, config_env_info_kwargs, lr_logging_kwargs, \
    grad_param_norm_kwargs, ray_report_kwargs, ray_tune_checkpoint_kwargs:
        Optional kwargs dicts forwarded to each callback constructor. Unknown keys
        are handled according to ``strict_callbacks``.
    extra_callbacks:
        User-provided callbacks appended at the end. ``None`` entries are ignored.
    strict_callbacks:
        Controls instantiation behavior:

        - If True: unknown kwargs raise (unless callback accepts **kwargs), and
          instantiation errors propagate.
        - If False: unknown kwargs are dropped, and failing callbacks are skipped.

    Returns
    -------
    CallbackList
        Ordered callback dispatcher.

    Notes
    -----
    Ordering rationale
    -----------------
    This factory places "informational/monitoring" callbacks earlier, then the
    evaluation/checkpoint family, then optional Ray callbacks, and finally any
    user-provided callbacks.

    In many training loops, the ordering mainly matters if callbacks interact
    through shared side effects (e.g., logging, checkpoints), or if a callback
    may request early stop and thus short-circuit later callbacks in the list.
    """
    callbacks: List[BaseCallback] = []

    # --- informational / monitoring callbacks (early) ---
    if use_config_env_info:
        cb = _instantiate_callback(ConfigAndEnvInfoCallback, config_env_info_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_episode_stats:
        cb = _instantiate_callback(EpisodeStatsCallback, episode_stats_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_timing:
        cb = _instantiate_callback(TimingCallback, timing_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_lr_logging:
        cb = _instantiate_callback(LRLoggingCallback, lr_logging_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_grad_param_norm:
        cb = _instantiate_callback(GradParamNormCallback, grad_param_norm_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_nan_guard:
        cb = _instantiate_callback(NaNGuardCallback, nan_guard_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    # --- evaluation / checkpointing family ---
    if use_eval:
        cb = _instantiate_callback(EvalCallback, eval_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_checkpoint:
        cb = _instantiate_callback(CheckpointCallback, checkpoint_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_best_model:
        cb = _instantiate_callback(BestModelCallback, best_model_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_early_stop:
        cb = _instantiate_callback(EarlyStopCallback, early_stop_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    # --- optional Ray callbacks ---
    if use_ray_report and RayReportCallback is not None:
        cb = _instantiate_callback(RayReportCallback, ray_report_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    if use_ray_tune_checkpoint and RayTuneCheckpointCallback is not None:
        cb = _instantiate_callback(RayTuneCheckpointCallback, ray_tune_checkpoint_kwargs, strict=strict_callbacks)
        if cb is not None:
            callbacks.append(cb)

    # --- user-provided callbacks last ---
    if extra_callbacks:
        for cb in extra_callbacks:
            if cb is not None:
                callbacks.append(cb)

    return CallbackList(callbacks)
