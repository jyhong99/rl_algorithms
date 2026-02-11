from __future__ import annotations

from typing import Any, Dict, Mapping

from ..utils.train_utils import _sync_normalize_state, _maybe_call


def run_evaluation(trainer: Any) -> Dict[str, Any]:
    """
    Run evaluation using ``trainer.evaluator`` (if present) and emit side effects.

    This helper consolidates the evaluation pipeline commonly found in RL trainers:

    1) (Optional) Synchronize NormalizeWrapper running statistics from train_env -> eval_env
    2) Run evaluator rollouts via ``trainer.evaluator.evaluate(trainer.algo)``
    3) Emit side effects:
         - log metrics to ``trainer.logger``
         - notify callbacks via ``callbacks.on_eval_end(...)``
         - optionally set ``trainer._stop_training = True`` if callbacks request stop

    Parameters
    ----------
    trainer : Any
        Trainer-like object (duck-typed) that may provide:
          - evaluator : object with ``evaluate(agent) -> Mapping[str, Any]``
          - algo : agent/policy object passed to evaluator
          - global_env_step : int, used as the logging step index
          - train_env, eval_env : env objects (for normalization sync)
          - _normalize_enabled : bool (optional; enables normalization sync)
          - logger : object with ``log(metrics: Mapping, step: int, prefix: str)`` (optional)
          - callbacks : object with ``on_eval_end(trainer, metrics)`` (optional)
          - _warn(msg: str) : warning hook (optional)
          - _msg_pbar : progress bar-like object (optional; supports set_description_str)

    Returns
    -------
    metrics : Dict[str, Any]
        Evaluation metrics dictionary.
        - Returns an empty dict if no evaluator is attached.
        - Returns an empty dict if evaluator returns a non-mapping result.

    Notes
    -----
    - Normalization sync is best-effort and warns once per trainer instance via
      ``trainer._warned_norm_sync``.
    - Logging and callback invocation are best-effort: failures are swallowed.
    - Callback return convention:
        If ``callbacks.on_eval_end(...)`` returns False, this helper sets
        ``trainer._stop_training = True`` as a stop request flag.
    """
    evaluator = getattr(trainer, "evaluator", None)
    if evaluator is None:
        return {}

    _maybe_sync_normalize_state(trainer)

    metrics = evaluator.evaluate(getattr(trainer, "algo", None))
    out = dict(metrics) if isinstance(metrics, Mapping) else {}

    if out:
        _maybe_log_eval_metrics(trainer, out)
        _maybe_fire_eval_callbacks(trainer, out)

    return out


# =============================================================================
# Internal helpers
# =============================================================================
def _maybe_sync_normalize_state(trainer: Any) -> None:
    """
    Synchronize NormalizeWrapper running stats from train_env -> eval_env (best-effort).

    Parameters
    ----------
    trainer : Any
        Trainer-like object that may provide:
          - _normalize_enabled : bool
          - train_env, eval_env : environment instances
          - _warn(msg) : warning hook (optional)
          - _warned_norm_sync : bool guard (optional; set on first failure)

    Notes
    -----
    - This is a best-effort operation. Any failure results in:
        - a single warning (once per trainer instance), and then
        - evaluation proceeds without sync (may be inconsistent).
    """
    if not bool(getattr(trainer, "_normalize_enabled", False)):
        return

    ok = False
    try:
        ok = bool(
            _sync_normalize_state(
                getattr(trainer, "train_env", None),
                getattr(trainer, "eval_env", None),
            )
        )
    except Exception:
        ok = False

    if ok:
        return

    _warn_once_norm_sync(trainer)


def _warn_once_norm_sync(trainer: Any) -> None:
    """
    Emit a one-time warning for normalization sync failure.

    This uses ``trainer._warned_norm_sync`` as a guard flag and calls ``trainer._warn``
    when available.
    """
    if bool(getattr(trainer, "_warned_norm_sync", False)):
        return

    try:
        setattr(trainer, "_warned_norm_sync", True)
        warn_fn = getattr(trainer, "_warn", None)
        if callable(warn_fn):
            warn_fn("NormalizeWrapper state sync train->eval failed (evaluation may be inconsistent).")
    except Exception:
        pass


def _maybe_log_eval_metrics(trainer: Any, metrics: Mapping[str, Any]) -> None:
    """
    Log evaluation metrics via ``trainer.logger`` if available.

    Parameters
    ----------
    trainer : Any
        Trainer-like object that may provide:
          - logger.log(metrics, step, prefix)
          - global_env_step : int
          - _msg_pbar : progress-bar like object (optional)
    metrics : Mapping[str, Any]
        Evaluation metric mapping.

    Notes
    -----
    - Uses step = trainer.global_env_step.
    - Uses prefix = "eval/" unless keys already start with "eval/".
    - Also updates a progress-bar description (if trainer._msg_pbar exists) with:
        eval step=<...> return_mean=<...> return_std=<...>
      when those keys are present.
    """
    logger = getattr(trainer, "logger", None)
    log_fn = getattr(logger, "log", None) if logger is not None else None
    if not callable(log_fn):
        return

    try:
        step = int(getattr(trainer, "global_env_step", 0))
        prefix = "" if _has_eval_prefix(metrics) else "eval/"
        log_fn(dict(metrics), step=step, prefix=prefix)
    except Exception:
        pass

    _maybe_update_eval_pbar(trainer, metrics)


def _has_eval_prefix(metrics: Mapping[str, Any]) -> bool:
    """
    Check whether any metric key is already prefixed with 'eval/'.

    Parameters
    ----------
    metrics : Mapping[str, Any]

    Returns
    -------
    already_prefixed : bool
        True if any string key starts with 'eval/'.
    """
    for k in metrics.keys():
        if isinstance(k, str) and k.startswith("eval/"):
            return True
    return False


def _maybe_update_eval_pbar(trainer: Any, metrics: Mapping[str, Any]) -> None:
    """
    Update trainer progress bar description with common eval stats (best-effort).

    Parameters
    ----------
    trainer : Any
        Trainer-like object that may provide `_msg_pbar` supporting `set_description_str`.
    metrics : Mapping[str, Any]
        Evaluation metrics. Looks for:
          - "eval/return_mean"
          - "eval/return_std"
    """
    msg_pbar = getattr(trainer, "_msg_pbar", None)
    if msg_pbar is None or not metrics:
        return

    try:
        step = int(getattr(trainer, "global_env_step", 0))
        rm = metrics.get("eval/return_mean", None)
        rs = metrics.get("eval/return_std", None)

        msg = f"eval step={step}"
        if rm is not None:
            msg += f" return_mean={float(rm):.4g}"
        if rs is not None:
            msg += f" return_std={float(rs):.4g}"

        msg_pbar.set_description_str(msg, refresh=True)
    except Exception:
        pass


def _maybe_fire_eval_callbacks(trainer: Any, metrics: Mapping[str, Any]) -> None:
    """
    Notify callbacks that evaluation has finished (best-effort).

    Parameters
    ----------
    trainer : Any
        Trainer-like object that may provide:
          - callbacks : object
          - _stop_training : bool flag (this function may set it)
    metrics : Mapping[str, Any]
        Raw evaluation metrics.

    Notes
    -----
    - Metric key compatibility:
        Some callback stacks expect all metrics to be namespaced with "eval/".
        This helper builds a payload that:
          - includes the original metrics as-is, and
          - additionally adds "eval/<k>" for any key not already prefixed.
    - Stop convention:
        If `_maybe_call(callbacks, "on_eval_end", ...)` returns False, we set
        `trainer._stop_training = True`.
    """
    callbacks = getattr(trainer, "callbacks", None)
    if callbacks is None:
        return

    payload = _build_eval_payload(metrics)

    try:
        cont = _maybe_call(callbacks, "on_eval_end", trainer, payload)
        if cont is False:
            setattr(trainer, "_stop_training", True)
    except Exception:
        pass


def _build_eval_payload(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build an evaluation payload for callbacks with key namespace compatibility.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Raw evaluation metrics (possibly already namespaced).

    Returns
    -------
    payload : Dict[str, Any]
        Copy of metrics plus prefixed aliases:
          - for every key k not starting with "eval/", add "eval/k" -> value
    """
    payload: Dict[str, Any] = dict(metrics)
    for k, v in metrics.items():
        if isinstance(k, str) and k and not k.startswith("eval/"):
            payload[f"eval/{k}"] = v
    return payload
