from __future__ import annotations

from typing import Any, Dict, Mapping

from ..utils.train_utils import sync_normalize_state, maybe_call


def run_evaluation(trainer: Any) -> Dict[str, Any]:
    """
    Run evaluation using `trainer.evaluator` (if provided) and emit side effects.

    This helper performs three responsibilities:
      1) (optional) Synchronize NormalizeWrapper running statistics from train_env -> eval_env
      2) Run evaluator rollouts via `trainer.evaluator.evaluate(trainer.algo)`
      3) Log metrics and notify callbacks

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing (duck-typed):
          - evaluator : object with evaluate(agent) -> Mapping[str, Any] (optional)
          - algo : agent/policy object passed to evaluator
          - global_env_step : int (used as logging step)
          - train_env, eval_env : env objects (used for normalization sync)
          - _normalize_enabled : bool (optional)
          - logger : object with log(metrics: Mapping, step: int, prefix: str) (optional)
          - callbacks : object with on_eval_end(trainer, metrics) (optional)
          - _warn(msg: str) method (best-effort)

    Returns
    -------
    metrics : Dict[str, Any]
        Evaluation metrics. Returns empty dict if no evaluator is attached or
        if evaluator returned a non-mapping result.

    Notes
    -----
    - Normalization sync is best-effort and warns once per trainer instance via
      `trainer._warned_norm_sync`.
    - Callback invocation is best-effort via `maybe_call`.
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
    Synchronize running statistics from train_env -> eval_env if normalization is enabled.

    Warns once on failure using `trainer._warned_norm_sync`.
    """
    if not bool(getattr(trainer, "_normalize_enabled", False)):
        return

    ok = False
    try:
        ok = bool(sync_normalize_state(getattr(trainer, "train_env", None), getattr(trainer, "eval_env", None)))
    except Exception:
        ok = False

    if ok:
        return

    if not bool(getattr(trainer, "_warned_norm_sync", False)):
        setattr(trainer, "_warned_norm_sync", True)
        warn_fn = getattr(trainer, "_warn", None)
        if callable(warn_fn):
            warn_fn("NormalizeWrapper state sync train->eval failed (evaluation may be inconsistent).")


def _maybe_log_eval_metrics(trainer: Any, metrics: Mapping[str, Any]) -> None:
    """
    Log evaluation metrics via trainer.logger if available.

    Notes
    -----
    - Uses step = trainer.global_env_step
    - Uses prefix = "eval/" (logger may further nest)
    """
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return
    log_fn = getattr(logger, "log", None)
    if not callable(log_fn):
        return

    try:
        step = int(getattr(trainer, "global_env_step", 0))

        keys = list(metrics.keys())
        already_prefixed = any(isinstance(k, str) and k.startswith("eval/") for k in keys)
        prefix = "" if already_prefixed else "eval/"

        log_fn(dict(metrics), step=step, prefix=prefix)
    except Exception:
        pass

    msg_pbar = getattr(trainer, "_msg_pbar", None)
    if msg_pbar is not None and metrics:
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
    callbacks = getattr(trainer, "callbacks", None)
    if callbacks is None:
        return

    # 콜백 키 호환 (이미 적용했을 가능성)
    payload = dict(metrics)
    for k, v in metrics.items():
        if isinstance(k, str) and k and not k.startswith("eval/"):
            payload[f"eval/{k}"] = v

    try:
        cont = maybe_call(callbacks, "on_eval_end", trainer, payload)
        if cont is False:
            # ✅ 여기서 학습 중단 요청을 trainer에 기록
            setattr(trainer, "_stop_training", True)
    except Exception:
        pass

