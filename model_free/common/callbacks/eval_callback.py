from __future__ import annotations

from typing import Any, Dict, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import IntervalGate, _safe_env_step


class EvalCallback(BaseCallback):
    """
    Periodic evaluation callback scheduled by environment steps.

    This callback triggers evaluation every ``eval_every`` environment steps by
    calling ``trainer.run_evaluation()`` (if provided). Optionally, it can
    dispatch an ``on_eval_end`` hook to the trainer's callback dispatcher as a
    fallback integration path for downstream callbacks (e.g., best-model saving,
    early stopping).

    Parameters
    ----------
    eval_every:
        Evaluation interval in **environment steps**.

        - If ``eval_every <= 0``: evaluation is disabled (no-op).
    dispatch_eval_end:
        If True and ``trainer.run_evaluation()`` returns a metrics dict, this
        callback will best-effort dispatch::

            trainer.callbacks.on_eval_end(trainer, metrics)

        This is useful when the trainer's evaluation path does not already
        broadcast evaluation results to callbacks.

    Attributes
    ----------
    eval_every:
        Environment-step evaluation interval.
    dispatch_eval_end:
        Whether to dispatch ``on_eval_end`` with returned metrics.
    _gate:
        Interval scheduler (``IntervalGate(mode="delta")``) that decides when
        evaluation is due.
    _last_eval_trigger_step:
        Local guard to prevent double-triggering at the same env-step.

    Notes
    -----
    Scheduling policy
    -----------------
    Uses ``IntervalGate(mode="delta")`` for robustness against:
    - step jumps (e.g., trainer increments steps in chunks),
    - irregular callback invocation frequency.

    Duplicate-trigger guard
    -----------------------
    If the callback is invoked twice for the same env-step (e.g., due to a bug in
    the trainer loop), ``_last_eval_trigger_step`` prevents running evaluation
    twice at the same step.
    """

    def __init__(self, eval_every: int = 50_000, *, dispatch_eval_end: bool = False) -> None:
        self.eval_every = int(eval_every)
        self.dispatch_eval_end = bool(dispatch_eval_end)

        # Gate uses delta-based triggering (not strictly absolute alignment).
        self._gate = IntervalGate(every=self.eval_every, mode="delta")

        # Prevent duplicate eval at the same step.
        self._last_eval_trigger_step: Optional[int] = None

    def on_train_start(self, trainer: Any) -> bool:
        """
        Initialize evaluation schedule when training starts (or resumes).

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Used to infer current env-step via
            :func:`safe_env_step`.

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        If evaluation is enabled, the gate is aligned such that the *next* evaluation
        will occur at the next boundary strictly greater than the current step.

        Example
        -------
        If ``eval_every = 100`` and current step is 250, the next eval should be
        at 300, therefore we set::

            gate.last = floor(250 / 100) * 100 = 200
        """
        if self.eval_every <= 0:
            # Disabled mode: keep the gate consistent and reset guards.
            self._gate.every = self.eval_every
            self._gate.last = 0
            self._last_eval_trigger_step = None
            return True

        step = _safe_env_step(trainer)
        if step < 0:
            step = 0

        self._gate.every = self.eval_every
        self._gate.last = (step // self.eval_every) * self.eval_every
        self._last_eval_trigger_step = None
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Potentially trigger evaluation based on the environment-step schedule.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Expected to provide:
            - ``run_evaluation() -> Optional[Dict[str, Any]]`` (optional)
            - ``callbacks.on_eval_end(trainer, metrics)`` if ``dispatch_eval_end=True`` (optional)
        transition:
            Unused transition payload (accepted for hook compatibility).

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Control flow
        ------------
        1. If evaluation is disabled (``eval_every <= 0``): no-op.
        2. Read current env-step via :func:`safe_env_step`; if non-positive: no-op.
        3. If the interval gate is not ready: no-op.
        4. If ready:
           - prevent double-trigger at the same step,
           - call ``trainer.run_evaluation()`` best-effort,
           - optionally dispatch ``on_eval_end`` with returned metrics dict.
        """
        if self.eval_every <= 0:
            return True

        step = _safe_env_step(trainer)
        if step <= 0:
            return True

        if not self._gate.ready(step):
            return True

        # Guard against double triggering at the same step.
        if self._last_eval_trigger_step == step:
            return True
        self._last_eval_trigger_step = step

        run_eval = getattr(trainer, "run_evaluation", None)
        if callable(run_eval):
            try:
                out = run_eval()
            except Exception:
                # Evaluation failures should not crash training.
                out = None

            if self.dispatch_eval_end and isinstance(out, dict):
                cbs = getattr(trainer, "callbacks", None)
                on_eval_end = getattr(cbs, "on_eval_end", None) if cbs is not None else None
                if callable(on_eval_end):
                    try:
                        on_eval_end(trainer, out)
                    except Exception:
                        # Downstream callback failures should not crash training.
                        pass

        return True
