from __future__ import annotations

from typing import Any, Dict, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, IntervalGate


class EvalCallback(BaseCallback):
    """
    Periodic evaluation callback scheduled by environment steps.

    This callback triggers evaluation every `eval_every` environment steps by calling
    `trainer.run_evaluation()` if it exists.

    Parameters
    ----------
    eval_every : int
        Evaluation interval in *environment steps*.
        - If <= 0: evaluation is disabled (no-op).
    dispatch_eval_end : bool
        If True, and if `trainer.run_evaluation()` returns a metrics dict,
        this callback will also dispatch `trainer.callbacks.on_eval_end(trainer, metrics)`
        as a fallback integration path for other callbacks (e.g., best model, early stop).

    Notes
    -----
    - Scheduling uses IntervalGate(mode="delta") to be robust against:
        * step jumps (e.g., when trainer increases step in chunks)
        * irregular callback invocation frequencies
    - A local guard `_last_eval_trigger_step` prevents duplicate eval triggers if the
      callback is accidentally invoked twice for the same env-step.
    """

    def __init__(self, eval_every: int = 50_000, *, dispatch_eval_end: bool = False):
        # User configuration
        self.eval_every = int(eval_every)
        self.dispatch_eval_end = bool(dispatch_eval_end)

        # Interval gate determines when evaluation is "due".
        # mode="delta" means it cares about step deltas, not absolute alignment only.
        self._gate = IntervalGate(every=self.eval_every, mode="delta")

        # Prevent double-trigger at the same `step` if callback is invoked twice.
        self._last_eval_trigger_step: Optional[int] = None

    def on_train_start(self, trainer: Any) -> bool:
        """
        Initialize evaluation schedule when training starts (or resumes).

        Key behaviors
        -------------
        - If eval is disabled (eval_every <= 0): reset internal state and return.
        - Else: align gate.last to the most recent completed boundary so the *next*
          evaluation happens at the next boundary strictly greater than current step.

          Example:
            eval_every = 100
            current step = 250
            next eval should be at 300
            therefore gate.last is set to 200 (the last completed boundary)
        """
        if self.eval_every <= 0:
            # Disabled mode: ensure gate is consistent and clear trigger guard.
            self._gate.every = self.eval_every
            self._gate.last = 0
            self._last_eval_trigger_step = None
            return True

        # Read current step from trainer in a robust way (best-effort).
        step = safe_int_attr(trainer)
        if step < 0:
            step = 0

        # Align scheduling so that next eval is at the next boundary:
        #   gate.last = floor(step / eval_every) * eval_every
        self._gate.every = self.eval_every
        self._gate.last = (step // self.eval_every) * self.eval_every

        # Reset duplicate-trigger guard.
        self._last_eval_trigger_step = None
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called on each environment step (or step batch, depending on trainer).

        Logic
        -----
        1) If evaluation disabled -> no-op.
        2) Fetch current step; if not positive -> no-op.
        3) If IntervalGate not ready -> no-op.
        4) If ready:
            - protect against double-trigger at the same step
            - call trainer.run_evaluation() best-effort
            - optionally dispatch on_eval_end with returned metrics dict
        """
        # Fast exit if evaluation is disabled.
        if self.eval_every <= 0:
            return True

        # Best-effort step acquisition; require positive step for meaningful scheduling.
        step = safe_int_attr(trainer)
        if step <= 0:
            return True

        # Gate decides whether we have reached the evaluation interval.
        if not self._gate.ready(step):
            return True

        # Guard: avoid running evaluation twice at the same step
        # (can happen if callbacks are invoked twice for the same env-step).
        if self._last_eval_trigger_step == step:
            return True
        self._last_eval_trigger_step = step

        # Call evaluation if trainer provides it.
        run_eval = getattr(trainer, "run_evaluation", None)
        if callable(run_eval):
            try:
                out = run_eval()
            except Exception:
                # Evaluation errors should not crash training; swallow and continue.
                out = None

            # Optional: dispatch eval-end callback for downstream consumers.
            # This is useful if the trainer's evaluation path doesn't automatically call it.
            if self.dispatch_eval_end and isinstance(out, dict):
                cbs = getattr(trainer, "callbacks", None)
                on_eval_end = getattr(cbs, "on_eval_end", None) if cbs is not None else None
                if callable(on_eval_end):
                    try:
                        on_eval_end(trainer, out)
                    except Exception:
                        # Do not let downstream callback errors crash training.
                        pass

        return True
