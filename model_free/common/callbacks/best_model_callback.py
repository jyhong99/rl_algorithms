from __future__ import annotations

from typing import Any, Dict, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, to_finite_float
from ..utils.log_utils import log


class BestModelCallback(BaseCallback):
    """
    Save a "best" checkpoint when evaluation finishes and a target metric improves.

    This callback hooks into `on_eval_end(...)` and:
      1) reads a scalar metric from the evaluation metrics dict
      2) compares it to the best value seen so far
      3) if improved, saves a checkpoint and logs the new best value

    Parameters
    ----------
    metric_key : str
        Key to read from the evaluation metrics dict passed to on_eval_end(...).
        Examples:
          - "eval/return_mean"
          - "eval_return_mean"
    save_path : str
        Path passed to trainer.save_checkpoint(...).
        If empty string, checkpoint saving is disabled (no-op).
    mode : str
        Optimization direction: either "max" or "min".
        - "max": larger metric is better (e.g., return, success rate)
        - "min": smaller metric is better (e.g., loss, error)

    Notes
    -----
    Trainer contract (duck-typed):
      - save_checkpoint(path: Optional[str] = None) -> Optional[str] (recommended)
      - logger.log(metrics: Dict[str, Any], step: int, prefix: str = "") (optional)

    Robustness policy
    -----------------
    - Missing / non-finite metrics are ignored.
    - All save/log errors are swallowed (best-effort).
    - The callback never throws to avoid interrupting training.

    Typical usage
    -------------
    - Callbacks are expected to be invoked by a Trainer:
        callbacks.on_eval_end(trainer, eval_metrics)
    """

    def __init__(
        self,
        metric_key: str = "eval_return_mean",
        save_path: str = "best.pt",
        *,
        mode: str = "max",
    ):
        # Which metric do we monitor as the "selection criterion" for best model.
        self.metric_key = str(metric_key)

        # Where to save the best checkpoint.
        # If "", saving is disabled (still tracks best metric value).
        self.save_path = str(save_path)

        # Normalize mode input and validate.
        mode = str(mode).lower().strip()
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got: {mode}")
        self.mode = mode

        # Stores the best metric value seen so far.
        # None means "no best yet" (first valid eval becomes best).
        self.best: Optional[float] = None

    def _is_better(self, value: float, best: Optional[float]) -> bool:
        """
        Compare `value` against `best` using mode semantics.

        Returns
        -------
        bool
            True if `value` should replace `best`.
        """
        # First valid metric always becomes the best.
        if best is None:
            return True

        # For "max": higher is better.
        if self.mode == "max":
            return value > best

        # For "min": lower is better.
        return value < best  # mode == "min"

    def _save_checkpoint(self, trainer: Any) -> None:
        """
        Best-effort checkpoint saving.

        Why the signature fallback?
        ---------------------------
        Different Trainer implementations may define:
          - save_checkpoint(path="...")      (keyword-only)
          - save_checkpoint("...")           (positional)
          - save_checkpoint() with internal naming

        This helper tries keyword first, then positional.
        Any exception results in a silent no-op.
        """
        # If save_path is empty, disable saving.
        if not self.save_path:
            return

        # Trainer must expose a callable save_checkpoint.
        save_fn = getattr(trainer, "save_checkpoint", None)
        if not callable(save_fn):
            return

        # Prefer keyword argument path=...
        # This is more stable if Trainer uses multiple parameters.
        try:
            save_fn(path=self.save_path)
            return
        except TypeError:
            # Signature mismatch -> fall back to positional attempt below.
            pass
        except Exception:
            # Any runtime failure should not stop training.
            return

        # Fallback: positional call save_checkpoint(self.save_path)
        try:
            save_fn(self.save_path)
        except Exception:
            return

    def _log_best(self, trainer: Any, best_value: float) -> None:
        """
        Best-effort logging of the best metric value.

        Logging format
        --------------
        Logs under a stable namespace:
            best/<metric_key>

        Example:
            best/eval_return_mean = 123.4

        Notes
        -----
        - Uses `safe_int_attr(trainer)` to get a robust step value
          (e.g., trainer.total_env_steps or similar).
        - Uses `log(trainer, ...)` wrapper so missing logger backends
          do not break training.
        """
        step = safe_int_attr(trainer)

        # Log the current best under "best/" namespace to avoid collisions.
        payload = {f"best/{self.metric_key}": float(best_value)}
        log(trainer, payload, step=step, prefix="")

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Called by the Trainer when an evaluation phase ends.

        Parameters
        ----------
        trainer : Any
            Trainer-like object containing save_checkpoint, step counters, logger, etc.
        metrics : Dict[str, Any]
            Evaluation metrics dictionary. Must contain `metric_key` to be tracked.

        Returns
        -------
        bool
            True to continue training. (Callback never requests stop.)
        """
        # Defensive: ignore non-dict metrics.
        if not isinstance(metrics, dict):
            return True

        # Read raw metric value from dict.
        raw = metrics.get(self.metric_key, None)

        # Convert to finite float:
        # - returns None if raw is missing, NaN, inf, non-numeric, etc.
        val = to_finite_float(raw)
        if val is None:
            return True

        # If improved, update best and persist + log.
        if self._is_better(val, self.best):
            self.best = val
            self._save_checkpoint(trainer)
            self._log_best(trainer, best_value=val)

        return True