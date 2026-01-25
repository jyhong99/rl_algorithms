from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, to_finite_float
from ..utils.log_utils import log


class EarlyStopCallback(BaseCallback):
    """
    Early stopping based on evaluation metric stagnation.

    This callback monitors a scalar evaluation metric reported at `on_eval_end(...)`.
    If the metric does not improve for `patience` consecutive evaluation events,
    it requests the Trainer to stop training by returning False.

    Parameters
    ----------
    metric_key : str
        Metric name to read from the `metrics` dict passed to `on_eval_end`.
        Example: "eval_return_mean", "eval/return_mean", etc.
    patience : int
        Number of consecutive "non-improving" eval events tolerated before stopping.
        Must be >= 1.
    min_delta : float
        Minimum required improvement magnitude.
        - For mode="max": improvement if val > best + min_delta
        - For mode="min": improvement if val < best - min_delta
        Must be >= 0.
    mode : Literal["max", "min"]
        Direction of improvement.
        - "max": higher is better (return, accuracy, success rate)
        - "min": lower is better (loss, error)
    log_prefix : str
        Prefix passed to the logger so early-stop metrics are grouped under a namespace.
        Example: "sys/" => logs "sys/early_stop/best", ...

    Behavior summary
    ----------------
    - If the metric is missing / invalid / non-finite => ignore this eval event.
    - On first valid eval => initialize best = current value, bad_count = 0
    - On improvement => update best, reset bad_count
    - On no improvement => increment bad_count
    - If bad_count >= patience => stop training (return False)
    """

    def __init__(
        self,
        metric_key: str = "eval_return_mean",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["max", "min"] = "max",
        *,
        log_prefix: str = "sys/",
    ):
        # -----------------------------
        # User configuration
        # -----------------------------
        self.metric_key = str(metric_key)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.log_prefix = str(log_prefix)

        # -----------------------------
        # Validation
        # -----------------------------
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode}")

        # -----------------------------
        # Internal state
        # -----------------------------
        # best: best metric observed so far (None until first valid eval)
        self.best: Optional[float] = None

        # bad_count: number of consecutive evals without improvement
        self.bad_count: int = 0

        # last: last valid metric value observed
        self.last: Optional[float] = None

    def _is_improved(self, val: float, best: float) -> bool:
        """
        Check whether `val` is an improvement over `best` considering mode and min_delta.

        Returns
        -------
        bool
            True if improved, else False.
        """
        # Higher-is-better improvement rule.
        if self.mode == "max":
            return val > (best + self.min_delta)

        # Lower-is-better improvement rule.
        return val < (best - self.min_delta)

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Called when an evaluation phase ends.

        Returns
        -------
        bool
            True  -> continue training
            False -> request early stop
        """
        # Defensive: ignore malformed payloads.
        if not isinstance(metrics, dict):
            return True

        # Extract the monitored metric.
        raw = metrics.get(self.metric_key, None)

        # Convert to a finite float (None if missing/NaN/inf/non-numeric).
        val = to_finite_float(raw)
        if val is None:
            return True

        # Track last-seen valid metric (useful for debugging/inspection).
        self.last = val

        # Best-effort step index for logging alignment.
        # Usually env steps are fine for callback-level diagnostics.
        step = safe_int_attr(trainer)

        # ================================================================
        # Case 1) First valid evaluation -> initialize best
        # ================================================================
        if self.best is None:
            self.best = val
            self.bad_count = 0

            # Log initialization state (useful to verify callback is active).
            log(
                trainer,
                {
                    "early_stop/init": 1.0,
                    "early_stop/best": float(self.best),
                    "early_stop/last": float(val),
                    "early_stop/bad_count": float(self.bad_count),
                    "early_stop/patience": float(self.patience),
                    "early_stop/min_delta": float(self.min_delta),
                },
                step=step,
                prefix=self.log_prefix,
            )
            return True

        # ================================================================
        # Case 2) Improvement -> update best and reset bad_count
        # ================================================================
        if self._is_improved(val, self.best):
            self.best = val
            self.bad_count = 0

            log(
                trainer,
                {
                    "early_stop/improved": 1.0,
                    "early_stop/best": float(self.best),
                    "early_stop/last": float(val),
                    "early_stop/bad_count": float(self.bad_count),
                },
                step=step,
                prefix=self.log_prefix,
            )
            return True

        # ================================================================
        # Case 3) No improvement -> increment bad_count
        # ================================================================
        self.bad_count += 1

        log(
            trainer,
            {
                "early_stop/no_improve": 1.0,
                "early_stop/best": float(self.best),
                "early_stop/last": float(val),
                "early_stop/bad_count": float(self.bad_count),
                "early_stop/patience": float(self.patience),
            },
            step=step,
            prefix=self.log_prefix,
        )

        # ================================================================
        # Case 4) Patience exceeded -> trigger early stop
        # ================================================================
        if self.bad_count >= self.patience:
            log(
                trainer,
                {
                    "early_stop/triggered": 1.0,
                    "early_stop/best": float(self.best),
                    "early_stop/last": float(val),
                    "early_stop/bad_count": float(self.bad_count),
                    "early_stop/patience": float(self.patience),
                    "early_stop/min_delta": float(self.min_delta),

                    # Helpful boolean flag for dashboards / filtering.
                    "early_stop/mode_max": 1.0 if self.mode == "max" else 0.0,
                },
                step=step,
                prefix=self.log_prefix,
            )

            # Returning False is the stop-signal to the Trainer/callback runner.
            return False

        return True
