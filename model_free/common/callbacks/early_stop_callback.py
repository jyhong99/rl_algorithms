from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import _safe_env_step, _to_finite_float


class EarlyStopCallback(BaseCallback):
    """
    Early stopping based on evaluation metric stagnation.

    This callback monitors a scalar evaluation metric reported via
    :meth:`BaseCallback.on_eval_end`. If the metric does not improve for
    ``patience`` consecutive evaluation events, the callback requests the trainer
    to stop training by returning ``False``.

    Parameters
    ----------
    metric_key:
        Key to read from the evaluation ``metrics`` dict passed to
        :meth:`on_eval_end`. Examples:

        - ``"eval_return_mean"``
        - ``"eval/return_mean"``
    patience:
        Number of consecutive "non-improving" evaluation events tolerated before
        stopping. Must be ``>= 1``.
    min_delta:
        Minimum required improvement magnitude. Must be ``>= 0``.

        Improvement rules:

        - If ``mode="max"``: improvement if ``val > best + min_delta``
        - If ``mode="min"``: improvement if ``val < best - min_delta``
    mode:
        Direction of improvement:

        - ``"max"``: higher is better (return, accuracy, success rate)
        - ``"min"``: lower is better (loss, error)
    log_prefix:
        Prefix passed to the logger so early-stop metrics are grouped under a
        namespace (e.g., ``"sys/"`` logs ``"sys/early_stop/best"``).

    Attributes
    ----------
    best:
        Best metric value observed so far. ``None`` until the first valid metric
        is encountered.
    bad_count:
        Number of consecutive evaluation events without improvement.
    last:
        Most recent valid metric value observed (useful for debugging).

    Behavior summary
    ----------------
    - Missing / invalid / non-finite metrics are ignored (no state change).
    - First valid eval initializes ``best`` and resets ``bad_count``.
    - Improvement updates ``best`` and resets ``bad_count``.
    - No improvement increments ``bad_count``.
    - If ``bad_count >= patience``, returns ``False`` to request early stop.

    Notes
    -----
    - Logging uses :func:`safe_env_step` for a best-effort step value suitable for
      dashboard alignment.
    - This callback never raises during normal operation; it may raise on invalid
      constructor arguments.
    """

    def __init__(
        self,
        metric_key: str = "eval_return_mean",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["max", "min"] = "max",
        *,
        log_prefix: str = "sys/",
    ) -> None:
        self.metric_key = str(metric_key)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode: Literal["max", "min"] = mode
        self.log_prefix = str(log_prefix)

        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode}")

        self.best: Optional[float] = None
        self.bad_count: int = 0
        self.last: Optional[float] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _is_improved(self, val: float, best: float) -> bool:
        """
        Determine whether ``val`` improves upon ``best`` according to the configured rule.

        Parameters
        ----------
        val:
            Candidate metric value from the latest evaluation (finite float).
        best:
            Current best metric value (finite float).

        Returns
        -------
        bool
            True if ``val`` is considered an improvement; otherwise False.
        """
        if self.mode == "max":
            return val > (best + self.min_delta)
        # self.mode == "min"
        return val < (best - self.min_delta)

    # -------------------------------------------------------------------------
    # Hook: evaluation end
    # -------------------------------------------------------------------------
    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Process evaluation metrics and decide whether to early stop.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Used for retrieving a step counter
            (via :func:`safe_env_step`) and for logging via :meth:`BaseCallback.log`.
        metrics:
            Evaluation metrics dictionary. Must contain ``metric_key`` for this
            callback to operate.

        Returns
        -------
        bool
            Control signal for the training loop:

            - True: continue training
            - False: request early stop
        """
        if not isinstance(metrics, dict):
            return True

        raw = metrics.get(self.metric_key, None)
        val = _to_finite_float(raw)
        if val is None:
            return True

        self.last = val
        step = _safe_env_step(trainer)

        # --------------------------------------------------------------
        # Case 1) First valid evaluation => initialize state
        # --------------------------------------------------------------
        if self.best is None:
            self.best = val
            self.bad_count = 0
            self.log(
                trainer,
                {
                    "early_stop/init": 1.0,
                    "early_stop/best": float(self.best),
                    "early_stop/last": float(val),
                    "early_stop/bad_count": float(self.bad_count),
                    "early_stop/patience": float(self.patience),
                    "early_stop/min_delta": float(self.min_delta),
                    "early_stop/mode_max": 1.0 if self.mode == "max" else 0.0,
                },
                step=step,
                prefix=self.log_prefix,
            )
            return True

        assert self.best is not None  # for type checkers

        # --------------------------------------------------------------
        # Case 2) Improvement => update best and reset bad_count
        # --------------------------------------------------------------
        if self._is_improved(val, self.best):
            self.best = val
            self.bad_count = 0
            self.log(
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

        # --------------------------------------------------------------
        # Case 3) No improvement => increment bad_count
        # --------------------------------------------------------------
        self.bad_count += 1
        self.log(
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

        # --------------------------------------------------------------
        # Case 4) Patience exceeded => trigger early stop
        # --------------------------------------------------------------
        if self.bad_count >= self.patience:
            self.log(
                trainer,
                {
                    "early_stop/triggered": 1.0,
                    "early_stop/best": float(self.best),
                    "early_stop/last": float(val),
                    "early_stop/bad_count": float(self.bad_count),
                    "early_stop/patience": float(self.patience),
                    "early_stop/min_delta": float(self.min_delta),
                    "early_stop/mode_max": 1.0 if self.mode == "max" else 0.0,
                },
                step=step,
                prefix=self.log_prefix,
            )
            return False

        return True
