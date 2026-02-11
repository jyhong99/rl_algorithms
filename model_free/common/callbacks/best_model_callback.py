from __future__ import annotations

from typing import Any, Dict, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import _safe_env_step, _to_finite_float


class BestModelCallback(BaseCallback):
    """
    Save a "best" checkpoint when evaluation completes and a target metric improves.

    This callback listens to :meth:`BaseCallback.on_eval_end` and implements a
    simple model-selection policy:

    1) Read a scalar metric from the evaluation ``metrics`` dict
    2) Compare it with the best value seen so far
    3) If improved, save a checkpoint and log the new best value

    Parameters
    ----------
    metric_key:
        Key to read from the evaluation metrics dict passed to
        :meth:`on_eval_end`. Examples:

        - ``"eval/return_mean"``
        - ``"eval_return_mean"``
    save_path:
        Path passed to ``trainer.save_checkpoint(...)``. If an empty string,
        checkpoint saving is disabled (but the best value is still tracked).
    mode:
        Optimization direction for the metric:

        - ``"max"``: larger is better (e.g., return, success rate)
        - ``"min"``: smaller is better (e.g., loss, error)

    Attributes
    ----------
    best:
        Best metric value observed so far. ``None`` means "no valid evaluation
        metric has been observed yet".

    Notes
    -----
    Trainer contract (duck-typed)
    -----------------------------
    The trainer is expected to provide some subset of:

    - ``save_checkpoint(path: Optional[str] = None) -> Optional[str]`` (recommended)
    - ``global_env_step`` or another step counter compatible with ``safe_env_step``
    - ``logger.log(metrics: Dict[str, Any], step: int, prefix: str = "")`` (optional)

    Robustness policy
    -----------------
    - Missing / non-finite metrics are ignored.
    - Save/log failures are swallowed (best-effort).
    - The callback never raises to avoid interrupting training.

    Typical usage
    -------------
    The trainer calls:

    >>> cb = BestModelCallback(metric_key="eval_return_mean", save_path="best.pt", mode="max")
    >>> cb.on_eval_end(trainer, eval_metrics)
    """

    def __init__(
        self,
        metric_key: str = "eval_return_mean",
        save_path: str = "best.pt",
        *,
        mode: str = "max",
    ) -> None:
        self.metric_key = str(metric_key)
        self.save_path = str(save_path)

        mode_norm = str(mode).lower().strip()
        if mode_norm not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got: {mode}")
        self.mode = mode_norm

        # Best metric so far; None means "no best yet".
        self.best: Optional[float] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _is_better(self, value: float, best: Optional[float]) -> bool:
        """
        Decide whether ``value`` should replace ``best``.

        Parameters
        ----------
        value:
            Candidate metric value from the latest evaluation (finite float).
        best:
            Current best value (or None if no best yet).

        Returns
        -------
        bool
            True if ``value`` improves upon ``best`` according to ``self.mode``.
        """
        if best is None:
            # First valid metric always becomes the best.
            return True

        if self.mode == "max":
            return value > best

        # self.mode == "min"
        return value < best

    def _save_checkpoint(self, trainer: Any) -> None:
        """
        Best-effort checkpoint saving.

        The method attempts to call ``trainer.save_checkpoint``. Since trainer
        implementations may expose different signatures, we try:

        1) Keyword call: ``save_checkpoint(path=...)``
        2) Positional call: ``save_checkpoint("...")``

        If ``save_path`` is empty, saving is disabled and this method is a no-op.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Must provide a callable attribute
            ``save_checkpoint`` for saving to occur.

        Notes
        -----
        Any exceptions are swallowed by design.
        """
        if not self.save_path:
            return

        save_fn = getattr(trainer, "save_checkpoint", None)
        if not callable(save_fn):
            return

        # Prefer keyword argument; more stable if the trainer has multiple params.
        try:
            save_fn(path=self.save_path)
            return
        except TypeError:
            # Signature mismatch -> fall back to positional call.
            pass
        except Exception:
            return

        try:
            save_fn(self.save_path)
        except Exception:
            return

    def _log_best(self, trainer: Any, best_value: float) -> None:
        """
        Best-effort logging of the updated best metric.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Used to fetch a robust step counter and
            to access the logger.
        best_value:
            Best metric value to report.

        Notes
        -----
        - The metric is logged under a stable key:

          ``best/<metric_key>``

          Example: ``best/eval_return_mean = 123.4``

        - Step is retrieved via :func:`safe_env_step` to tolerate differing trainer
          attribute names.
        - Uses :meth:`BaseCallback.log` which swallows logger errors.
        """
        step = _safe_env_step(trainer)
        payload = {f"best/{self.metric_key}": float(best_value)}
        self.log(trainer, payload, step=step, prefix="")

    # -------------------------------------------------------------------------
    # Hook: evaluation end
    # -------------------------------------------------------------------------
    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Called when an evaluation phase ends.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed) potentially providing:
            - ``save_checkpoint(...)``
            - step counters used by :func:`safe_env_step`
            - ``logger`` used by :meth:`BaseCallback.log`
        metrics:
            Evaluation metrics dict. The value associated with ``self.metric_key``
            is interpreted as the selection criterion.

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        - If ``metrics`` is not a dict, the callback ignores the call.
        - The metric value is converted to a finite float using
          :func:`to_finite_float`. Non-numeric/NaN/inf values are ignored.
        """
        if not isinstance(metrics, dict):
            return True

        raw = metrics.get(self.metric_key, None)
        val = _to_finite_float(raw)
        if val is None:
            return True

        if self._is_better(val, self.best):
            self.best = val
            self._save_checkpoint(trainer)
            self._log_best(trainer, best_value=val)

        return True
