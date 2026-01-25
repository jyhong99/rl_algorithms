from __future__ import annotations

from typing import Any, Dict, Optional, Mapping

from .base_callback import BaseCallback
from ..utils.callback_utils import coerce_scalar_mapping


class RayReportCallback(BaseCallback):
    """
    Report training/evaluation metrics to Ray Tune (Ray AIR session or legacy tune).

    Motivation
    ----------
    When running experiments with Ray Tune/AIR, metrics must be reported through Ray's
    reporting API so the tuner can:
      - record results (progress, scores, losses)
      - drive schedulers (ASHA/PBT) and early stopping
      - select best trials / checkpoint policies

    Behavior
    --------
    - on_update(): reports update metrics (default prefix: "train/...")
    - on_eval_end(): reports evaluation metrics (default prefix: "eval/...")
    - Optionally adds global step counters (env/update) if present on trainer.
    - Optionally drops non-scalar values (Ray sinks usually expect scalars).

    Ray backends (preference order)
    -------------------------------
    1) ray.air.session.report   (Ray AIR / newer API)
    2) ray.tune.report          (legacy Tune API)

    Parameters
    ----------
    report_on_update : bool, default=True
        Whether to report metrics in on_update().
    report_on_eval : bool, default=True
        Whether to report metrics in on_eval_end().
    prefix_update : str, default="train"
        Prefix applied to update metrics.
        Example: {"loss": 1.2} -> {"train/loss": 1.2}
    prefix_eval : str, default="eval"
        Prefix applied to evaluation metrics.
        Example: {"return_mean": 100} -> {"eval/return_mean": 100}
    include_steps : bool, default=True
        If True, include additional counters if present:
          - sys/global_env_step
          - sys/global_update_step
        These are helpful for aligning curves and scheduler decisions.
    drop_non_scalars : bool, default=True
        If True, keep only scalar-like values using `coerce_scalar_mapping()`.
        This prevents Ray report failures due to non-serializable objects.
    """

    def __init__(
        self,
        *,
        report_on_update: bool = True,
        report_on_eval: bool = True,
        prefix_update: str = "train",
        prefix_eval: str = "eval",
        include_steps: bool = True,
        drop_non_scalars: bool = True,
    ) -> None:
        # User configuration
        self.report_on_update = bool(report_on_update)
        self.report_on_eval = bool(report_on_eval)
        self.prefix_update = str(prefix_update)
        self.prefix_eval = str(prefix_eval)
        self.include_steps = bool(include_steps)
        self.drop_non_scalars = bool(drop_non_scalars)

        # Ray runtime state:
        # - _ray_available indicates whether importing/reporting is possible
        # - _report_fn is the chosen reporting function (session.report or tune.report)
        self._ray_available: bool = False
        self._report_fn: Any = None  # session.report or tune.report

        # Initialize Ray reporting backend immediately (best-effort).
        self._try_init_ray()

    # =========================================================================
    # Internal helpers
    # =========================================================================
    @staticmethod
    def _maybe_add_global_steps(trainer: Any, payload: Dict[str, Any]) -> None:
        """
        Add global step counters if present on trainer.

        Keys are intentionally explicit and sink-agnostic:
          - sys/global_env_step
          - sys/global_update_step

        Notes
        -----
        - Uses setdefault to avoid overwriting user-provided keys.
        - Any attribute access failure is silently ignored.
        """
        try:
            payload.setdefault("sys/global_env_step", float(getattr(trainer, "global_env_step", 0)))
        except Exception:
            pass
        try:
            payload.setdefault("sys/global_update_step", float(getattr(trainer, "global_update_step", 0)))
        except Exception:
            pass

    @staticmethod
    def _add_prefix(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
        """
        Add a string prefix to metric keys in "train/loss" style.

        Rules
        -----
        - If prefix is empty/None -> do not modify keys.
        - Ensures prefix ends with "/".
        - Always stringifies keys to avoid non-string key issues downstream.

        Examples
        --------
        prefix="train", metrics={"loss": 1.0}  -> {"train/loss": 1.0}
        prefix="train/", metrics={"loss": 1.0} -> {"train/loss": 1.0}
        prefix="", metrics={"loss": 1.0}       -> {"loss": 1.0}
        """
        p = str(prefix) if prefix else ""
        if not p:
            return {str(k): v for k, v in metrics.items()}
        if not p.endswith("/"):
            p = p + "/"
        return {f"{p}{str(k)}": v for k, v in metrics.items()}

    # =========================================================================
    # Ray init (callback-local)
    # =========================================================================
    def _try_init_ray(self) -> None:
        """
        Detect Ray reporting API and store a reporting function.

        Preference order:
        - ray.air.session.report (newer)
        - ray.tune.report        (legacy)

        If neither import succeeds, reporting is disabled (no-op).
        """
        # Preferred: Ray AIR session.report
        try:
            from ray.air import session  # type: ignore

            self._report_fn = session.report
            self._ray_available = True
            return
        except Exception:
            pass

        # Fallback: legacy tune.report
        try:
            from ray import tune  # type: ignore

            self._report_fn = tune.report
            self._ray_available = True
            return
        except Exception:
            self._ray_available = False
            self._report_fn = None

    # =========================================================================
    # Metric normalization + report
    # =========================================================================
    def _coerce_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Normalize metrics into a Ray-friendly mapping.

        If drop_non_scalars=True:
          - Keep only scalar-like values
          - Convert keys to strings
          - Filter out invalid values (implementation-dependent in coerce_scalar_mapping)

        If drop_non_scalars=False:
          - Keep values as-is (stringify keys only)
          - May still fail in Ray if values are not serializable
        """
        if not self.drop_non_scalars:
            return {str(k): v for k, v in metrics.items()}
        return coerce_scalar_mapping(metrics)

    def _report(self, payload: Dict[str, Any]) -> None:
        """
        Best-effort reporting to Ray.

        Notes
        -----
        - Never raises exceptions (callback must not crash training).
        - No-op if Ray is not available or payload is empty.
        """
        if not self._ray_available or self._report_fn is None or not payload:
            return
        try:
            self._report_fn(payload)
        except Exception:
            return

    # =========================================================================
    # Hooks
    # =========================================================================
    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Report update-time metrics to Ray.

        Flow
        ----
        1) Check feature flag + metrics exist
        2) Coerce metrics (optionally drop non-scalars)
        3) Add prefix (train/...)
        4) Optionally add global steps
        5) Report to Ray
        """
        if not self.report_on_update or not metrics:
            return True

        coerced = self._coerce_metrics(metrics)
        if not coerced:
            return True

        payload = self._add_prefix(coerced, self.prefix_update)

        # Add global counters so Ray dashboards/schedulers can align progress.
        if self.include_steps:
            self._maybe_add_global_steps(trainer, payload)

        self._report(payload)
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Report evaluation-time metrics to Ray.

        Same policy as on_update(), but uses prefix_eval (eval/...).
        """
        if not self.report_on_eval or not metrics:
            return True

        coerced = self._coerce_metrics(metrics)
        if not coerced:
            return True

        payload = self._add_prefix(coerced, self.prefix_eval)

        if self.include_steps:
            self._maybe_add_global_steps(trainer, payload)

        self._report(payload)
        return True