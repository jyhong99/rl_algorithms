from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import _coerce_scalar_mapping


class RayReportCallback(BaseCallback):
    """
    Report metrics to Ray Tune / Ray AIR (best-effort).

    This callback bridges a framework-agnostic Trainer loop to Ray's reporting APIs.
    When running trials under Ray Tune/AIR, you must call Ray's reporting function so
    the driver can:

    - record progress curves and scalar results
    - drive schedulers (e.g., ASHA/PBT) and early stopping decisions
    - select best trials and manage checkpoint policies

    Reporting hooks
    ---------------
    - ``on_update(trainer, metrics)``:
        Reports update-time metrics (typically "train/*") if enabled.
    - ``on_eval_end(trainer, metrics)``:
        Reports evaluation metrics (typically "eval/*") if enabled.

    Ray backends (preference order)
    -------------------------------
    1) ``ray.air.session.report`` (Ray AIR / newer API)
    2) ``ray.tune.report``        (legacy Tune API)

    If Ray is not importable or no reporting API is available, the callback becomes a no-op.

    Parameters
    ----------
    report_on_update:
        If True, report metrics provided to ``on_update``.
    report_on_eval:
        If True, report metrics provided to ``on_eval_end``.
    prefix_update:
        Prefix applied to update metrics keys. Typical: "train".
        Example: {"loss": 1.2} -> {"train/loss": 1.2}
    prefix_eval:
        Prefix applied to eval metrics keys. Typical: "eval".
        Example: {"return_mean": 100} -> {"eval/return_mean": 100}
    include_steps:
        If True, attach global step counters (if present on trainer) into the payload:
          - "sys/global_env_step"
          - "sys/global_update_step"
        This helps Ray align progress curves and scheduler decisions.
    drop_non_scalars:
        If True, keep only scalar-like values (and stringify keys) using
        ``coerce_scalar_mapping``. This reduces the chance of Ray reporting failures
        due to non-serializable objects.

    Notes
    -----
    Trainer contract (duck-typed)
    -----------------------------
    This callback may read:
      - trainer.global_env_step (optional)
      - trainer.global_update_step (optional)

    It does *not* require a concrete Trainer implementation.

    Robustness policy
    -----------------
    - Never raises exceptions (best-effort reporting must not crash training).
    - If metrics are missing/empty or coercion produces an empty mapping, it does nothing.
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
        self.report_on_update = bool(report_on_update)
        self.report_on_eval = bool(report_on_eval)
        self.prefix_update = str(prefix_update)
        self.prefix_eval = str(prefix_eval)
        self.include_steps = bool(include_steps)
        self.drop_non_scalars = bool(drop_non_scalars)

        # Runtime state for Ray reporting.
        self._ray_available: bool = False
        self._report_fn: Optional[Any] = None  # session.report or tune.report

        # Initialize Ray reporting backend immediately (best-effort).
        self._try_init_ray()

    # =========================================================================
    # Internal helpers
    # =========================================================================
    @staticmethod
    def _maybe_add_global_steps(trainer: Any, payload: Dict[str, Any]) -> None:
        """
        Attach global step counters to the payload if present on the trainer.

        Parameters
        ----------
        trainer:
            Trainer-like object that may expose step counters.
        payload:
            Mutable dict to be augmented in-place.

        Notes
        -----
        - Uses ``setdefault`` so user-provided keys are not overwritten.
        - Values are cast to float for consistency with typical Ray scalar sinks.
        - All attribute-access errors are swallowed (best-effort).
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
        Add a prefix to metric keys in "prefix/key" format.

        Parameters
        ----------
        metrics:
            Input metrics mapping.
        prefix:
            Prefix string. If empty/falsey, keys are not prefixed.

        Returns
        -------
        Dict[str, Any]
            New dict with stringified keys and optional "prefix/" prepended.

        Notes
        -----
        - Ensures the prefix ends with "/" if non-empty.
        - Always stringifies keys to avoid downstream issues with non-string keys.

        Examples
        --------
        >>> RayReportCallback._add_prefix({"loss": 1.0}, "train")
        {'train/loss': 1.0}

        >>> RayReportCallback._add_prefix({"loss": 1.0}, "train/")
        {'train/loss': 1.0}

        >>> RayReportCallback._add_prefix({"loss": 1.0}, "")
        {'loss': 1.0}
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
        Detect a Ray reporting backend and store the chosen report function.

        Preference order
        ----------------
        1) ``ray.air.session.report``
        2) ``ray.tune.report``

        If neither is importable, reporting is disabled and this callback becomes a no-op.
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

        Parameters
        ----------
        metrics:
            Raw metrics mapping.

        Returns
        -------
        Dict[str, Any]
            A dict with string keys that is intended to be safe for Ray reporting.

        Notes
        -----
        - If ``drop_non_scalars=True``, the output contains scalar-like values only,
          as decided by ``coerce_scalar_mapping``.
        - If ``drop_non_scalars=False``, values are passed through unchanged, which
          may still fail at report-time if values are not serializable.
        """
        if not self.drop_non_scalars:
            return {str(k): v for k, v in metrics.items()}
        return _coerce_scalar_mapping(metrics)

    def _report(self, payload: Dict[str, Any]) -> None:
        """
        Report payload to Ray (best-effort).

        Parameters
        ----------
        payload:
            Final payload dict to be reported.

        Notes
        -----
        - No-op if Ray is unavailable or payload is empty.
        - Swallows all exceptions so reporting can never crash training.
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

        Parameters
        ----------
        trainer:
            Trainer-like object (duck-typed). Used for optional step counters.
        metrics:
            Update metrics dictionary (typically produced after one optimizer update).

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Flow:
          1) Check feature flag and that metrics exist
          2) Coerce metrics (optionally drop non-scalars)
          3) Add prefix (e.g., "train/*")
          4) Optionally attach global step counters
          5) Report to Ray
        """
        if not self.report_on_update or not metrics:
            return True

        coerced = self._coerce_metrics(metrics)
        if not coerced:
            return True

        payload = self._add_prefix(coerced, self.prefix_update)

        if self.include_steps:
            self._maybe_add_global_steps(trainer, payload)

        self._report(payload)
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Report evaluation-time metrics to Ray.

        Parameters
        ----------
        trainer:
            Trainer-like object (duck-typed). Used for optional step counters.
        metrics:
            Evaluation metrics dictionary.

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Same policy as ``on_update`` but uses ``prefix_eval`` (e.g., "eval/*").
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
