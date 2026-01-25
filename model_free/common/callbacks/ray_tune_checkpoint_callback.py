from __future__ import annotations

from typing import Any, Dict, Optional

from .base_callback import BaseCallback


class RayTuneCheckpointCallback(BaseCallback):
    """
    Bridge Trainer-created checkpoints to Ray Tune / Ray AIR (best-effort).

    Design principles
    -----------------
    - No persistent Ray context or handles.
    - Safe to import/use even when Ray is not installed.
    - On each checkpoint event:
        1) Try Ray AIR (ray.air.session.report)
        2) Fallback to legacy Ray Tune (tune.report)
        3) Otherwise: no-op

    Trainer contract
    ----------------
    on_checkpoint(self, trainer, path: str) is called with:
      - path: directory path of the saved checkpoint
    """

    def __init__(self, *, report_empty_metrics: bool = True) -> None:
        self.report_empty_metrics = bool(report_empty_metrics)

    # ------------------------------------------------------------------
    # Internal helpers (all best-effort, no persistent state)
    # ------------------------------------------------------------------
    @staticmethod
    def _checkpoint_from_directory(path: str) -> Optional[Any]:
        """
        Best-effort: create ray.air.Checkpoint from directory path.
        Returns None if Ray is unavailable or path is invalid.
        """
        if not isinstance(path, str) or not path:
            return None
        try:
            from ray.air import Checkpoint  # type: ignore
            return Checkpoint.from_directory(path)
        except Exception:
            return None

    @staticmethod
    def _report_air(metrics: Dict[str, Any], checkpoint: Any) -> bool:
        """
        Try Ray AIR reporting.
        Returns True if successful.
        """
        try:
            from ray.air import session  # type: ignore
            session.report(metrics, checkpoint=checkpoint)
            return True
        except Exception:
            return False

    @staticmethod
    def _report_legacy(metrics: Dict[str, Any], checkpoint: Any) -> bool:
        """
        Try legacy Ray Tune reporting.
        Returns True if successful.
        """
        try:
            from ray import tune  # type: ignore
            # Legacy Tune supports passing checkpoint as kwarg
            tune.report(**metrics, checkpoint=checkpoint)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------
    def on_checkpoint(self, trainer: Any, path: str) -> bool:
        """
        Called by Trainer after a checkpoint is saved.

        Behavior
        --------
        - Converts checkpoint directory -> Ray AIR Checkpoint (if possible)
        - Reports to Ray AIR first, then legacy Tune
        - Always returns True (never interrupts training)
        """
        ckpt = self._checkpoint_from_directory(path)
        if ckpt is None:
            return True

        metrics: Dict[str, Any] = {} if self.report_empty_metrics else {}

        # Prefer Ray AIR
        if self._report_air(metrics, ckpt):
            return True

        # Fallback: legacy Tune
        self._report_legacy(metrics, ckpt)
        return True