from __future__ import annotations

from typing import Dict

from .base_writer import Writer
from ..utils.log_utils import split_meta, get_step

# TensorBoard is optional; keep import best-effort so the package
# can be used without torch / tensorboard installed.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class TensorBoardWriter(Writer):
    """
    TensorBoard writer backend.

    This writer consumes a flat metric row (Dict[str, float]) and logs
    scalar values to TensorBoard using `add_scalar`.

    Design notes
    ------------
    - Assumes the input row may contain *meta fields* (e.g., step, time),
      which are separated via `split_meta`.
    - Uses `get_step(row)` to determine the global_step consistently with
      other writers/sinks.
    - Best-effort flush/close to avoid breaking training loops.
    """

    def __init__(self, run_dir: str) -> None:
        """
        Parameters
        ----------
        run_dir : str
            Directory where TensorBoard event files will be written.

        Raises
        ------
        RuntimeError
            If torch.utils.tensorboard is not available.
        """
        if SummaryWriter is None:
            raise RuntimeError(
                "TensorBoard is not available (torch.utils.tensorboard missing)."
            )

        # Create TensorBoard SummaryWriter bound to run directory
        self._tb = SummaryWriter(log_dir=run_dir)

    def write(self, row: Dict[str, float]) -> None:
        """
        Write a single metric row to TensorBoard.

        Parameters
        ----------
        row : Dict[str, float]
            Metric mapping. May include meta keys (e.g., step).

        Behavior
        --------
        - Extracts global_step via `get_step(row)`
        - Splits meta vs. metric values using `split_meta`
        - Logs each metric as a scalar
        """
        # Determine the global step for TensorBoard
        step = get_step(row)

        # Separate metadata (e.g., step/time) from actual scalar metrics
        _, metrics = split_meta(row)

        # Log each metric independently
        for k, v in metrics.items():
            self._tb.add_scalar(str(k), float(v), global_step=step)

    def flush(self) -> None:
        """
        Flush pending TensorBoard events to disk (best-effort).

        Useful for:
        - long-running jobs
        - reducing data loss on crashes
        """
        try:
            self._tb.flush()
        except Exception:
            # Never allow logging failures to interrupt training
            pass

    def close(self) -> None:
        """
        Close the underlying TensorBoard writer (best-effort).

        Safe to call multiple times.
        """
        try:
            self._tb.close()
        except Exception:
            pass
