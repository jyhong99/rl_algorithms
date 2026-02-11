from __future__ import annotations

from typing import Dict

from .base_writer import Writer
from ..utils.logger_utils import _get_step, _split_meta

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


class TensorBoardWriter(Writer):
    """
    TensorBoard writer backend for scalar metrics.

    This writer consumes a flat metric row (typically produced by `Logger.log`)
    and emits each metric as a TensorBoard scalar via :meth:`SummaryWriter.add_scalar`.

    The input row may include metadata fields (e.g., ``step``, ``wall_time``,
    ``timestamp``). These are separated from metric keys using `_split_meta`,
    and the step is determined using `_get_step` to remain consistent across
    backends (CSV/JSONL/TensorBoard).

    Parameters
    ----------
    run_dir : str
        Directory where TensorBoard event files will be written.

    Attributes
    ----------
    _tb : SummaryWriter
        Underlying TensorBoard summary writer.

    Raises
    ------
    RuntimeError
        If TensorBoard support is not available (i.e., `torch.utils.tensorboard`
        cannot be imported).

    Notes
    -----
    - This class is intentionally minimal and does not implement custom naming
      policies beyond `str(key)` conversion.
    - Exceptions from `flush()` and `close()` are swallowed to avoid interrupting
      training loops. (Write failures are expected to be handled by the outer
      `Logger` or `SafeWriter` depending on your architecture.)
    """

    def __init__(self, run_dir: str) -> None:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard is not available (torch.utils.tensorboard missing).")
        self._tb = SummaryWriter(log_dir=run_dir)

    def write(self, row: Dict[str, float]) -> None:
        """
        Write one metric row to TensorBoard.

        Parameters
        ----------
        row : Dict[str, float]
            Flat mapping of keys to scalar floats. May include meta keys such as
            ``step``, ``wall_time``, and ``timestamp``.

        Notes
        -----
        The write procedure is:
        1. Infer the TensorBoard `global_step` using `_get_step(row)`.
        2. Split meta keys from metric keys using `_split_meta(row)`.
        3. Emit each metric as a scalar:
           ``add_scalar(tag=str(key), scalar_value=float(value), global_step=step)``.

        Raises
        ------
        Exception
            Propagates exceptions thrown by TensorBoard `add_scalar`.
            (Callers typically wrap this writer with `SafeWriter` or rely on
            `Logger` to handle exceptions.)
        """
        step = _get_step(row)
        _, metrics = _split_meta(row)

        for k, v in metrics.items():
            self._tb.add_scalar(str(k), float(v), global_step=int(step))

    def flush(self) -> None:
        """
        Best-effort flush of pending TensorBoard events to disk.

        Notes
        -----
        - This method suppresses all exceptions to avoid disrupting training.
        - Flushing improves durability for long-running jobs but does not strictly
          guarantee persistence to physical storage (OS-dependent).
        """
        try:
            self._tb.flush()
        except Exception:
            pass

    def close(self) -> None:
        """
        Best-effort close of the underlying TensorBoard writer.

        Notes
        -----
        - Safe to call multiple times.
        - Exceptions are suppressed.
        """
        try:
            self._tb.close()
        except Exception:
            pass
