from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Mapping


class Writer(ABC):
    """
    Abstract base class for metric writer backends.

    This interface defines a minimal, side-effecting sink for scalar metrics
    produced during training or evaluation loops (e.g., RL, optimization).

    The writer is intentionally *stateless from the caller's perspective*:
    it consumes rows of key–value pairs and manages its own buffering,
    serialization, and persistence strategy.

    Contract
    --------
    - `write(row)` must accept a mapping of metric names to scalar floats.
    - Writers MAY support special meta-keys (e.g., "step", "wall_time"),
      but this is writer-specific and not enforced at the interface level.
    - `flush()` and `close()` should be best-effort and idempotent.

    Notes
    -----
    - Implementations are expected to be *side-effecting* (I/O, IPC, etc.).
    - Implementations SHOULD raise exceptions on failure; suppression is
      handled explicitly by wrappers such as `SafeWriter`.
    """

    @abstractmethod
    def write(self, row: Mapping[str, float]) -> None:
        """
        Consume a single row of scalar metrics.

        Parameters
        ----------
        row : Mapping[str, float]
            Dictionary-like object mapping metric names to scalar values.
            Values are expected to be finite floats; validation is implementation-defined.

        Raises
        ------
        Exception
            Implementations may raise on I/O failure, serialization errors,
            or invalid input.
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """
        Flush any internal buffers to the underlying sink.

        Notes
        -----
        - This method should be safe to call multiple times.
        - No guarantees are made about durability beyond the implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release any resources held by the writer.

        Notes
        -----
        - This method should be idempotent.
        - After `close()`, subsequent calls to `write()` are undefined behavior
          unless explicitly supported by the implementation.
        """
        raise NotImplementedError


class SafeWriter(Writer):
    """
    Exception-swallowing wrapper for a :class:`Writer`.

    This class enforces *failure isolation*: failures in the underlying writer
    MUST NOT propagate to the training loop or caller. All exceptions raised
    by the wrapped writer are silently suppressed.

    Typical use cases include:
    - Best-effort logging (CSV, TensorBoard, remote logging)
    - Non-critical diagnostics
    - Long-running training where logging failures must not interrupt progress

    Parameters
    ----------
    inner : Writer
        The concrete writer instance to wrap.
    name : str, optional
        Human-readable identifier for diagnostics or debugging.
        Defaults to ``inner.__class__.__name__``.

    Notes
    -----
    - This wrapper intentionally discards all exceptions.
    - If diagnostics are required, extend this class to record failure counts
      or last-seen exceptions.
    - This class does NOT attempt retries or recovery.
    """

    def __init__(self, inner: Writer, *, name: Optional[str] = None) -> None:
        self._inner = inner
        self._name = name or inner.__class__.__name__

    def write(self, row: Mapping[str, float]) -> None:
        """
        Forward a metric row to the inner writer, suppressing all exceptions.
        """
        try:
            self._inner.write(row)
        except Exception:
            pass

    def flush(self) -> None:
        """
        Flush the inner writer, suppressing all exceptions.
        """
        try:
            self._inner.flush()
        except Exception:
            pass

    def close(self) -> None:
        """
        Close the inner writer, suppressing all exceptions.
        """
        try:
            self._inner.close()
        except Exception:
            pass
