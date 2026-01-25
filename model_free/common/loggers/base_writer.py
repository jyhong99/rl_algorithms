from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class Writer(ABC):
    """
    Abstract base class for logger writer backends.

    Contract
    --------
    - write(row): consume a dict of float metrics (may include meta keys like step/wall_time/timestamp).
    - flush(): best-effort flush of underlying buffers.
    - close(): release resources; should be idempotent (best-effort).
    """

    @abstractmethod
    def write(self, row: Dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class SafeWriter(Writer):
    """
    Exception-swallowing wrapper for any Writer.

    Notes
    -----
    - This wrapper NEVER raises exceptions from inner writer calls.
    - `name` is stored for debugging/diagnostics but not used by default.
    """

    def __init__(self, inner: Writer, *, name: Optional[str] = None) -> None:
        self._inner = inner
        self._name = name or inner.__class__.__name__

    def write(self, row: Dict[str, float]) -> None:
        try:
            self._inner.write(row)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._inner.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception:
            pass
