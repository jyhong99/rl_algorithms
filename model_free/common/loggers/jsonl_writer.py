from __future__ import annotations

from typing import Dict

import os

from .base_writer import Writer
from ..utils.log_utils import (
    open_append,
    safe_call,
    json_dumps,
)


class JSONLWriter(Writer):
    """
    Lightweight JSONL (JSON Lines) writer for metric logging.

    Each call to `write()` appends exactly one JSON object as a single line:
        {"key": value, ...}

    Design goals
    ------------
    - Append-only (safe for long-running training jobs)
    - Minimal overhead (no buffering logic beyond file handle)
    - Robust to partial failures via safe_flush / safe_close
    - Compatible with common log processors (jq, pandas, Ray, etc.)

    Typical usage
    -------------
    writer = JSONLWriter(run_dir="./runs/exp1")
    writer.write({"loss": 0.12, "step": 100})
    writer.flush()
    writer.close()
    """

    def __init__(self, run_dir: str, filename: str = "metrics.jsonl") -> None:
        """
        Parameters
        ----------
        run_dir : str
            Directory where the JSONL file will be created.
        filename : str, default="metrics.jsonl"
            Log file name inside `run_dir`.

        Notes
        -----
        - The file is opened in *append* mode.
        - Parent directory must already exist.
        """
        # Absolute path to the JSONL log file
        self._path = os.path.join(run_dir, filename)

        # Open file handle in append-text mode (best-effort utility)
        self._f = open_append(self._path)

    def write(self, row: Dict[str, float]) -> None:
        """
        Append one JSON object as a single line.

        Parameters
        ----------
        row : Dict[str, float]
            Mapping of metric names to scalar values.

        Notes
        -----
        - `row` is defensively copied via dict(row).
        - Serialization is handled by `json_dumps` for consistency
          (e.g., float formatting, NaN handling if implemented there).
        """
        self._f.write(json_dumps(dict(row)) + "\n")

    def flush(self) -> None:
        """
        Flush buffered data to disk (best-effort).

        Useful for:
        - periodic durability guarantees
        - reducing data loss on unexpected termination
        """
        safe_call(self._f, "flush")

    def close(self) -> None:
        """
        Close the underlying file handle (best-effort).

        Safe to call multiple times.
        """
        safe_call(self._f, "close")
