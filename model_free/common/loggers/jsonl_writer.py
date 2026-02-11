from __future__ import annotations

import os
from typing import Dict, TextIO

from .base_writer import Writer
from ..utils.logger_utils import (
    _open_append,
    _safe_call,
    _json_dumps,
)


class JSONLWriter(Writer):
    """
    JSON Lines (JSONL) writer for scalar metric logging.

    This writer appends **exactly one JSON object per `write()` call**,
    serialized onto a single line (newline-delimited JSON):

    ``{"key": value, ...}\\n``

    The JSONL format is convenient because it is:
    - Append-friendly (streaming / tailing / resume-safe)
    - Easy to parse with common tooling (e.g., pandas, jq)
    - Robust to schema changes over time (new keys can appear at any step)

    Parameters
    ----------
    run_dir : str
        Directory where the JSONL file will be created/appended.
    filename : str, default="metrics.jsonl"
        JSONL filename inside ``run_dir``.

    Attributes
    ----------
    _path : str
        Absolute path to the JSONL file.
    _f : TextIO
        Open file handle in append mode.

    Notes
    -----
    - The file is opened in append-text mode.
    - Parent directory is expected to exist. If you want auto-create behavior,
      ensure `run_dir` exists before constructing this writer.
    - Serialization is delegated to ``_json_dumps`` to centralize policies such as:
      float formatting, NumPy scalar handling, and NaN/Inf handling (if supported).
    """

    def __init__(self, run_dir: str, filename: str = "metrics.jsonl") -> None:
        self._path = os.path.join(run_dir, filename)
        self._f: TextIO = _open_append(self._path)

    def write(self, row: Dict[str, float]) -> None:
        """
        Append one metric row as a single JSON object line.

        Parameters
        ----------
        row : Dict[str, float]
            Mapping from metric names to scalar float values. Common examples:
            ``{"step": 1000, "loss": 0.23, "reward_mean": 1.5}``.

        Notes
        -----
        - The input is defensively copied via ``dict(row)`` to decouple from
          caller-side mutations.
        - Exactly one newline is appended per call, producing strict JSONL.
        - This method performs no explicit flushing; call `flush()` if you need
          stronger durability guarantees.
        """
        self._f.write(_json_dumps(dict(row)) + "\n")

    def flush(self) -> None:
        """
        Best-effort flush of the underlying file buffer.

        This is useful when you want to reduce the amount of log data lost if the
        process crashes or is preempted.

        Notes
        -----
        - Exceptions are swallowed by ``_safe_call``.
        - Flushing does not guarantee durability on all systems unless the OS
          also syncs buffers to storage (implementation/system-dependent).
        """
        _safe_call(self._f, "flush")

    def close(self) -> None:
        """
        Close the underlying file handle (best-effort).

        Notes
        -----
        - Safe to call multiple times.
        - After closing, further writes are undefined behavior.
        - Exceptions are swallowed by ``_safe_call``.
        """
        _safe_call(self._f, "close")
        # Optional: help prevent accidental reuse after close
        # (kept minimal; remove if you intentionally want reuse)
        # self._f = None  # type: ignore[assignment]
