from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

from .base_writer import Writer
from ..utils.logger_utils import (
    _open_append, 
    _safe_call, 
    _safe_file_size,
    _read_csv_header, 
    _seek_eof, 
    _extract_meta
)


@dataclass(frozen=True)
class CSVWriterConfig:
    """
    Configuration for :class:`CSVWriter`.

    Parameters
    ----------
    run_dir : str
        Directory where CSV files will be created/appended.
    wide : bool
        Enable the "wide" CSV (one row per step; fixed schema).
    long : bool
        Enable the "long" CSV (one row per key; lossless).
    wide_filename : str
        Filename for wide CSV.
    long_filename : str
        Filename for long CSV.
    encoding : str
        Text encoding used when reading an existing wide header.
        (Writing uses the OS default from `open_append` unless it sets encoding.)
    meta_keys : Tuple[str, ...]
        Keys treated as metadata. In long format, these are replicated into each row.
        In wide format, they participate like any other column if present in schema.
    """

    run_dir: str
    wide: bool = True
    long: bool = True
    wide_filename: str = "metrics.csv"
    long_filename: str = "metrics_long.csv"
    encoding: str = "utf-8"
    meta_keys: Tuple[str, ...] = ("step", "wall_time", "timestamp")


class CSVWriter(Writer):
    """
    CSV backend writer supporting two complementary logging formats.

    This writer is optimized for long-running training jobs that may be resumed.
    It provides two append-only outputs:

    1) Wide CSV (e.g., ``metrics.csv``)
       - One row per logging call.
       - A *frozen* column schema determined once:
         * If the file is new/empty: schema is taken from the first emitted row keys.
         * If the file exists: schema is taken from the existing header row.
       - Any keys not present in the frozen schema are ignored (best-effort stability).

    2) Long CSV (e.g., ``metrics_long.csv``)
       - One row per metric key/value (lossless w.r.t. changing metric names).
       - Fixed schema: ``[step, wall_time, timestamp, key, value]``.
       - Suitable for later pivoting/aggregation.

    Parameters
    ----------
    run_dir : str
        Directory where CSV files will be created/appended.
    wide : bool, default=True
        Enable wide CSV output.
    long : bool, default=True
        Enable long CSV output.
    wide_filename : str, default="metrics.csv"
        Wide CSV filename (relative to ``run_dir``).
    long_filename : str, default="metrics_long.csv"
        Long CSV filename (relative to ``run_dir``).

    Notes
    -----
    - This writer is append-only. It does not rewrite existing rows.
    - `flush()` and `close()` are best-effort; errors are swallowed via `safe_call`.
    - Schema freezing in wide format avoids "column drift" across time and resumes.
    """

    def __init__(
        self,
        run_dir: str,
        *,
        wide: bool = True,
        long: bool = True,
        wide_filename: str = "metrics.csv",
        long_filename: str = "metrics_long.csv",
    ) -> None:
        self._cfg = CSVWriterConfig(
            run_dir=run_dir,
            wide=bool(wide),
            long=bool(long),
            wide_filename=wide_filename,
            long_filename=long_filename,
        )

        # ------------------------------
        # Paths
        # ------------------------------
        self._wide_path = os.path.join(self._cfg.run_dir, self._cfg.wide_filename)
        self._long_path = os.path.join(self._cfg.run_dir, self._cfg.long_filename)

        # ------------------------------
        # Wide CSV state
        # ------------------------------
        self._wide_file: Optional[TextIO] = _open_append(self._wide_path, newline="") if self._cfg.wide else None
        self._wide_writer: Optional[csv.DictWriter] = None
        self._wide_fieldnames: List[str] = []
        self._wide_header_ready: bool = False

        # ------------------------------
        # Long CSV state
        # ------------------------------
        self._long_file: Optional[TextIO] = _open_append(self._long_path, newline="") if self._cfg.long else None
        self._long_writer: Optional[Any] = None  # csv.writer
        self._long_header_ready: bool = False

    # ---------------------------------------------------------------------
    # Writer interface
    # ---------------------------------------------------------------------
    def write(self, row: Dict[str, float]) -> None:
        """
        Write one logical logging row to all enabled CSV formats.

        Parameters
        ----------
        row : Dict[str, float]
            Mapping of metric names to scalar floats. Common meta keys include:
            ``"step"``, ``"wall_time"``, and ``"timestamp"``. Additional keys
            (e.g., ``"loss"``, ``"reward_mean"``, ``"q1"``) are supported.

        Notes
        -----
        - Wide CSV writes exactly one output row per call (fixed columns).
        - Long CSV may write many output rows per call (one per metric key).
        """
        if self._wide_file is not None:
            self._write_wide(row)

        if self._long_file is not None:
            self._write_long(row)

    def flush(self) -> None:
        """
        Best-effort flush of underlying file buffers.

        Notes
        -----
        - This method is safe to call even if a file handle is missing/closed.
        - Exceptions are swallowed by `safe_call`.
        """
        _safe_call(self._wide_file, "flush")
        _safe_call(self._long_file, "flush")

    def close(self) -> None:
        """
        Close file handles and release writer resources.

        Notes
        -----
        - Always attempts `flush()` first.
        - Exceptions during flush/close are suppressed (best-effort).
        - After `close()`, the instance should be considered unusable.
        """
        try:
            self.flush()
        finally:
            _safe_call(self._wide_file, "close")
            _safe_call(self._long_file, "close")

            self._wide_file = None
            self._long_file = None
            self._wide_writer = None
            self._long_writer = None

    # ---------------------------------------------------------------------
    # Wide CSV (fixed schema, one row per step)
    # ---------------------------------------------------------------------
    def _write_wide(self, row: Mapping[str, float]) -> None:
        """
        Append one row to the wide CSV.

        Parameters
        ----------
        row : Mapping[str, float]
            Metrics for the current step.

        Notes
        -----
        - The schema is determined exactly once (header negotiation).
        - Keys not present in the frozen schema are ignored.
        - Missing keys are emitted as empty strings.
        """
        assert self._wide_file is not None

        if not self._wide_header_ready:
            self._prepare_wide_schema(first_row=row)

        if self._wide_writer is None:
            # Best-effort: schema preparation failed.
            return

        out = {k: row.get(k, "") for k in self._wide_fieldnames}
        self._wide_writer.writerow(out)

    def _prepare_wide_schema(self, first_row: Mapping[str, float]) -> None:
        """
        Initialize wide CSV schema and header in a resume-safe manner.

        Parameters
        ----------
        first_row : Mapping[str, float]
            The first row observed by this writer instance. Used only when the
            wide CSV file is new/empty or when header recovery is required.

        Resume-safe logic
        -----------------
        1. If wide file is empty:
           - Use ``first_row.keys()`` as the schema.
           - Write a header row immediately.
        2. If wide file is non-empty (resuming):
           - Read the existing header from disk and freeze schema to it.
           - Do not attempt to merge new keys (prevents schema drift).

        Implementation detail
        ---------------------
        A separate read-only handle is used to read the header because an append
        handle is positioned at EOF and is not reliable for reading the first line.
        """
        assert self._wide_file is not None

        size = _safe_file_size(self._wide_file)

        if size == 0:
            self._wide_fieldnames = list(first_row.keys())
            self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)
            self._wide_writer.writeheader()
            self._wide_header_ready = True
            return

        header = _read_csv_header(
            path=self._wide_path,
            encoding=self._cfg.encoding,
        )

        if header:
            self._wide_fieldnames = [h for h in header if h]
            self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)
            self._wide_header_ready = True
            _seek_eof(self._wide_file)
            return

        # Fallback: malformed file or unreadable header -> append a header best-effort.
        self._wide_fieldnames = list(first_row.keys())
        self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)
        self._wide_writer.writeheader()
        self._wide_header_ready = True
        _seek_eof(self._wide_file)

    # ---------------------------------------------------------------------
    # Long CSV (lossless, one row per key)
    # ---------------------------------------------------------------------
    def _write_long(self, row: Mapping[str, float]) -> None:
        """
        Append key/value rows to the long CSV.

        Each metric key (except meta keys) becomes one row:

        ``[step, wall_time, timestamp, key, value]``

        Parameters
        ----------
        row : Mapping[str, float]
            Metrics and optional meta information.

        Notes
        -----
        - Long schema is fixed and does not depend on which metric keys appear.
        - New metric keys can appear at any time without schema negotiation.
        """
        assert self._long_file is not None

        if not self._long_header_ready:
            self._prepare_long_schema()

        if self._long_writer is None:
            return

        meta = _extract_meta(row, self._cfg.meta_keys)
        step, wall_time, timestamp = meta

        meta_key_set = set(self._cfg.meta_keys)
        for k, v in row.items():
            if k in meta_key_set:
                continue
            self._long_writer.writerow([step, wall_time, timestamp, str(k), str(v)])

    def _prepare_long_schema(self) -> None:
        """
        Initialize long CSV writer and write header if the file is empty.

        Notes
        -----
        The long format uses a fixed schema independent of metric keys:
        ``["step", "wall_time", "timestamp", "key", "value"]``.
        """
        assert self._long_file is not None

        size = _safe_file_size(self._long_file)

        self._long_writer = csv.writer(self._long_file)
        if size == 0:
            self._long_writer.writerow(["step", "wall_time", "timestamp", "key", "value"])

        self._long_header_ready = True
        _seek_eof(self._long_file)