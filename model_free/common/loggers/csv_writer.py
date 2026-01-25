from __future__ import annotations

from typing import Any, Dict, List, Optional, TextIO
import csv
import os

from .base_writer import Writer
from ..utils.log_utils import open_append, safe_call


class CSVWriter(Writer):
    """
    CSV backend writer supporting two complementary logging formats.

    Formats
    -------
    1) Wide CSV (metrics.csv)
       - One row per logging step.
       - Columns are fixed after initialization (schema is frozen).
       - Extra keys that appear later are ignored (best-effort / stable schema).

    2) Long CSV (metrics_long.csv)
       - One row per (step, key).
       - Lossless: can store arbitrary new keys without schema changes.
       - Schema: [step, wall_time, timestamp, key, value]

    Why two formats?
    ---------------
    - Wide CSV is convenient for quick spreadsheet viewing and plotting.
      However, it cannot easily handle dynamic metric keys over time.
    - Long CSV is robust and lossless, and can be pivoted later if needed.

    Resume behavior
    ---------------
    - Wide CSV:
      * If file is empty, create header from the first emitted row.
      * If file exists, read header from the first line and keep it fixed.
    - Long CSV:
      * If file is empty, write the fixed long header once.
      * Otherwise append rows (no schema negotiation needed).

    Notes
    -----
    - All writes are append-only.
    - This writer is intended to be safe when resuming training runs.
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
        # Feature flags (enable/disable each CSV format independently).
        self._wide_enabled = bool(wide)
        self._long_enabled = bool(long)

        # ================================================================
        # Wide CSV state
        # ================================================================
        self._wide_file: Optional[TextIO] = None
        self._wide_writer: Optional[csv.DictWriter] = None
        self._wide_path = os.path.join(run_dir, wide_filename)

        # Fieldnames/schema is fixed once the header is determined.
        self._wide_fieldnames: List[str] = []

        # Indicates whether wide header/schema has been initialized.
        self._wide_header_ready = False

        if self._wide_enabled:
            # Open in append mode. `newline=""` is important for csv module
            # to avoid extra blank lines on some platforms.
            self._wide_file = open_append(self._wide_path, newline="")

        # ================================================================
        # Long CSV state
        # ================================================================
        self._long_file: Optional[TextIO] = None
        self._long_writer: Optional[Any] = None  # csv.writer instance
        self._long_path = os.path.join(run_dir, long_filename)

        # Indicates whether long header has been written (if necessary).
        self._long_header_ready = False

        if self._long_enabled:
            self._long_file = open_append(self._long_path, newline="")

    # ------------------------------------------------------------------
    # Writer interface
    # ------------------------------------------------------------------
    def write(self, row: Dict[str, float]) -> None:
        """
        Write one logical logging "row" (a dict of metrics) to enabled CSVs.

        Parameters
        ----------
        row : Dict[str, float]
            Metrics dictionary for one logging step.
            Typical keys include:
              - "step"
              - "wall_time"
              - "timestamp"
              - arbitrary metric names (loss, reward, etc.)
        """
        # Wide: 1 row per step (fixed columns).
        if self._wide_file is not None:
            self._write_wide(row)

        # Long: 1 row per key/value (lossless).
        if self._long_file is not None:
            self._write_long(row)

    def flush(self) -> None:
        """
        Best-effort flush for both CSV files.
        Safe even if a file handle is None.
        """
        safe_call(self._wide_file, "flush")
        safe_call(self._long_file, "flush")

    def close(self) -> None:
        """
        Close both CSV files safely.

        Notes
        -----
        - Always attempts to flush first.
        - Exceptions during close are swallowed by safe_call.
        """
        try:
            self.flush()
        finally:
            safe_call(self._wide_file, "close")
            safe_call(self._long_file, "close")

            # Release references to prevent accidental reuse after close.
            self._wide_file = None
            self._long_file = None
            self._wide_writer = None
            self._long_writer = None

    # ------------------------------------------------------------------
    # Wide CSV
    # ------------------------------------------------------------------
    def _write_wide(self, row: Dict[str, float]) -> None:
        """
        Write a single row to the wide CSV.

        Behavior
        --------
        - Ensures header/schema is prepared once (resume-safe).
        - Drops keys not present in the frozen schema.
        """
        assert self._wide_file is not None

        # Schema negotiation occurs once on the first call.
        if not self._wide_header_ready:
            self._prepare_wide_schema(first_row=row)

        # If schema preparation failed, skip (best-effort).
        if self._wide_writer is None:
            return

        # Wide format requires a fixed set of columns.
        # Unknown keys are ignored to prevent schema drift.
        out = {k: row.get(k, "") for k in self._wide_fieldnames}
        self._wide_writer.writerow(out)

    def _prepare_wide_schema(self, first_row: Dict[str, float]) -> None:
        """
        Prepare wide CSV schema (fieldnames) and DictWriter.

        Resume-safe approach
        --------------------
        - If the file is empty:
            * fieldnames = keys from `first_row`
            * write header immediately
        - If the file is non-empty (resume):
            * read the existing header from a separate read handle
            * keep header fixed forever

        Why open a separate read handle?
        --------------------------------
        The append handle might be positioned at EOF and not suitable for
        reliably reading the first line. Opening a separate read-only handle
        ensures we can fetch the header deterministically.
        """
        assert self._wide_file is not None

        # Detect whether file is empty.
        # If empty, we must write a header before writing any rows.
        try:
            self._wide_file.seek(0, os.SEEK_END)
            size = self._wide_file.tell()
        except Exception:
            # Conservative fallback: treat as empty.
            size = 0

        # Case 1) brand new file: initialize schema from first row keys
        if size == 0:
            self._wide_fieldnames = list(first_row.keys())
            self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)

            # Header row defines the schema of the wide CSV.
            self._wide_writer.writeheader()
            self._wide_header_ready = True
            return

        # Case 2) resume: read existing header and freeze schema to it
        header: Optional[List[str]] = None
        try:
            with open(self._wide_path, "r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                header = next(reader, None)
        except Exception:
            header = None

        if header:
            # Filter out empty column names (defensive).
            self._wide_fieldnames = [h for h in header if h]
            self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)
            self._wide_header_ready = True

            # Ensure the append handle is at EOF before writing.
            try:
                self._wide_file.seek(0, os.SEEK_END)
            except Exception:
                pass
            return

        # Fallback: if header cannot be read, treat it as a malformed file.
        # We append a header at the end as a best-effort recovery.
        self._wide_fieldnames = list(first_row.keys())
        self._wide_writer = csv.DictWriter(self._wide_file, fieldnames=self._wide_fieldnames)
        self._wide_writer.writeheader()
        self._wide_header_ready = True

        try:
            self._wide_file.seek(0, os.SEEK_END)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Long CSV
    # ------------------------------------------------------------------
    def _write_long(self, row: Dict[str, float]) -> None:
        """
        Write key/value rows to the long CSV.

        Each metric key (except meta keys) becomes one row:
          [step, wall_time, timestamp, key, value]

        This format is append-only and lossless:
        - New metric keys can appear at any time with no schema changes.
        """
        assert self._long_file is not None

        # Ensure header exists (only written once if file is empty).
        if not self._long_header_ready:
            self._prepare_long_schema()

        if self._long_writer is None:
            return

        # Meta columns are copied into every long-format row.
        step = row.get("step", "")
        wall_time = row.get("wall_time", "")
        timestamp = row.get("timestamp", "")

        # Emit one record per metric key.
        for k, v in row.items():
            if k in ("step", "wall_time", "timestamp"):
                continue
            self._long_writer.writerow([step, wall_time, timestamp, str(k), str(v)])

    def _prepare_long_schema(self) -> None:
        """
        Prepare long CSV writer and write header if file is empty.

        Long CSV schema is fixed and does not depend on emitted metric keys.
        """
        assert self._long_file is not None

        # Detect empty file.
        try:
            self._long_file.seek(0, os.SEEK_END)
            size = self._long_file.tell()
        except Exception:
            size = 0

        # csv.writer has stable fixed columns for long format.
        self._long_writer = csv.writer(self._long_file)

        # If new file -> write the header once.
        if size == 0:
            self._long_writer.writerow(["step", "wall_time", "timestamp", "key", "value"])

        self._long_header_ready = True

        # Ensure append position.
        try:
            self._long_file.seek(0, os.SEEK_END)
        except Exception:
            pass