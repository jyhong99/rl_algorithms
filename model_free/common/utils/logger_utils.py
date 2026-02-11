from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, TextIO, Tuple, List
import csv
import json
import os
import uuid


# =============================================================================
# Metadata convention
# =============================================================================
# These keys are treated as "meta" fields (not plotted as typical metrics).
# Writers can use them for indexing, timestamps, etc.
META_KEYS: Tuple[str, str, str] = ("step", "wall_time", "timestamp")


# =============================================================================
# Run directory utilities
# =============================================================================
def _generate_run_id() -> str:
    """
    Generate a unique run identifier suitable for filesystem paths.

    Returns
    -------
    run_id : str
        Run id in the form ``"{YYYY-mm-dd_HH-MM-SS}_{8-hex}"``, e.g.
        ``"2026-01-22_14-03-12_a1b2c3d4"``.

    Notes
    -----
    - The timestamp prefix improves human readability and directory sort order.
    - The random suffix reduces collision risk in fast repeated launches.
    - Uses local time via ``datetime.now()``.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _make_run_dir(
    log_dir: str,
    exp_name: str,
    *,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = False,
    require_resume_exists: bool = True,
) -> str:
    """
    Resolve a run directory path for an experiment.

    This function computes a run directory path of the form::

        {log_dir}/{exp_name}/{rid}

    where ``rid`` is chosen by precedence rules (see Notes). If the computed path
    already exists and this is a fresh run (resume=False, overwrite=False), it
    appends an incremental suffix ``_{k}`` to avoid collisions.

    Parameters
    ----------
    log_dir : str
        Root logging directory (e.g., ``"./runs"``).
    exp_name : str
        Experiment name (subdirectory under ``log_dir``).
    run_id : Optional[str], default=None
        Explicit run identifier. Highest priority if provided.
    run_name : Optional[str], default=None
        Alternative identifier. Used only if ``run_id`` is None.
    overwrite : bool, default=False
        If True, reuse the computed directory even if it exists.
        If False, create a suffix ``"_{k}"`` to avoid collisions.
    resume : bool, default=False
        If True, return the computed path directly without generating suffixes.
        Intended for resuming an existing run.
    require_resume_exists : bool, default=True
        If True and ``resume=True``, raise FileNotFoundError when the directory
        does not exist. This prevents silently starting a fresh run when the
        user intended to resume.

    Returns
    -------
    run_dir : str
        Resolved run directory path.

    Raises
    ------
    FileNotFoundError
        If ``resume=True`` and ``require_resume_exists=True`` but the directory
        does not exist.

    Notes
    -----
    Identifier precedence (rid):
      1) ``run_id`` (explicit)
      2) ``run_name`` (explicit)
      3) auto-generated id via `_generate_run_id()`

    Collision behavior (fresh run only):
      - If overwrite=True: keep the original path even if it exists.
      - Else: find the first available ``{path}_{k}``.
    """
    base = os.path.join(str(log_dir), str(exp_name))
    rid = run_id or run_name or _generate_run_id()
    path = os.path.join(base, str(rid))

    if resume:
        if require_resume_exists and (not os.path.exists(path)):
            raise FileNotFoundError(f"resume=True but run_dir does not exist: {path}")
        return path

    if overwrite or (not os.path.exists(path)):
        return path

    i = 1
    while True:
        cand = f"{path}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


# =============================================================================
# Metric row helpers
# =============================================================================
def _split_meta(row: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Split a row into (meta, metrics) based on META_KEYS.

    Parameters
    ----------
    row : Mapping[str, Any]
        Input row that may contain both meta fields and metric fields.

    Returns
    -------
    meta : Dict[str, float]
        Meta fields extracted from META_KEYS and converted to float.
        Missing keys are omitted.
    metrics : Dict[str, float]
        All non-meta keys converted to float.

    Raises
    ------
    (Implicit)
        This function assumes values are float-castable. If a value cannot be
        cast, `float(...)` will raise.

    Notes
    -----
    - This is useful when downstream tools treat "step/wall_time/timestamp" as
      indexing rather than plot series.
    - If you need to preserve original types (e.g., timestamp strings), do not
      use this helper; prefer `_extract_meta(...)` below.
    """
    meta = {k: float(row[k]) for k in META_KEYS if k in row}
    metrics = {str(k): float(v) for k, v in row.items() if k not in META_KEYS}
    return meta, metrics


def _get_step(row: Mapping[str, Any]) -> int:
    """
    Extract an integer "step" from a metric row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Row that may contain a "step" key.

    Returns
    -------
    step : int
        Integer step if present and int-castable, else 0.

    Notes
    -----
    - This is intentionally best-effort and never raises.
    - If "step" is a float string (e.g., "10.0"), `int("10.0")` fails; consider
      normalizing upstream if that is a common case.
    """
    try:
        return int(row.get("step", 0))
    except Exception:
        return 0


# =============================================================================
# Serialization helpers
# =============================================================================
def _json_dumps(obj: Any) -> str:
    """
    Serialize an object to JSON with practical defaults for logging.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    s : str
        JSON string with:
        - ``ensure_ascii=False`` to keep Unicode readable
        - ``default=str`` for best-effort serialization of non-JSON objects

    Notes
    -----
    - `default=str` is intended for logging/debugging. It is not a strict schema.
    - If you require stable structured JSON, provide custom encoding logic.
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


# =============================================================================
# Filesystem helpers for writers
# =============================================================================
def _ensure_dir(path: str) -> None:
    """
    Create a directory (and parents) if missing.

    Parameters
    ----------
    path : str
        Directory path.

    Notes
    -----
    No-op if the directory already exists.
    """
    os.makedirs(path, exist_ok=True)


def _open_append(
    path: str,
    *,
    newline: Optional[str] = None,
    encoding: str = "utf-8",
) -> TextIO:
    """
    Open a file in append mode, ensuring the parent directory exists.

    Parameters
    ----------
    path : str
        File path.
    newline : str or None, default=None
        Newline handling. For CSV, prefer ``newline=""`` to avoid blank lines on
        Windows when using ``csv.writer``.
    encoding : str, default="utf-8"
        Text encoding.

    Returns
    -------
    f : TextIO
        Opened file handle in append mode.

    Notes
    -----
    - Caller owns the returned file handle and must close it.
    - Parent directory is created if needed.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        _ensure_dir(dirpath)
    return open(path, "a", newline=newline, encoding=encoding)


def _safe_call(obj: Optional[Any], method: str) -> None:
    """
    Best-effort method call; never raises.

    Parameters
    ----------
    obj : Any or None
        Target object. If None, this is a no-op.
    method : str
        Method name to call if present and callable.

    Notes
    -----
    - Silently ignores missing methods and all exceptions.
    - Useful for optional dependencies (e.g., "flush", "close", "finish").
    """
    if obj is None:
        return
    try:
        fn = getattr(obj, method, None)
        if callable(fn):
            fn()
    except Exception:
        pass


# =============================================================================
# Helpers (small, testable units)
# =============================================================================
def _safe_file_size(f: TextIO) -> int:
    """
    Return file size using seek/tell; fall back to 0 on failure.

    Parameters
    ----------
    f : TextIO
        Open file handle.

    Returns
    -------
    size : int
        Size in bytes (best-effort).

    Notes
    -----
    This uses `seek(0, SEEK_END)` and `tell()`. The original file pointer is
    moved to EOF on success.
    """
    try:
        f.seek(0, os.SEEK_END)
        return int(f.tell())
    except Exception:
        return 0


def _seek_eof(f: TextIO) -> None:
    """
    Best-effort seek to end-of-file (EOF).

    Parameters
    ----------
    f : TextIO
        Open file handle.

    Notes
    -----
    This function never raises.
    """
    try:
        f.seek(0, os.SEEK_END)
    except Exception:
        pass


def _read_csv_header(*, path: str, encoding: str = "utf-8") -> Optional[List[str]]:
    """
    Read the first CSV row as a header.

    Parameters
    ----------
    path : str
        CSV file path.
    encoding : str, default="utf-8"
        Text encoding used for reading.

    Returns
    -------
    header : list[str] or None
        Header columns if readable and non-empty, otherwise None.

    Notes
    -----
    - Uses `csv.reader` and reads only the first row.
    - Returns None on any I/O or parsing failure.
    """
    try:
        with open(path, "r", newline="", encoding=encoding) as rf:
            reader = csv.reader(rf)
            header = next(reader, None)
            if not header:
                return None
            return [str(h) for h in header]
    except Exception:
        return None


def _extract_meta(
    row: Mapping[str, Any],
    meta_keys: Sequence[str] = META_KEYS,
) -> Tuple[Any, Any, Any]:
    """
    Extract (step, wall_time, timestamp) meta fields with empty-string defaults.

    Parameters
    ----------
    row : Mapping[str, Any]
        Input row containing metrics and optional meta keys.
    meta_keys : Sequence[str], default=META_KEYS
        Expected ordering of meta keys. This implementation expects three keys
        corresponding to (step, wall_time, timestamp).

    Returns
    -------
    step : Any
        Value for "step" (or first meta key), defaulting to "" if absent.
    wall_time : Any
        Value for "wall_time" (or second meta key), defaulting to "" if absent.
    timestamp : Any
        Value for "timestamp" (or third meta key), defaulting to "" if absent.

    Notes
    -----
    - Return types are `Any` on purpose: writers may store step as int, wall_time
      as float, and timestamp as str. CSV writing will serialize these later.
    - If `meta_keys` does not have length 3, the conventional trio is used.
    """
    if len(meta_keys) != 3:
        meta_keys = ("step", "wall_time", "timestamp")

    step = row.get(meta_keys[0], "")
    wall_time = row.get(meta_keys[1], "")
    timestamp = row.get(meta_keys[2], "")
    return step, wall_time, timestamp
