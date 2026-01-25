from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple, Mapping, Any, Optional, TextIO
import os
import uuid
import json


# =============================================================================
# Metadata convention
# =============================================================================
# These keys are treated as "meta" fields (not plotted as typical metrics).
# Writers can use them for indexing, timestamps, etc.
META_KEYS = ("step", "wall_time", "timestamp")


# =============================================================================
# Run directory utilities
# =============================================================================
def generate_run_id() -> str:
    """
    Generate a unique run identifier.

    Returns
    -------
    str
        Run id in the form "{YYYY-mm-dd_HH-MM-SS}_{8-hex}", e.g.
        "2026-01-22_14-03-12_a1b2c3d4".

    Notes
    -----
    - Timestamp improves human readability and makes directory sorting meaningful.
    - Random suffix reduces collision risk in fast repeated launches.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def make_run_dir(
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

    Parameters
    ----------
    log_dir : str
        Root logging directory (e.g., "./runs").
    exp_name : str
        Experiment name (subdirectory under log_dir).
    run_id : Optional[str]
        Explicit run identifier. Highest priority if provided.
    run_name : Optional[str]
        Alternative identifier. Used only if run_id is None.
    overwrite : bool
        If True, reuse the same directory even if it exists.
        If False, create a suffix "_{k}" to avoid collisions.
    resume : bool
        If True, return the computed path without generating a new suffixed path.
        Used when resuming a previous run.
    require_resume_exists : bool
        If True and resume=True, raise FileNotFoundError when the directory doesn't exist.
        Recommended default for safety to avoid silently starting a fresh run.

    Returns
    -------
    str
        Resolved run directory path:
          "{log_dir}/{exp_name}/{rid}" or "{...}/{rid}_{k}".

    Notes
    -----
    Precedence for selecting the identifier (rid):
      1) run_id (explicit)
      2) run_name (explicit)
      3) auto-generated id (generate_run_id)
    """
    # Base experiment directory
    base = os.path.join(str(log_dir), str(exp_name))

    # Pick a run identifier
    rid = run_id or run_name or generate_run_id()

    # Candidate run directory
    path = os.path.join(base, str(rid))

    # Resume mode: do not create new suffixes
    if resume:
        if require_resume_exists and (not os.path.exists(path)):
            raise FileNotFoundError(f"resume=True but run_dir does not exist: {path}")
        return path

    # Fresh run mode:
    # - If overwrite: reuse the directory name even if it exists
    # - If path does not exist: safe to use
    if overwrite or (not os.path.exists(path)):
        return path

    # Collision avoidance: append incremental suffix until a free directory is found.
    i = 1
    while True:
        cand = f"{path}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


# =============================================================================
# Metric row helpers
# =============================================================================
def split_meta(row: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Split a metric row into (meta, metrics).

    Parameters
    ----------
    row : Mapping[str, float]
        Input row that may contain both meta fields and metric fields.

    Returns
    -------
    meta : Dict[str, float]
        Only META_KEYS extracted and cast to float.
    metrics : Dict[str, float]
        All remaining keys cast to float.

    Notes
    -----
    - This function assumes metric values are float-castable.
    - If a writer needs to preserve original types, do not use this function.
    """
    meta = {k: float(row[k]) for k in META_KEYS if k in row}
    metrics = {k: float(v) for k, v in row.items() if k not in META_KEYS}
    return meta, metrics


def get_step(row: Mapping[str, float]) -> int:
    """
    Extract integer step from a metric row.

    Returns 0 if step is missing or not int-castable.
    """
    try:
        return int(row.get("step", 0))
    except Exception:
        return 0


# =============================================================================
# Serialization helpers
# =============================================================================
def json_dumps(obj: Any) -> str:
    """
    JSON serialize with:
    - ensure_ascii=False (keep unicode readable)
    - default=str (best-effort for non-JSON objects)
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


# =============================================================================
# Filesystem helpers for writers
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create directory (and parents) if missing. No-op if exists."""
    os.makedirs(path, exist_ok=True)


def open_append(
    path: str,
    *,
    newline: str | None = None,
    encoding: str = "utf-8",
) -> TextIO:
    """
    Open a file in append mode, ensuring the parent directory exists.

    Parameters
    ----------
    path : str
        File path.
    newline : str | None, optional
        Newline handling (use "" for CSV on Windows).
        If None, Python default newline behavior is used.
    encoding : str, optional
        File encoding, by default "utf-8".

    Returns
    -------
    f : TextIO
        Opened file handle in append mode.

    Notes
    -----
    - Caller owns the returned file handle and must close it.
    - Use `newline=""` when writing CSV via `csv.writer`.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)

    return open(path, "a", newline=newline, encoding=encoding)


def safe_call(obj: Optional[Any], method: str) -> None:
    """
    Best-effort method call; never raises.

    Parameters
    ----------
    obj : Any or None
        Target object.
    method : str
        Method name to call if present.

    Notes
    -----
    - Safe to call with None.
    - Silently ignores missing methods and all exceptions.
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
# Logging dispatch (trainer-agnostic)
# =============================================================================
def log(
    trainer: Any,
    metrics: Dict[str, Any],
    *,
    step: int,
    prefix: str = "",
) -> None:
    """
    Best-effort logger dispatch.

    Expected logger interface (duck-typed)
    --------------------------------------
    trainer.logger.log(metrics: dict, step: int, prefix: str = "")

    Parameters
    ----------
    trainer : Any
        Trainer object holding a `.logger`.
    metrics : Dict[str, Any]
        Metrics payload (typically scalars and small JSON-friendly objects).
    step : int
        Global step to associate with these metrics (env_step or update_step).
    prefix : str, default=""
        Optional prefix namespace (e.g., "train/", "eval/", "sys/").

    Notes
    -----
    - This function intentionally swallows all exceptions so callbacks cannot crash training.
    - If logger is missing or does not implement `.log`, it becomes a no-op.
    """
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return

    fn = getattr(logger, "log", None)
    if not callable(fn):
        return

    try:
        fn(metrics, step=step, prefix=prefix)
    except Exception:
        # Callbacks should not crash training.
        return