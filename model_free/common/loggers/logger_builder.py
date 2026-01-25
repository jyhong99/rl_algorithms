from __future__ import annotations

from typing import Any, List, Optional, Sequence
import os

from .logger import Logger
from .csv_writer import CSVWriter
from .jsonl_writer import JSONLWriter
from .tensorboard_writer import TensorBoardWriter
from .wandb_writer import WandBWriter


def build_logger(
    *,
    log_dir: str = "./runs",
    exp_name: str = "exp",
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = False,
    require_resume_exists: bool = True,
    # backends
    use_tensorboard: bool = True,
    use_csv: bool = True,
    use_csv_long: bool = True,
    use_jsonl: bool = True,
    use_wandb: bool = False,
    # wandb flat options (match build_trainer args)
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_run_name: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_resume: Optional[str] = None,
    # logger behavior
    console_every: int = 1,
    flush_every: int = 200,
    drop_non_finite: bool = False,
    strict: bool = False,
) -> "Logger":
    """
    Create a Logger and attach selected writer backends.

    This version uses *flat* W&B arguments (no WandBOptions dataclass) so it
    matches the signature style used in your build_trainer factory.

    Parameters
    ----------
    log_dir : str
        Root logging directory.
    exp_name : str
        Experiment name (subdirectory name under log_dir).
    run_id : Optional[str]
        Explicit run id (filesystem-level). Highest priority.
    run_name : Optional[str]
        Alternative identifier (filesystem-level). Used only if run_id is None.
    overwrite : bool
        If True, reuse the same run directory even if it exists.
    resume : bool
        If True, do not create a new run directory; instead, use the existing one.
    require_resume_exists : bool
        If True and resume=True, raise FileNotFoundError if the directory is missing.

    use_tensorboard/use_csv/use_jsonl/use_wandb : bool
        Enable each writer backend.

    wandb_project : Optional[str]
        Required when use_wandb=True.
    wandb_entity, wandb_group, wandb_tags, wandb_run_name, wandb_mode, wandb_resume : Optional
        Passed through to your WandBWriter.

    console_every : int
        Console logging frequency (semantics depend on Logger implementation).
    flush_every : int
        Writer flush frequency.
    drop_non_finite : bool
        If True, drop NaN/Inf scalars instead of raising.
    strict : bool
        If True, enforce stricter checks inside Logger.

    Returns
    -------
    Logger
        Initialized logger with requested writers attached.

    Notes
    -----
    - Logger is created first because it owns the resolved `run_dir`.
    - Writers are constructed afterwards using `logger.run_dir`.
    - Prefer a public method like `logger.add_writers(...)` if available.
    """
    # ---- construct Logger first (it resolves run_dir) ----
    logger = Logger(
        log_dir=str(log_dir),
        exp_name=str(exp_name),
        run_id=run_id,
        run_name=run_name,
        overwrite=bool(overwrite),
        resume=bool(resume),
        writers=None,  # attach later
        console_every=int(console_every),
        flush_every=int(flush_every),
        drop_non_finite=bool(drop_non_finite),
        strict=bool(strict),
    )

    # ---- optional resume existence validation ----
    if resume and require_resume_exists and (not os.path.exists(logger.run_dir)):
        raise FileNotFoundError(f"resume=True but run_dir does not exist: {logger.run_dir}")

    writers: List[Any] = []

    if use_tensorboard:
        writers.append(TensorBoardWriter(logger.run_dir))

    if use_csv:
        writers.append(CSVWriter(logger.run_dir, wide=True, long=bool(use_csv_long)))

    if use_jsonl:
        writers.append(JSONLWriter(logger.run_dir))

    if use_wandb:
        if not wandb_project:
            raise ValueError("wandb_project must be provided when use_wandb=True.")
        writers.append(
            WandBWriter(
                run_dir=logger.run_dir,
                project=str(wandb_project),
                entity=wandb_entity,
                group=wandb_group,
                tags=wandb_tags,
                name=wandb_run_name,
                mode=wandb_mode,
                resume=wandb_resume,
            )
        )

    # ---- attach writers (prefer public API; fallback to _writers for compatibility) ----
    add_writers = getattr(logger, "add_writers", None)
    if callable(add_writers):
        add_writers(writers)
        return logger

    add_writer = getattr(logger, "add_writer", None)
    if callable(add_writer):
        for w in writers:
            add_writer(w)
        return logger

    # Minimal fallback (keeps behavior close to your existing code).
    if getattr(logger, "_writers", None) is None:
        raise AttributeError(
            "Logger has no add_writer(s) method and no attribute '_writers'. "
            "Please add Logger.add_writer(s) or expose a writers container."
        )
    logger._writers.extend(writers)
    return logger