from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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
    # backend enable flags
    use_tensorboard: bool = True,
    use_csv: bool = True,
    use_jsonl: bool = True,
    use_wandb: bool = False,
    # backend kwargs
    csv_kwargs: Optional[Dict[str, Any]] = None,
    jsonl_kwargs: Optional[Dict[str, Any]] = None,
    wandb_kwargs: Optional[Dict[str, Any]] = None,
    tensorboard_kwargs: Optional[Dict[str, Any]] = None,
    # logger behavior
    console_every: int = 1,
    flush_every: int = 200,
    drop_non_finite: bool = False,
    strict: bool = False,
) -> Logger:
    """
    Construct a :class:`~logger.Logger` and attach selected writer backends.

    This factory centralizes logger initialization and backend wiring so that
    training scripts can enable/disable outputs (CSV/JSONL/TensorBoard/W&B)
    without duplicating setup code.

    Parameters
    ----------
    log_dir : str, default="./runs"
        Root directory under which experiment run directories are created.
    exp_name : str, default="exp"
        Experiment name used by :class:`~logger.Logger` to form run paths.
    run_id : str, optional
        Optional run identifier. If provided, the logger uses it to produce a
        deterministic run directory (exact semantics depend on `Logger`).
    run_name : str, optional
        Optional human-friendly run name. Typically used for display (e.g., W&B),
        and sometimes included in directory naming depending on `Logger`.
    overwrite : bool, default=False
        If True, allow overwriting an existing run directory (semantics are owned
        by `Logger`). For safety, keep this False in most training jobs.
    resume : bool, default=False
        If True, attempt to resume logging into an existing run directory.
        This enables append-only writers (CSV/JSONL) to continue where they left off.
    require_resume_exists : bool, default=True
        If True and `resume=True`, raise if the resolved run directory does not exist.
        This prevents accidentally creating a fresh directory when you intended to resume.

    use_tensorboard : bool, default=True
        Attach :class:`~tensorboard_writer.TensorBoardWriter`.
    use_csv : bool, default=True
        Attach :class:`~csv_writer.CSVWriter`.
    use_jsonl : bool, default=True
        Attach :class:`~jsonl_writer.JSONLWriter`.
    use_wandb : bool, default=False
        Attach :class:`~wandb_writer.WandBWriter`.

    csv_kwargs : dict, optional
        Keyword arguments forwarded to ``CSVWriter(run_dir=..., **csv_kwargs)``.
        By default, this factory enables both wide and long CSV formats unless
        explicitly overridden.
    jsonl_kwargs : dict, optional
        Keyword arguments forwarded to ``JSONLWriter(run_dir=..., **jsonl_kwargs)``.
    tensorboard_kwargs : dict, optional
        Keyword arguments forwarded to ``TensorBoardWriter(run_dir=..., **tensorboard_kwargs)``.
    wandb_kwargs : dict, optional
        Keyword arguments forwarded to ``WandBWriter(run_dir=..., **wandb_kwargs)``.
        When ``use_wandb=True``, ``wandb_kwargs["project"]`` must be provided.

    console_every : int, default=1
        Logger-side console printing interval (in "write calls" or steps, depending on Logger).
    flush_every : int, default=200
        Logger-side flush interval. Writers may still buffer internally.
    drop_non_finite : bool, default=False
        If True, the logger may drop NaN/Inf values before dispatching to writers
        (behavior depends on `Logger`).
    strict : bool, default=False
        If True, the logger may raise on invalid inputs or writer failures
        (behavior depends on `Logger`).

    Returns
    -------
    Logger
        A configured logger instance with requested writer backends attached.

    Raises
    ------
    FileNotFoundError
        If ``resume=True`` and ``require_resume_exists=True`` but the resolved
        run directory does not exist.
    ValueError
        If ``use_wandb=True`` but ``wandb_kwargs`` does not include a non-empty
        ``"project"`` entry.
    AttributeError
        If the `Logger` instance does not expose an attachment mechanism
        (``add_writers``, ``add_writer``, or an internal ``_writers`` list).

    Notes
    -----
    - The :class:`~logger.Logger` is instantiated first because it "owns" the logic
      of resolving ``logger.run_dir`` from ``log_dir``, ``exp_name``, and run identifiers.
    - Writer instances are then constructed using that resolved directory.
    - Attachment is performed using the first available method:
        1) ``logger.add_writers(writers)`` (preferred bulk attach)
        2) ``logger.add_writer(writer)`` (iterative attach)
        3) Fallback: extend ``logger._writers`` (last resort)
    """
    # Normalize kwarg dicts (avoid mutating caller objects).
    csv_kwargs = dict(csv_kwargs or {})
    jsonl_kwargs = dict(jsonl_kwargs or {})
    wandb_kwargs = dict(wandb_kwargs or {})
    tensorboard_kwargs = dict(tensorboard_kwargs or {})

    # ------------------------------------------------------------------
    # Construct Logger first (it resolves run_dir).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Optional resume existence validation (guardrail).
    # ------------------------------------------------------------------
    if resume and require_resume_exists and (not os.path.exists(logger.run_dir)):
        raise FileNotFoundError(f"resume=True but run_dir does not exist: {logger.run_dir}")

    # ------------------------------------------------------------------
    # Instantiate selected writer backends.
    # ------------------------------------------------------------------
    writers: List[Any] = []

    if use_tensorboard:
        writers.append(TensorBoardWriter(logger.run_dir, **tensorboard_kwargs))

    if use_csv:
        # Default behavior: enable both wide and long formats unless overridden.
        csv_defaults: Dict[str, Any] = {"wide": True, "long": True}
        csv_defaults.update(csv_kwargs)
        writers.append(CSVWriter(logger.run_dir, **csv_defaults))

    if use_jsonl:
        writers.append(JSONLWriter(logger.run_dir, **jsonl_kwargs))

    if use_wandb:
        project = wandb_kwargs.get("project")
        if not project:
            raise ValueError("wandb_kwargs must include non-empty 'project' when use_wandb=True.")
        writers.append(WandBWriter(run_dir=logger.run_dir, **wandb_kwargs))

    # ------------------------------------------------------------------
    # Attach writers to Logger using the best available interface.
    # ------------------------------------------------------------------
    add_writers = getattr(logger, "add_writers", None)
    if callable(add_writers):
        add_writers(writers)
        return logger

    add_writer = getattr(logger, "add_writer", None)
    if callable(add_writer):
        for w in writers:
            add_writer(w)
        return logger

    # Fallback: extend internal container (last resort).
    if getattr(logger, "_writers", None) is None:
        raise AttributeError(
            "Logger has no add_writer(s) method and no attribute '_writers'. "
            "Please implement Logger.add_writer(s) or expose a writers container."
        )

    logger._writers.extend(writers)
    return logger
