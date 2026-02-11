"""
Loggers
====================

This package provides:
- Logger frontend (step inference, buffering, console printing)
- Multiple writer backends (CSV, JSONL, TensorBoard, W&B)
- A builder utility to construct a Logger with selected backends

Typical usage
-------------
from logger import build_logger

logger = build_logger(
    log_dir="./runs",
    exp_name="exp1",
    use_csv=True,
    use_tensorboard=True,
)

logger.log({"loss": 0.1}, step=1)
logger.close()
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Core logger
# -----------------------------------------------------------------------------
from .logger import Logger

# -----------------------------------------------------------------------------
# Writer base + concrete writers
# -----------------------------------------------------------------------------
from .base_writer import Writer, SafeWriter
from .csv_writer import CSVWriter
from .jsonl_writer import JSONLWriter
from .tensorboard_writer import TensorBoardWriter
from .wandb_writer import WandBWriter

# -----------------------------------------------------------------------------
# Builder utility
# -----------------------------------------------------------------------------
from .build_logger import build_logger

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # core
    "Logger",

    # writer base
    "Writer",
    "SafeWriter",

    # writers
    "CSVWriter",
    "JSONLWriter",
    "TensorBoardWriter",
    "WandBWriter",

    # builder
    "build_logger",
]
