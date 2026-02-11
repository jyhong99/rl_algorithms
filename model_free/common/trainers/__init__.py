"""
Trainers
====================

This package provides a unified, high-level training interface for
reinforcement learning algorithms, supporting:

- Single-environment training loops
- Ray-based multi-worker rollout and learning
- Evaluation, checkpointing, and callback-driven extensibility

The design goal is to keep the public API minimal and stable, while
isolating optional dependencies (e.g., Ray) behind guarded imports.

Public API
----------
Trainer
    Unified training orchestrator that coordinates environments, algorithms,
    logging, evaluation, callbacks, checkpointing, and (optionally) Ray workers.

build_trainer
    High-level factory that constructs environments, logger, evaluator,
    callbacks, and returns a fully-configured Trainer instance.

Evaluator
    Episode-based policy evaluator that runs rollouts and aggregates metrics.

save_checkpoint
    Persist trainer state (counters, env state, algorithm artifact).

load_checkpoint
    Restore trainer state from a saved checkpoint.

run_evaluation
    Execute evaluation via the trainer's attached Evaluator and emit side effects.

Optional Ray API
----------------
RayEnvRunner
    Worker-side environment runner used by Ray actors.

RayLearner
    Learner-side orchestrator that manages RayEnvRunner actors.

_RAY_AVAILABLE
    Boolean flag indicating whether Ray-dependent symbols are available.

Notes
-----
- Ray is an optional dependency. This module remains importable even when
  Ray is not installed.
- Ray-related classes are exposed only if the import succeeds.
- This file intentionally contains *no heavy logic*; it only defines
  the public surface of the training subsystem.
"""

from __future__ import annotations

# =============================================================================
# Core public objects
# =============================================================================
from .trainer import Trainer
from .evaluator import Evaluator
from .trainer_builder import build_trainer

# =============================================================================
# Convenience re-exports (lightweight helpers)
# =============================================================================
from .train_checkpoint import save_checkpoint, load_checkpoint
from .train_eval import run_evaluation

# =============================================================================
# Optional Ray integration (guarded import)
# =============================================================================
try:  # pragma: no cover
    from .ray_workers import RayEnvRunner, RayLearner
    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    # Ray is not installed or failed to import.
    RayEnvRunner = None  # type: ignore
    RayLearner = None  # type: ignore
    _RAY_AVAILABLE = False


# =============================================================================
# Public export list
# =============================================================================
__all__ = [
    # Core API
    "Trainer",
    "build_trainer",
    "Evaluator",
    # Evaluation / checkpoint helpers
    "save_checkpoint",
    "load_checkpoint",
    "run_evaluation",
    # Ray (optional)
    "RayEnvRunner",
    "RayLearner",
    "_RAY_AVAILABLE",
]
