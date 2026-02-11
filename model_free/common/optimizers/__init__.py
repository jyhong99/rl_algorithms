"""
Optimizers
====================

This package provides:
- Optimizer implementations (e.g., Lion, KFAC)
- Factory functions to build optimizers and schedulers from string identifiers
- Practical utilities for parameter grouping, AMP-safe gradient clipping,
  and checkpoint (de)serialization.

The intent of this module is to expose a *stable* import surface for the rest
of the codebase. Downstream modules should import from this package (or from
this `__init__.py`) rather than reaching into internal builder/implementation
files.

Notes
-----
- The actual optimizer classes (e.g., ``Lion``, ``KFAC``) live in their own
  modules and are intentionally not re-exported here unless you add them to
  ``__all__``.
- Checkpoint helpers return plain Python dicts compatible with torch.save().
- Scheduler builders return ``None`` for constant LR schedules.

Public API
----------
Optimizers
- build_optimizer
- make_param_groups
- clip_grad_norm
- optimizer_state_dict
- load_optimizer_state_dict

Schedulers
- build_scheduler
- scheduler_state_dict
- load_scheduler_state_dict
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Optimizer factories / utilities
# -----------------------------------------------------------------------------
from .optimizer_builder import (
    build_optimizer,
    clip_grad_norm,
    load_optimizer_state_dict,
    make_param_groups,
    optimizer_state_dict,
)

# -----------------------------------------------------------------------------
# Scheduler factories / utilities
# -----------------------------------------------------------------------------
from .scheduler_builder import (
    build_scheduler,
    load_scheduler_state_dict,
    scheduler_state_dict,
)

__all__ = [
    # optimizer utils
    "build_optimizer",
    "make_param_groups",
    "clip_grad_norm",
    "optimizer_state_dict",
    "load_optimizer_state_dict",
    # scheduler utils
    "build_scheduler",
    "scheduler_state_dict",
    "load_scheduler_state_dict",
]
