"""
Optimizers and LR schedulers.

Public API
----------
Optimizers
- Lion
- KFAC
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


# ---- Factory / utilities (optimizers) ----
from .optimizer_builder import (
    build_optimizer,
    make_param_groups,
    clip_grad_norm,
    optimizer_state_dict,
    load_optimizer_state_dict,
)

# ---- Factory / utilities (schedulers) ----
from .scheduler_builder import (
    build_scheduler,
    scheduler_state_dict,
    load_scheduler_state_dict,
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
