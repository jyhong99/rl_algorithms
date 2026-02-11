"""
SAC
=======

This subpackage exposes the minimal, stable surface area for constructing and
using the SAC implementation in this codebase.

Exports
-------
sac : callable
    Factory/builder function that assembles a complete off-policy SAC algorithm.
    The builder typically wires together:

    - :class:`~.head.SACHead` : policy/value networks (actor + twin critics + target critics)
    - :class:`~.core.SACCore` : update engine (losses, optimizers, target updates, etc.)
    - an outer algorithm driver (e.g., OffPolicyAlgorithm) that manages replay and
      update scheduling (exact composition depends on your `sac` builder)

SACHead : type
    Head module that owns SAC neural networks and inference utilities.

SACCore : type
    Core update engine that implements SAC training rules on top of shared
    ActorCriticCore infrastructure.

Notes
-----
- `__all__` defines the intended public API for `from ... import *` usage and
  helps avoid accidental symbol leakage from internal modules.
- Internal modules may contain additional helpers, but only the names listed
  in `__all__` are considered stable API.
"""

from __future__ import annotations

from .sac import sac
from .head import SACHead
from .core import SACCore

__all__ = [
    "sac",
    "SACHead",
    "SACCore",
]
