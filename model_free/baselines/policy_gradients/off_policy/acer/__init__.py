"""
ACER
=======

This package provides a minimal, stable public API for a discrete-action ACER
implementation built on the project's head/core/algorithm composition pattern.

The package is intentionally small:

- **Head** (:class:`~.head.ACERHead`)
    Owns the neural networks and policy/value helpers:
    categorical actor π(a|s), Q critic Q(s,a), and a target critic for stable TD
    targets. Also provides persistence helpers (save/load) and (optionally) Ray
    reconstruction utilities.

- **Core** (:class:`~.core.ACERCore`)
    Implements the optimization step(s) for ACER-style training:
    TD(0) critic regression, truncated importance sampling actor update, optional
    bias correction (if behavior probabilities are stored), entropy regularization,
    and target-network update cadence. It also owns optimizer/scheduler plumbing
    via :class:`~model_free.common.policies.base_core.ActorCriticCore`.

- **Builder** (:func:`~.acer.acer`)
    Convenience factory that wires together:
    ``ACERHead`` + ``ACERCore`` + ``OffPolicyAlgorithm`` (replay buffer and update
    scheduling). Use this if you want a "ready-to-setup" algorithm instance.

Public API
----------
acer
    Factory function returning a configured
    :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`.

ACERHead
    Discrete actor-critic head (actor + critic + target critic).

ACERCore
    ACER update engine (losses, target updates, optimizers/schedulers).

Notes
-----
- This ``__init__`` deliberately re-exports only the stable surface area.
  Internal helpers and submodules should be imported from their respective files
  (e.g., ``.head`` or ``.core``) if needed for advanced customization.
- Keeping ``__all__`` explicit helps avoid accidental public API expansion when
  new files are added to the package.
"""

from __future__ import annotations

from .acer import acer
from .core import ACERCore
from .head import ACERHead

__all__ = [
    "acer",
    "ACERHead",
    "ACERCore",
]
