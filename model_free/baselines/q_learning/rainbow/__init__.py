"""
Rainbow
=======

This subpackage provides a *config-free* constructor for the Rainbow algorithm
(C51-style distributional Q-learning with target networks, optional Double DQN
action selection, PER integration, and NoisyNet exploration depending on the
head implementation).

Public API
----------
rainbow : callable
    Builder function that returns a fully-wired
    :class:`model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
    instance (head + core + algorithm driver).
RainbowHead : type
    Q-learning head that owns the online/target distributional Q-networks and
    the fixed categorical support (C51 atoms). May also support NoisyNet.
RainbowCore : type
    Update engine implementing the C51 distributional Bellman projection and
    optimizer step, typically built on top of a shared QLearningCore base.

Notes
-----
- This module is intended to be the main import surface for the Rainbow variant:
    >>> from model_free.algorithms.rainbow import rainbow
- Internal modules (head/core/builder) can still be imported directly, but the
  symbols exported here are considered the stable API.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
from .rainbow import rainbow
from .head import RainbowHead
from .core import RainbowCore

__all__ = [
    "rainbow",
    "RainbowHead",
    "RainbowCore",
]
