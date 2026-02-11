"""
DQN
=======

This subpackage provides a minimal, stable import surface for the DQN family:

- :func:`dqn`
    High-level builder that wires together:
      * :class:`~.head.DQNHead`  (online + target Q networks)
      * :class:`~.core.DQNCore`  (TD update / Double DQN / target updates / PER weighting)
      * :class:`model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
        (replay buffer + update scheduling)

- :class:`DQNHead`
    Q-network container for discrete control. Owns:
      * online Q-network
      * target Q-network
    and provides Ray-friendly reconstruction metadata (if supported by your base head).

- :class:`DQNCore`
    Update engine implementing the DQN/Double DQN TD loss, optimizer step,
    optional AMP path, and periodic hard/soft target updates (delegated to base core).

Design notes
------------
- This module intentionally re-exports a small set of symbols via ``__all__`` to:
  * keep user-facing imports clean and stable (e.g., ``from ...dqn import dqn``),
  * avoid leaking internal modules,
  * reduce circular-import risk across the larger codebase.

Examples
--------
>>> from model_free.algos.dqn import dqn
>>> algo = dqn(obs_dim=8, n_actions=4, device="cuda")
"""

from __future__ import annotations

from .core import DQNCore
from .dqn import dqn
from .head import DQNHead

__all__ = [
    "dqn",
    "DQNHead",
    "DQNCore",
]
