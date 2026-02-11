"""
buffers
=======

Public exports for the ``buffers`` subpackage.

This subpackage provides buffer implementations used by both:
- **Off-policy** algorithms (e.g., DQN/DDPG/TD3/SAC) via replay buffers
- **On-policy** algorithms (e.g., PPO/A2C/TRPO) via rollout buffers

The module-level imports below define the public API of the package so users can do:

>>> from yourpkg.buffers import ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer

Notes
-----
- :class:`~buffers.ReplayBuffer` implements uniform sampling.
- :class:`~buffers.PrioritizedReplayBuffer` implements Prioritized Experience Replay (PER).
- :class:`~buffers.RolloutBuffer` implements fixed-horizon rollouts with return/advantage
  computation (e.g., GAE(λ)).
"""

from __future__ import annotations

from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .replay_buffer import ReplayBuffer
from .rollout_buffer import RolloutBuffer

__all__ = (
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "RolloutBuffer",
)