"""
DDPG package.

Public API
----------
- DDPGHead  : actor/critic + target networks (head)
- DDPGCore  : update engine (critic/actor update + target soft update)
- ddpg      : builder that assembles Head + Core + OffPolicyAlgorithm
"""

from __future__ import annotations

# Head
from .head import DDPGHead

# Core
from .core import DDPGCore

# Builder / entrypoint
from .ddpg import ddpg

__all__ = [
    "DDPGHead",
    "DDPGCore",
    "ddpg",
]
