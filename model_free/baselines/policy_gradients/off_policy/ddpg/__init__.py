"""
DDPG
=======

This package implements a deterministic off-policy actor-critic stack in the
DDPG (Deep Deterministic Policy Gradient) family using the project's modular
composition pattern:

- **Head**: network definitions + target networks + (optional) exploration noise
- **Core**: gradient update logic (critic TD regression + actor policy update)
- **Builder**: convenience factory that assembles head + core + replay/scheduling
  wrapper into a ready-to-train algorithm object

Public API
----------
DDPGHead
    Deterministic actor-critic head that owns:
    - actor π(s) -> a
    - critic Q(s,a) -> scalar
    - target copies (actor_target, critic_target)
    - persistence helpers (save/load) and (optionally) Ray reconstruction support

DDPGCore
    Update engine built on :class:`~model_free.common.policies.base_core.ActorCriticCore`:
    - critic TD(0) regression toward targets computed using target networks
    - actor update via deterministic policy gradient (maximize Q(s, π(s)))
    - target network updates (hard/soft, controlled by tau and update cadence)
    - optimizer/scheduler plumbing, AMP, gradient clipping

ddpg
    Factory function that wires together:
    ``DDPGHead`` + ``DDPGCore`` + ``OffPolicyAlgorithm`` (replay + scheduling + PER).

Notes
-----
- This ``__init__`` re-exports only the stable surface area of the package.
  Internal helpers should be imported directly from their defining modules if
  needed for advanced customization.
- The explicit ``__all__`` prevents accidental API expansion when adding new
  modules to the package.
"""

from __future__ import annotations

from .core import DDPGCore
from .ddpg import ddpg
from .head import DDPGHead

__all__ = [
    "DDPGHead",
    "DDPGCore",
    "ddpg",
]
