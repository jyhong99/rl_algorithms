"""
Discrete PPO
=======

This subpackage provides the discrete-action variant of Proximal Policy Optimization
(PPO) in a "head + core + algorithm" composition style.

Architecture
------------
The implementation is split into three cooperating components:

- **PPODiscreteHead**
  Owns the neural networks (categorical actor + value critic) and provides the
  inference/evaluation APIs expected by on-policy trainers.

- **PPODiscreteCore**
  Implements the PPO update rule for discrete actions (clipped surrogate objective,
  value loss with optional value clipping, entropy bonus) and applies optimizer steps.

- **ppo_discrete**
  A config-free factory that wires together:
    head (PPODiscreteHead) + core (PPODiscreteCore) + algorithm (OnPolicyAlgorithm).

Public API
----------
Only the following symbols are intended to be imported from this package:

- ``ppo_discrete`` : builder function returning an ``OnPolicyAlgorithm`` instance
- ``PPODiscreteHead`` : actor-critic network container for discrete actions
- ``PPODiscreteCore`` : PPO update engine for discrete actions

Notes
-----
- This file is the package-level ``__init__.py``. It re-exports the public API and
  defines ``__all__`` so tools like linters, IDEs, and wildcard imports behave
  predictably.
- Keep exports stable to avoid breaking downstream code that imports from the
  package root.
"""

from __future__ import annotations

from .core import PPODiscreteCore
from .head import PPODiscreteHead
from .ppo_discrete import ppo_discrete

__all__ = [
    "ppo_discrete",
    "PPODiscreteHead",
    "PPODiscreteCore",
]
