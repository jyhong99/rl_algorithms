"""
Discrete SAC
=======

This subpackage exposes the *discrete-action* SAC implementation as a small,
stable surface for external users while keeping internal module structure
flexible.

Exports
-------
sac_discrete : Callable[..., OffPolicyAlgorithm]
    Factory/builder that assembles a complete discrete SAC algorithm instance
    (typically: Head + Core + OffPolicyAlgorithm driver).
SACDiscreteHead : type
    Policy "head" that owns neural networks used for discrete SAC:
    - actor: categorical policy over actions
    - critic: twin Q networks producing Q(s, ·) over all actions
    - critic_target: target copy used for stable bootstrap targets
SACDiscreteCore : type
    Update engine that implements discrete SAC loss/optimization logic:
    - critic regression to soft Bellman targets
    - actor update using entropy-regularized objective
    - optional temperature (alpha) auto-tuning
    - target network Polyak updates
"""

from __future__ import annotations

# Public builder / entrypoint
from .sac_discrete import sac_discrete

# Core components
from .head import SACDiscreteHead
from .core import SACDiscreteCore

__all__ = [
    "sac_discrete",
    "SACDiscreteHead",
    "SACDiscreteCore",
]
