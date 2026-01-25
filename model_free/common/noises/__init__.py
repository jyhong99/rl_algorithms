"""
Exploration noise package for continuous-control RL.

This package provides:
- Action-independent noise (Gaussian, OU, Uniform)
- Action-dependent noise (GaussianAction, Multiplicative, ClippedGaussian)
- A unified factory function `build_noise`

Public API
----------
Base interfaces
- BaseNoise
- BaseActionNoise

Action-independent noises
- GaussianNoise
- OrnsteinUhlenbeckNoise
- UniformNoise

Action-dependent noises
- GaussianActionNoise
- MultiplicativeActionNoise
- ClippedGaussianActionNoise

Factory
- build_noise
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Base interfaces
# ---------------------------------------------------------------------
from .base_noise import BaseNoise, BaseActionNoise

# ---------------------------------------------------------------------
# Action-independent noises
# ---------------------------------------------------------------------
from .noises import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    UniformNoise,
)

# ---------------------------------------------------------------------
# Action-dependent noises
# ---------------------------------------------------------------------
from .action_noises import (
    GaussianActionNoise,
    MultiplicativeActionNoise,
    ClippedGaussianActionNoise,
)

# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
from .noise_builder import build_noise

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    # base
    "BaseNoise",
    "BaseActionNoise",
    # independent
    "GaussianNoise",
    "OrnsteinUhlenbeckNoise",
    "UniformNoise",
    # action-dependent
    "GaussianActionNoise",
    "MultiplicativeActionNoise",
    "ClippedGaussianActionNoise",
    # factory
    "build_noise",
]
