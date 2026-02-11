"""
Noises
====================

This package provides reusable exploration-noise processes commonly used in
continuous-action RL (e.g., DDPG/TD3/SAC variants), including both:

- **Action-independent noise**: noise sampled without conditioning on the current
  action (e.g., i.i.d. Gaussian, Ornstein–Uhlenbeck, Uniform).
- **Action-dependent noise**: noise sampled as a function of the deterministic
  action (e.g., multiplicative noise, clipped additive noise).

A small factory function :func:`build_noise` is also provided to construct noise
objects from a string identifier and hyperparameters.

Extended Summary
----------------
Typical usage patterns include:

- **Action-independent** (historically common in DDPG):
  `a_noisy = a_det + noise.sample()`
- **Action-dependent** (scale-aware or bounded exploration):
  `a_noisy = a_det + noise.sample(a_det)`

The base interfaces define the minimal contracts for these two categories.
Concrete noise classes implement the sampling behavior and optional lifecycle
hook :meth:`reset` (useful for stateful processes like OU noise).

Public API
----------
Base interfaces
    - :class:`BaseNoise`
    - :class:`BaseActionNoise`

Action-independent noises
    - :class:`GaussianNoise`
    - :class:`OrnsteinUhlenbeckNoise`
    - :class:`UniformNoise`

Action-dependent noises
    - :class:`GaussianActionNoise`
    - :class:`MultiplicativeActionNoise`
    - :class:`ClippedGaussianActionNoise`

Factory
    - :func:`build_noise`

Notes
-----
- Action-independent noise objects typically own their device/dtype configuration
  and return samples directly on that device.
- Action-dependent noise objects typically infer device/dtype from the provided
  action tensor at sampling time, though bounded variants may store bounds
  internally.

Examples
--------
Create OU noise and reset at episode boundaries:

>>> from model_free.noise import OrnsteinUhlenbeckNoise
>>> ou = OrnsteinUhlenbeckNoise(size=3, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2)
>>> ou.reset()
>>> n = ou.sample()

Use the factory to build a noise object from configuration:

>>> from model_free.noise import build_noise
>>> noise = build_noise(kind="gaussian_action", action_dim=8, noise_sigma=0.1)

See Also
--------
build_noise : Factory function for constructing noise objects.
"""

from __future__ import annotations

# =============================================================================
# Base interfaces
# =============================================================================
from .base_noise import BaseActionNoise, BaseNoise

# =============================================================================
# Action-independent noises
# =============================================================================
from .noises import GaussianNoise, OrnsteinUhlenbeckNoise, UniformNoise

# =============================================================================
# Action-dependent noises
# =============================================================================
from .action_noises import (
    ClippedGaussianActionNoise,
    GaussianActionNoise,
    MultiplicativeActionNoise,
)

# =============================================================================
# Factory
# =============================================================================
from .noise_builder import build_noise


__all__ = [
    # ---- base ----
    "BaseNoise",
    "BaseActionNoise",
    # ---- action-independent ----
    "GaussianNoise",
    "OrnsteinUhlenbeckNoise",
    "UniformNoise",
    # ---- action-dependent ----
    "GaussianActionNoise",
    "MultiplicativeActionNoise",
    "ClippedGaussianActionNoise",
    # ---- factory ----
    "build_noise",
]
