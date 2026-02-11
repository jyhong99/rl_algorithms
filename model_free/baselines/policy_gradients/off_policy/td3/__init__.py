"""
TD3
=======

This subpackage implements Twin Delayed Deep Deterministic Policy Gradient (TD3)
as a clean, modular stack built from three layers:

- Head (:class:`~.head.TD3Head`)
  Owns neural networks and inference-time utilities:
  - deterministic actor and twin critics
  - target actor and target critics
  - optional action bounds handling
  - (optional) exploration noise application during action selection
  - Ray worker factory support via a JSON-safe export spec

- Core (:class:`~.core.TD3Core`)
  Owns update/optimization logic:
  - critic regression every update call
  - delayed actor updates (policy delay)
  - TD3 target policy smoothing
  - Polyak target network updates
  - logging scalars and (optionally) PER TD-error feedback

- Builder (:func:`~.td3.td3`)
  Factory that wires together:
  ``TD3Head`` + ``TD3Core`` + :class:`model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`.

Public API
----------
td3 : callable
    Factory that returns a fully assembled TD3 OffPolicyAlgorithm.
TD3Head : type
    Actor/critic networks + target networks + inference helpers.
TD3Core : type
    TD3 loss computation + optimizer steps + target updates.

Notes
-----
- This file is intentionally lightweight: it only re-exports the public symbols
  and documents the package-level contract.
- All heavy logic lives in ``head.py``, ``core.py``, and ``td3.py``.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Public re-exports
# -----------------------------------------------------------------------------
from .td3 import td3
from .head import TD3Head
from .core import TD3Core

__all__ = [
    "td3",
    "TD3Head",
    "TD3Core",
]
