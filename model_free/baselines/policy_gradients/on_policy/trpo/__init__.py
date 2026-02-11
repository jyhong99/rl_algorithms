"""
TRPO
=======

This subpackage provides a minimal, explicit public API for Trust Region Policy
Optimization (TRPO) in continuous-control settings.

Public interface
----------------
The TRPO implementation is split into three main pieces:

- ``trpo``:
    A high-level builder that wires together the head (networks), core (update
    engine), and the on-policy driver algorithm (rollout + GAE + update loop).

- ``TRPOHead``:
    Actor-critic network container (Gaussian policy + state-value baseline),
    including optional Ray worker reconstruction utilities.

- ``TRPOCore``:
    TRPO update engine implementing:
      * critic regression update (supervised value fitting),
      * natural-gradient step via conjugate gradient on Fisher-vector products,
      * KL-constrained backtracking line search.

Notes
-----
- Only the names in ``__all__`` are considered stable public API.
- Internal helpers and implementation details should be imported from their
  respective modules (e.g., ``.core`` / ``.head``) rather than relying on
  implicit exports.
"""

from __future__ import annotations

from .trpo import trpo
from .head import TRPOHead
from .core import TRPOCore

__all__ = [
    "trpo",
    "TRPOHead",
    "TRPOCore",
]
