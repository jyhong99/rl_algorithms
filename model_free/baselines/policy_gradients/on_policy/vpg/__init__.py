"""
VPG
=======

This subpackage provides a small, explicit public API for Vanilla Policy Gradient (VPG)
in continuous-control settings.

Public interface
----------------
The implementation is organized into three main entry points:

- ``vpg``:
    High-level builder that wires together the head (networks), core (update engine),
    and the on-policy driver algorithm (rollout + advantage/return computation + update loop).

- ``VPGHead``:
    Actor-critic network container for continuous actions:
      * Actor: diagonal Gaussian policy π(a|s) (unsquashed)
      * Critic: optional value baseline V(s) controlled by ``use_baseline``

- ``VPGCore``:
    Update engine implementing:
      * policy-gradient loss (REINFORCE / VPG),
      * optional entropy regularization,
      * optional critic regression update when a baseline is enabled.

Notes
-----
- Only the names listed in ``__all__`` are considered stable public API.
- Internal helpers and implementation details should be imported directly from
  their defining modules if needed (e.g., ``.core`` / ``.head``).
"""

from __future__ import annotations

from .vpg import vpg
from .head import VPGHead
from .core import VPGCore

__all__ = [
    "vpg",
    "VPGHead",
    "VPGCore",
]
