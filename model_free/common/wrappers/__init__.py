"""
Wrappers
====================

This subpackage contains lightweight environment wrappers that modify or
augment an underlying RL environment. Typical use-cases include:

- Online observation / reward normalization
- Action rescaling or clipping for continuous-control (Box-like) spaces
- Compatibility shims across Gym and Gymnasium variants

Public API
----------
NormalizeWrapper
    Online normalization wrapper that maintains running statistics for
    observations and/or (discounted-return) rewards, and optionally handles
    Box-like action rescaling/clipping and time-limit truncation semantics.

Notes
-----
- Keep this module importable in minimal environments. Wrappers should avoid
  importing heavy optional dependencies at import time.
- Re-export wrapper classes here to provide a stable import path, e.g.::

      from your_pkg.wrappers import NormalizeWrapper
"""

from __future__ import annotations

# =============================================================================
# Public re-exports
# =============================================================================
from .normalize_wrapper import NormalizeWrapper

__all__ = [
    "NormalizeWrapper",
]