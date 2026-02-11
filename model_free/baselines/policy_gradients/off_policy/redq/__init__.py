"""
REDQ
=======

This subpackage exposes a minimal, stable surface area for external users while
keeping internal module structure (head/core/builder) flexible.

Exports
-------
redq : callable
    Factory function that constructs a complete REDQ off-policy algorithm by
    composing:
      - :class:`~.head.REDQHead` (networks + action sampling / inference)
      - :class:`~.core.REDQCore` (losses + optimizers + target updates)
      - :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
        (replay buffer + update scheduling + PER plumbing)

REDQHead : type
    Policy "head" that owns model components:
      - stochastic actor (SAC-style squashed Gaussian)
      - critic ensemble (online)
      - target critic ensemble (Polyak-updated, frozen)

REDQCore : type
    Update engine that owns learning logic:
      - critic ensemble regression to REDQ/SAC targets
      - actor update with entropy regularization
      - optional automatic temperature tuning (alpha)
      - TD-error computation for prioritized replay (PER), if enabled

Notes
-----
- This file intentionally re-exports only the primary entrypoints.
  Importing from the package root is recommended for downstream code stability:
    >>> from ...redq import redq, REDQHead, REDQCore
- The public names are enumerated in ``__all__`` to support:
  - `from package import *` hygiene
  - static analysis / IDE autocomplete consistency
"""

from __future__ import annotations

from .core import REDQCore
from .head import REDQHead
from .redq import redq

__all__ = [
    "redq",
    "REDQHead",
    "REDQCore",
]
