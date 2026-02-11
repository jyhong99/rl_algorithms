"""
TQC
=======

This subpackage exposes the public API for the Truncated Quantile Critics (TQC)
algorithm implementation.

TQC is an off-policy actor-critic method that combines:
- a SAC-style stochastic actor (squashed Gaussian policy), and
- an ensemble of distributional critics that predict quantiles of the return.

The key TQC idea is *target truncation*: when building the bootstrap target
distribution, the largest quantiles are dropped after sorting the flattened
ensemble quantiles. This reduces overestimation bias while retaining a
distributional learning signal.

Public objects
--------------
tqc : callable
    Factory function that wires together:
    - :class:`~.head.TQCHead` (networks + action sampling utilities),
    - :class:`~.core.TQCCore` (losses, optimizers, target updates, alpha tuning),
    - :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
      (replay buffer + scheduling + PER integration).

TQCHead : type
    Neural-network container (actor + quantile critic ensemble + target critic),
    plus serialization and (optionally) Ray worker factory spec.

TQCCore : type
    Update engine implementing:
    - quantile regression loss (Huber quantile loss),
    - TQC truncation for target quantiles,
    - SAC-style actor update and optional temperature (alpha) learning,
    - periodic Polyak updates for target critic.

Notes
-----
- This module is intended to be the stable import surface for TQC:
    ``from ...tqc import tqc, TQCHead, TQCCore``.
- Keep ``__all__`` aligned with the symbols you consider part of the public API.
"""

from __future__ import annotations

# Re-export the factory and main components so users can import from the package
# root rather than individual modules.
from .core import TQCCore
from .head import TQCHead
from .tqc import tqc

__all__ = [
    "tqc",
    "TQCHead",
    "TQCCore",
]
