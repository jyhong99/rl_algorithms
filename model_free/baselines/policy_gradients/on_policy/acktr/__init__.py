"""
ACKTR
=======

This subpackage provides the components required to build and train an
ACKTR-style (Actor Critic using Kronecker-Factored Trust Region) agent for
**continuous** action spaces.

Public API
----------
acktr : callable
    High-level builder that wires together the ACKTR head, core, and the
    :class:`model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`.

ACKTRHead : torch.nn.Module
    Actor-critic network container:
    - actor: diagonal Gaussian policy network (typically unsquashed)
    - critic: state-value network V(s)

ACKTRCore : ActorCriticCore
    Update engine implementing one ACKTR optimization step:
    - A2C-style policy/value/entropy losses
    - K-FAC optimizer wiring (damping, trust region, factor update cadence, etc.)
    - optional AMP
    - optional global gradient clipping
    - optional learning-rate schedulers (via the base core)

Notes
-----
- This implementation is **continuous-only**. For discrete action spaces, use a
  categorical policy head/core pair.
- ACKTR's defining feature primarily lives in the **optimizer/core** (K-FAC /
  natural-gradient approximation). The head is a standard on-policy actor-critic
  container.
- Only the symbols listed in ``__all__`` are considered part of the stable,
  public import surface of this package.

Examples
--------
Construct an algorithm instance::

    from model_free.algos.acktr import acktr

    algo = acktr(obs_dim=obs_dim, action_dim=action_dim, device="cuda:0")

Or import individual components::

    from model_free.algos.acktr import ACKTRHead, ACKTRCore
"""

from __future__ import annotations

# Re-export public symbols for a clean user-facing import surface.
from .acktr import acktr
from .core import ACKTRCore
from .head import ACKTRHead

__all__ = [
    "acktr",
    "ACKTRHead",
    "ACKTRCore",
]
