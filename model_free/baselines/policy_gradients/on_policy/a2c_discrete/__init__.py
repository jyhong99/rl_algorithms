"""
Discrete A2C
=======

This subpackage provides the components required to build and train an
Advantage Actor-Critic (A2C) agent for **discrete** action spaces using a
categorical policy.

Public API
----------
a2c_discrete : callable
    High-level builder that wires together the discrete A2C head, core, and the
    :class:`model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`.

A2CDiscreteHead : torch.nn.Module
    Actor-critic network container for discrete actions:
    - actor: categorical policy network producing logits over `n_actions`
    - critic: state-value network V(s)

A2CDiscreteCore : ActorCriticCore
    Update engine implementing one A2C optimization step for categorical policies:
    - policy/value/entropy losses
    - optimizer steps (actor + critic)
    - optional AMP
    - global gradient clipping
    - optional learning-rate schedulers (via the base core)

Notes
-----
- This implementation is **discrete-only**. For continuous action spaces, use the
  continuous A2C package (Gaussian policy head/core).
- Only the symbols listed in ``__all__`` are considered part of the stable,
  public import surface of this package.

Examples
--------
Construct an A2C algorithm instance::

    from model_free.algos.a2c_discrete import a2c_discrete

    algo = a2c_discrete(obs_dim=obs_dim, n_actions=n_actions, device="cuda:0")

Or import individual components::

    from model_free.algos.a2c_discrete import A2CDiscreteHead, A2CDiscreteCore
"""

from __future__ import annotations

# Re-export public symbols for a clean user-facing import surface.
from .a2c_discrete import a2c_discrete
from .core import A2CDiscreteCore
from .head import A2CDiscreteHead

__all__ = [
    "a2c_discrete",
    "A2CDiscreteHead",
    "A2CDiscreteCore",
]
