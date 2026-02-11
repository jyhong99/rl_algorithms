"""
PPO
=======

This subpackage exposes a small, stable public API for Proximal Policy
Optimization (PPO) with **continuous** action spaces.

Public API
----------
ppo : callable
    Config-free builder that constructs a complete on-policy training stack:
    ``PPOHead`` + ``PPOCore`` + :class:`model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`.

PPOHead : torch.nn.Module
    Actor-critic network container for continuous control:
    - Actor : diagonal Gaussian policy π(a|s) (typically unsquashed)
    - Critic: state-value function V(s)

PPOCore : ActorCriticCore
    PPO update engine:
    - clipped policy objective (PPO-Clip)
    - optional value clipping
    - entropy bonus
    - optimizer steps for actor and critic
    - optional target-KL early stopping (minibatch-level signal)
    - optional schedulers via the base core

Notes
-----
- This implementation is **continuous-only** (Gaussian policy). Discrete PPO
  requires a categorical head/core variant.
- PPO-specific logic lives primarily in the **core** (ratio/clip/KL/value-clip).
  The head is intentionally a thin actor-critic container.
- Only symbols listed in ``__all__`` are considered part of the supported,
  public import surface.

Examples
--------
Build an algorithm instance::

    from model_free.algos.ppo import ppo
    algo = ppo(obs_dim=obs_dim, action_dim=action_dim, device="cuda:0")

Import individual components::

    from model_free.algos.ppo import PPOHead, PPOCore
"""

from __future__ import annotations

# Re-export user-facing symbols for a clean import path.
from .core import PPOCore
from .head import PPOHead
from .ppo import ppo

__all__ = [
    "ppo",
    "PPOHead",
    "PPOCore",
]
