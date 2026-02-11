"""
Policies
====================

This package provides the high-level RL "policy stack":

- **Heads**: neural-network containers that implement action selection and
  distribution/value evaluation (duck-typed interfaces used by algorithms).
- **Cores**: update engines that own optimizers/schedulers and implement
  `update_from_batch(batch)` (gradient steps).
- **Algorithms**: env-facing drivers that manage data collection (rollout/replay)
  and call cores for learning updates.

Public API
----------
Algorithms
- BaseAlgorithm
- BasePolicyAlgorithm
- OnPolicyAlgorithm
- OffPolicyAlgorithm

Cores
- BaseCore
- ActorCriticCore
- QLearningCore

Heads
- BaseHead
- OnPolicyContinuousActorCriticHead
- OnPolicyDiscreteActorCriticHead
- OffPolicyContinuousActorCriticHead
- OffPolicyDiscreteActorCriticHead
- DeterministicActorCriticHead
- QLearningHead

Notes
-----
- The design is intentionally **duck-typed**: heads/cores need to expose
  expected methods/attributes, not inherit concrete base classes.
- Checkpointing is handled by BaseAlgorithm via `head.state_dict()` and
  `core.state_dict()` when available.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Algorithms (env-facing drivers)
# -----------------------------------------------------------------------------
from .base_policy import BaseAlgorithm, BasePolicyAlgorithm
from .on_policy_algorithm import OnPolicyAlgorithm
from .off_policy_algorithm import OffPolicyAlgorithm

# -----------------------------------------------------------------------------
# Cores (update engines)
# -----------------------------------------------------------------------------
from .base_core import BaseCore, ActorCriticCore, QLearningCore

# -----------------------------------------------------------------------------
# Heads (network containers / action interfaces)
# -----------------------------------------------------------------------------
from .base_head import (
    BaseHead,
    OnPolicyContinuousActorCriticHead,
    OnPolicyDiscreteActorCriticHead,
    OffPolicyContinuousActorCriticHead,
    OffPolicyDiscreteActorCriticHead,
    DeterministicActorCriticHead,
    QLearningHead,
)

__all__ = [
    # Algorithms
    "BaseAlgorithm",
    "BasePolicyAlgorithm",
    "OnPolicyAlgorithm",
    "OffPolicyAlgorithm",
    # Cores
    "BaseCore",
    "ActorCriticCore",
    "QLearningCore",
    # Heads
    "BaseHead",
    "OnPolicyContinuousActorCriticHead",
    "OnPolicyDiscreteActorCriticHead",
    "OffPolicyContinuousActorCriticHead",
    "OffPolicyDiscreteActorCriticHead",
    "DeterministicActorCriticHead",
    "QLearningHead",
]
