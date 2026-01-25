"""
model_free

Public API surface for the library.

Design goals
------------
- Provide stable, minimal imports for end-users.
- Avoid importing heavy/optional dependencies eagerly (Ray, tqdm, etc.).
- Keep internal module layout flexible while preserving external API.

Usage examples
--------------
from model_free import OnPolicyAlgorithm, OffPolicyAlgorithm
from model_free import BaseHead, OnPolicyActorCriticHead, QLearningHead
"""

from __future__ import annotations

from importlib import metadata as _metadata

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
try:
    __version__ = _metadata.version("model_free")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


# -----------------------------------------------------------------------------
# Algorithms (public)
# -----------------------------------------------------------------------------
from .base_policy import BaseAlgorithm, BasePolicyAlgorithm
from .on_policy_algorithm import OnPolicyAlgorithm
from .off_policy_algorithm import OffPolicyAlgorithm

# -----------------------------------------------------------------------------
# Heads (public)
# -----------------------------------------------------------------------------
from .base_head import (
    BaseHead,
    OnPolicyContinuousActorCriticHead,
    OffPolicyContinuousActorCriticHead,
    DeterministicActorCriticHead,
    QLearningHead,
)

# -----------------------------------------------------------------------------
# Cores (public)
# -----------------------------------------------------------------------------
from .base_core import BaseCore, ActorCriticCore, QLearningCore

# -----------------------------------------------------------------------------
# Buffers (public)
# -----------------------------------------------------------------------------
from ..buffers.rollout_buffer import RolloutBuffer
from ..buffers.replay_buffer import ReplayBuffer
from ..buffers.prioritized_replay_buffer import PrioritizedReplayBuffer

# -----------------------------------------------------------------------------
# Common utilities (selectively exported)
# -----------------------------------------------------------------------------
from ..utils.common_utils import (
    to_tensor, 
    to_numpy, 
    to_flat_np, 
    to_action_np, 
    to_scalar, 
    is_scalar_like, 
    require_scalar_like, 
    require_mapping, 
    infer_shape
)

# -----------------------------------------------------------------------------
# Optional: Ray hook types (guarded)
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from ..utils.ray_utils import PolicyFactorySpec
except Exception:  # pragma: no cover
    PolicyFactorySpec = None  # type: ignore


# -----------------------------------------------------------------------------
# __all__ (controls `from model_free import *`)
# -----------------------------------------------------------------------------
__all__ = [
    # version
    "__version__",

    # algorithms
    "BaseAlgorithm",
    "BasePolicyAlgorithm",
    "OnPolicyAlgorithm",
    "OffPolicyAlgorithm",

    # heads
    "BaseHead",
    "OnPolicyContinuousActorCriticHead",
    "OffPolicyContinuousActorCriticHead",
    "DeterministicActorCriticHead",
    "QLearningHead",

    # cores
    "BaseCore",
    "ActorCriticCore",
    "QLearningCore",

    # buffers
    "RolloutBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",

    # utils
    "to_tensor",
    "to_numpy",
    "to_flat_np",
    "to_action_np",
    "to_scalar",
    "is_scalar_like",
    "require_scalar_like",
    "require_mapping",
    "infer_shape",

    # optional
    "PolicyFactorySpec",
]
