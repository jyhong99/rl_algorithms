"""
model_free

Top-level package initializer.

This file exposes the stable public API intended for end-users, while keeping the
internal module layout flexible.

The canonical public API definitions live in:
    model_free.common.policies

Usage
-----
from model_free import OnPolicyAlgorithm, OffPolicyAlgorithm
from model_free import BaseHead, RolloutBuffer
"""

from __future__ import annotations

# Re-export the intended public API from the internal "public surface" module.
# This keeps heavy/optional imports centralized and avoids duplicating __all__.
from .common.policies import *  # noqa: F401,F403

# Ensure `model_free.__all__` matches the canonical list.
from .common.policies import __all__  # noqa: F401
