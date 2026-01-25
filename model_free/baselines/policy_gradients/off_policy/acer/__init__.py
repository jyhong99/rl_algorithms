"""
ACER (Discrete) package public API.

Exports
-------
- acer: builder that returns OffPolicyAlgorithm
- ACERHead: discrete actor + (double) Q critic + target critic head
- ACERCore: ACER update engine
"""

from __future__ import annotations

from .acer import acer
from .head import ACERHead
from .core import ACERCore

__all__ = [
    "acer",
    "ACERHead",
    "ACERCore",
]
