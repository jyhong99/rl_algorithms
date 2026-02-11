"""
QRDQN
=======

This subpackage exposes a minimal, stable public API for the QR-DQN algorithm
implementation in this repository.

Public API
----------
qrdqn : callable
    Config-free builder that wires together:
      - QRDQNHead  : online/target quantile Q-networks (distributional critic)
      - QRDQNCore  : QR-DQN update engine (quantile regression TD update)
      - OffPolicyAlgorithm : replay buffer + scheduling driver (returned by builder)

QRDQNHead : nn.Module
    Head module that owns the online and target quantile networks and provides
    helper APIs such as:
      - quantiles(obs) / quantiles_target(obs)
      - q_values(obs) / q_values_target(obs)   (expected Q = mean over quantiles)
      - save/load, and Ray reconstruction spec (if enabled in head)

QRDQNCore : BaseCore
    Core update engine that performs gradient updates for the online quantile
    network, manages optimizer/scheduler state (via QLearningCore inheritance),
    and triggers periodic target network updates.

Notes
-----
- This file is intentionally small and only re-exports the public symbols.
- Keep imports lightweight to avoid side effects at package import time.
- The public API is controlled by ``__all__`` to support:
    * `from ... import *` hygiene
    * static analysis / IDE auto-completion
    * stable external imports across refactors
"""

from __future__ import annotations

from .core import QRDQNCore
from .head import QRDQNHead
from .qrdqn import qrdqn

__all__ = [
    "qrdqn",
    "QRDQNHead",
    "QRDQNCore",
]
