"""
Discrete VPG
=======

This subpackage defines the public API for **Vanilla Policy Gradient (VPG)** in
**discrete-action** environments.

Conceptual structure
--------------------
The discrete VPG implementation is split into three layers:

- ``vpg_discrete``:
    High-level builder that assembles the full algorithm (head + core + driver)
    into a ready-to-train :class:`model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`.

- ``VPGDiscreteHead``:
    Discrete actor-critic *network container*.
    - Actor: categorical policy :math:`\\pi(a\\mid s)` (logits -> Categorical distribution)
    - Critic (optional): value baseline :math:`V(s)` controlled by ``use_baseline``

- ``VPGDiscreteCore``:
    Update engine implementing:
    - policy-gradient loss (REINFORCE/VPG),
    - optional entropy regularization,
    - optional critic regression update (baseline enabled only),
    - optimizer/scheduler stepping and checkpointable optimizer state.

Public API policy
-----------------
Only the names listed in ``__all__`` are considered stable public imports for this
subpackage. Internal helpers and implementation details should be imported from
their defining modules directly when needed.

Examples
--------
>>> from model_free.algos.vpg_discrete import vpg_discrete
>>> algo = vpg_discrete(obs_dim=8, n_actions=4, device="cpu", use_baseline=True)
>>> # algo.setup(env)  # depends on your OnPolicyAlgorithm interface
"""

from __future__ import annotations

from .vpg_discrete import vpg_discrete
from .head import VPGDiscreteHead
from .core import VPGDiscreteCore

__all__ = [
    "vpg_discrete",
    "VPGDiscreteHead",
    "VPGDiscreteCore",
]
