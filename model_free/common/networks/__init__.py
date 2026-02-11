"""
Networks
====================

This package provides reusable neural-network components commonly used in
reinforcement learning (RL) agents. It exposes a stable, import-friendly API
via re-exports so that downstream code can simply do::

    from model_free.networks import ContinuousPolicyNetwork, QNetwork

without importing internal submodules directly.

Modules
-------
distributions
    Action-distribution wrappers with a small unified interface used by policies.

    - :class:`~networks.distributions.BaseDistribution`
    - :class:`~networks.distributions.DiagGaussianDistribution`
    - :class:`~networks.distributions.SquashedDiagGaussianDistribution`
    - :class:`~networks.distributions.CategoricalDistribution`

policy_networks
    Concrete actor/policy networks.

    - :class:`~networks.policy_networks.DeterministicPolicyNetwork`
      (DDPG/TD3-style deterministic actor; tanh + optional bounds scaling)
    - :class:`~networks.policy_networks.ContinuousPolicyNetwork`
      (Gaussian actor; unsquashed for PPO/A2C or tanh-squashed for SAC)
    - :class:`~networks.policy_networks.DiscretePolicyNetwork`
      (Categorical actor for discrete actions)

q_networks
    DQN-family value networks for discrete-action RL.

    - :class:`~networks.q_networks.QNetwork` (optionally dueling)
    - :class:`~networks.q_networks.DoubleQNetwork`
    - :class:`~networks.q_networks.QuantileQNetwork` (QR-DQN; optionally dueling)
    - :class:`~networks.q_networks.RainbowQNetwork` (C51 + NoisyNet + dueling)

value_networks
    Critic/value networks for actor-critic methods (continuous-control and general).

    - :class:`~networks.value_networks.StateValueNetwork`          (V(s))
    - :class:`~networks.value_networks.StateActionValueNetwork`    (Q(s,a))
    - :class:`~networks.value_networks.DoubleStateActionValueNetwork` (twin Q)
    - :class:`~networks.value_networks.QuantileStateActionValueNetwork`
      (quantile ensemble; useful for TQC / quantile critics)

Notes
-----
- This file intentionally does **not** re-export low-level internals such as
  trunk building blocks or initialization helpers. Those are considered internal
  implementation details and should be imported from their respective modules
  only when necessary.
- `__all__` defines the public, stable surface area of this package.
  Anything not listed there may be changed without notice.

See Also
--------
base_networks
    Shared trunks, base classes for policies/critics/value networks, and NoisyNet blocks.
network_utils
    Helper utilities such as initialization helpers, dueling mixin, and bijectors.
"""

# =============================================================================
# Distributions
# =============================================================================
from .distributions import (
    BaseDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)

# =============================================================================
# Concrete policy networks
# =============================================================================
from .policy_networks import (
    ContinuousPolicyNetwork,
    DeterministicPolicyNetwork,
    DiscretePolicyNetwork,
)

# =============================================================================
# DQN-family Q networks
# =============================================================================
from .q_networks import (
    DoubleQNetwork,
    QNetwork,
    QuantileQNetwork,
    RainbowQNetwork,
)

# =============================================================================
# Actor-Critic value/critic networks
# =============================================================================
from .value_networks import (
    DoubleStateActionValueNetwork,
    QuantileStateActionValueNetwork,
    StateActionValueNetwork,
    StateValueNetwork,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # ---- distributions ----
    "BaseDistribution",
    "DiagGaussianDistribution",
    "SquashedDiagGaussianDistribution",
    "CategoricalDistribution",
    # ---- policy_networks ----
    "DeterministicPolicyNetwork",
    "ContinuousPolicyNetwork",
    "DiscretePolicyNetwork",
    # ---- q_networks ----
    "QNetwork",
    "DoubleQNetwork",
    "QuantileQNetwork",
    "RainbowQNetwork",
    # ---- value_networks ----
    "StateValueNetwork",
    "StateActionValueNetwork",
    "DoubleStateActionValueNetwork",
    "QuantileStateActionValueNetwork",
]
