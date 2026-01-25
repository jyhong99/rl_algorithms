"""
Networks package public API.

This package groups reusable neural-network components for RL:
- base_networks: shared trunks, base policy/critic/value nets, NoisyNet blocks
- network_utils : tanh bijector, init helpers, dueling mixin
- distributions : action distributions (Gaussian / squashed Gaussian / categorical)
- policy_networks: concrete actor policies (deterministic / Gaussian / categorical)
- q_networks     : DQN-family value networks (Q / DoubleQ / QuantileQ / Rainbow)
- value_networks : critic/value networks for actor-critic methods (V / Q / double / quantile ensemble)
"""

# =============================================================================
# Distributions
# =============================================================================
from .distributions import (
    BaseDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    CategoricalDistribution,
)

# =============================================================================
# Concrete policy networks
# =============================================================================
from .policy_networks import (
    DeterministicPolicyNetwork,
    ContinuousPolicyNetwork,
    DiscretePolicyNetwork,
)

# =============================================================================
# DQN-family Q networks
# =============================================================================
from .q_networks import (
    QNetwork,
    DoubleQNetwork,
    QuantileQNetwork,
    RainbowQNetwork,
)

# =============================================================================
# Actor-Critic value/critic networks
# =============================================================================
from .value_networks import (
    StateValueNetwork,
    StateActionValueNetwork,
    DoubleStateActionValueNetwork,
    QuantileStateActionValueNetwork,
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
