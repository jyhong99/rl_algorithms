from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch as th
from torch.distributions import Categorical, Normal

from model_free.common.utils.network_utils import TanhBijector


# =============================================================================
# Numerical constants
# =============================================================================

LOG_STD_MIN = -20.0
"""
Minimum clamp value for log standard deviation.

Notes
-----
Used to prevent extremely small standard deviations that can cause
numerical instability (e.g., exploding log-probabilities).
"""

LOG_STD_MAX = 2.0
"""
Maximum clamp value for log standard deviation.

Notes
-----
Used to prevent overly large standard deviations that can destabilize training.
"""

EPS = 1e-6
"""
Small constant for numerical stability (tanh inverse / boundary clamping).
"""


# =============================================================================
# Base interface
# =============================================================================

class BaseDistribution(ABC):
    """
    Base interface for policy action distributions.

    This module provides a lightweight abstraction layer so that policy networks
    can expose a unified distribution API regardless of action type
    (continuous vs. discrete).

    Contract
    --------
    Implementations must follow these shape conventions:

    - `sample()`:
        Returns an action tensor with batch dimension.
        Typical shapes:
          * continuous: (B, A)
          * discrete  : (B, 1)  (this repo uses 2D storage for discrete actions)
    - `rsample()`:
        Reparameterized sample (pathwise gradient), if supported.
        Discrete distributions typically do not support this.
    - `log_prob(action, ...)`:
        Must return shape (B, 1). For multidimensional actions, log-probabilities
        should be summed over the last dimension.
    - `entropy()`:
        Should return (B, 1) when available.
    - `mode()`:
        Deterministic action (mean / tanh(mean) / argmax).

    Notes
    -----
    - This interface is intentionally minimal. Algorithms may require extra
      distribution-specific methods (e.g., KL divergence for TRPO/PPO),
      which can be added by concrete subclasses.
    """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Sample an action without reparameterization.

        Returns
        -------
        torch.Tensor
            Sampled action tensor (batch-first).
        """
        raise NotImplementedError

    def rsample(self, *args, **kwargs) -> th.Tensor:
        """
        Sample an action with reparameterization (pathwise gradient), if supported.

        Raises
        ------
        NotImplementedError
            If the concrete distribution does not support reparameterized sampling.

        Notes
        -----
        - Continuous distributions (Normal-based) typically support `rsample`.
        - Discrete distributions (Categorical) do not support `rsample` in PyTorch.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support rsample().")

    @abstractmethod
    def log_prob(self, action: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        Compute log π(a|s) for given actions.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor. Shape depends on distribution type.

        Returns
        -------
        torch.Tensor
            Log-probability tensor of shape (B, 1).
        """
        raise NotImplementedError

    @abstractmethod
    def entropy(self) -> th.Tensor:
        """
        Compute entropy of the distribution.

        Returns
        -------
        torch.Tensor
            Entropy tensor of shape (B, 1) when available.
        """
        raise NotImplementedError

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Deterministic action.

        Returns
        -------
        torch.Tensor
            Deterministic action tensor (batch-first).
        """
        raise NotImplementedError


# =============================================================================
# Continuous: Diagonal Gaussian (unsquashed)
# =============================================================================

class DiagGaussianDistribution(BaseDistribution):
    """
    Diagonal Gaussian distribution for continuous actions (no squashing).

    This distribution models each action dimension as an independent Normal:
        a ~ Normal(mean, std), with diagonal covariance.

    Parameters
    ----------
    mean : torch.Tensor
        Mean tensor of shape (B, A).
    log_std : torch.Tensor
        Log standard deviation tensor of shape (B, A). Values are clamped to
        [LOG_STD_MIN, LOG_STD_MAX] for numerical stability.

    Attributes
    ----------
    mean : torch.Tensor
        Mean tensor, shape (B, A).
    log_std : torch.Tensor
        Clamped log standard deviation, shape (B, A).
    std : torch.Tensor
        Standard deviation, shape (B, A).
    dist : torch.distributions.Normal
        Underlying PyTorch Normal distribution.

    Notes
    -----
    - `log_prob(action)` returns summed log-prob over action dims, shape (B, 1).
    - Suitable for unbounded action spaces, or when clipping is handled externally.
    """

    def __init__(self, mean: th.Tensor, log_std: th.Tensor) -> None:
        self.mean = mean
        self.log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = th.exp(self.log_std)
        self.dist = Normal(self.mean, self.std)

    def sample(self) -> th.Tensor:
        """
        Sample actions (non-reparameterized).

        Returns
        -------
        torch.Tensor
            Sampled action tensor of shape (B, A).
        """
        return self.dist.sample()

    def rsample(self) -> th.Tensor:
        """
        Sample actions with reparameterization.

        Returns
        -------
        torch.Tensor
            Reparameterized action tensor of shape (B, A).
        """
        return self.dist.rsample()

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        """
        Compute log-probability for given actions.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor of shape (B, A) or (A,).

        Returns
        -------
        torch.Tensor
            Summed log-probability over action dims, shape (B, 1).

        Notes
        -----
        - If `action` is 1D (A,), it is treated as a single batch element.
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.dist.log_prob(action).sum(dim=-1, keepdim=True)

    def entropy(self) -> th.Tensor:
        """
        Compute entropy.

        Returns
        -------
        torch.Tensor
            Summed entropy over action dims, shape (B, 1).
        """
        return self.dist.entropy().sum(dim=-1, keepdim=True)

    def mode(self) -> th.Tensor:
        """
        Deterministic action.

        Returns
        -------
        torch.Tensor
            Mean action (no squashing), shape (B, A).
        """
        return self.mean

    def kl(self, other: "DiagGaussianDistribution") -> th.Tensor:
        """
        Compute KL divergence KL(self || other).

        Parameters
        ----------
        other : DiagGaussianDistribution
            Reference distribution (often the old policy distribution).

        Returns
        -------
        torch.Tensor
            Summed KL divergence over action dims, shape (B, 1).

        Raises
        ------
        TypeError
            If `other` is not a DiagGaussianDistribution.
        """
        if not isinstance(other, DiagGaussianDistribution):
            raise TypeError(f"KL requires DiagGaussianDistribution, got {type(other)}")

        kl_per_dim = th.distributions.kl_divergence(self.dist, other.dist)  # (B, A)
        return kl_per_dim.sum(dim=-1, keepdim=True)


# =============================================================================
# Continuous: Squashed Diagonal Gaussian via tanh
# =============================================================================

class SquashedDiagGaussianDistribution(BaseDistribution):
    """
    Squashed diagonal Gaussian distribution using a tanh bijector.

    Sampling procedure:
        z ~ Normal(mean, std)
        a = tanh(z)

    Parameters
    ----------
    mean : torch.Tensor
        Pre-squash mean tensor, shape (B, A).
    log_std : torch.Tensor
        Pre-squash log-std tensor, shape (B, A). Clamped for stability.
    eps : float, optional
        Numerical stability constant used when clamping actions for inversion
        and inside the bijector (default: EPS).

    Attributes
    ----------
    mean : torch.Tensor
        Pre-squash mean, shape (B, A).
    log_std : torch.Tensor
        Clamped pre-squash log-std, shape (B, A).
    std : torch.Tensor
        Pre-squash std, shape (B, A).
    dist : torch.distributions.Normal
        Base Normal distribution over pre-squash variable z.
    bijector : TanhBijector
        Bijector implementing tanh forward/inverse and log-det-Jacobian correction.
    eps : float
        Stability constant.

    Notes
    -----
    Change-of-variables for log-probability:
        log π(a|s) = log p(z) - log |det(da/dz)|
    where a = tanh(z).

    Practical stability tips:
    - Prefer passing `pre_tanh` returned by `rsample(return_pre_tanh=True)`
      to avoid atanh computations.
    - `entropy()` returns the base Gaussian entropy (not the exact squashed entropy).
      This is a common approximation used in SAC-style implementations.
    """

    def __init__(self, mean: th.Tensor, log_std: th.Tensor, *, eps: float = EPS) -> None:
        self.mean = mean
        self.log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = th.exp(self.log_std)
        self.dist = Normal(self.mean, self.std)
        self.bijector = TanhBijector(float(eps))
        self.eps = float(eps)

    def sample(self) -> th.Tensor:
        """
        Sample squashed actions (non-reparameterized).

        Returns
        -------
        torch.Tensor
            Squashed action tensor in [-1, 1], shape (B, A).
        """
        z = self.dist.sample()
        return self.bijector.forward(z)

    def rsample(
        self, *, return_pre_tanh: bool = False
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Sample squashed actions with reparameterization.

        Parameters
        ----------
        return_pre_tanh : bool, optional
            If True, also return the pre-squash variable z (default: False).

        Returns
        -------
        action : torch.Tensor
            Squashed action tensor in [-1, 1], shape (B, A).
        pre_tanh : torch.Tensor, optional
            Pre-squash z tensor, shape (B, A). Returned iff return_pre_tanh=True.
        """
        z = self.dist.rsample()
        a = self.bijector.forward(z)
        return (a, z) if return_pre_tanh else a

    def log_prob(self, action: th.Tensor, pre_tanh: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Compute log-probability with tanh change-of-variables correction.

        Parameters
        ----------
        action : torch.Tensor
            Squashed action tensor in [-1, 1], shape (B, A) or (A,).
        pre_tanh : torch.Tensor, optional
            Pre-squash z tensor of shape (B, A). If not provided, it is computed
            via inverse tanh (atanh) from `action`.

        Returns
        -------
        torch.Tensor
            Log-probability tensor of shape (B, 1).

        Notes
        -----
        - If `pre_tanh` is None, we compute z = atanh(a). To avoid overflow near ±1,
          actions are clamped to [-1+eps, 1-eps] before inversion.
        - The correction term subtracts log|det J| where J = d(tanh(z))/dz.
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        if pre_tanh is None:
            a = th.clamp(action, -1.0 + self.eps, 1.0 - self.eps)
            pre_tanh = self.bijector.inverse(a)

        # Base log-prob under z ~ Normal(mean, std)
        logp_z = self.dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)

        # Change-of-variables correction: sum log|d tanh(z) / dz|
        log_abs_det_jac = self.bijector.log_prob_correction(pre_tanh).sum(dim=-1, keepdim=True)
        return logp_z - log_abs_det_jac

    def entropy(self) -> th.Tensor:
        """
        Approximate entropy (base Gaussian entropy).

        Returns
        -------
        torch.Tensor
            Summed base Gaussian entropy, shape (B, 1).

        Notes
        -----
        This is not the exact entropy of the squashed distribution.
        """
        return self.dist.entropy().sum(dim=-1, keepdim=True)

    def mode(self) -> th.Tensor:
        """
        Deterministic action.

        Returns
        -------
        torch.Tensor
            Squashed mean action tanh(mean), shape (B, A).
        """
        return self.bijector.forward(self.mean)


# =============================================================================
# Discrete: Categorical
# =============================================================================

class CategoricalDistribution(BaseDistribution):
    """
    Categorical distribution wrapper for discrete actions.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized logits tensor of shape (B, K), where K is number of actions.

    Attributes
    ----------
    logits : torch.Tensor
        Stored logits, shape (B, K).
    dist : torch.distributions.Categorical
        Underlying PyTorch categorical distribution.

    Notes
    -----
    - `rsample()` is not supported for categorical distributions (no pathwise gradient).
    - This wrapper returns actions as shape (B, 1) to align with replay-buffer designs
      that store actions consistently as 2D tensors across action types.
      If you prefer actions of shape (B,), modify `sample()` and `mode()` accordingly.
    """

    def __init__(self, logits: th.Tensor) -> None:
        self.logits = logits
        self.dist = Categorical(logits=logits)

    def sample(self) -> th.Tensor:
        """
        Sample discrete action indices.

        Returns
        -------
        torch.Tensor
            Sampled action indices, shape (B, 1), dtype long.
        """
        a = self.dist.sample()  # (B,)
        return a.unsqueeze(-1)  # (B, 1)

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        """
        Compute log-probability of discrete actions.

        Parameters
        ----------
        action : torch.Tensor
            Action indices. Supported shapes:
            - (B,)   : batch of indices
            - (B, 1) : batch with trailing singleton dim
            - ()     : scalar (will be treated as a batch of size 1)
            dtype long recommended.

        Returns
        -------
        torch.Tensor
            Log-probability tensor of shape (B, 1).
        """
        if action.dim() == 2 and action.size(-1) == 1:
            action = action.squeeze(-1)  # (B,)
        elif action.dim() == 0:
            action = action.view(1)      # (1,)
        return self.dist.log_prob(action).unsqueeze(-1)

    def entropy(self) -> th.Tensor:
        """
        Compute entropy.

        Returns
        -------
        torch.Tensor
            Entropy tensor of shape (B, 1).
        """
        return self.dist.entropy().unsqueeze(-1)

    def mode(self) -> th.Tensor:
        """
        Deterministic action = argmax over probabilities.

        Returns
        -------
        torch.Tensor
            Action indices, shape (B, 1), dtype long.
        """
        a = th.argmax(self.dist.probs, dim=-1)  # (B,)
        return a.unsqueeze(-1)
