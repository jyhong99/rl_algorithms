from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch as th
from torch.distributions import Categorical, Normal

from model_free.common.utils.network_utils import TanhBijector


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


# =============================================================================
# Base interface
# =============================================================================
class BaseDistribution(ABC):
    """
    Base interface for policy action distributions.

    Concrete subclasses implement either continuous or discrete action
    distributions and expose a unified interface used by policies/agents.

    Contract
    --------
    - `sample()`   : non-reparameterized sampling (no pathwise gradient).
    - `rsample()`  : reparameterized sampling (pathwise gradient), if supported.
    - `log_prob()` : must return shape (B, 1) (summed over action dims if needed).
    - `entropy()`  : return shape (B, 1) when available.
    - `mode()`     : deterministic action (mean/tanh(mean)/argmax).

    Notes
    -----
    This interface is intentionally minimal; algorithms may require extra methods
    (e.g., KL for PPO) that are distribution-specific.
    """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """Sample an action without reparameterization."""
        raise NotImplementedError

    def rsample(self, *args, **kwargs) -> th.Tensor:
        """
        Sample an action with reparameterization.

        Notes
        -----
        - Not all distributions support rsample (e.g., categorical).
        - Subclasses should override if supported.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support rsample().")

    @abstractmethod
    def log_prob(self, action: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        Compute log π(a|s) for given actions.

        Returns
        -------
        log_prob : torch.Tensor
            Log-probability, shape (B, 1).
        """
        raise NotImplementedError

    @abstractmethod
    def entropy(self) -> th.Tensor:
        """
        Returns
        -------
        entropy : torch.Tensor
            Entropy, shape (B, 1) when available.
        """
        raise NotImplementedError

    @abstractmethod
    def mode(self) -> th.Tensor:
        """Deterministic action (mean/argmax)."""
        raise NotImplementedError


# =============================================================================
# Continuous: Diagonal Gaussian (unsquashed)
# =============================================================================
class DiagGaussianDistribution(BaseDistribution):
    """
    Diagonal Gaussian distribution (continuous actions, no squashing).

    Parameters
    ----------
    mean : torch.Tensor
        Mean tensor, shape (B, A).
    log_std : torch.Tensor
        Log standard deviation tensor, shape (B, A). Clamped to
        [LOG_STD_MIN, LOG_STD_MAX] for numerical stability.

    Notes
    -----
    - `log_prob(action)` returns summed log-prob over action dims, shape (B, 1).
    - Suitable for unbounded actions, or when clipping is handled externally.
    """

    def __init__(self, mean: th.Tensor, log_std: th.Tensor) -> None:
        self.mean = mean
        self.log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = th.exp(self.log_std)
        self.dist = Normal(self.mean, self.std)

    def sample(self) -> th.Tensor:
        """Sample action, shape (B, A)."""
        return self.dist.sample()

    def rsample(self) -> th.Tensor:
        """Reparameterized sample, shape (B, A)."""
        return self.dist.rsample()

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        action : torch.Tensor
            Action tensor, shape (B, A) or (A,).

        Returns
        -------
        log_prob : torch.Tensor
            Summed log-prob, shape (B, 1).
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.dist.log_prob(action).sum(dim=-1, keepdim=True)

    def entropy(self) -> th.Tensor:
        """
        Returns
        -------
        entropy : torch.Tensor
            Summed entropy, shape (B, 1).
        """
        return self.dist.entropy().sum(dim=-1, keepdim=True)

    def mode(self) -> th.Tensor:
        """Deterministic action = mean, shape (B, A)."""
        return self.mean

    def kl(self, other: "DiagGaussianDistribution") -> th.Tensor:
        """
        Compute KL divergence KL(self || other).

        Parameters
        ----------
        other : DiagGaussianDistribution
            Reference distribution (often the old policy).

        Returns
        -------
        kl : torch.Tensor
            Summed KL divergence, shape (B, 1).
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
    Squashed diagonal Gaussian distribution via tanh.

    Generates:
        z ~ Normal(mean, std)
        a = tanh(z)

    Parameters
    ----------
    mean : torch.Tensor
        Pre-squash mean, shape (B, A).
    log_std : torch.Tensor
        Pre-squash log std, shape (B, A). Clamped for stability.
    eps : float, optional
        Small constant used by bijector for numerical stability, by default EPS.

    Notes
    -----
    - `log_prob(a, pre_tanh=None)` applies change-of-variables correction:
        log π(a|s) = log p(z) - sum log|d tanh(z) / dz|
    - Prefer passing `pre_tanh` returned by `rsample(return_pre_tanh=True)`
      to avoid inverse tanh and improve stability.
    - `entropy()` returns base Gaussian entropy (not the exact squashed entropy).
    """

    def __init__(self, mean: th.Tensor, log_std: th.Tensor, *, eps: float = EPS) -> None:
        self.mean = mean
        self.log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = th.exp(self.log_std)
        self.dist = Normal(self.mean, self.std)
        self.bijector = TanhBijector(float(eps))
        self.eps = float(eps)

    def sample(self) -> th.Tensor:
        """Non-reparameterized squashed sample, shape (B, A)."""
        z = self.dist.sample()
        return self.bijector.forward(z)

    def rsample(
        self, *, return_pre_tanh: bool = False
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Parameters
        ----------
        return_pre_tanh : bool, optional
            If True, also return pre-squash z, by default False.

        Returns
        -------
        action : torch.Tensor
            Squashed action, shape (B, A).
        pre_tanh : torch.Tensor, optional
            Pre-squash z, shape (B, A). Returned iff return_pre_tanh=True.
        """
        z = self.dist.rsample()
        a = self.bijector.forward(z)
        return (a, z) if return_pre_tanh else a

    def log_prob(self, action: th.Tensor, pre_tanh: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Parameters
        ----------
        action : torch.Tensor
            Squashed action in [-1, 1], shape (B, A) or (A,).
        pre_tanh : Optional[torch.Tensor], optional
            Pre-squash z, shape (B, A). If None, inverse(action) is used.

        Returns
        -------
        log_prob : torch.Tensor
            Log-prob with tanh correction, shape (B, 1).
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # If caller didn't provide pre_tanh, we must invert tanh(a).
        # Clamp actions slightly away from ±1 to avoid atanh overflow.
        if pre_tanh is None:
            a = th.clamp(action, -1.0 + self.eps, 1.0 - self.eps)
            pre_tanh = self.bijector.inverse(a)

        logp_z = self.dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        corr = self.bijector.log_prob_correction(pre_tanh).sum(dim=-1, keepdim=True)
        return logp_z - corr

    def entropy(self) -> th.Tensor:
        """
        Returns
        -------
        entropy : torch.Tensor
            Base Gaussian entropy (summed), shape (B, 1).
        """
        return self.dist.entropy().sum(dim=-1, keepdim=True)

    def mode(self) -> th.Tensor:
        """Deterministic squashed action = tanh(mean), shape (B, A)."""
        return self.bijector.forward(self.mean)


# =============================================================================
# Discrete: Categorical
# =============================================================================
class CategoricalDistribution(BaseDistribution):
    """
    Categorical distribution (discrete actions).

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized logits, shape (B, A) where A is number of actions.

    Notes
    -----
    - `rsample()` is not supported for categorical distributions (no pathwise gradient).
    - This wrapper returns actions as shape (B, 1) for consistency with many
      buffer designs that store actions as 2D tensors regardless of action type.
      If you prefer (B,), change `sample()` and `mode()` accordingly.
    """

    def __init__(self, logits: th.Tensor) -> None:
        self.logits = logits
        self.dist = Categorical(logits=logits)

    def sample(self) -> th.Tensor:
        """
        Sample discrete action indices.

        Returns
        -------
        action : torch.Tensor
            Action indices, shape (B, 1), dtype long.
        """
        a = self.dist.sample()  # (B,)
        return a.unsqueeze(-1)  # (B,1)

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        action : torch.Tensor
            Discrete action indices, shape (B,), (B,1), or scalar ().
            dtype long recommended.

        Returns
        -------
        log_prob : torch.Tensor
            Log-prob, shape (B, 1).
        """
        if action.dim() == 2 and action.size(-1) == 1:
            action = action.squeeze(-1)
        elif action.dim() == 0:
            action = action.view(1)
        return self.dist.log_prob(action).unsqueeze(-1)

    def entropy(self) -> th.Tensor:
        """
        Returns
        -------
        entropy : torch.Tensor
            Entropy, shape (B, 1).
        """
        return self.dist.entropy().unsqueeze(-1)

    def mode(self) -> th.Tensor:
        """
        Deterministic action = argmax(probs).

        Returns
        -------
        action : torch.Tensor
            Action indices, shape (B, 1).
        """
        a = th.argmax(self.dist.probs, dim=-1)  # (B,)
        return a.unsqueeze(-1)