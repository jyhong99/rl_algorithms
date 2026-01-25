from __future__ import annotations

from typing import Union

import torch as th

from .base_noise import BaseActionNoise
from ..utils.common_utils import to_tensor


class GaussianActionNoise(BaseActionNoise):
    """
    Gaussian noise scaled by action magnitude.

    Samples noise proportional to the absolute action value:
        noise = sigma * max(|action|, eps) * N(0, 1)

    This is useful for scale-aware exploration when different action dimensions
    have different magnitudes during training.

    Parameters
    ----------
    sigma : float, optional
        Global noise scale factor (>= 0), by default 0.1.
    eps : float, optional
        Minimum scale to avoid vanishing noise at action=0. Must be > 0,
        by default 1e-6.

    Notes
    -----
    - Returns a noise tensor with the same shape/device/dtype as `action`.
    - This class is stateless: `reset()` is a no-op (inherited).
    """

    def __init__(self, sigma: float = 0.1, eps: float = 1e-6) -> None:
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.sigma = float(sigma)
        self.eps = float(eps)

    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw action-dependent Gaussian noise.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor. Shape: (B, act_dim) or (act_dim,).

        Returns
        -------
        noise : torch.Tensor
            Noise tensor with the same shape as `action`.
        """
        if self.sigma == 0.0:
            return th.zeros_like(action)

        scale = action.abs().clamp_min(self.eps)
        return self.sigma * scale * th.randn_like(action)


class MultiplicativeActionNoise(BaseActionNoise):
    """
    Multiplicative Gaussian action noise.

    Samples:
        noise = sigma * action * N(0, 1)

    Parameters
    ----------
    sigma : float, optional
        Noise scale factor (>= 0), by default 0.1.

    Notes
    -----
    - If `action` is near zero, exploration noise vanishes accordingly.
    - Returns a noise tensor with the same shape/device/dtype as `action`.
    """

    def __init__(self, sigma: float = 0.1) -> None:
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw multiplicative Gaussian noise.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor. Shape: (B, act_dim) or (act_dim,).

        Returns
        -------
        noise : torch.Tensor
            Noise tensor with the same shape as `action`.
        """
        if self.sigma == 0.0:
            return th.zeros_like(action)
        return self.sigma * action * th.randn_like(action)


class ClippedGaussianActionNoise(BaseActionNoise):
    """
    Additive Gaussian noise with post-clipping to bounds.

    Produces:
        a_noisy = clip(action + sigma * N(0,1), low, high)
        noise   = a_noisy - action

    Parameters
    ----------
    sigma : float, optional
        Standard deviation scale (>= 0), by default 0.1.
    low : float or torch.Tensor, optional
        Lower bound(s). If tensor, must be broadcastable to `action`.
        By default -1.0.
    high : float or torch.Tensor, optional
        Upper bound(s). If tensor, must be broadcastable to `action`.
        By default 1.0.

    Notes
    -----
    - This returns the *effective* noise after clipping, not the raw Gaussian sample.
    - `low/high` tensors are moved to `action.device` and cast to `action.dtype`.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        low: Union[float, th.Tensor] = -1.0,
        high: Union[float, th.Tensor] = 1.0,
    ) -> None:
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)
        self.low = low
        self.high = high

    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw clipped additive Gaussian noise.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor. Shape: (B, act_dim) or (act_dim,).

        Returns
        -------
        noise : torch.Tensor
            Effective noise tensor after clipping. Same shape as `action`.
        """
        if self.sigma == 0.0:
            return th.zeros_like(action)

        low = to_tensor(self.low, device=action.device)
        high = to_tensor(self.high, device=action.device)

        # Optional sanity check: if both are floats, validate ordering once.
        if not isinstance(low, th.Tensor) and not isinstance(high, th.Tensor):
            if high <= low:
                raise ValueError(f"high must be > low, got low={low}, high={high}")

        raw = self.sigma * th.randn_like(action)
        noisy_action = (action + raw).clamp(low, high)
        return noisy_action - action