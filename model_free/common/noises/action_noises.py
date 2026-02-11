from __future__ import annotations

from typing import Union

import torch as th

from .base_noise import BaseActionNoise
from ..utils.common_utils import _to_tensor


# =============================================================================
# Gaussian noise variants for continuous-action exploration
# =============================================================================

class GaussianActionNoise(BaseActionNoise):
    """
    Action-dependent Gaussian noise scaled by action magnitude.

    This noise model scales exploration per action dimension using the current
    action magnitude:

        noise = sigma * max(|action|, eps) * N(0, 1)

    where:
    - `sigma` controls the global noise intensity
    - `eps` prevents the scale from collapsing to zero when action is near 0

    Parameters
    ----------
    sigma : float, optional
        Global noise scale factor (must be >= 0), by default 0.1.
    eps : float, optional
        Minimum scale to avoid vanishing noise at action = 0
        (must be > 0), by default 1e-6.

    Attributes
    ----------
    sigma : float
        Global noise scale factor.
    eps : float
        Minimum multiplicative scale applied to |action|.

    Returns
    -------
    torch.Tensor
        Noise tensor with the same shape, dtype, and device as the input action.

    Notes
    -----
    - This class is stateless; `reset()` is typically a no-op (inherited).
    - This is useful when different action coordinates naturally live at
      different magnitudes (scale-aware exploration).
    - If `sigma == 0`, `sample()` returns zeros.

    Examples
    --------
    >>> noise = GaussianActionNoise(sigma=0.2)
    >>> a = th.tensor([[0.1, 2.0]])
    >>> n = noise.sample(a)  # same shape as a
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
            Action tensor of shape (A,) or (B, A).

        Returns
        -------
        noise : torch.Tensor
            Noise tensor with the same shape as `action`.

        Notes
        -----
        The per-element noise scale is:
            scale = clamp_min(|action|, eps)
        """
        if self.sigma == 0.0:
            return th.zeros_like(action)

        scale = action.abs().clamp_min(self.eps)
        return self.sigma * scale * th.randn_like(action)


class MultiplicativeActionNoise(BaseActionNoise):
    """
    Multiplicative Gaussian action noise.

    This noise model is purely multiplicative:

        noise = sigma * action * N(0, 1)

    Parameters
    ----------
    sigma : float, optional
        Noise scale factor (must be >= 0), by default 0.1.

    Attributes
    ----------
    sigma : float
        Noise scale factor.

    Returns
    -------
    torch.Tensor
        Noise tensor with the same shape, dtype, and device as the input action.

    Notes
    -----
    - If the action is near zero, noise magnitude also vanishes.
      This can be desirable (stable near 0) or undesirable (insufficient exploration),
      depending on the task.
    - If `sigma == 0`, `sample()` returns zeros.
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
            Action tensor of shape (A,) or (B, A).

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

    This noise model adds Gaussian noise in action space and then clips the
    resulting noisy action to user-provided bounds:

        a_noisy = clip(action + sigma * N(0, 1), low, high)
        noise   = a_noisy - action

    Importantly, this returns the *effective* noise after clipping (i.e., the
    difference between the clipped noisy action and the original action).

    Parameters
    ----------
    sigma : float, optional
        Standard deviation scale (must be >= 0), by default 0.1.
    low : float or torch.Tensor, optional
        Lower bound(s). If a tensor, must be broadcastable to `action`.
        By default -1.0.
    high : float or torch.Tensor, optional
        Upper bound(s). If a tensor, must be broadcastable to `action`.
        By default 1.0.

    Attributes
    ----------
    sigma : float
        Noise scale factor.
    low : float or torch.Tensor
        Stored lower bound(s) (not necessarily on the same device as actions).
    high : float or torch.Tensor
        Stored upper bound(s).
    _bounds_are_floats : bool
        Whether `low` and `high` were provided as Python floats.

    Notes
    -----
    - This returns *effective noise*, not the raw Gaussian sample.
      If clipping is active, the returned noise may have a smaller magnitude than
      `sigma * N(0,1)` due to saturation at bounds.
    - Bounds are converted/moved to the action device at sampling time via `_to_tensor`.
      Dtype casting behavior depends on `_to_tensor`; typically you want bounds in the
      same dtype as `action`.
    - If `sigma == 0`, `sample()` returns zeros.
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

        # If bounds are scalar floats, validate ordering once.
        self._bounds_are_floats = isinstance(low, (float, int)) and isinstance(high, (float, int))
        if self._bounds_are_floats and float(high) <= float(low):
            raise ValueError(f"high must be > low, got low={low}, high={high}")

    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw clipped additive Gaussian noise.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor of shape (A,) or (B, A).

        Returns
        -------
        noise : torch.Tensor
            Effective noise tensor after clipping, same shape as `action`.

        Notes
        -----
        Let `raw = sigma * N(0,1)` and `a_noisy = clamp(action + raw, low, high)`.
        This function returns:
            noise = a_noisy - action
        """
        if self.sigma == 0.0:
            return th.zeros_like(action)

        # Convert bounds to tensors on the correct device.
        # If you want strict dtype matching, ensure `_to_tensor` supports dtype
        # or cast explicitly: low = low.to(dtype=action.dtype).
        low = _to_tensor(self.low, device=action.device)
        high = _to_tensor(self.high, device=action.device)

        raw = self.sigma * th.randn_like(action)
        a_noisy = (action + raw).clamp(low, high)
        return a_noisy - action
