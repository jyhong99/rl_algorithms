from __future__ import annotations

from typing import Optional, Tuple, Union

import math
import torch as th

from .base_noise import BaseNoise
from ..utils.noise_utils import normalize_size


class GaussianNoise(BaseNoise):
    """
    i.i.d. Gaussian noise (stateless).

    Samples independent Gaussian noise at each call:
        noise ~ Normal(mu, sigma^2)

    Parameters
    ----------
    size : int or Tuple[int, ...]
        Output shape of the noise tensor (typically matches action shape).
    mu : float, optional
        Mean, by default 0.0.
    sigma : float, optional
        Standard deviation (>= 0), by default 0.2.
    device : str or torch.device, optional
        Device to generate noise on, by default "cpu".
    dtype : torch.dtype, optional
        Tensor dtype, by default torch.float32.

    Notes
    -----
    - Stateless: `reset()` is a no-op (inherited).
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        mu: float = 0.0,
        sigma: float = 0.2,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")

        self.size = normalize_size(size)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.device = device
        self.dtype = dtype

    def sample(self) -> th.Tensor:
        """
        Draw i.i.d. Gaussian noise.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape `size`.
        """
        if self.sigma == 0.0:
            return th.full(self.size, self.mu, device=self.device, dtype=self.dtype)
        return self.mu + self.sigma * th.randn(self.size, device=self.device, dtype=self.dtype)


class OrnsteinUhlenbeckNoise(BaseNoise):
    """
    Ornstein–Uhlenbeck (OU) temporally-correlated noise (stateful).

    Discrete-time approximation:
        x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, 1)

    Parameters
    ----------
    size : int or Tuple[int, ...]
        Noise tensor shape (typically matches action shape).
    mu : float, optional
        Long-run mean, by default 0.0.
    theta : float, optional
        Mean reversion speed (>= 0), by default 0.15.
    sigma : float, optional
        Volatility (>= 0), by default 0.2.
    dt : float, optional
        Discrete time step (> 0), by default 1e-2.
    x0 : Optional[float], optional
        Initial state value. If None, initializes at `mu`, by default None.
    return_copy : bool, optional
        If True, return a detached copy of internal state to prevent external
        in-place modification from corrupting the noise state, by default True.
    device : str or torch.device, optional
        Device for state/noise tensors, by default "cpu".
    dtype : torch.dtype, optional
        Tensor dtype, by default torch.float32.

    Notes
    -----
    - Stateful: call `reset()` at episode boundaries.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[float] = None,
        *,
        return_copy: bool = True,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        if theta < 0.0:
            raise ValueError(f"theta must be >= 0, got {theta}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        self.size = normalize_size(size)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.return_copy = bool(return_copy)
        self.device = device
        self.dtype = dtype

        self._x0 = self.mu if x0 is None else float(x0)
        self.state: th.Tensor
        self.reset()

    def sample(self) -> th.Tensor:
        """
        Draw one temporally-correlated OU noise sample.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape `size`.

        Notes
        -----
        - The internal state is updated in-place.
        - If `return_copy=True`, returns `state.clone()` to avoid aliasing.
        """
        if self.sigma == 0.0 and self.theta == 0.0:
            # Degenerate case: constant state.
            return self.state.clone() if self.return_copy else self.state

        eps = th.randn(self.size, device=self.device, dtype=self.dtype)
        dx = (self.theta * (self.mu - self.state) * self.dt) + (self.sigma * math.sqrt(self.dt) * eps)
        self.state.add_(dx)
        return self.state.clone() if self.return_copy else self.state

    def reset(self) -> None:
        """
        Reset the OU internal state to the initial value.
        """
        self.state = th.full(self.size, self._x0, device=self.device, dtype=self.dtype)


class UniformNoise(BaseNoise):
    """
    i.i.d. Uniform noise (stateless).

    Samples independent uniform noise:
        noise ~ Uniform(low, high)

    Parameters
    ----------
    size : int or Tuple[int, ...]
        Output shape of the noise tensor.
    low : float, optional
        Lower bound, by default -1.0.
    high : float, optional
        Upper bound, by default 1.0.
    device : str or torch.device, optional
        Device to generate noise on, by default "cpu".
    dtype : torch.dtype, optional
        Tensor dtype, by default torch.float32.

    Notes
    -----
    Stateless: primarily useful for debugging/sanity checks.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        low: float = -1.0,
        high: float = 1.0,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        if high <= low:
            raise ValueError(f"high must be > low, got low={low}, high={high}")

        self.size = normalize_size(size)
        self.low = float(low)
        self.high = float(high)
        self.device = device
        self.dtype = dtype

    def sample(self) -> th.Tensor:
        """
        Draw i.i.d. uniform noise.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape `size`.
        """
        u = th.rand(self.size, device=self.device, dtype=self.dtype)
        return (self.high - self.low) * u + self.low