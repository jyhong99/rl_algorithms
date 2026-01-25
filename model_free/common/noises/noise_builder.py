from __future__ import annotations

from typing import Optional, Sequence, Union, Tuple

import torch as th

from .noises import GaussianNoise, OrnsteinUhlenbeckNoise, UniformNoise
from ..utils.noise_utils import normalize_kind, as_flat_bounds
from .action_noises import (
    GaussianActionNoise,
    ClippedGaussianActionNoise,
    MultiplicativeActionNoise,
)

# If you have a common base interface like NoiseProcess, prefer:
# from base_noise import NoiseProcess, BaseNoise, BaseActionNoise
# and then return Optional[Union[BaseNoise, BaseActionNoise]].
NoiseObj = Union[
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    UniformNoise,
    GaussianActionNoise,
    MultiplicativeActionNoise,
    ClippedGaussianActionNoise,
]


def build_noise(
    *,
    kind: Optional[str],
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # shared params
    noise_mu: float = 0.0,
    noise_sigma: float = 0.2,
    # OU params
    ou_theta: float = 0.15,
    ou_dt: float = 1e-2,
    # uniform params
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    # action-noise params
    action_noise_eps: float = 1e-6,
    action_noise_low: Optional[Union[float, Sequence[float]]] = None,
    action_noise_high: Optional[Union[float, Sequence[float]]] = None,
    dtype: th.dtype = th.float32,
) -> Optional[NoiseObj]:
    """
    Factory to construct exploration noise objects.

    Parameters
    ----------
    kind : Optional[str]
        Noise type identifier. If None/"none"/"" -> returns None.
        Supported kinds (aliases allowed):
          - "gaussian"
          - "ou" / "ornstein_uhlenbeck"
          - "uniform"
          - "gaussian_action"
          - "multiplicative" / "multiplicative_action"
          - "clipped_gaussian" / "clipped_gaussian_action"
    action_dim : int
        Action dimension (> 0). Used to set noise tensor size for action-independent noises.
    device : str or torch.device, optional
        Device for noise tensors/state, by default "cpu".
    noise_mu : float, optional
        Mean for Gaussian/OU, by default 0.0.
    noise_sigma : float, optional
        Stddev/scale parameter (>= 0), by default 0.2.
    ou_theta : float, optional
        OU mean reversion speed (>= 0), by default 0.15.
    ou_dt : float, optional
        OU time step (> 0), by default 1e-2.
    uniform_low, uniform_high : float, optional
        Uniform bounds, requires high > low.
    action_noise_eps : float, optional
        Epsilon for GaussianActionNoise scale clamp (> 0), by default 1e-6.
    action_noise_low, action_noise_high : Optional[float or Sequence[float]]
        Required for clipped Gaussian action noise. If sequences, length must equal action_dim.
    dtype : torch.dtype, optional
        Dtype for created tensors, by default torch.float32.

    Returns
    -------
    noise : Optional[NoiseObj]
        Noise instance or None.

    Raises
    ------
    ValueError
        If parameters are invalid or `kind` is unknown.
    """
    nt = normalize_kind(kind)
    if nt is None:
        return None

    if action_dim <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")

    if noise_sigma < 0.0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")

    size: Tuple[int, ...] = (int(action_dim),)

    if nt == "gaussian":
        return GaussianNoise(
            size=size,
            mu=float(noise_mu),
            sigma=float(noise_sigma),
            device=device,
            dtype=dtype,
        )

    if nt in ("ou", "ornstein_uhlenbeck"):
        if ou_theta < 0.0:
            raise ValueError(f"ou_theta must be >= 0, got {ou_theta}")
        if ou_dt <= 0.0:
            raise ValueError(f"ou_dt must be > 0, got {ou_dt}")
        return OrnsteinUhlenbeckNoise(
            size=size,
            mu=float(noise_mu),
            theta=float(ou_theta),
            sigma=float(noise_sigma),
            dt=float(ou_dt),
            device=device,
            dtype=dtype,
        )

    if nt == "uniform":
        if uniform_high <= uniform_low:
            raise ValueError(
                f"uniform_high must be > uniform_low, got low={uniform_low}, high={uniform_high}"
            )
        return UniformNoise(
            size=size,
            low=float(uniform_low),
            high=float(uniform_high),
            device=device,
            dtype=dtype,
        )

    if nt == "gaussian_action":
        return GaussianActionNoise(sigma=float(noise_sigma), eps=float(action_noise_eps))

    if nt in ("multiplicative", "multiplicative_action"):
        return MultiplicativeActionNoise(sigma=float(noise_sigma))

    if nt in ("clipped_gaussian", "clipped_gaussian_action"):
        if action_noise_low is None or action_noise_high is None:
            raise ValueError(
                "clipped_gaussian requires action_noise_low and action_noise_high "
                "(scalar or sequence of length action_dim)."
            )
        low_t = as_flat_bounds(
            action_noise_low, action_dim=action_dim, device=device, dtype=dtype, name="action_noise_low"
        )
        high_t = as_flat_bounds(
            action_noise_high, action_dim=action_dim, device=device, dtype=dtype, name="action_noise_high"
        )

        # If both are scalar tensors, validate ordering early.
        if low_t.ndim == 0 and high_t.ndim == 0 and (high_t <= low_t).item():
            raise ValueError(f"action_noise_high must be > action_noise_low, got {high_t.item()} <= {low_t.item()}")

        return ClippedGaussianActionNoise(sigma=float(noise_sigma), low=low_t, high=high_t)

    raise ValueError(
        f"Unknown exploration noise kind='{kind}'. "
        "Supported: None|'gaussian'|'ou'|'uniform'|'gaussian_action'|'multiplicative'|'clipped_gaussian'."
    )