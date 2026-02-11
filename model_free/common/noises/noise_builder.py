from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch as th

from .action_noises import (
    ClippedGaussianActionNoise,
    GaussianActionNoise,
    MultiplicativeActionNoise,
)
from .noises import GaussianNoise, OrnsteinUhlenbeckNoise, UniformNoise
from ..utils.noise_utils import _as_flat_bounds, _normalize_kind


# =============================================================================
# Types
# =============================================================================

NoiseObj = Union[
    # action-independent
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    UniformNoise,
    # action-dependent
    GaussianActionNoise,
    MultiplicativeActionNoise,
    ClippedGaussianActionNoise,
]


# =============================================================================
# Factory
# =============================================================================

_SUPPORTED_KINDS = (
    "gaussian",
    "ou",
    "ornstein_uhlenbeck",
    "uniform",
    "gaussian_action",
    "multiplicative",
    "multiplicative_action",
    "clipped_gaussian",
    "clipped_gaussian_action",
)


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
    Construct an exploration-noise object.

    This is a small factory that builds either:
    - **action-independent noise** (returns a tensor of shape (action_dim,))
      such as Gaussian / OU / Uniform, or
    - **action-dependent noise** (expects an action tensor at sampling time)
      such as multiplicative or clipped Gaussian action noise.

    Parameters
    ----------
    kind : Optional[str]
        Noise type identifier. If `None`, empty, or an alias of "none",
        this returns ``None`` (no exploration noise).
        Supported kinds (aliases may be normalized by `_normalize_kind`):

        Action-independent
            - ``"gaussian"``
            - ``"ou"`` or ``"ornstein_uhlenbeck"``
            - ``"uniform"``

        Action-dependent
            - ``"gaussian_action"``
            - ``"multiplicative"`` or ``"multiplicative_action"``
            - ``"clipped_gaussian"`` or ``"clipped_gaussian_action"``

    action_dim : int
        Action dimension ``A`` (must be > 0).

        Notes
        -----
        - For action-independent noises, this determines the returned sample size.
        - For action-dependent noises, this is used only for validating bounds
          (e.g., clipped Gaussian with per-dimension low/high).

    device : str or torch.device, optional
        Device used for action-independent noise buffers/state (default: "cpu").
        For clipped action noise, bounds will be constructed on this device.

    noise_mu : float, optional
        Mean for Gaussian/OU noise (default: 0.0).

    noise_sigma : float, optional
        Standard deviation / scale parameter (must be >= 0) (default: 0.2).

    ou_theta : float, optional
        OU mean reversion speed (must be >= 0) (default: 0.15).

    ou_dt : float, optional
        OU time step (must be > 0) (default: 1e-2).

    uniform_low, uniform_high : float, optional
        Uniform bounds (require ``uniform_high > uniform_low``).

    action_noise_eps : float, optional
        Epsilon used by :class:`~GaussianActionNoise` to prevent vanishing scale at
        action=0 (must be > 0) (default: 1e-6).

    action_noise_low, action_noise_high : Optional[float or Sequence[float]]
        Bounds for clipped Gaussian action noise.
        Must be provided iff kind is ``"clipped_gaussian"`` / ``"clipped_gaussian_action"``.

        - If scalar: treated as a shared bound for all action dimensions.
        - If a sequence: must have length ``action_dim``.

    dtype : torch.dtype, optional
        Dtype used for tensors created inside the noise objects (default: torch.float32).

    Returns
    -------
    Optional[NoiseObj]
        The constructed noise instance, or ``None`` if `kind` indicates no noise.

    Raises
    ------
    ValueError
        If:
        - `action_dim` <= 0
        - `noise_sigma` < 0
        - OU parameters are invalid
        - uniform bounds are invalid
        - clipped bounds are missing/invalid
        - `kind` is unknown

    Notes
    -----
    Device and dtype rules
        - Action-independent noises typically create and return tensors on `device`
          with `dtype`.
        - Action-dependent noises typically infer device/dtype from the input action
          at sampling time (except clipped bounds, which are validated/constructed here).

    Kind normalization
        `_normalize_kind(kind)` is expected to:
        - return ``None`` for None/"none"/"" (no noise)
        - return a canonical kind string for aliases (e.g., "OU" -> "ou")
    """
    nt = _normalize_kind(kind)
    if nt is None:
        return None

    if action_dim <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")
    if noise_sigma < 0.0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")

    size: Tuple[int, ...] = (int(action_dim),)

    # -------------------------------------------------------------------------
    # Action-independent noises
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Action-dependent noises
    # -------------------------------------------------------------------------
    if nt == "gaussian_action":
        if action_noise_eps <= 0.0:
            raise ValueError(f"action_noise_eps must be > 0, got {action_noise_eps}")
        return GaussianActionNoise(sigma=float(noise_sigma), eps=float(action_noise_eps))

    if nt in ("multiplicative", "multiplicative_action"):
        return MultiplicativeActionNoise(sigma=float(noise_sigma))

    if nt in ("clipped_gaussian", "clipped_gaussian_action"):
        if action_noise_low is None or action_noise_high is None:
            raise ValueError(
                "clipped_gaussian requires action_noise_low and action_noise_high "
                "(scalar or sequence of length action_dim)."
            )

        low_t = _as_flat_bounds(
            action_noise_low,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
            name="action_noise_low",
        )
        high_t = _as_flat_bounds(
            action_noise_high,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
            name="action_noise_high",
        )

        # Ordering check:
        # - If both are scalar bounds, validate once.
        # - If per-dimension bounds, you might optionally validate elementwise.
        if low_t.numel() == 1 and high_t.numel() == 1 and (high_t <= low_t).item():
            raise ValueError(
                f"action_noise_high must be > action_noise_low, got {high_t.item()} <= {low_t.item()}"
            )

        return ClippedGaussianActionNoise(sigma=float(noise_sigma), low=low_t, high=high_t)

    raise ValueError(
        f"Unknown exploration noise kind={kind!r} (normalized={nt!r}). "
        f"Supported kinds include: {', '.join(_SUPPORTED_KINDS)}."
    )
