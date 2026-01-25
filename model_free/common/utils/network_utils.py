from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import math
import torch as th
import torch.nn as nn


# =============================================================================
# Squashed Gaussian utilities
# =============================================================================
class TanhBijector:
    """
    Tanh bijector for squashed Gaussian policies.

    Implements the transformation:
        a = tanh(z),  z ∈ R, a ∈ (-1, 1)

    Commonly used in continuous-control policies (e.g., SAC) where the policy
    samples `z` from an unbounded Gaussian and squashes it into action bounds.

    Parameters
    ----------
    epsilon : float, optional
        Numerical stability constant used in:
        - inverse clamp margin (avoid atanh(±1))
        - Jacobian term: log(1 - tanh(z)^2 + epsilon)
        by default 1e-6.

    Notes
    -----
    Change-of-variables (per-dimension correction):
        log |da/dz| = log(1 - tanh(z)^2)

    Full squashed log-prob:
        log π(a|s) = log p(z) - Σ log(1 - tanh(z)^2 + eps)
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = float(epsilon)

    def forward(self, z: th.Tensor) -> th.Tensor:
        """
        Apply tanh squashing.

        Parameters
        ----------
        z : torch.Tensor
            Pre-squash tensor, shape (..., A).

        Returns
        -------
        a : torch.Tensor
            Squashed tensor, shape (..., A), range (-1, 1).
        """
        return th.tanh(z)

    def inverse(self, a: th.Tensor) -> th.Tensor:
        """
        Invert tanh squashing (recover z from a).

        Parameters
        ----------
        a : torch.Tensor
            Squashed tensor, nominally in [-1, 1], shape (..., A).

        Returns
        -------
        z : torch.Tensor
            Pre-squash tensor, shape (..., A).

        Notes
        -----
        - We clamp `a` away from ±1 to avoid atanh overflow.
        - Clamp margin uses both dtype epsilon and the user-provided epsilon.
        """
        # Prefer a practical margin, not the extremely tiny dtype eps alone.
        finfo_eps = th.finfo(a.dtype).eps if a.is_floating_point() else 1e-12
        margin = max(self.epsilon, float(finfo_eps))

        a = a.clamp(min=-1.0 + margin, max=1.0 - margin)
        return self.atanh(a)

    @staticmethod
    def atanh(a: th.Tensor) -> th.Tensor:
        """
        Numerically stable inverse hyperbolic tangent.

        Implements:
            atanh(a) = 0.5 * (log1p(a) - log1p(-a))

        Parameters
        ----------
        a : torch.Tensor
            Values strictly in (-1, 1), shape (..., A).

        Returns
        -------
        z : torch.Tensor
            Pre-squash tensor, shape (..., A).
        """
        return 0.5 * (th.log1p(a) - th.log1p(-a))

    def log_prob_correction(self, z: th.Tensor) -> th.Tensor:
        """
        Per-dimension log-Jacobian correction for a = tanh(z).

        Parameters
        ----------
        z : torch.Tensor
            Pre-squash tensor, shape (..., A).

        Returns
        -------
        correction : torch.Tensor
            Per-dimension correction term, shape (..., A).

        Notes
        -----
        Caller typically does:
            correction.sum(dim=-1, keepdim=True)
        """
        t = th.tanh(z)
        return th.log(1.0 - t * t + self.epsilon)


# =============================================================================
# Network utils
# =============================================================================
def validate_hidden_sizes(hidden_sizes: Sequence[int]) -> Tuple[int, ...]:
    """
    Validate MLP hidden layer specification.

    Parameters
    ----------
    hidden_sizes : Sequence[int]
        Hidden layer sizes (e.g., (64, 64) or [256, 256]).

    Returns
    -------
    hidden_sizes : Tuple[int, ...]
        Validated sizes as a tuple of positive ints.

    Raises
    ------
    ValueError
        If empty or contains non-positive entries.
    """
    hs = tuple(int(h) for h in hidden_sizes)
    if len(hs) == 0:
        raise ValueError("hidden_sizes must have at least one layer (e.g., (64, 64)).")
    if any(h <= 0 for h in hs):
        raise ValueError(f"hidden_sizes must be positive integers, got: {hs}")
    return hs


def make_weights_init(
    init_type: str = "xavier_uniform",
    gain: float = 1.0,
    bias: float = 0.0,
    kaiming_a: float = math.sqrt(5.0),
) -> Callable[[nn.Module], None]:
    """
    Create a module initializer compatible with `nn.Module.apply()`.

    Parameters
    ----------
    init_type : str, optional
        Initialization scheme identifier (case-insensitive), by default "xavier_uniform".

        Supported for `nn.Linear`:
        - "xavier_uniform"
        - "xavier_normal"
        - "kaiming_uniform"
        - "kaiming_normal"
        - "orthogonal"
        - "normal"   (std = gain)
        - "uniform"  (range = [-gain, +gain])
    gain : float, optional
        Gain used by Xavier/Orthogonal initializers (and as std/range for normal/uniform),
        by default 1.0.
    bias : float, optional
        Constant used to initialize linear biases (if present), by default 0.0.
    kaiming_a : float, optional
        Negative slope parameter `a` for Kaiming initialization, by default sqrt(5.0).

    Returns
    -------
    init_fn : Callable[[nn.Module], None]
        Function intended to be used as:
            model.apply(init_fn)

    Notes
    -----
    - Only `nn.Linear` modules are initialized; other modules are ignored.
    - Centralizing init helps reproducibility and controlled ablations.
    """
    name = str(init_type).lower().strip()
    gain = float(gain)
    bias = float(bias)
    kaiming_a = float(kaiming_a)

    def init_fn(module: nn.Module) -> None:
        """Initialize a single module in-place (called by `apply`)."""
        if not isinstance(module, nn.Linear):
            return

        if name == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif name == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif name == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, a=kaiming_a)
        elif name == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, a=kaiming_a)
        elif name == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=gain)
        elif name == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=gain)
        elif name == "uniform":
            nn.init.uniform_(module.weight, -gain, gain)
        else:
            raise ValueError(f"Unknown init_type: {init_type!r}")

        if module.bias is not None:
            nn.init.constant_(module.bias, bias)

    return init_fn


# =============================================================================
# Dueling combination
# =============================================================================
class DuelingMixin:
    """
    Utility for dueling combination.

    Combines value and advantage streams as:
        Q = V + (A - mean(A, over action dimension))
    """

    @staticmethod
    def combine_dueling(v: th.Tensor, a: th.Tensor, *, mean_dim: int = -1) -> th.Tensor:
        """
        Parameters
        ----------
        v : torch.Tensor
            Value stream. Common shapes:
            - (B, 1) for standard dueling Q
            - (B, N, 1) for quantile dueling
            - (B, 1, K) for C51 logits dueling
        a : torch.Tensor
            Advantage stream:
            - (B, A)
            - (B, N, A)
            - (B, A, K)
        mean_dim : int, optional
            Dimension over which to mean-reduce advantages (typically action dim),
            by default -1.

        Returns
        -------
        q : torch.Tensor
            Combined tensor matching `a` broadcast shape.
        """
        return v + (a - a.mean(dim=mean_dim, keepdim=True))


def ensure_batch(x: Any, device: th.device | str) -> th.Tensor:
    x_t = x if isinstance(x, th.Tensor) else th.as_tensor(x)

    # float policy면 float로
    if not x_t.is_floating_point():
        x_t = x_t.float()

    # device 정렬 (self.device가 있거나 파라미터 device에 맞춤)
    try:
        x_t = x_t.to(device)
    except Exception:
        pass

    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)

    return x_t