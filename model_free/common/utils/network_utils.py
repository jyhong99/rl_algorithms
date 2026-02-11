from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union
import math

import torch as th
import torch.nn as nn


# =============================================================================
# Squashed Gaussian utilities
# =============================================================================
class TanhBijector:
    """
    Tanh bijector for squashed Gaussian policies.

    This bijector implements the element-wise transformation:

        a = tanh(z),   z ∈ R,   a ∈ (-1, 1)

    It is commonly used in continuous-control policies (e.g., SAC) where the policy
    samples an unconstrained Gaussian variable `z` and then squashes it into a
    bounded action range. If the environment action bounds are not (-1, 1), a
    further affine transformation is usually applied outside this bijector.

    Parameters
    ----------
    epsilon : float, default=1e-6
        Numerical stability constant used for:
        - inverse clamp margin (avoid atanh(±1))
        - Jacobian term: log(1 - tanh(z)^2 + epsilon)

    Notes
    -----
    Change-of-variables correction (per dimension):

        log |da/dz| = log(1 - tanh(z)^2)

    For squashed log-probabilities:

        log π(a|s) = log p(z) - Σ log(1 - tanh(z)^2 + eps)

    The correction returned by `log_prob_correction` is per-dimension; callers
    typically sum across action dimensions.
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
            Squashed tensor, shape (..., A), with values in (-1, 1).
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
        - We clamp `a` away from ±1 to avoid numerical overflow in atanh.
        - The clamp margin uses both dtype epsilon and the user-provided epsilon,
          taking the larger for practical stability.
        """
        if not a.is_floating_point():
            a = a.float()

        finfo_eps = th.finfo(a.dtype).eps
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

        Notes
        -----
        Prefer this over `0.5*log((1+a)/(1-a))` for stability near 0.
        """
        return 0.5 * (th.log1p(a) - th.log1p(-a))

    def log_prob_correction(self, z: th.Tensor) -> th.Tensor:
        """
        Compute the per-dimension log-Jacobian correction for `a = tanh(z)`.

        Parameters
        ----------
        z : torch.Tensor
            Pre-squash tensor, shape (..., A).

        Returns
        -------
        corr : torch.Tensor
            Per-dimension correction term, shape (..., A), equal to:
                log(1 - tanh(z)^2 + epsilon)

        Notes
        -----
        For squashed Gaussian policies, log-prob is typically computed as:

            logp_a = logp_z - corr.sum(dim=-1, keepdim=True)

        where `logp_z` is the Gaussian log-prob in z-space.
        """
        t = th.tanh(z)
        return th.log(1.0 - t * t + self.epsilon)


# =============================================================================
# Network utilities
# =============================================================================
def _validate_hidden_sizes(hidden_sizes: Sequence[int]) -> Tuple[int, ...]:
    """
    Validate an MLP hidden layer size specification.

    Parameters
    ----------
    hidden_sizes : Sequence[int]
        Hidden layer sizes (e.g., (64, 64) or [256, 256]).

    Returns
    -------
    hs : Tuple[int, ...]
        Validated sizes as a tuple of positive integers.

    Raises
    ------
    ValueError
        If empty or contains non-positive entries.
    TypeError
        If entries cannot be cast to int.
    """
    hs = tuple(int(h) for h in hidden_sizes)
    if len(hs) == 0:
        raise ValueError("hidden_sizes must have at least one layer (e.g., (64, 64)).")
    if any(h <= 0 for h in hs):
        raise ValueError(f"hidden_sizes must be positive integers, got: {hs}")
    return hs


def _make_weights_init(
    init_type: str = "xavier_uniform",
    gain: float = 1.0,
    bias: float = 0.0,
    kaiming_a: float = math.sqrt(5.0),
) -> Callable[[nn.Module], None]:
    """
    Create an initializer function compatible with `nn.Module.apply()`.

    Parameters
    ----------
    init_type : str, default="xavier_uniform"
        Initialization scheme identifier (case-insensitive).

        Supported for `nn.Linear`:
        - "xavier_uniform"
        - "xavier_normal"
        - "kaiming_uniform"
        - "kaiming_normal"
        - "orthogonal"
        - "normal"   (std = gain)
        - "uniform"  (range = [-gain, +gain])
    gain : float, default=1.0
        Gain used by Xavier/Orthogonal initializers, and used as:
        - std for "normal"
        - magnitude for "uniform" ([-gain, +gain])
    bias : float, default=0.0
        Constant value for initializing linear biases (if present).
    kaiming_a : float, default=sqrt(5.0)
        Negative slope parameter `a` for Kaiming initialization.

    Returns
    -------
    init_fn : Callable[[nn.Module], None]
        Function intended to be used as:
            `model.apply(init_fn)`

    Raises
    ------
    ValueError
        If `init_type` is unknown.

    Notes
    -----
    - Only `nn.Linear` modules are initialized; other modules are ignored.
    - Centralizing init logic helps reproducibility and controlled ablations.
    """
    name = str(init_type).lower().strip()
    gain = float(gain)
    bias = float(bias)
    kaiming_a = float(kaiming_a)

    def init_fn(module: nn.Module) -> None:
        """
        Initialize a single module in-place (called by `nn.Module.apply`).

        Parameters
        ----------
        module : nn.Module
            Module instance to initialize.
        """
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
    Utility mixin for dueling value/advantage combination.

    Dueling networks decompose Q-values into:
        - a scalar value stream V
        - an advantage stream A over actions

    The canonical combination is:

        Q = V + (A - mean(A))

    which makes Q identifiable by forcing advantages to have zero mean across the
    action dimension.

    Notes
    -----
    This mixin only provides the combination; it does not impose any architectural
    constraints on how V and A are produced.
    """

    @staticmethod
    def combine_dueling(v: th.Tensor, a: th.Tensor, *, mean_dim: int = -1) -> th.Tensor:
        """
        Combine value and advantage streams.

        Parameters
        ----------
        v : torch.Tensor
            Value stream. Common shapes:
            - (B, 1)     for standard dueling Q
            - (B, N, 1)  for quantile dueling (e.g., IQN)
            - (B, 1, K)  for categorical logits dueling (e.g., C51)
        a : torch.Tensor
            Advantage stream. Common shapes:
            - (B, A)
            - (B, N, A)
            - (B, A, K)
        mean_dim : int, default=-1
            Dimension over which to mean-reduce advantages (typically the action dim).

        Returns
        -------
        q : torch.Tensor
            Combined tensor with broadcasted shape compatible with `a`.

        Notes
        -----
        Broadcasting rules must align between `v` and `a`. Typically `v` includes
        singleton dimensions where appropriate.
        """
        return v + (a - a.mean(dim=mean_dim, keepdim=True))


# =============================================================================
# Input shape/device normalization
# =============================================================================
def _ensure_batch(x: Any, device: Union[th.device, str]) -> th.Tensor:
    """
    Convert input to a floating-point tensor on `device` and ensure a batch dimension.

    This is a common utility for policy/value networks that accept either:
      - a single sample (D,)  -> converted to (1, D)
      - a batch        (B, D) -> unchanged

    Parameters
    ----------
    x : Any
        Input data. Accepted forms include:
        - torch.Tensor
        - numpy array
        - Python list/tuple
        - scalar (becomes shape (1, 1) or (1,) depending on conversion)
    device : torch.device or str
        Target device.

    Returns
    -------
    x_t : torch.Tensor
        Floating-point tensor on `device` with shape:
        - (1, D) if input is 1D
        - (B, ...) if input already has batch dimension
        - (1, 1) if input is scalar (depends on torch's as_tensor rules)

    Notes
    -----
    - Non-floating tensors are cast to float32 to match typical neural network inputs.
    - Device move errors are swallowed (best-effort) to be robust in edge cases,
      but in most training code you may prefer to raise to catch misconfigurations.
    """
    x_t = x if isinstance(x, th.Tensor) else th.as_tensor(x)

    if not x_t.is_floating_point():
        x_t = x_t.float()

    try:
        x_t = x_t.to(device)
    except Exception:
        pass

    if x_t.dim() == 0:
        # Scalar -> (1, 1)
        x_t = x_t.view(1, 1)
    elif x_t.dim() == 1:
        # (D,) -> (1, D)
        x_t = x_t.unsqueeze(0)

    return x_t
