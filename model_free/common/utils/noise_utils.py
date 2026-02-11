from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch as th


# =============================================================================
# Normalization helpers
# =============================================================================
def _normalize_size(size: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    Normalize a size specification into a tuple of positive integers.

    This helper is useful for APIs that accept both:
      - an integer (interpreted as a 1D shape), or
      - an explicit shape tuple.

    Parameters
    ----------
    size : int or Tuple[int, ...]
        Desired tensor shape specification.
        - If `int`, it is interpreted as a 1D shape `(size,)`.
        - If `tuple`, it is used as-is after validation.

    Returns
    -------
    shape : Tuple[int, ...]
        Validated shape tuple with strictly positive integer dimensions.

    Raises
    ------
    ValueError
        If:
        - `size` is an int <= 0,
        - `size` is an empty tuple,
        - any dimension is not an int or is <= 0.

    Examples
    --------
    >>> _normalize_size(8)
    (8,)
    >>> _normalize_size((2, 3))
    (2, 3)
    """
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        return (size,)

    shape = tuple(size)
    if len(shape) == 0:
        raise ValueError("size must be non-empty")
    if any((not isinstance(s, int)) or (s <= 0) for s in shape):
        raise ValueError(f"all size dims must be positive ints, got {shape}")
    return shape


def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    """
    Normalize a "kind" string into a canonical snake_case identifier.

    This function is designed for configuration parsing, where users may provide
    values with mixed casing and separators. It also treats empty/none-like
    strings as `None`.

    Parameters
    ----------
    kind : str or None
        Input kind specifier.

    Returns
    -------
    norm : str or None
        Normalized kind string, or None if `kind` is None/empty/"none"/"null".

    Notes
    -----
    Normalization steps:
      1) strip and lowercase
      2) map {"", "none", "null"} -> None
      3) unify separators: "-" and whitespace -> "_"
      4) collapse repeated underscores

    Examples
    --------
    >>> _normalize_kind(" Ornstein-Uhlenbeck ")
    'ornstein_uhlenbeck'
    >>> _normalize_kind("gaussian-action")
    'gaussian_action'
    >>> _normalize_kind("none")
    None
    """
    if kind is None:
        return None

    s = str(kind).strip().lower()
    if s in ("", "none", "null"):
        return None

    # Unify separators to underscore.
    s = s.replace("-", "_").replace(" ", "_")

    # Collapse repeated underscores for a clean canonical form.
    while "__" in s:
        s = s.replace("__", "_")

    return s


def _as_flat_bounds(
    x: Union[float, Sequence[float]],
    *,
    action_dim: int,
    device: Union[str, th.device],
    dtype: th.dtype,
    name: str,
) -> th.Tensor:
    """
    Convert scalar or per-dimension bounds to a tensor compatible with `action_dim`.

    This helper supports two common configurations for action bounds/noise scales:
      - a scalar bound shared across all action dimensions, or
      - a vector bound specified per action dimension.

    Parameters
    ----------
    x : float or Sequence[float]
        Bound value(s).
        - Scalar -> returned as a scalar tensor (ndim=0), broadcastable to (action_dim,).
        - Sequence -> must have length exactly `action_dim`.
    action_dim : int
        Action dimension (> 0).
    device : str or torch.device
        Target device for the returned tensor.
    dtype : torch.dtype
        Target dtype for the returned tensor.
    name : str
        Human-readable name used in error messages (e.g., "low", "high", "sigma").

    Returns
    -------
    t : torch.Tensor
        Tensor on `device` with dtype `dtype`.
        Shape is either:
        - ()            for scalar bounds
        - (action_dim,) for per-dimension bounds

    Raises
    ------
    ValueError
        If `action_dim <= 0`, or if `x` is neither:
        - scalar-like, nor
        - a 1D sequence/tensor of length `action_dim`.

    Notes
    -----
    Keeping scalars as 0-d tensors is intentional: it lets callers rely on
    PyTorch broadcasting rules when combining with action tensors of shape
    (..., action_dim).

    Examples
    --------
    >>> _as_flat_bounds(0.2, action_dim=3, device="cpu", dtype=th.float32, name="sigma").shape
    torch.Size([])
    >>> _as_flat_bounds([0.1, 0.2, 0.3], action_dim=3, device="cpu", dtype=th.float32, name="sigma").shape
    torch.Size([3])
    """
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")

    t = th.as_tensor(x, dtype=dtype, device=device)

    if t.ndim == 0:
        # Scalar bound: broadcastable to (action_dim,)
        return t

    if t.ndim == 1 and t.shape[0] == int(action_dim):
        # Per-dimension bound
        return t

    raise ValueError(
        f"{name} must be a scalar or a 1D tensor/sequence of length action_dim={action_dim}. "
        f"Got shape={tuple(t.shape)}."
    )
