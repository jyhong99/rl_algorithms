from __future__ import annotations

from typing import Optional, Sequence, Union, Tuple

import torch as th


def normalize_size(size: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    Normalize a size specification into a tuple.

    Parameters
    ----------
    size : int or Tuple[int, ...]
        Desired tensor shape.

    Returns
    -------
    shape : Tuple[int, ...]
        Shape as a tuple.
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


def normalize_kind(kind: Optional[str]) -> Optional[str]:
    """
    Normalize noise kind string.

    Examples
    --------
    " Ornstein-Uhlenbeck " -> "ornstein_uhlenbeck"
    "gaussian-action"      -> "gaussian_action"
    """
    if kind is None:
        return None
    s = str(kind).strip().lower()
    if s in ("", "none", "null"):
        return None
    # Unify separators
    s = s.replace("-", "_").replace(" ", "_")
    # Collapse repeated underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def as_flat_bounds(
    x: Union[float, Sequence[float]],
    *,
    action_dim: int,
    device: Union[str, th.device],
    dtype: th.dtype,
    name: str,
) -> th.Tensor:
    """
    Convert scalar/sequence bounds into a 1D tensor broadcastable to (action_dim,).

    Parameters
    ----------
    x : float or Sequence[float]
        Bound value(s).
    action_dim : int
        Action dimension (> 0).
    device : str or torch.device
    dtype : torch.dtype
    name : str
        For error messages.

    Returns
    -------
    t : torch.Tensor
        Tensor on `device` with `dtype`.
        Shape is either () (scalar) or (action_dim,).
    """
    t = th.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 0:
        return t
    if t.ndim == 1 and t.shape[0] == action_dim:
        return t
    raise ValueError(
        f"{name} must be a scalar or a 1D tensor/sequence of length action_dim={action_dim}. "
        f"Got shape={tuple(t.shape)}."
    )