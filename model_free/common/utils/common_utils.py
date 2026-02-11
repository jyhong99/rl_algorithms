from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import math

import numpy as np
import torch as th
import torch.nn.functional as F


# =============================================================================
# NumPy / Torch conversion utilities
# =============================================================================
def _to_numpy(x: Any, *, ensure_1d: bool = False) -> np.ndarray:
    """
    Convert an input to a NumPy array on CPU.

    Parameters
    ----------
    x : Any
        Input object. Common cases include:
        - ``np.ndarray``
        - ``torch.Tensor``
        - Python scalars, lists, tuples
    ensure_1d : bool, default=False
        If True and the resulting array is a scalar (0-d), it is converted to a
        1D array of shape (1,). This is often useful for Gym-style APIs that
        expect actions/observations to be at least 1D.

    Returns
    -------
    arr : np.ndarray
        NumPy array on CPU. Dtype is not forced.

    Notes
    -----
    - If ``x`` is a ``torch.Tensor``, it is detached and moved to CPU before
      converting via ``.numpy()``.
    - For non-array inputs, ``np.asarray`` is used (may create a view).
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif th.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if ensure_1d and arr.shape == ():
        arr = np.asarray([arr])

    return arr


def _to_tensor(
    x: Any,
    device: Union[str, th.device],
    dtype: th.dtype = th.float32,
) -> th.Tensor:
    """
    Convert input to a torch.Tensor on the given device and dtype.

    Parameters
    ----------
    x : Any
        Input object. Common cases include:
        - ``np.ndarray`` (CPU)
        - ``torch.Tensor``
        - Python scalars / lists
    device : Union[str, torch.device]
        Target device (e.g., "cpu", "cuda:0").
    dtype : torch.dtype, default=torch.float32
        Target dtype. Applied even if ``x`` is already a tensor.

    Returns
    -------
    t : torch.Tensor
        Tensor placed on ``device`` with dtype ``dtype``.

    Notes
    -----
    - If ``x`` is a tensor, this calls ``x.to(device=..., dtype=...)``.
      If you want to preserve an existing tensor dtype, add a flag like
      ``preserve_tensor_dtype=True`` and branch accordingly.
    - If ``x`` is a NumPy array, ``torch.from_numpy`` is used first (CPU sharing),
      and then moved to the target device.
    """
    dev = th.device(device)

    if th.is_tensor(x):
        return x.to(device=dev, dtype=dtype)

    if isinstance(x, np.ndarray):
        return th.from_numpy(x).to(device=dev, dtype=dtype)

    return th.as_tensor(x, dtype=dtype, device=dev)


def _to_flat_np(x: Any, *, dtype: Optional[np.dtype] = np.float32) -> np.ndarray:
    """
    Convert input to a flattened (1D) NumPy array.

    Parameters
    ----------
    x : Any
        Input object.
    dtype : Optional[np.dtype], default=np.float32
        If not None, cast output to this dtype (without copy when possible).

    Returns
    -------
    arr : np.ndarray, shape (D,)
        Flattened NumPy array.

    Notes
    -----
    - Torch tensors are detached and moved to CPU.
    - The returned array is always contiguous only if the underlying representation
      is contiguous; otherwise NumPy may return a view.
    """
    if th.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _to_scalar(x: Any) -> Optional[float]:
    """
    Convert a scalar-like input to a Python float.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    s : float or None
        Python float if convertible, else None.

    Accepted inputs
    ---------------
    - Python scalars: int/float/bool
    - NumPy scalars (np.number)
    - 0-d NumPy arrays or 1-element arrays
    - 0-d torch tensors or 1-element tensors

    Notes
    -----
    This is intentionally conservative: tensors/arrays with more than one element
    return None to avoid silently discarding data.
    """
    if th.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return None

    if isinstance(x, (bool, int, float, np.number)):
        return float(x)

    try:
        arr = np.asarray(x)
        if arr.shape == () or arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        return None

    return None


def _is_scalar_like(x: Any) -> bool:
    """
    Return True if ``x`` can be safely converted by `_to_scalar`.

    Parameters
    ----------
    x : Any
        Input.

    Returns
    -------
    ok : bool
        True if `_to_scalar(x)` would return a float (not None).
    """
    return _to_scalar(x) is not None


def _is_sequence(x: Any) -> bool:
    """
    Return True if ``x`` is a list/tuple (a light-weight "sequence" check).

    Parameters
    ----------
    x : Any
        Input.

    Returns
    -------
    is_seq : bool
        True if x is an instance of (list, tuple).
    """
    return isinstance(x, (list, tuple))


def _require_scalar_like(x: Any, *, name: str) -> float:
    """
    Require scalar-like input and return a Python float.

    Parameters
    ----------
    x : Any
        Input expected to be scalar-like (see `_to_scalar`).
    name : str
        Name used in error messages.

    Returns
    -------
    s : float
        Converted scalar value.

    Raises
    ------
    TypeError
        If `x` is not scalar-like.
    """
    s = _to_scalar(x)
    if s is None:
        raise TypeError(f"{name} must be scalar-like, got: {type(x)}")
    return float(s)


def _to_action_np(action: Any, *, action_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Convert a policy action into a NumPy array suitable for ``env.step(...)``.

    Parameters
    ----------
    action : Any
        Action output from a policy. Common cases:
        - scalar
        - ``np.ndarray``
        - ``torch.Tensor``
        - list/tuple
    action_shape : Optional[Tuple[int, ...]], default=None
        If provided, reshape the output to exactly this shape.

    Returns
    -------
    a : np.ndarray
        NumPy action array.

    Notes
    -----
    - Scalar actions are converted to shape (1,) (via ``ensure_1d=True``) because
      many Gym-style APIs prefer/expect at least 1D arrays.
    - If reshaping fails due to non-matching shapes, we fall back to flatten-then-reshape.
      Reshape errors are still surfaced if the total number of elements is incompatible.
    """
    a = _to_numpy(action, ensure_1d=True)
    a = np.asarray(a)

    if action_shape is not None:
        try:
            a = a.reshape(action_shape)
        except Exception:
            a = a.reshape(-1).reshape(action_shape)

    return a


def _to_column(x: th.Tensor) -> th.Tensor:
    """
    Ensure a 1D batch tensor becomes a column tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
        - If shape is (B,), it will be converted to (B, 1).
        - Otherwise, it is returned unchanged.

    Returns
    -------
    y : torch.Tensor
        Output tensor with normalized shape.
        - (B,)   -> (B, 1)
        - (B, 1) -> (B, 1)
        - (B, A) -> (B, A)

    Notes
    -----
    This is a small shape-normalization helper for consistent broadcasting and
    concatenation when you conceptually have "per-sample scalars".
    """
    return x.unsqueeze(1) if x.dim() == 1 else x


def _reduce_joint(x: th.Tensor) -> th.Tensor:
    """
    Reduce per-sample "joint" values into a single scalar per batch element.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor expected to be either:
        - (B,)   : already reduced
        - (B, 1) : column tensor
        - (B, A) : per-component tensor

    Returns
    -------
    y : torch.Tensor, shape (B,)
        Reduced tensor.

    Shape Rules
    -----------
    - (B, 1) -> sum over last dim -> (B,)
    - (B, A) -> sum over last dim -> (B,)
    - (B,)   -> unchanged         -> (B,)

    Notes
    -----
    Reduction uses `sum(dim=-1)` by design. If you want mean/max/etc.,
    create separate helpers for explicitness.
    """
    if x.dim() == 2:
        return x.sum(dim=-1)
    return x


# =============================================================================
# CPU-safe serialization helpers
# =============================================================================
def _to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors to CPU and detach (serialization-friendly).

    Parameters
    ----------
    obj : Any
        A tensor or a nested structure containing tensors.
        Supported containers:
        - Mapping (dict-like)
        - list / tuple

    Returns
    -------
    out : Any
        Same structure where all tensors are replaced with ``tensor.detach().cpu()``.

    Notes
    -----
    This is useful for:
    - storing snapshots in replay buffers / checkpoints
    - JSON-friendly logging (after additional scalar coercion)
    - avoiding GPU tensor references in long-lived python objects
    """
    if th.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, Mapping):
        return {k: _to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        vals = [_to_cpu(v) for v in obj]
        return type(obj)(vals)

    return obj


def _to_cpu_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert a module state_dict to a pure-CPU, detached form.

    Parameters
    ----------
    state_dict : Mapping[str, Any]
        State dict mapping parameter/buffer names to tensors (or tensor-like).

    Returns
    -------
    cpu_state : Dict[str, Any]
        A dict with the same keys where tensors are CPU+detached.

    Notes
    -----
    - This is typically used prior to checkpointing when you want CPU-only artifacts.
    - If the incoming mapping is not a plain dict, it is first copied to dict.
    """
    return _to_cpu(dict(state_dict))


# =============================================================================
# Observation formatting
# =============================================================================
def _obs_to_cpu_tensor(obs: Any) -> th.Tensor:
    """
    Convert an observation to a CPU float32 tensor with a batch dimension.

    Parameters
    ----------
    obs : Any
        Observation input. Common shapes:
        - scalar                    -> returns (1, 1)
        - (obs_dim,)                -> returns (1, obs_dim)
        - (B, obs_dim)              -> returns (B, obs_dim)
        - higher-rank (B, ...)      -> preserved as (B, ...)

    Returns
    -------
    t : torch.Tensor
        CPU float32 tensor with batch dimension.

    Notes
    -----
    - This function standardizes "single observation" inputs to be batched.
    - No normalization (mean/std) is applied here; it's purely a shape/device cast.
    """
    t = _to_tensor(obs, device="cpu", dtype=th.float32)

    # Normalize to (B, ...)
    if t.dim() == 0:
        t = t.view(1, 1)
    elif t.dim() == 1:
        t = t.unsqueeze(0)

    return t


# =============================================================================
# Target network / EMA utilities
# =============================================================================
@th.no_grad()
def _polyak_update(target: th.Tensor, source: th.Tensor, tau: float) -> None:
    """
    In-place Polyak update (source-weight convention).

    Performs:
        ``target <- (1 - tau) * target + tau * source``

    Parameters
    ----------
    target : torch.Tensor
        Tensor updated in-place (e.g., target network parameter).
    source : torch.Tensor
        Tensor providing new values (e.g., online network parameter).
    tau : float
        Interpolation factor in [0, 1].
        Typical values for target networks: 0.005, 0.01.

    Raises
    ------
    ValueError
        If ``tau`` is outside [0, 1].

    Notes
    -----
    - ``tau -> 0``: very slow update (target changes little).
    - ``tau -> 1``: immediate copy from source.
    - Assumes `target` and `source` are shape-compatible.
    """
    tau = float(tau)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1], got: {tau}")

    target.mul_(1.0 - tau).add_(source, alpha=tau)


@th.no_grad()
def _ema_update(old: th.Tensor, new: th.Tensor, beta: float) -> None:
    """
    In-place exponential moving average (EMA) update (keep-ratio convention).

    Performs:
        ``old <- beta * old + (1 - beta) * new``

    Parameters
    ----------
    old : torch.Tensor
        Running statistic updated in-place.
    new : torch.Tensor
        Freshly computed statistic.
    beta : float
        Keep ratio in [0, 1]. Typical values: 0.95, 0.99, 0.999.

    Raises
    ------
    ValueError
        If ``beta`` is outside [0, 1].

    Notes
    -----
    - ``beta -> 1``: heavy smoothing, slow adaptation.
    - ``beta -> 0``: fast tracking (nearly overwrite with new).
    """
    beta = float(beta)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got: {beta}")
    old.mul_(beta).add_(new, alpha=(1.0 - beta))


# =============================================================================
# Simple stats helpers (pure Python)
# =============================================================================
def _mean(xs: Sequence[float]) -> float:
    """
    Compute mean for a sequence, returning 0.0 for empty sequences.

    Parameters
    ----------
    xs : Sequence[float]
        Input sequence.

    Returns
    -------
    m : float
        Mean of xs, or 0.0 if xs is empty.
    """
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _std(xs: Sequence[float]) -> float:
    """
    Compute population standard deviation (ddof=0).

    Parameters
    ----------
    xs : Sequence[float]
        Input sequence.

    Returns
    -------
    s : float
        Population standard deviation.
        Returns 0.0 for n <= 1 to avoid divide-by-zero.

    Notes
    -----
    For short windows in logging, population std is typically stable and sufficient.
    """
    n = len(xs)
    if n <= 1:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) * (x - m) for x in xs) / n
    return float(math.sqrt(var))


# =============================================================================
# Type / shape helpers
# =============================================================================
def _require_mapping(x: Any, *, name: str) -> Mapping[str, Any]:
    """
    Require `x` to be a Mapping[str, Any].

    Parameters
    ----------
    x : Any
        Input object.
    name : str
        Name used in error messages.

    Returns
    -------
    m : Mapping[str, Any]
        Same object, typed as a mapping.

    Raises
    ------
    TypeError
        If `x` is not a Mapping.
    """
    if not isinstance(x, Mapping):
        raise TypeError(f"{name} must be a Mapping[str, Any], got: {type(x)}")
    return x


def _infer_shape(space: Any, *, name: str) -> Tuple[int, ...]:
    """
    Infer a tensor shape from a Gym/Gymnasium-like space.

    Parameters
    ----------
    space : Any
        Space-like object.
        Supported patterns:
        - Box-like: attribute ``space.shape`` exists and is not None
        - Discrete-like: attribute ``space.n`` exists, mapped to shape (1,)
    name : str
        Name used in error messages (e.g., "obs_space", "act_space").

    Returns
    -------
    shape : Tuple[int, ...]
        Inferred shape tuple.

    Raises
    ------
    ValueError
        If neither `shape` nor `n` is available.
    """
    if hasattr(space, "shape") and space.shape is not None:
        return tuple(int(s) for s in space.shape)
    if hasattr(space, "n"):
        return (1,)
    raise ValueError(f"Unsupported {name} (no shape or n): {space}")


# =============================================================================
# (Optional) Vision / Conv helpers
# =============================================================================
def _img2col(
    x: th.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> th.Tensor:
    """
    Convert a Conv2d input tensor into a 2D patch matrix (im2col).

    Parameters
    ----------
    x : torch.Tensor, shape (N, C, H, W)
        Input tensor.
    kernel_size : Tuple[int, int]
        Kernel size (kH, kW).
    stride : Tuple[int, int]
        Stride (sH, sW).
    padding : Tuple[int, int]
        Symmetric padding (pH, pW) applied to (H, W).

    Returns
    -------
    cols : torch.Tensor, shape (N * H_out * W_out, C * kH * kW)
        Unfolded patch matrix.

    Raises
    ------
    ValueError
        If `x` is not 4D (N, C, H, W).

    Notes
    -----
    - This is functionally similar to ``torch.nn.functional.unfold``:
        ``F.unfold(x, kernel_size, dilation=1, padding=padding, stride=stride)``
      followed by reshaping/transposing.
    - Keep this function only if you want explicit unfolding logic or custom
      behavior; otherwise prefer `F.unfold` for clarity and maintenance.

    Shape details
    -------------
    Let:
      H_out = floor((H + 2*pH - kH) / sH) + 1
      W_out = floor((W + 2*pW - kW) / sW) + 1

    Then each row in `cols` corresponds to one spatial location across the batch.
    """
    if x.dim() != 4:
        raise ValueError(f"x must be (N,C,H,W), got shape: {tuple(x.shape)}")

    pH, pW = padding
    if pH > 0 or pW > 0:
        # F.pad uses (left, right, top, bottom) for 2D spatial padding
        x = F.pad(x, (pW, pW, pH, pH))

    kH, kW = kernel_size
    sH, sW = stride

    # Unfold H then W -> (N, C, H_out, W_out, kH, kW)
    patches = x.unfold(2, kH, sH).unfold(3, kW, sW)

    # Move patch dims to the end, and channels before them:
    # (N, H_out, W_out, C, kH, kW)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    # Flatten per-patch features -> (N*H_out*W_out, C*kH*kW)
    cols = patches.view(-1, patches.size(3) * patches.size(4) * patches.size(5))
    return cols
