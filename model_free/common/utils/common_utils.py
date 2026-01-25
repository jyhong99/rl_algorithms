from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F


# =============================================================================
# NumPy / Torch conversion utilities
# =============================================================================

def to_numpy(x: Any, *, ensure_1d: bool = False) -> np.ndarray:
    """
    Convert an input to a NumPy array (CPU).

    Parameters
    ----------
    x : Any
        Input object. Common cases: np.ndarray, torch.Tensor, python scalar/list.
    ensure_1d : bool, default=False
        If True and the result is a scalar (0-d), convert to shape (1,).

    Returns
    -------
    arr : np.ndarray
        NumPy array on CPU.

    Notes
    -----
    - torch.Tensor is detached and moved to CPU.
    - This function does not enforce dtype.
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


def to_tensor(
    x: Any,
    device: Union[str, th.device],
    dtype: th.dtype = th.float32,
) -> th.Tensor:
    """
    Convert input to a torch.Tensor on the given device and dtype.

    Parameters
    ----------
    x : Any
        Input object. Common cases: np.ndarray, torch.Tensor, python scalar/list.
    device : Union[str, torch.device]
        Target device.
    dtype : torch.dtype, default=torch.float32
        Target dtype.

    Returns
    -------
    t : torch.Tensor
        Tensor on `device` with dtype `dtype`.

    Notes
    -----
    - If `x` is a torch.Tensor, `.to(device=..., dtype=...)` is applied.
      If you want to preserve dtype for tensors, consider adding a flag
      like `preserve_dtype_for_tensor=True`.
    """
    dev = th.device(device)

    if th.is_tensor(x):
        return x.to(device=dev, dtype=dtype)

    if isinstance(x, np.ndarray):
        # from_numpy shares CPU memory; then moved to target device
        return th.from_numpy(x).to(device=dev, dtype=dtype)

    return th.as_tensor(x, dtype=dtype, device=dev)


def to_flat_np(x: Any, *, dtype: Optional[np.dtype] = np.float32) -> np.ndarray:
    """
    Convert input to a flattened (1D) NumPy array.

    Parameters
    ----------
    x : Any
        Input object.
    dtype : Optional[np.dtype], default=np.float32
        If not None, cast output to this dtype.

    Returns
    -------
    arr : np.ndarray
        Flattened array of shape (D,).
    """
    if th.is_tensor(x):
        t = x.detach().cpu()
        arr = t.numpy()
    else:
        arr = np.asarray(x)

    arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_scalar(x: Any) -> Optional[float]:
    """
    Convert a scalar-like input to Python float.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    s : Optional[float]
        float if convertible, else None.

    Accepted inputs
    ---------------
    - Python scalars: int/float/bool
    - NumPy scalars
    - 0-d / 1-element NumPy arrays
    - 0-d / 1-element torch tensors
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


def is_scalar_like(x: Any) -> bool:
    """Return True if `x` can be converted by `to_scalar`."""
    return to_scalar(x) is not None


def require_scalar_like(x: Any, *, name: str) -> float:
    """
    Require scalar-like input and return float.

    Raises
    ------
    TypeError
        If `x` is not scalar-like.
    """
    s = to_scalar(x)
    if s is None:
        raise TypeError(f"{name} must be scalar-like, got: {type(x)}")
    return float(s)


def to_action_np(action: Any, *, action_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Convert an agent action to a NumPy array suitable for env.step().

    Parameters
    ----------
    action : Any
        Action output from a policy (scalar, np.ndarray, torch.Tensor, list, ...).
    action_shape : Optional[Tuple[int, ...]], default=None
        If given, reshape output exactly to this shape.

    Returns
    -------
    a : np.ndarray
        NumPy action array.

    Notes
    -----
    - If policy returns a scalar, Gym/Gymnasium often expects shape (1,),
      so `ensure_1d=True` is applied.
    - Reshape errors are surfaced unless shape fallback can fix it.
    """
    a = to_numpy(action, ensure_1d=True)
    a = np.asarray(a)

    if action_shape is not None:
        try:
            a = a.reshape(action_shape)
        except Exception:
            a = a.reshape(-1).reshape(action_shape)

    return a


def to_column(x: th.Tensor) -> th.Tensor:
    """
    Ensure a 1D batch tensor becomes a column tensor.

    This is a small shape-normalization utility frequently used when you want
    "per-sample scalars" to have explicit feature dimension = 1.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
        - If shape is (B,), it will be converted to (B, 1).
        - Otherwise, it is returned unchanged.

    Returns
    -------
    y : torch.Tensor
        Output tensor.
        - (B,)   -> (B, 1)
        - (B, 1) -> (B, 1)
        - (B, A) -> (B, A)

    Notes
    -----
    - This function assumes the first dimension is the batch dimension.
    - It is typically used to make broadcasting / concatenation consistent.
    """
    # If x is (B,), add a feature dimension so it becomes (B, 1).
    return x.unsqueeze(1) if x.dim() == 1 else x


def reduce_joint(x: th.Tensor) -> th.Tensor:
    """
    Reduce "joint" per-sample values into a single scalar per batch element.

    This is useful when:
    - you have a vector action space and store per-dimension values (B, A),
      but later want a single score/value per sample (B,)
    - you have a column tensor (B, 1) and want to squeeze it to (B,)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor expected to be either:
        - (B,)   : already reduced
        - (B, 1) : column tensor
        - (B, A) : per-action(or per-component) tensor

    Returns
    -------
    y : torch.Tensor
        Reduced tensor of shape (B,).

    Shape Rules
    -----------
    - (B, 1) -> sum over last dim -> (B,)
    - (B, A) -> sum over last dim -> (B,)
    - (B,)   -> unchanged         -> (B,)

    Notes
    -----
    - Using sum(dim=-1) is a design choice.
      If you want mean or other reductions, create a separate function
      (e.g., reduce_joint_mean).
    """
    # If x has a feature/action dimension, collapse it into one scalar per sample.
    if x.dim() == 2:
        return x.sum(dim=-1)
    # Already (B,) (or more generally not 2D) -> return as-is.
    return x

    
# =============================================================================
# CPU-safe serialization helpers
# =============================================================================

def to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors to CPU and detach.

    Parameters
    ----------
    obj : Any
        Tensor / nested structure.

    Returns
    -------
    out : Any
        Same structure with tensors converted to CPU tensors.
    """
    if th.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, Mapping):
        return {k: to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        vals = [to_cpu(v) for v in obj]
        return type(obj)(vals)

    return obj


def to_cpu_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert a module state_dict to a pure-CPU form.

    Parameters
    ----------
    state_dict : Mapping[str, Any]
        State dict.

    Returns
    -------
    cpu_state : Dict[str, Any]
        CPU / detached version of the state dict.
    """
    return to_cpu(dict(state_dict))


# =============================================================================
# Observation formatting
# =============================================================================

def obs_to_cpu_tensor(obs: Any) -> th.Tensor:
    """
    Convert an observation to a CPU float32 tensor with batch dimension.

    Parameters
    ----------
    obs : Any
        Observation. Common shapes:
        - (obs_dim,) -> returns (1, obs_dim)
        - (B, obs_dim) -> returns (B, obs_dim)
        - scalar -> returns (1, 1)

    Returns
    -------
    t : torch.Tensor
        CPU float32 tensor with batch dimension.
    """
    t = to_tensor(obs, device="cpu", dtype=th.float32)

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
def polyak_update(target: th.Tensor, source: th.Tensor, tau: float) -> None:
    """
    In-place Polyak update (source-weight convention):
        target <- (1 - tau) * target + tau * source

    Parameters
    ----------
    target : torch.Tensor
        Tensor to update in-place (e.g., target parameter).
    source : torch.Tensor
        Tensor providing new values (e.g., online parameter).
    tau : float
        Source interpolation factor in [0, 1].
        Typical for target networks: 0.005.

    Notes
    -----
    - tau close to 0.0: very slow update (keeps target).
    - tau close to 1.0: fast copy from source.
    """
    tau = float(tau)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1], got: {tau}")

    target.mul_(1.0 - tau).add_(source, alpha=tau)


@th.no_grad()
def ema_update(old: th.Tensor, new: th.Tensor, beta: float) -> None:
    """
    In-place EMA update (keep-ratio convention):
        old <- beta * old + (1 - beta) * new

    Parameters
    ----------
    old : torch.Tensor
        Running statistic updated in-place.
    new : torch.Tensor
        Freshly computed statistic.
    beta : float
        Keep ratio in [0, 1]. Typical: 0.95, 0.99, 0.999 depending on smoothing.

    Notes
    -----
    - beta close to 1.0: slow update (heavily keeps old).
    - beta close to 0.0: fast replace with new.
    """
    beta = float(beta)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got: {beta}")
    old.mul_(beta).add_(new, alpha=(1.0 - beta))


# =============================================================================
# Type / shape helpers
# =============================================================================

def require_mapping(x: Any, *, name: str) -> Mapping[str, Any]:
    """
    Require `x` to be a Mapping[str, Any].

    Raises
    ------
    TypeError
        If `x` is not a Mapping.
    """
    if not isinstance(x, Mapping):
        raise TypeError(f"{name} must be a Mapping[str, Any], got: {type(x)}")
    return x


def infer_shape(space: Any, *, name: str) -> Tuple[int, ...]:
    """
    Infer tensor shape from a Gym/Gymnasium-like space.

    Supported
    ---------
    - Box-like spaces: `space.shape` exists and is not None
    - Discrete-like spaces: `space.n` exists, mapped to shape (1,)

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
# (Optional) Vision / Conv helpers (consider moving to a separate module)
# =============================================================================

def img2col(
    x: th.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> th.Tensor:
    """
    Convert Conv2d input tensor into a 2D patch matrix (im2col).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C, H, W).
    kernel_size : Tuple[int, int]
        (kH, kW).
    stride : Tuple[int, int]
        (sH, sW).
    padding : Tuple[int, int]
        (pH, pW), symmetric padding.

    Returns
    -------
    cols : torch.Tensor
        Patch matrix of shape (N * H_out * W_out, C * kH * kW).

    Notes
    -----
    PyTorch provides `torch.nn.functional.unfold` which can replace this.
    Keeping this only if you want explicit unfolding logic.
    """
    if x.dim() != 4:
        raise ValueError(f"x must be (N,C,H,W), got shape: {tuple(x.shape)}")

    pH, pW = padding
    if pH > 0 or pW > 0:
        x = F.pad(x, (pW, pW, pH, pH))

    kH, kW = kernel_size
    sH, sW = stride

    x = x.unfold(2, kH, sH).unfold(3, kW, sW)          # (N,C,H_out,W_out,kH,kW)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()       # (N,H_out,W_out,C,kH,kW)
    cols = x.view(-1, x.size(3) * x.size(4) * x.size(5))
    return cols