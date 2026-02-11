from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type
import importlib

try:
    import ray  # type: ignore
except Exception:
    ray = None  # type: ignore

import torch.nn as nn

from .common_utils import _to_cpu_state_dict


# =============================================================================
# Entrypoint-based policy factory
# =============================================================================
@dataclass(frozen=True)
class PolicyFactorySpec:
    """
    Pure-Python description of how to build a worker-side policy.

    This is designed for distributed execution (e.g., Ray workers) where you
    cannot ship live Python function objects or nn.Modules reliably, but you can
    ship a string entrypoint + simple kwargs and re-create the object in the worker
    process.

    Parameters
    ----------
    entrypoint : str
        Factory entrypoint in the form ``"package.module:function_name"``.
        The target must be importable on remote workers and must be a top-level
        function (not a nested function or lambda).
    kwargs : Dict[str, Any]
        Keyword arguments passed to the factory.

        Notes
        -----
        When used with Ray, `kwargs` must be pickle-serializable (and ideally JSON-safe).
        Do NOT include:
          - torch.Tensor
          - nn.Module
          - torch.device
          - CUDA storages / non-serializable handles
    """
    entrypoint: str
    kwargs: Dict[str, Any]


def _make_entrypoint(fn: Callable[..., Any]) -> str:
    """
    Convert a top-level function into an entrypoint string.

    Parameters
    ----------
    fn : Callable[..., Any]
        A *top-level* function object.

    Returns
    -------
    entrypoint : str
        Entrypoint string in the form ``"module:function"``.

    Raises
    ------
    TypeError
        If `fn` is not callable.
    ValueError
        If module/name cannot be inferred, or if `fn` is not a top-level function.

    Notes
    -----
    Ray workers must be able to import the factory by name. Nested functions have
    qualnames containing ``"<locals>"`` and cannot be imported by module attribute
    access. This function rejects nested functions accordingly.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")

    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    name = getattr(fn, "__name__", None)

    if not mod or not qual or not name:
        raise ValueError("Cannot infer module/qualname/name for entrypoint.")

    if "<locals>" in qual:
        raise ValueError(
            "Entrypoint must be a top-level function (not nested). "
            f"Got qualname={qual!r}"
        )

    return f"{mod}:{name}"


def _resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """
    Resolve an entrypoint string ``"package.module:function_name"`` to a callable.

    Parameters
    ----------
    entrypoint : str
        Entrypoint string of the form ``"module:function"``.

    Returns
    -------
    fn : Callable[..., Any]
        Resolved callable object.

    Raises
    ------
    ValueError
        If `entrypoint` format is invalid (missing/empty module or function token).
    ImportError
        If the module cannot be imported.
    AttributeError
        If the function name does not exist in the module.
    TypeError
        If the resolved object is not callable.

    Notes
    -----
    This function does not validate the callable signature; it only resolves and
    checks `callable(obj)`.
    """
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint format: {entrypoint!r} (expected 'module:function')"
        )

    mod_name, fn_name = entrypoint.split(":", 1)
    if not mod_name or not fn_name:
        raise ValueError(f"Invalid entrypoint format: {entrypoint!r} (empty module/function)")

    mod = importlib.import_module(mod_name)
    obj = getattr(mod, fn_name)  # may raise AttributeError

    if not callable(obj):
        raise TypeError(f"Entrypoint is not callable: {entrypoint!r} -> {type(obj)}")

    return obj


def _build_policy_from_spec(spec: PolicyFactorySpec) -> nn.Module:
    """
    Build a policy module from a PolicyFactorySpec.

    Parameters
    ----------
    spec : PolicyFactorySpec
        Factory spec describing how to build the policy.

    Returns
    -------
    policy : nn.Module
        Instantiated policy module moved to CPU and set to eval() mode.

    Raises
    ------
    TypeError
        If the factory does not return an `nn.Module`.
    ImportError, AttributeError, ValueError
        Propagated from `_resolve_entrypoint` if the entrypoint cannot be resolved.

    Notes
    -----
    CPU+eval is a conservative default for worker-side usage:
      - avoids accidental GPU allocation on remote workers
      - avoids training-mode stochasticity (dropout, batchnorm updates) by default
    """
    fn = _resolve_entrypoint(spec.entrypoint)
    policy = fn(**dict(spec.kwargs))

    if not isinstance(policy, nn.Module):
        raise TypeError(
            "Policy factory must return torch.nn.Module. "
            f"Got: {type(policy)} from entrypoint={spec.entrypoint!r}"
        )

    policy = policy.to("cpu")
    policy.eval()
    return policy


# =============================================================================
# Activation function resolver
# =============================================================================
def _normalize_activation_name(name: str) -> str:
    """
    Normalize an activation name token into a registry key.

    Parameters
    ----------
    name : str
        Input activation name string. Examples:
        - "torch.nn.ReLU"
        - "nn.LeakyReLU"
        - "Leaky ReLU"
        - "gaussian-action" (generally for config parsing, though not an activation)

    Returns
    -------
    key : str
        Normalized key, lowercased with separators unified.

    Notes
    -----
    Normalization rules:
      - Strip known prefixes: "torch.nn.", "nn."
      - Lowercase
      - Convert "-" and "." to underscores where relevant
      - Remove spaces
      - Keep underscores to support common aliases
    """
    n = name.strip()
    if n.startswith("torch.nn."):
        n = n[len("torch.nn.") :]
    if n.startswith("nn."):
        n = n[len("nn.") :]

    # "Leaky ReLU" -> "leakyrelu"
    key = n.lower().replace("-", "_").replace(" ", "").replace(".", "_")
    return key


def _resolve_activation_fn(act: Any, *, default: Type[nn.Module] = nn.ReLU) -> Type[nn.Module]:
    """
    Resolve an activation specification to an `nn.Module` **class** (not an instance).

    Parameters
    ----------
    act : Any
        Activation specification. Supported inputs:
        - None:
            Return `default`.
        - nn.Module subclass:
            Return as-is.
        - nn.Module instance:
            Return `instance.__class__`.
        - str:
            A class name or alias (e.g., "relu", "torch.nn.ReLU", "LeakyReLU",
            "Leaky ReLU", "swish").
    default : Type[nn.Module], default=nn.ReLU
        Default activation if `act` is None.

    Returns
    -------
    cls : Type[nn.Module]
        Activation module class (e.g., `nn.ReLU`, `nn.SiLU`, ...).

    Raises
    ------
    ValueError
        If `act` cannot be resolved to an `nn.Module` subclass.
    TypeError
        If the resolved object is not an `nn.Module` subclass.

    Notes
    -----
    Returning a class (not an instance) is convenient for MLP builders that want to
    create fresh activations per layer:

        act_cls = _resolve_activation_fn("silu")
        layers.append(act_cls())

    The resolver tries:
      1) a canonical alias registry
      2) attribute lookup on `torch.nn` using the raw name token
    """
    if act is None:
        return default

    if isinstance(act, type) and issubclass(act, nn.Module):
        return act

    if isinstance(act, nn.Module):
        return act.__class__

    if isinstance(act, str):
        key = _normalize_activation_name(act)

        aliases = {
            "relu": "relu",
            "silu": "silu",
            "swish": "silu",
            "gelu": "gelu",
            "tanh": "tanh",
            "sigmoid": "sigmoid",
            "elu": "elu",
            "selu": "selu",
            "prelu": "prelu",
            "leakyrelu": "leakyrelu",
            "leaky_relu": "leakyrelu",
            "relu6": "relu6",
            "softplus": "softplus",
            "softsign": "softsign",
            "mish": "mish",
            "hardtanh": "hardtanh",
            "hard_tanh": "hardtanh",
            "hardswish": "hardswish",
            "hard_swish": "hardswish",
            "logsigmoid": "logsigmoid",
            "log_sigmoid": "logsigmoid",
            "identity": "identity",
            "linear": "identity",
        }
        canonical = aliases.get(key, key)

        registry: Dict[str, Optional[Type[nn.Module]]] = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "prelu": nn.PReLU,
            "leakyrelu": nn.LeakyReLU,
            "relu6": nn.ReLU6,
            "softplus": nn.Softplus,
            "softsign": nn.Softsign,
            "mish": nn.Mish if hasattr(nn, "Mish") else None,
            "hardtanh": nn.Hardtanh,
            "hardswish": nn.Hardswish if hasattr(nn, "Hardswish") else None,
            "logsigmoid": nn.LogSigmoid,
            "identity": nn.Identity,
        }

        cls = registry.get(canonical, None)

        # Fallback: try attribute on torch.nn with the raw final token
        if cls is None:
            raw = act.strip().split(".")[-1]  # e.g., "ReLU", "LeakyReLU"
            cls = getattr(nn, raw, None)

        if cls is None:
            supported = sorted(k for k, v in registry.items() if v is not None)
            raise ValueError(f"Unknown activation_fn string: {act!r}. Supported: {supported}")

        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError(f"Resolved activation is not an nn.Module class: {act!r} -> {cls!r}")

        return cls

    raise ValueError(
        "activation_fn must be None, an nn.Module subclass/instance, or a string. "
        f"Got: {type(act)}"
    )


# =============================================================================
# Ray gating
# =============================================================================
def _require_ray() -> None:
    """
    Raise a clear error if Ray features are requested but Ray is not installed.

    Raises
    ------
    RuntimeError
        If Ray is not importable in the current environment.

    Notes
    -----
    This is meant to be called at the boundary where Ray-only code paths are entered
    (e.g., RayRunner construction). It avoids late, confusing NameError/ImportError
    failures inside worker setup code.
    """
    if ray is None:
        raise RuntimeError(
            "Ray is not installed, but RayRunner/RayEnvWorker was requested. "
            "Install Ray (e.g., `pip install ray`) or run with n_envs=1."
        )


# =============================================================================
# Policy weight export helpers
# =============================================================================
def _locate_head_module(algo: Any) -> nn.Module:
    """
    Locate a policy head module from an algorithm object.

    Supported patterns
    ------------------
    - `algo.policy.head`
    - `algo.head`

    Parameters
    ----------
    algo : Any
        Algorithm-like object.

    Returns
    -------
    head : nn.Module
        Located head module.

    Raises
    ------
    ValueError
        If no head module can be found or the found object is not an `nn.Module`.

    Notes
    -----
    This is intentionally permissive to support different algorithm wrappers
    (custom trainers, SB3-style, internal frameworks). If you have a stable
    interface, prefer a direct attribute access.
    """
    head = getattr(getattr(algo, "policy", None), "head", None)
    if head is None:
        head = getattr(algo, "head", None)

    if head is None or not isinstance(head, nn.Module):
        raise ValueError("Cannot locate algo.policy.head (or algo.head) to export weights.")

    return head


def _get_policy_state_dict_cpu(algo: Any) -> Dict[str, Any]:
    """
    Export policy/head weights as a CPU-only state dict.

    Parameters
    ----------
    algo : Any
        Algorithm object which contains `policy.head` or `head`.

    Returns
    -------
    state_dict : Dict[str, Any]
        CPU / detached weights as a plain Python dict.

    Notes
    -----
    - This is useful when sending weights across process boundaries
      (e.g., Ray workers) or storing checkpoints in a device-agnostic manner.
    - `_to_cpu_state_dict` should detach tensors and move them to CPU recursively.
    """
    head = _locate_head_module(algo)
    return _to_cpu_state_dict(head.state_dict())
