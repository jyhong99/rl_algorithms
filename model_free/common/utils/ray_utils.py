from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type
import importlib

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover
    # Ray is an optional dependency. This module must remain importable without it.
    ray = None

import torch.nn as nn

from .common_utils import to_cpu_state_dict


# =============================================================================
# Entrypoint-based policy factory
# =============================================================================

@dataclass(frozen=True)
class PolicyFactorySpec:
    """
    Pure-Python description of how to build a worker-side policy.

    Parameters
    ----------
    entrypoint : str
        Factory entrypoint in the form "package.module:function_name".
        The target must be importable on Ray workers and must be a top-level
        function (not a nested function or lambda).
    kwargs : Dict[str, Any]
        Keyword arguments passed to the factory.

        Notes
        -----
        When used with Ray, `kwargs` must be pickle-serializable (and ideally JSON-safe).
        Do NOT include:
          - torch.Tensor
          - nn.Module
          - th.device
          - CUDA storages / non-serializable handles
    """
    entrypoint: str
    kwargs: Dict[str, Any]


def make_entrypoint(fn: Callable[..., Any]) -> str:
    """
    Convert a top-level function to an entrypoint string "module:function".

    Parameters
    ----------
    fn : Callable[..., Any]
        Top-level function.

    Returns
    -------
    entrypoint : str
        Entrypoint string.

    Raises
    ------
    TypeError
        If `fn` is not callable.
    ValueError
        If `fn` is not a top-level function or module/name cannot be inferred.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")

    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    name = getattr(fn, "__name__", None)

    if not mod or not qual or not name:
        raise ValueError("Cannot infer module/qualname/name for entrypoint.")

    # Ray workers must import it; nested functions cannot be imported by name.
    if "<locals>" in qual:
        raise ValueError(
            "Entrypoint must be a top-level function (not nested). "
            f"Got qualname={qual!r}"
        )

    return f"{mod}:{name}"


def resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """
    Resolve "package.module:function_name" to a callable.

    Parameters
    ----------
    entrypoint : str
        Entrypoint string.

    Returns
    -------
    fn : Callable[..., Any]
        Resolved callable.

    Raises
    ------
    ValueError
        If format is invalid.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the function is missing in the module.
    TypeError
        If the resolved object is not callable.
    """
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint format: {entrypoint!r} (expected 'module:function')")

    mod_name, fn_name = entrypoint.split(":", 1)
    if not mod_name or not fn_name:
        raise ValueError(f"Invalid entrypoint format: {entrypoint!r} (empty module/function)")

    mod = importlib.import_module(mod_name)
    obj = getattr(mod, fn_name)  # may raise AttributeError

    if not callable(obj):
        raise TypeError(f"Entrypoint is not callable: {entrypoint!r} -> {type(obj)}")

    return obj


def build_policy_from_spec(spec: PolicyFactorySpec) -> nn.Module:
    """
    Build a policy module from a PolicyFactorySpec.

    Parameters
    ----------
    spec : PolicyFactorySpec
        Factory spec.

    Returns
    -------
    policy : nn.Module
        Instantiated policy, moved to CPU and set to eval mode.

    Raises
    ------
    TypeError
        If the factory does not return an nn.Module.
    """
    fn = resolve_entrypoint(spec.entrypoint)
    policy = fn(**dict(spec.kwargs))

    if not isinstance(policy, nn.Module):
        raise TypeError(
            "Policy factory must return torch.nn.Module. "
            f"Got: {type(policy)} from entrypoint={spec.entrypoint!r}"
        )

    # Worker-safe default: CPU + eval
    policy = policy.to("cpu")
    policy.eval()
    return policy


# =============================================================================
# Activation function resolver
# =============================================================================

def _normalize_activation_name(name: str) -> str:
    n = name.strip()
    if n.startswith("torch.nn."):
        n = n[len("torch.nn.") :]
    if n.startswith("nn."):
        n = n[len("nn.") :]

    # "Leaky ReLU" -> "leakyrelu"
    key = n.lower().replace("-", "_").replace(" ", "").replace(".", "_")
    return key


def resolve_activation_fn(act: Any, *, default: Type[nn.Module] = nn.ReLU) -> Type[nn.Module]:
    """
    Resolve activation spec to an nn.Module class (not instance).

    Parameters
    ----------
    act : Any
        One of:
        - None: returns `default`
        - nn.Module subclass: returned as-is
        - nn.Module instance: returns its class
        - str: class name or alias (e.g., "relu", "torch.nn.ReLU", "LeakyReLU")
    default : Type[nn.Module], default=nn.ReLU
        Default activation if act is None.

    Returns
    -------
    cls : Type[nn.Module]
        Activation module class.

    Raises
    ------
    ValueError
        If `act` cannot be resolved to an nn.Module subclass.
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

        # Fallback: try attribute on torch.nn with the *original* tokenized name
        if cls is None:
            # attempt with "ReLU", "LeakyReLU", etc.
            raw = act.strip().split(".")[-1]
            cls = getattr(nn, raw, None)

        if cls is None:
            supported = sorted(k for k, v in registry.items() if v is not None)
            raise ValueError(f"Unknown activation_fn string: {act!r}. Supported: {supported}")

        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise ValueError(f"Resolved activation is not an nn.Module class: {act!r} -> {cls!r}")

        return cls

    raise ValueError(
        "activation_fn must be None, an nn.Module subclass/instance, or a string. "
        f"Got: {type(act)}"
    )


# =============================================================================
# Ray gating
# =============================================================================

def require_ray() -> None:
    """
    Raise a clear error when Ray features are requested without Ray installed.

    Raises
    ------
    RuntimeError
        If Ray is not installed.
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
    Locate the policy head module from an algorithm object.

    Supported patterns
    ------------------
    - algo.policy.head
    - algo.head

    Raises
    ------
    ValueError
        If no nn.Module head is found.
    """
    head = getattr(getattr(algo, "policy", None), "head", None)
    if head is None:
        head = getattr(algo, "head", None)

    if head is None or not isinstance(head, nn.Module):
        raise ValueError("Cannot locate algo.policy.head (or algo.head) to export weights.")

    return head


def get_policy_state_dict_cpu(algo: Any) -> Dict[str, Any]:
    """
    Export policy/head weights as a CPU-only state_dict.

    Parameters
    ----------
    algo : Any
        Algorithm object which contains `policy.head` or `head`.

    Returns
    -------
    state_dict : Dict[str, Any]
        CPU / detached weights.
    """
    head = _locate_head_module(algo)
    return to_cpu_state_dict(head.state_dict())
