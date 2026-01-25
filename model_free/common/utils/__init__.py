"""
Core utilities for policy construction, activation resolution, and
CPU-safe tensor / state handling.

This module aggregates frequently used infrastructure utilities, including:

- Entrypoint-based policy factory specifications for Ray workers
- Activation function resolution from flexible user specifications
- CPU-safe serialization and state_dict export helpers
- Observation formatting and NumPy ↔ Torch conversion utilities
- Common RL helpers (Polyak averaging, action formatting, shape inference)

Design goals
------------
- Safe to import without optional dependencies (e.g., Ray)
- Explicit, stable public API via __all__
- Suitable for both single-process and distributed (Ray-based) execution
"""


from .ray_utils import (
    PolicyFactorySpec,
    make_entrypoint,
    resolve_entrypoint,
    build_policy_from_spec,
    resolve_activation_fn,
    require_ray,
    get_policy_state_dict_cpu,
)

from .common_utils import (
    to_numpy,
    to_tensor,
    to_flat_np,
    to_scalar,
    is_scalar_like,
    require_scalar_like,
    to_action_np,
    to_column,
    to_cpu,
    to_cpu_state_dict,
    obs_to_cpu_tensor,
    polyak_update,
    ema_update,
    require_mapping,
    infer_shape,
    img2col,
)

from .wrapper_utils import (
    MinimalWrapper,
    RunningMeanStd,
    RunningMeanStdState
)

__all__ = [
    "PolicyFactorySpec",
    "make_entrypoint",
    "resolve_entrypoint",
    "build_policy_from_spec",
    "resolve_activation_fn",
    "require_ray",
    "get_policy_state_dict_cpu",

    "to_numpy",
    "to_tensor",
    "to_flat_np",
    "to_scalar",
    "is_scalar_like",
    "require_scalar_like",
    "to_action_np",
    "to_column",
    "to_cpu",
    "to_cpu_state_dict",
    "obs_to_cpu_tensor",
    "require_mapping",
    "infer_shape",
    "polyak_update",
    "ema_update",
    "img2col",

    "MinimalWrapper",
    "RunningMeanStd",
    "RunningMeanStdState"
]
