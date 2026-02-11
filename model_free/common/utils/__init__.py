"""
Utils
====================

This subpackage aggregates small, reusable helpers used across the codebase.

Modules included
----------------
- buffer_utils
    Prioritized Experience Replay (PER) segment trees and related utilities.
- callback_utils
    Scheduling gates and trainer-step inspection helpers for callbacks/logging.
- common_utils
    NumPy/Torch conversion helpers, scalar coercion, and small tensor utilities.
- logger_utils
    Run-directory management and lightweight CSV/JSON serialization helpers.
- network_utils
    Neural-network helpers (tanh bijector, dueling combine, weight init, etc.).
- noise_utils
    Noise-configuration normalization utilities for exploration/noise modules.
- policy_utils
    Policy/critic helpers, including distributional RL utilities (QR/C51) and
    target-network update helpers.
- ray_utils
    Ray-related helpers: entrypoint-based factory specs, activation resolver,
    and CPU-safe policy export utilities.
- train_utils
    Training helpers: seeding, gym/gymnasium compatibility adapters, normalize
    wrapper synchronization, and action formatting utilities.
- wrapper_utils
    Minimal wrapper base and running mean/std utilities for normalization.

Design policy
-------------
- This package exposes a curated set of names via `__all__`.
- Functions prefixed with '_' are intentionally treated as "semi-private":
  importable for internal use, but not guaranteed as a stable public API.
  (They are still exported here for convenience in research code.)

Notes
-----
- Importing this package will import all listed submodules and bind their
  exported symbols into this namespace. If import time becomes important,
  consider moving optional dependencies (e.g., Ray) behind lazy imports.
"""

from __future__ import annotations

# =============================================================================
# Replay buffer / PER utilities
# =============================================================================
from .buffer_utils import MinSegmentTree, SegmentTree, SumSegmentTree

# =============================================================================
# Callback / scheduling utilities
# =============================================================================
from .callback_utils import (
    IntervalGate,
    _coerce_scalar_mapping,
    _infer_step,
    _safe_env_step,
    _safe_update_step,
)

# =============================================================================
# Common NumPy/Torch utilities
# =============================================================================
from .common_utils import (
    _ema_update,
    _polyak_update,
    _to_action_np,
    _to_cpu_state_dict,
    _to_flat_np,
    _to_numpy,
    _to_scalar,
    _to_tensor,
)

# =============================================================================
# Logger utilities
# =============================================================================
from .logger_utils import (
    _ensure_dir,
    _extract_meta,
    _generate_run_id,
    _get_step,
    _json_dumps,
    _make_run_dir,
    _open_append,
    _read_csv_header,
    _safe_call,
    _safe_file_size,
    _seek_eof,
    _split_meta,
)

# =============================================================================
# Network utilities
# =============================================================================
from .network_utils import (
    DuelingMixin,
    TanhBijector,
    _make_weights_init,
    _validate_hidden_sizes,
)

# =============================================================================
# Noise utilities
# =============================================================================
from .noise_utils import _as_flat_bounds, _normalize_kind, _normalize_size

# =============================================================================
# Policy / distributional RL utilities
# =============================================================================
from .policy_utils import (
    _distribution_projection,
    _freeze_target,
    _get_per_weights,
    _hard_update,
    _infer_n_actions_from_env,
    _quantile_huber_loss,
    _soft_update,
    _unfreeze_target,
    _validate_action_bounds,
)

# =============================================================================
# Ray utilities
# =============================================================================
from .ray_utils import (
    PolicyFactorySpec,
    _build_policy_from_spec,
    _get_policy_state_dict_cpu,
    _make_entrypoint,
    _require_ray,
    _resolve_activation_fn,
    _resolve_entrypoint,
)

# =============================================================================
# Training / env compatibility utilities
# =============================================================================
from .train_utils import (
    _env_reset,
    _format_env_action,
    _set_random_seed,
    _sync_normalize_state,
    _unpack_step,
)

# =============================================================================
# Wrappers / running statistics
# =============================================================================
from .wrapper_utils import MinimalWrapper, RunningMeanStd, RunningMeanStdState


__all__ = [
    # -------------------------------------------------------------------------
    # Replay / PER
    # -------------------------------------------------------------------------
    "SegmentTree",
    "SumSegmentTree",
    "MinSegmentTree",
    # -------------------------------------------------------------------------
    # Callback / scheduling
    # -------------------------------------------------------------------------
    "IntervalGate",
    "_safe_env_step",
    "_safe_update_step",
    "_infer_step",
    "_coerce_scalar_mapping",
    # -------------------------------------------------------------------------
    # Common utils (NumPy/Torch)
    # -------------------------------------------------------------------------
    "_to_numpy",
    "_to_tensor",
    "_to_flat_np",
    "_to_scalar",
    "_to_action_np",
    "_to_cpu_state_dict",
    "_polyak_update",
    "_ema_update",
    # -------------------------------------------------------------------------
    # Logger utilities
    # -------------------------------------------------------------------------
    "_generate_run_id",
    "_make_run_dir",
    "_split_meta",
    "_get_step",
    "_json_dumps",
    "_ensure_dir",
    "_open_append",
    "_safe_call",
    "_safe_file_size",
    "_seek_eof",
    "_read_csv_header",
    "_extract_meta",
    # -------------------------------------------------------------------------
    # Network utilities
    # -------------------------------------------------------------------------
    "TanhBijector",
    "DuelingMixin",
    "_make_weights_init",
    "_validate_hidden_sizes",
    # -------------------------------------------------------------------------
    # Noise utilities
    # -------------------------------------------------------------------------
    "_normalize_size",
    "_normalize_kind",
    "_as_flat_bounds",
    # -------------------------------------------------------------------------
    # Policy / distributional RL utilities
    # -------------------------------------------------------------------------
    "_quantile_huber_loss",
    "_distribution_projection",
    "_freeze_target",
    "_unfreeze_target",
    "_hard_update",
    "_soft_update",
    "_get_per_weights",
    "_infer_n_actions_from_env",
    "_validate_action_bounds",
    # -------------------------------------------------------------------------
    # Ray utilities
    # -------------------------------------------------------------------------
    "PolicyFactorySpec",
    "_make_entrypoint",
    "_resolve_entrypoint",
    "_build_policy_from_spec",
    "_resolve_activation_fn",
    "_require_ray",
    "_get_policy_state_dict_cpu",
    # -------------------------------------------------------------------------
    # Training / env compatibility utilities
    # -------------------------------------------------------------------------
    "_set_random_seed",
    "_env_reset",
    "_unpack_step",
    "_sync_normalize_state",
    "_format_env_action",
    # -------------------------------------------------------------------------
    # Wrappers
    # -------------------------------------------------------------------------
    "MinimalWrapper",
    "RunningMeanStd",
    "RunningMeanStdState",
]
