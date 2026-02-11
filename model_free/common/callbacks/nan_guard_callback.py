from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch as th

from .base_callback import BaseCallback
from ..utils.callback_utils import _infer_step


@dataclass(frozen=True)
class NonFiniteDetector:
    """
    Best-effort recursive detector for NaN/Inf values.

    This helper is intentionally conservative: it checks a limited set of types
    (scalars, torch tensors, numpy arrays, dicts, lists/tuples) and recurses up to
    `max_depth` to avoid expensive or risky deep traversal.

    Parameters
    ----------
    max_depth:
        Maximum recursion depth for container traversal. When `depth >= max_depth`,
        the detector stops descending and returns False for nested objects.

    Notes
    -----
    - Scalars are detected by attempting `float(x)` and checking `math.isfinite`.
    - Torch tensors are detected with `torch.isfinite`.
    - Numpy arrays are detected with `numpy.isfinite`.
    - Dicts recurse into values only.
    - Lists/tuples recurse into elements.
    - Unknown objects are treated as finite (False) to avoid crashes.
    """

    max_depth: int = 3

    def __call__(self, x: Any, *, depth: int = 0) -> bool:
        """
        Check whether `x` contains any non-finite value.

        Parameters
        ----------
        x:
            Value to inspect. May be a scalar, tensor/array, or a nested container.
        depth:
            Current recursion depth (used internally).

        Returns
        -------
        bool
            True if any non-finite (NaN/Inf) value is detected, otherwise False.
        """
        # Scalar-like: if convertible to float, treat as a scalar and finish.
        scalar_flag = NaNGuardCallback._is_non_finite_scalar(x)
        if scalar_flag is not None:
            return scalar_flag

        # Stop recursion if we've reached the maximum depth.
        if depth >= self.max_depth:
            return False

        # Torch tensor
        if isinstance(x, th.Tensor):
            try:
                return bool((~th.isfinite(x)).any().item())
            except Exception:
                return False

        # Numpy array
        if isinstance(x, np.ndarray):
            try:
                return bool((~np.isfinite(x)).any())
            except Exception:
                return False

        # Dict: recurse into values
        if isinstance(x, dict):
            for vv in x.values():
                if self(vv, depth=depth + 1):
                    return True
            return False

        # List/tuple: recurse into elements (only these containers by policy)
        it = NaNGuardCallback._iter_container(x)
        if it is not None:
            for y in it:
                if self(y, depth=depth + 1):
                    return True
            return False

        # Unknown types are treated as finite (best-effort, conservative).
        return False


class NaNGuardCallback(BaseCallback):
    """
    Request early stop when NaN/Inf is detected in step transitions or update metrics.

    Purpose
    -------
    Numerical issues (NaN/Inf) often indicate exploding gradients, invalid log/exp usage,
    division-by-zero, unstable optimizers, or environment bugs. This callback implements a
    lightweight, framework-agnostic "tripwire":

    - If any selected update metric becomes non-finite, `on_update()` returns False.
    - If any inspected step transition contains non-finite values, `on_step()` returns False.

    A Trainer (or callback runner) should treat a False return value as a request to stop
    training gracefully.

    Parameters
    ----------
    keys:
        If provided, only inspect these metric keys in `on_update(metrics=...)`.
        If None, inspect all items in the provided `metrics` dict.
    log_prefix:
        Prefix passed to the logger so NaN-guard diagnostics are namespaced (e.g. "sys/").
    max_key_len:
        Maximum length for the logged metric key string (truncated with ellipsis).
        This keeps dashboards readable and avoids UI limitations.
    max_depth:
        Maximum recursion depth for nested containers (lists/tuples and dicts) when scanning
        values for non-finite entries.

    Notes
    -----
    Detection policy (best-effort)
    ------------------------------
    The detector checks the following value types:

    - Scalar-like values convertible to float:
        uses ``math.isfinite(float(x))``
    - ``torch.Tensor``:
        uses ``torch.isfinite`` (any non-finite triggers)
    - ``numpy.ndarray``:
        uses ``numpy.isfinite`` (any non-finite triggers)
    - ``dict``:
        recurses into values up to `max_depth`
    - ``list``/``tuple``:
        recurses into elements up to `max_depth`
    - Other objects:
        treated as "unknown" and assumed finite (not checked)

    Logged diagnostics (minimal)
    ----------------------------
    When triggered, logs a compact payload:

    - ``nan_guard/triggered`` : 1.0
    - ``nan_guard/where``     : "metrics" or "transition"
    - ``nan_guard/key_code``  : CRC32 of the key string (stable numeric id; metrics only)
    - ``nan_guard/key``       : truncated key string (metrics only)
    - ``nan_guard/value``     : float value when scalar-like and convertible (metrics only)

    The callback swallows all logging errors (via BaseCallback.log) and never raises.

    See Also
    --------
    NonFiniteDetector:
        Recursive best-effort non-finite scanning utility used by this callback.
    """

    # =========================================================================
    # Small helpers
    # =========================================================================
    @staticmethod
    def _key_code_crc32(k: Any) -> int:
        """
        Compute a stable numeric code for a key using CRC32.

        Parameters
        ----------
        k:
            Key-like object to hash (converted to string).

        Returns
        -------
        int
            Unsigned 32-bit CRC value (0..2^32-1). Returns 0 on any failure.

        Notes
        -----
        CRC32 is useful when:
        - keys are long/noisy (e.g., nested metric names)
        - you want a compact numeric identifier for dashboards or grouping
        """
        try:
            s = str(k).encode("utf-8", errors="ignore")
            return int(zlib.crc32(s) & 0xFFFFFFFF)
        except Exception:
            return 0

    @staticmethod
    def _truncate_key(k: Any, *, max_len: int) -> str:
        """
        Convert a key to string and truncate it for logging.

        Parameters
        ----------
        k:
            Key-like object.
        max_len:
            Maximum length to keep. If `max_len <= 0`, returns an empty string.

        Returns
        -------
        str
            Truncated string representation of the key.
        """
        try:
            s = str(k)
        except Exception:
            s = "<unprintable>"

        if max_len <= 0:
            return ""
        if len(s) <= max_len:
            return s
        return s[:max_len] + "..."

    @staticmethod
    def _is_non_finite_scalar(x: Any) -> Optional[bool]:
        """
        Determine if `x` is a non-finite scalar (NaN/Inf), if scalar-like.

        Parameters
        ----------
        x:
            Candidate scalar-like value.

        Returns
        -------
        Optional[bool]
            - True/False if `x` can be converted to float.
            - None if `x` cannot be converted to float (treated as non-scalar/unknown).

        Notes
        -----
        This method is intentionally permissive: anything convertible by `float(x)` is
        treated as scalar-like.
        """
        try:
            fx = float(x)
            return not math.isfinite(fx)
        except Exception:
            return None

    @staticmethod
    def _iter_container(x: Any) -> Optional[Iterable[Any]]:
        """
        Return an iterable view for container types we recurse into.

        Parameters
        ----------
        x:
            Candidate container.

        Returns
        -------
        Optional[Iterable[Any]]
            Returns `x` if it is a list/tuple, else None.

        Notes
        -----
        Current policy is conservative:
        - Only list/tuple are treated as containers for recursion.
        - Other containers (set, dict) are handled elsewhere (dict explicitly),
          or ignored to avoid surprising traversal semantics.
        """
        if isinstance(x, (list, tuple)):
            return x
        return None

    # =========================================================================
    # Init / hooks
    # =========================================================================
    def __init__(
        self,
        keys: Optional[Sequence[str]] = None,
        *,
        log_prefix: str = "sys/",
        max_key_len: int = 120,
        max_depth: int = 3,
    ) -> None:
        self.keys = None if keys is None else [str(k) for k in keys]
        self.log_prefix = str(log_prefix)
        self.max_key_len = int(max_key_len)

        # Recursive detector instance controlling traversal depth.
        self._detector = NonFiniteDetector(max_depth=int(max_depth))

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Inspect update metrics for NaN/Inf and request stop if detected.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Used only for logging and step inference.
        metrics:
            Update metrics dict produced by an algorithm/trainer.

        Returns
        -------
        bool
            True  -> continue training
            False -> request stop (non-finite detected)

        Notes
        -----
        - If `keys` was provided at construction, only those metrics are checked.
        - Missing keys are treated as absent and skipped.
        """
        if not metrics or not isinstance(metrics, dict):
            return True

        # Select which items to scan.
        if self.keys is None:
            items = list(metrics.items())
        else:
            items = [(k, metrics.get(k, None)) for k in self.keys]

        for k, v in items:
            if v is None:
                continue

            if self._detector(v):
                payload: Dict[str, Any] = {
                    "nan_guard/triggered": 1.0,
                    "nan_guard/where": "metrics",
                    "nan_guard/key_code": float(self._key_code_crc32(k)),
                    "nan_guard/key": self._truncate_key(k, max_len=self.max_key_len),
                }

                # If scalar-like, include the offending value for quick diagnosis (NaN vs Inf).
                scalar_flag = self._is_non_finite_scalar(v)
                if scalar_flag is not None:
                    try:
                        payload["nan_guard/value"] = float(v)
                    except Exception:
                        pass

                self.log(trainer, payload, step=_infer_step(trainer), prefix=self.log_prefix)
                return False

        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Inspect per-step transition payload for NaN/Inf and request stop if detected.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Used only for logging and step inference.
        transition:
            Per-step transition payload emitted by the training loop. This may contain
            observations/actions/rewards/dones/infos, etc.

        Returns
        -------
        bool
            True  -> continue training
            False -> request stop (non-finite detected)

        Notes
        -----
        - Transition scanning can be useful to catch NaNs originating from environments,
          observation normalization, reward shaping, or action sampling.
        - This uses the same best-effort detector and depth policy as `on_update`.
        """
        if transition is None:
            return True

        if self._detector(transition):
            payload: Dict[str, Any] = {
                "nan_guard/triggered": 1.0,
                "nan_guard/where": "transition",
            }
            self.log(trainer, payload, step=_infer_step(trainer), prefix=self.log_prefix)
            return False

        return True
