from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Iterable

import math
import zlib

from .base_callback import BaseCallback
from ..utils.log_utils import log
from ..utils.callback_utils import infer_step


@dataclass(frozen=True)
class NonFiniteDetector:
    """
    Best-effort recursive non-finite detector.

    Rules
    -----
    1) If scalar convertible to float -> non-finite if not math.isfinite(...)
    2) If list/tuple -> recurse into elements up to `max_depth`
    3) Otherwise -> treat as finite (unknown)
    """
    max_depth: int = 3

    def __call__(self, x: Any, *, depth: int = 0) -> bool:
        # First try scalar path (fast path).
        s = NaNGuardCallback._is_non_finite_scalar(x)
        if s is not None:
            return s

        # Stop recursion at max depth (treat deeper structures as safe/unknown).
        if depth >= self.max_depth:
            return False

        # Recurse into list/tuple if applicable.
        it = NaNGuardCallback._iter_container(x)
        if it is not None:
            for y in it:
                if self(y, depth=depth + 1):
                    return True
            return False

        # Unknown type: do not attempt heavy inspection.
        return False
        

class NaNGuardCallback(BaseCallback):
    """
    Stop training when NaN/Inf is detected in update metrics (best-effort).

    Purpose
    -------
    Numerical issues (NaN/Inf) often indicate exploding gradients, invalid log/exp,
    division by zero, or unstable optimizers. This callback provides a simple,
    framework-agnostic safety stop: if any selected update metric becomes non-finite,
    it logs a small diagnostic payload and returns False from `on_update()`, which
    should stop training by Trainer/callback contract.

    Typical usage
    -------------
    - Attach this callback to a Trainer that calls callbacks.on_update(trainer, metrics).
    - If a metric becomes non-finite (NaN/Inf), this callback logs:
        * nan_guard/triggered = 1
        * nan_guard/key_code  (CRC32 of key string, stable numeric identifier)
        * nan_guard/key       (truncated key string)
        * nan_guard/value     (if scalar-like and convertible)
      and returns False (stop signal).

    Detection policy (best-effort)
    ------------------------------
    - Scalar-like values convertible to float: check `math.isfinite(float(x))`.
    - list/tuple containers: recurse into elements up to `max_depth`.
    - Other objects (dicts, tensors, custom classes, etc.): treated as "unknown"
      and assumed finite (not checked) to avoid heavy logic or accidental crashes.

    Parameters
    ----------
    keys : Optional[Sequence[str]]
        If provided, only check these metric keys.
        If None, check all key/value pairs in `metrics`.
    log_prefix : str, default="sys/"
        Prefix used in logger.
    max_key_len : int, default=120
        Maximum length for the logged key string (for readability / UI limits).
    max_depth : int, default=3
        Recursion depth for list/tuple containers.
    """

    # =========================================================================
    # Small helpers
    # =========================================================================
    @staticmethod
    def _key_code_crc32(k: Any) -> int:
        """
        Generate a stable numeric code for a key.

        Why CRC32?
        ----------
        - Always loggable (numeric)
        - Stable across runs given the same key string
        - Useful when full key strings are too long or noisy
        """
        try:
            s = str(k).encode("utf-8", errors="ignore")
            return int(zlib.crc32(s) & 0xFFFFFFFF)
        except Exception:
            return 0

    @staticmethod
    def _truncate_key(k: Any, *, max_len: int) -> str:
        """
        Convert a key to string and truncate for logging.

        Notes
        -----
        - Logging systems often benefit from bounded key lengths (UI/readability).
        - If conversion fails, uses a placeholder string.
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
        Check whether `x` is a non-finite scalar (NaN/Inf), if scalar-like.

        Returns
        -------
        Optional[bool]
            - True/False if `x` can be converted to float (scalar-like)
            - None if `x` is not convertible to float (unknown / not checked)
        """
        try:
            fx = float(x)
            return not math.isfinite(fx)
        except Exception:
            return None

    @staticmethod
    def _iter_container(x: Any) -> Optional[Iterable[Any]]:
        """
        Return an iterable view for common containers we want to recurse into.

        Current policy
        --------------
        - Only list/tuple are treated as containers for recursion.
        - dict/set/etc. are intentionally ignored to keep behavior conservative.
        """
        if isinstance(x, (list, tuple)):
            return x
        return None

    # =========================================================================
    # Init / hook
    # =========================================================================
    def __init__(
        self,
        keys: Optional[Sequence[str]] = None,
        *,
        log_prefix: str = "sys/",
        max_key_len: int = 120,
        max_depth: int = 3,
    ) -> None:
        # If keys are provided, normalize them to strings once.
        # This avoids repeated conversions inside on_update.
        self.keys = None if keys is None else [str(k) for k in keys]

        # Logging configuration.
        self.log_prefix = str(log_prefix)
        self.max_key_len = int(max_key_len)

        # Recursive detector instance (max_depth controls list/tuple recursion).
        self._detector = NonFiniteDetector(max_depth=int(max_depth))

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Inspect update metrics for NaN/Inf and stop training if detected.

        Returns
        -------
        bool
            True  -> continue training
            False -> request stop (non-finite detected)
        """
        # Defensive: ignore missing/non-dict metrics.
        if not metrics or not isinstance(metrics, dict):
            return True

        # Determine which key/value pairs to inspect:
        # - If self.keys is None: inspect all metrics
        # - Else: inspect only the requested keys (missing keys yield None)
        if self.keys is None:
            items = list(metrics.items())
        else:
            items = [(k, metrics.get(k, None)) for k in self.keys]

        # Scan selected metrics for any non-finite value.
        for k, v in items:
            if v is None:
                continue

            # Best-effort recursion into scalar/list/tuple structures.
            if self._detector(v):
                # Minimal diagnostic payload:
                # - triggered flag
                # - stable numeric key code (CRC32)
                # - truncated key string for readability
                payload: Dict[str, Any] = {
                    "nan_guard/triggered": 1.0,
                    "nan_guard/key_code": float(self._key_code_crc32(k)),
                    "nan_guard/key": self._truncate_key(k, max_len=self.max_key_len),
                }

                # If the problematic value is scalar-like, include its float value.
                # This is useful for quickly seeing whether it is NaN vs Inf.
                s = self._is_non_finite_scalar(v)
                if s is not None:
                    try:
                        payload["nan_guard/value"] = float(v)
                    except Exception:
                        pass

                # Log at current trainer step (best-effort).
                log(trainer, payload, step=infer_step(trainer), prefix=self.log_prefix)

                # Returning False is the stop-signal to the Trainer/callback runner.
                return False

        return True
