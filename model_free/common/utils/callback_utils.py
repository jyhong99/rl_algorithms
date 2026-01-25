from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import math


def safe_int_attr(trainer: Any, default: int = 0) -> int:
    """
    Best-effort integer step accessor.

    Priority:
      1) update_step / global_update_step  (for on_update-scheduled callbacks)
      2) step / global_env_step            (for on_step-scheduled callbacks)

    Rationale:
      Some callbacks (e.g., LRLoggingCallback) call safe_int_attr(trainer) inside on_update.
      If we always read env step first, upd may be 0 and gates never fire.
    """
    keys = ("update_step", "global_update_step", "step", "global_env_step")
    for k in keys:
        try:
            v = getattr(trainer, k, None)
            if v is None:
                continue
            iv = int(v)
            # allow 0, but return the first available integer
            return iv
        except Exception:
            continue
    return int(default)


# =============================================================================
# Scheduling gate
# =============================================================================
@dataclass
class IntervalGate:
    """
    Gate that decides whether to run an action based on a monotonically increasing counter.

    Modes
    -----
    "mod":
        Trigger when counter % every == 0.
        Suitable when the hook is called exactly once per increment and you want exact multiples.

    "delta":
        Trigger when counter - last >= every, then set last = counter.
        More robust to:
          - irregular hook calls
          - step jumps (e.g., resumed training, batched stepping)
          - missed callbacks

    Attributes
    ----------
    every : int
        Interval for triggering. If <= 0, gate is disabled (always False).
    mode : str, default="mod"
        "mod" or "delta".
    last : int, default=0
        Only used for mode="delta" to track the last triggering counter.

    Example
    -------
    gate = IntervalGate(every=200, mode="mod")
    if gate.ready(update_step):
        ...
    """
    every: int
    mode: str = "mod"   # "mod" or "delta"
    last: int = 0

    def ready(self, counter: int) -> bool:
        """
        Returns True if the gate condition is met for the given counter.
        """
        e = int(self.every)
        if e <= 0:
            return False

        c = int(counter)
        if c <= 0:
            return False

        if self.mode == "mod":
            return (c % e) == 0

        if self.mode == "delta":
            # Only trigger when enough distance since last trigger
            if (c - int(self.last)) < e:
                return False
            self.last = c
            return True

        raise ValueError(f"Unknown gate mode: {self.mode}")


# =============================================================================
# Scalar coercion utilities
# =============================================================================
def to_finite_float(x: Any) -> Optional[float]:
    """
    Convert x to a finite float if possible; else return None.

    Notes
    -----
    - Filters out NaN/Inf to keep logs JSON-friendly and stable.
    - Safe for numpy/torch scalar-likes (via float(x)) if they implement __float__.
    """
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def coerce_scalar_mapping(m: Mapping[str, Any]) -> Dict[str, float]:
    """
    Keep only finite float scalars from mapping (log-sink friendly).

    Parameters
    ----------
    m : Mapping[str, Any]
        Input metrics mapping.

    Returns
    -------
    out : Dict[str, float]
        Dictionary containing only entries where value can be converted to a finite float.
        Keys are stringified with str(k).

    Notes
    -----
    - This is useful before sending metrics to sinks that require scalar floats
      (e.g., Ray Tune, many dashboards).
    """
    out: Dict[str, float] = {}
    for k, v in m.items():
        fv = to_finite_float(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def infer_step(trainer: Any) -> int:
    """
    Best-effort resolve a logging step.
    Prefer env-step counters; fallback to 0.
    """
    # 환경 step 카운터가 가장 흔함
    for k in ("env_steps", "env_step", "global_step", "step", "timesteps", "total_steps"):
        v = getattr(trainer, k, None)
        if isinstance(v, int) and v >= 0:
            return int(v)
    return 0
