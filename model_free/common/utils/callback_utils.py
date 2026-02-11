from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional
import math


# =============================================================================
# Safe step accessors (trainer-agnostic)
# =============================================================================
def _safe_first_int_attr(trainer: Any, keys: Iterable[str], default: int = 0) -> int:
    """
    Read the first available integer-like attribute from a prioritized key list.

    This is a defensive helper for "trainer" objects that may come from different
    frameworks (custom Trainer, RLlib-like trainers, etc.) and therefore expose
    counters under different attribute names.

    Parameters
    ----------
    trainer : Any
        Trainer-like object from which to read attributes.
    keys : Iterable[str]
        Attribute names to try in order of priority. The first attribute that
        exists, is not None, and can be cast to ``int`` is returned.
    default : int, default=0
        Fallback integer returned if no suitable attribute is found.

    Returns
    -------
    out : int
        The first successfully retrieved and int-cast attribute value, otherwise
        ``default``.

    Notes
    -----
    - Errors during attribute access or casting are swallowed intentionally.
    - This function treats any int-castable object (e.g., numpy scalar) as valid.
    """
    for k in keys:
        try:
            v = getattr(trainer, k, None)
            if v is None:
                continue
            return int(v)
        except Exception:
            continue
    return int(default)


def _safe_env_step(trainer: Any, default: int = 0) -> int:
    """
    Best-effort environment-step accessor.

    Intended use cases
    ------------------
    - Callback hooks called per environment interaction (``on_step``-style).
    - Scheduling logging / evaluation based on number of environment steps.
    - Reporting progress in terms of interactions rather than gradient updates.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with some environment step counter attribute.
    default : int, default=0
        Fallback integer returned if no suitable attribute is found.

    Returns
    -------
    env_step : int
        Environment step count (best-effort), else ``default``.

    Notes
    -----
    The key list is intentionally redundant to cover common naming conventions.
    """
    env_keys = (
        "global_env_step",
        "env_step",
        "env_steps",
        "step",
        "total_env_steps",
        "total_timesteps",
        "timesteps",
        "total_steps",
    )
    return _safe_first_int_attr(trainer, env_keys, default=default)


def _safe_update_step(trainer: Any, default: int = 0) -> int:
    """
    Best-effort update-step accessor (i.e., gradient / optimizer update counter).

    Intended use cases
    ------------------
    - Callback hooks called per update (``on_update``-style).
    - Learning-rate schedules keyed by number of updates.
    - Logging optimizer statistics per update iteration.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with some update counter attribute.
    default : int, default=0
        Fallback integer returned if no suitable attribute is found.

    Returns
    -------
    update_step : int
        Update step count (best-effort), else ``default``.
    """
    upd_keys = (
        "global_update_step",
        "update_step",
        "updates",
        "upd",
        "total_updates",
        "num_updates",
    )
    return _safe_first_int_attr(trainer, upd_keys, default=default)


# =============================================================================
# Scheduling gate
# =============================================================================
@dataclass
class IntervalGate:
    """
    Interval-based trigger gate for callbacks and periodic actions.

    The gate decides whether an action should run based on a monotonically
    increasing counter (e.g., env steps, update steps).

    Parameters
    ----------
    every : int
        Trigger interval.
        If ``every <= 0``, the gate is disabled and always returns False.
    mode : str, default="mod"
        Triggering mode. One of:
        - ``"mod"``  : triggers when ``counter % every == 0``.
        - ``"delta"``: triggers when ``counter - last >= every`` and updates ``last``.
    last : int, default=0
        Last-trigger counter value (used only in ``mode="delta"``).

    Notes
    -----
    ``mode="mod"``
        Best when the hook is called exactly once per increment and you want
        triggers at exact multiples (e.g., 200, 400, 600, ...).

    ``mode="delta"``
        Robust to irregular calls and counter jumps:
          - batched stepping (counter increases by >1 between hook calls)
          - missed callbacks
          - resuming from checkpoints with a larger counter

        It behaves like an "at least every N steps" throttle.

    Examples
    --------
    >>> gate = IntervalGate(every=200, mode="mod")
    >>> gate.ready(200)
    True

    >>> gate = IntervalGate(every=200, mode="delta")
    >>> [gate.ready(s) for s in (50, 199, 200, 250, 399, 400)]
    [False, False, True, False, False, True]
    """
    every: int
    mode: str = "mod"  # "mod" or "delta"
    last: int = 0

    def ready(self, counter: int) -> bool:
        """
        Check whether the gate should trigger at the given counter value.

        Parameters
        ----------
        counter : int
            Current monotonically increasing counter.

        Returns
        -------
        trigger : bool
            True if the gate condition is met.

        Raises
        ------
        ValueError
            If ``mode`` is not one of {"mod", "delta"}.

        Notes
        -----
        - If ``counter <= 0``, this returns False (treats non-positive counters as
          "not started yet").
        - For ``mode="delta"``, triggering updates ``self.last`` to ``counter``.
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
            if (c - int(self.last)) < e:
                return False
            self.last = c
            return True

        raise ValueError(f"Unknown gate mode: {self.mode!r}")


# =============================================================================
# Scalar coercion utilities (metrics/logging)
# =============================================================================
def _to_finite_float(x: Any) -> Optional[float]:
    """
    Convert an object to a finite Python float, else return None.

    Parameters
    ----------
    x : Any
        Input value (may be a Python scalar, numpy scalar, or torch-like scalar).

    Returns
    -------
    v : float or None
        Finite float value if conversion succeeds and the value is finite.
        Returns None if conversion fails or results in NaN/Inf.

    Notes
    -----
    - Filters out NaN/Inf to keep logs JSON-friendly and downstream-safe.
    - Accepts scalar-likes that implement ``__float__`` (numpy/torch scalars).
    """
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _coerce_scalar_mapping(m: Mapping[str, Any]) -> Dict[str, float]:
    """
    Extract only finite float scalars from a mapping (log-sink friendly).

    Parameters
    ----------
    m : Mapping[str, Any]
        Input metrics mapping.

    Returns
    -------
    out : Dict[str, float]
        Dictionary containing only entries whose values can be converted to a
        finite float. Keys are converted to strings with ``str(k)``.

    Notes
    -----
    Useful before sending metrics to sinks that require scalar floats
    (e.g., Ray Tune, TensorBoard scalars, many dashboards).
    """
    out: Dict[str, float] = {}
    for k, v in m.items():
        fv = _to_finite_float(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def _infer_step(trainer: Any, default: int = 0) -> int:
    """
    Infer a reasonable "logging step" from a trainer-like object.

    This is a convenience wrapper that prefers environment-step semantics (the
    most common notion of "global step" in RL logging), but falls back to a
    default value if not available.

    Parameters
    ----------
    trainer : Any
        Trainer-like object.
    default : int, default=0
        Fallback step if no suitable attribute is found.

    Returns
    -------
    step : int
        Best-effort resolved step (non-negative). If nothing is found, returns
        ``default``.

    Notes
    -----
    If you want explicit control, prefer calling:
      - ``_safe_env_step(trainer)``
      - ``_safe_update_step(trainer)``
    depending on your scheduling semantics.
    """
    step = _safe_env_step(trainer, default=default)
    if step > 0:
        return step

    # Secondary fallback: some trainers use different naming for "global step".
    step = _safe_first_int_attr(
        trainer,
        keys=("global_step", "steps", "iteration", "iters", "train_step"),
        default=default,
    )
    return step if step >= 0 else int(default)
