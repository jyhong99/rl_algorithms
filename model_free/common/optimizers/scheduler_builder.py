from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
    _LRScheduler,
)


# =============================================================================
# Public API
# =============================================================================
def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str = "none",
    # common / lambda-based
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    # step / multistep / exp
    step_size: int = 1000,
    gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # onecycle
    max_lr: Optional[Union[float, Sequence[float]]] = None,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Optional[_LRScheduler]:
    """
    Construct a PyTorch learning-rate scheduler.

    This factory builds a learning-rate scheduler from a string identifier.
    It supports both:
    - **LambdaLR-based** schedules (linear/cosine/poly, with optional warmup)
    - **Classic** schedules (StepLR, MultiStepLR, ExponentialLR)
    - **OneCycleLR**

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Target optimizer whose param group learning rates will be scheduled.

        Notes
        -----
        LambdaLR, StepLR, etc. all operate by mutating `optimizer.param_groups[i]["lr"]`.
        Therefore:
        - You must call `scheduler.step()` at the correct cadence (usually once per
          `optimizer.step()` for step-based schedules).
        - If you call it per-epoch, interpret "steps" below as epochs instead.

    name : str, default="none"
        Scheduler identifier (case-insensitive). Hyphens/spaces are normalized.
        Supported values:

        - "none" / "constant"
            No scheduling (returns None).
        - "linear"
            Linear warmup (optional) then linear decay to `min_lr_ratio`.
        - "cosine"
            Linear warmup (optional) then cosine decay to `min_lr_ratio`.
        - "warmup_cosine"
            Same as cosine but requires `warmup_steps > 0`.
        - "poly"
            Linear warmup (optional) then polynomial decay to `min_lr_ratio`.
        - "step"
            StepLR (decay every `step_size` by factor `gamma`).
        - "multistep"
            MultiStepLR (decay at specified `milestones` by factor `gamma`).
        - "exponential"
            ExponentialLR (multiply by `gamma` every step).
        - "onecycle"
            OneCycleLR (cosine anneal strategy).

    total_steps : int, default=0
        Global training horizon in optimizer steps.

        Required for:
        - linear / cosine / warmup_cosine / poly (LambdaLR-based)
        - onecycle

        Important
        ---------
        For LambdaLR, the scheduler step index is the number of times you call
        `scheduler.step()`. If you want "global optimizer step" semantics, you
        must call `scheduler.step()` exactly once per `optimizer.step()`.

    warmup_steps : int, default=0
        Warmup steps for LambdaLR-based schedules. If > 0, the LR multiplier
        ramps from ~0 to 1 over `warmup_steps`.

        Notes
        -----
        Warmup is clamped to `total_steps` to avoid negative horizons.

    min_lr_ratio : float, default=0.0
        Final LR floor expressed as a fraction of base LR (the optimizer group's
        initial LR). Must be in [0, 1].

        Examples
        --------
        - min_lr_ratio=0.0 : decay to zero (or near-zero)
        - min_lr_ratio=0.1 : decay to 10% of base LR

    poly_power : float, default=1.0
        Polynomial decay exponent used for "poly". Must be > 0.
        - power=1.0 : linear decay
        - power>1.0 : faster decay near the end
        - power<1.0 : slower decay near the end (but still >0)

    step_size : int, default=1000
        Step period for StepLR. Must be > 0.

    gamma : float, default=0.99
        Multiplicative decay factor for StepLR/MultiStepLR/ExponentialLR.
        Must be > 0.

    milestones : Sequence[int], default=()
        Milestones for MultiStepLR. Must be non-empty when name="multistep".
        Duplicates are removed and the result is sorted.

    max_lr : Optional[float | Sequence[float]], default=None
        Maximum learning rate for OneCycleLR.
        - If None: uses current optimizer group LRs as max_lr (compat mode).
        - If float: applies same max_lr to all param groups.
        - If sequence: must match len(optimizer.param_groups).

    pct_start : float, default=0.3
        Fraction of total steps spent increasing LR in OneCycleLR.
        Must be in (0, 1).

    div_factor : float, default=25.0
        OneCycleLR initial_lr = max_lr / div_factor. Must be > 0.

    final_div_factor : float, default=1e4
        OneCycleLR min_lr = initial_lr / final_div_factor. Must be > 0.

    Returns
    -------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        - None if name is "none" / "constant"
        - otherwise a scheduler instance

    Raises
    ------
    ValueError
        If required parameters are missing or invalid for the selected scheduler.
    """
    if optimizer is None:
        raise ValueError("optimizer must not be None")

    sched = _normalize_scheduler_name(name)

    if sched in ("none", "constant"):
        return None

    # Shared validation
    min_lr_ratio_f = float(min_lr_ratio)
    if not (0.0 <= min_lr_ratio_f <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got: {min_lr_ratio_f}")

    warmup_steps_i = int(warmup_steps)
    if warmup_steps_i < 0:
        raise ValueError(f"warmup_steps must be >= 0, got: {warmup_steps_i}")

    # ------------------------------------------------------------------
    # LambdaLR-based schedules
    # ------------------------------------------------------------------
    if sched in ("linear", "cosine", "warmup_cosine", "poly"):
        _require_total_steps(total_steps, sched)
        total_steps_i = int(total_steps)

        # Clamp warmup to horizon (common config mistake)
        if warmup_steps_i > total_steps_i:
            warmup_steps_i = total_steps_i

        if sched == "linear":
            fn = _lr_lambda_linear(
                total_steps=total_steps_i,
                warmup_steps=warmup_steps_i,
                min_lr_ratio=min_lr_ratio_f,
            )
            return LambdaLR(optimizer, lr_lambda=fn)

        if sched in ("cosine", "warmup_cosine"):
            if sched == "warmup_cosine" and warmup_steps_i <= 0:
                raise ValueError("warmup_cosine requires warmup_steps > 0")

            fn = _lr_lambda_cosine(
                total_steps=total_steps_i,
                warmup_steps=warmup_steps_i,
                min_lr_ratio=min_lr_ratio_f,
            )
            return LambdaLR(optimizer, lr_lambda=fn)

        # poly
        power_f = float(poly_power)
        if power_f <= 0.0:
            raise ValueError(f"poly_power must be > 0, got: {power_f}")

        fn = _lr_lambda_poly(
            total_steps=total_steps_i,
            warmup_steps=warmup_steps_i,
            min_lr_ratio=min_lr_ratio_f,
            power=power_f,
        )
        return LambdaLR(optimizer, lr_lambda=fn)

    # ------------------------------------------------------------------
    # Classic schedules
    # ------------------------------------------------------------------
    if sched == "step":
        step_size_i = int(step_size)
        if step_size_i <= 0:
            raise ValueError(f"step_size must be > 0, got: {step_size_i}")

        gamma_f = float(gamma)
        if gamma_f <= 0.0:
            raise ValueError(f"gamma must be > 0, got: {gamma_f}")

        return StepLR(optimizer, step_size=step_size_i, gamma=gamma_f)

    if sched == "multistep":
        ms = sorted({int(m) for m in milestones})
        if len(ms) == 0:
            raise ValueError("multistep requires non-empty milestones")

        gamma_f = float(gamma)
        if gamma_f <= 0.0:
            raise ValueError(f"gamma must be > 0, got: {gamma_f}")

        return MultiStepLR(optimizer, milestones=list(ms), gamma=gamma_f)

    if sched == "exponential":
        gamma_f = float(gamma)
        if gamma_f <= 0.0:
            raise ValueError(f"gamma must be > 0, got: {gamma_f}")

        return ExponentialLR(optimizer, gamma=gamma_f)

    # ------------------------------------------------------------------
    # OneCycle
    # ------------------------------------------------------------------
    if sched == "onecycle":
        _require_total_steps(total_steps, sched)
        total_steps_i = int(total_steps)
        if total_steps_i <= 0:
            raise ValueError(f"onecycle requires total_steps > 0, got: {total_steps_i}")

        pct_start_f = float(pct_start)
        if not (0.0 < pct_start_f < 1.0):
            raise ValueError(f"pct_start must be in (0, 1), got: {pct_start_f}")

        div_factor_f = float(div_factor)
        final_div_factor_f = float(final_div_factor)
        if div_factor_f <= 0.0:
            raise ValueError(f"div_factor must be > 0, got: {div_factor_f}")
        if final_div_factor_f <= 0.0:
            raise ValueError(f"final_div_factor must be > 0, got: {final_div_factor_f}")

        max_lr_resolved = _resolve_onecycle_max_lr(optimizer, max_lr)

        return OneCycleLR(
            optimizer,
            max_lr=max_lr_resolved,
            total_steps=total_steps_i,
            pct_start=pct_start_f,
            div_factor=div_factor_f,
            final_div_factor=final_div_factor_f,
            anneal_strategy="cos",
        )

    raise ValueError(f"Unknown scheduler name: {name!r}")


def scheduler_state_dict(scheduler: Optional[_LRScheduler]) -> Dict[str, Any]:
    """
    Return a checkpoint-ready scheduler state dict.

    Parameters
    ----------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Scheduler instance, or None.

    Returns
    -------
    state : Dict[str, Any]
        - {} if scheduler is None
        - otherwise scheduler.state_dict()
    """
    return {} if scheduler is None else scheduler.state_dict()


def load_scheduler_state_dict(scheduler: Optional[_LRScheduler], state: Mapping[str, Any]) -> None:
    """
    Restore scheduler state from a checkpoint.

    Parameters
    ----------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Scheduler instance, or None. If None, this function is a no-op.
    state : Mapping[str, Any]
        Scheduler state dict, typically produced by `scheduler_state_dict()`.

    Returns
    -------
    None
    """
    if scheduler is None:
        return
    scheduler.load_state_dict(dict(state))


# =============================================================================
# Internal helpers
# =============================================================================
def _normalize_scheduler_name(name: str) -> str:
    """
    Normalize scheduler name to a canonical identifier.

    Parameters
    ----------
    name : str
        Raw scheduler name provided by the user.

    Returns
    -------
    sched : str
        Normalized identifier (lowercased, whitespace/hyphen normalized).

    Notes
    -----
    This normalizer is intentionally conservative:
    - It normalizes separators ("-", " ", "__") into "_"
    - It does not attempt fuzzy matching beyond that
    """
    return str(name).lower().strip().replace("-", "_").replace(" ", "_")


def _require_total_steps(total_steps: int, name: str) -> None:
    """
    Validate that `total_steps` is specified for progress-based schedulers.

    Parameters
    ----------
    total_steps : int
        Global training horizon in optimizer steps. Must be > 0.
    name : str
        Scheduler name used for error messages.

    Raises
    ------
    ValueError
        If total_steps <= 0.
    """
    if int(total_steps) <= 0:
        raise ValueError(f"{name} scheduler requires total_steps > 0")


def _lr_lambda_linear(*, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """
    Create a LambdaLR function for linear warmup + linear decay.

    The returned function `f(step)` produces a multiplicative LR factor.

    Schedule definition
    -------------------
    - Warmup phase (if warmup_steps > 0):
        f ramps from ~0 to 1 over warmup_steps.
    - Decay phase:
        f decays linearly from 1 to min_lr_ratio over remaining steps.

    Parameters
    ----------
    total_steps : int
        Total number of scheduler steps (horizon).
    warmup_steps : int
        Number of warmup steps. Must satisfy 0 <= warmup_steps <= total_steps.
    min_lr_ratio : float
        Final LR multiplier at step=total_steps, in [0, 1].

    Returns
    -------
    f : Callable[[int], float]
        LambdaLR multiplier function.
    """
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    min_lr_ratio = float(min_lr_ratio)

    def f(step: int) -> float:
        s = max(0, int(step))

        # Warmup: 1/warmup_steps, 2/warmup_steps, ..., 1.0
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / float(max(1, warmup_steps))

        # Decay: t in [0, 1]
        denom = max(1, total_steps - warmup_steps)
        t = min(1.0, (s - warmup_steps) / float(denom))
        return (1.0 - t) + t * min_lr_ratio

    return f


def _lr_lambda_cosine(*, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """
    Create a LambdaLR function for linear warmup + cosine decay.

    The returned function `f(step)` produces a multiplicative LR factor.

    Schedule definition
    -------------------
    - Warmup phase (optional):
        linear ramp from ~0 to 1
    - Decay phase:
        cosine decay from 1 to min_lr_ratio

    Parameters
    ----------
    total_steps : int
        Total number of scheduler steps (horizon).
    warmup_steps : int
        Number of warmup steps. Must satisfy 0 <= warmup_steps <= total_steps.
    min_lr_ratio : float
        Final LR multiplier at step=total_steps, in [0, 1].

    Returns
    -------
    f : Callable[[int], float]
        LambdaLR multiplier function.
    """
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    min_lr_ratio = float(min_lr_ratio)

    def f(step: int) -> float:
        s = max(0, int(step))

        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / float(max(1, warmup_steps))

        denom = max(1, total_steps - warmup_steps)
        t = min(1.0, (s - warmup_steps) / float(denom))

        # cosine in [0, 1]: 1 at t=0, 0 at t=1
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return f


def _lr_lambda_poly(
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    power: float,
) -> Callable[[int], float]:
    """
    Create a LambdaLR function for linear warmup + polynomial decay.

    The returned function `f(step)` produces a multiplicative LR factor.

    Schedule definition
    -------------------
    - Warmup phase (optional):
        linear ramp from ~0 to 1
    - Decay phase:
        polynomial decay from 1 to min_lr_ratio:

            f = min_lr_ratio + (1 - min_lr_ratio) * (1 - t)^power

        where t is normalized progress in [0, 1].

    Parameters
    ----------
    total_steps : int
        Total number of scheduler steps (horizon).
    warmup_steps : int
        Number of warmup steps. Must satisfy 0 <= warmup_steps <= total_steps.
    min_lr_ratio : float
        Final LR multiplier at step=total_steps, in [0, 1].
    power : float
        Polynomial exponent (> 0).

    Returns
    -------
    f : Callable[[int], float]
        LambdaLR multiplier function.
    """
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    min_lr_ratio = float(min_lr_ratio)
    power = float(power)

    def f(step: int) -> float:
        s = max(0, int(step))

        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / float(max(1, warmup_steps))

        denom = max(1, total_steps - warmup_steps)
        t = min(1.0, (s - warmup_steps) / float(denom))

        poly = (1.0 - t) ** power
        return min_lr_ratio + (1.0 - min_lr_ratio) * poly

    return f


def _resolve_onecycle_max_lr(
    optimizer: Optimizer,
    max_lr: Optional[Union[float, Sequence[float]]],
) -> Union[float, Sequence[float]]:
    """
    Resolve the `max_lr` argument for OneCycleLR.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer containing one or more parameter groups.
    max_lr : Optional[float | Sequence[float]]
        User-provided `max_lr` specification.

        - None:
            Use the current learning rates in `optimizer.param_groups` as max_lr.
            This makes OneCycle "compat" behavior easier when the caller already
            set group LRs as the intended peak.
        - float:
            Use the same max_lr for all parameter groups.
        - sequence[float]:
            Per-group max_lr. Must match len(optimizer.param_groups).

    Returns
    -------
    resolved : float | Sequence[float]
        Value suitable to pass to `torch.optim.lr_scheduler.OneCycleLR`.

    Raises
    ------
    ValueError
        If a sequence is provided but its length does not match param_groups.
    """
    if max_lr is None:
        return [float(g["lr"]) for g in optimizer.param_groups]

    if isinstance(max_lr, (int, float)):
        return float(max_lr)

    vals = [float(v) for v in max_lr]
    if len(vals) != len(optimizer.param_groups):
        raise ValueError(
            f"onecycle max_lr sequence length mismatch: got {len(vals)}, "
            f"expected {len(optimizer.param_groups)}"
        )
    return vals
