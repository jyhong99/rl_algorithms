from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    OneCycleLR,
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
    Build a PyTorch learning-rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Target optimizer whose param_group learning rates will be scheduled.

    name : str, optional
        Scheduler identifier (case-insensitive), by default "none".
        Supported:
          - "none" / "constant"
          - "linear"
          - "cosine"
          - "warmup_cosine"
          - "poly"
          - "step"
          - "multistep"
          - "exponential"
          - "onecycle"

    total_steps : int, optional
        Total number of optimizer steps in the run (global horizon).
        Required for: linear/cosine/warmup_cosine/poly/onecycle.

        Note:
        - For LambdaLR-based schedules, the step index is the number of times
          you call `scheduler.step()`. For “global optimizer step” semantics,
          call `scheduler.step()` exactly once per `optimizer.step()`.

    warmup_steps : int, optional
        Warmup steps for lambda-based schedules. If > 0, factor ramps from ~0 to 1.

    min_lr_ratio : float, optional
        Final LR floor as a fraction of base LR. Used by linear/cosine/poly.
        Must be in [0, 1]. Example: 0.1 means LR floors at 10% of base LR.

    poly_power : float, optional
        Polynomial decay exponent for "poly" schedule, must be > 0, by default 1.0.

    step_size : int, optional
        StepLR period in steps, by default 1000.

    gamma : float, optional
        Multiplicative decay factor for StepLR/MultiStepLR/ExponentialLR, by default 0.99.

    milestones : Sequence[int], optional
        MultiStepLR decay steps, must be non-empty for "multistep".

    max_lr : Optional[float | Sequence[float]], optional
        OneCycleLR maximum learning rate(s).
        - If None: uses current optimizer group lrs as max_lr (compat mode).
        - If float: applies same max_lr to all param groups.
        - If sequence: must match len(optimizer.param_groups).

    pct_start : float, optional
        OneCycleLR fraction of steps spent increasing LR, by default 0.3.
        Must be in (0, 1).

    div_factor : float, optional
        OneCycleLR initial_lr = max_lr / div_factor, by default 25.0.

    final_div_factor : float, optional
        OneCycleLR min_lr = initial_lr / final_div_factor, by default 1e4.

    Returns
    -------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        - None if name is "none" / "constant"
        - otherwise a PyTorch LR scheduler instance

    Raises
    ------
    ValueError
        If required parameters are missing/invalid for the selected scheduler.
    """
    if optimizer is None:
        raise ValueError("optimizer must not be None")

    # normalize name: allow hyphen/underscore variants
    sched = str(name).lower().strip().replace("-", "_").replace(" ", "_")

    if sched in ("none", "constant"):
        return None

    # shared validation for ratio/power
    min_lr_ratio = float(min_lr_ratio)
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got: {min_lr_ratio}")

    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got: {warmup_steps}")

    if sched in ("linear", "cosine", "warmup_cosine", "poly"):
        _require_total_steps(total_steps, sched)
        total_steps = int(total_steps)

        # Clamp warmup to horizon (common config mistake)
        if warmup_steps > total_steps:
            warmup_steps = total_steps

        if sched == "linear":
            fn = _lr_lambda_linear(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            )
            return LambdaLR(optimizer, lr_lambda=fn)

        if sched in ("cosine", "warmup_cosine"):
            if sched == "warmup_cosine" and warmup_steps <= 0:
                raise ValueError("warmup_cosine requires warmup_steps > 0")
            fn = _lr_lambda_cosine(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            )
            return LambdaLR(optimizer, lr_lambda=fn)

        # poly
        poly_power = float(poly_power)
        if poly_power <= 0.0:
            raise ValueError(f"poly_power must be > 0, got: {poly_power}")

        fn = _lr_lambda_poly(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            power=poly_power,
        )
        return LambdaLR(optimizer, lr_lambda=fn)

    if sched == "step":
        step_size = int(step_size)
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got: {step_size}")
        gamma_f = float(gamma)
        if gamma_f <= 0.0:
            raise ValueError(f"gamma must be > 0, got: {gamma_f}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma_f)

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

    if sched == "onecycle":
        _require_total_steps(total_steps, sched)
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError(f"onecycle requires total_steps > 0, got: {total_steps}")

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
            total_steps=total_steps,
            pct_start=pct_start_f,
            div_factor=div_factor_f,
            final_div_factor=final_div_factor_f,
            anneal_strategy="cos",
        )

    raise ValueError(f"Unknown scheduler name: {name!r}")


def scheduler_state_dict(scheduler: Optional[_LRScheduler]) -> Dict[str, Any]:
    """
    Get a checkpoint-ready scheduler state.

    Parameters
    ----------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Scheduler instance or None.

    Returns
    -------
    state : Dict[str, Any]
        - {} if scheduler is None
        - otherwise scheduler.state_dict()
    """
    return {} if scheduler is None else scheduler.state_dict()


def load_scheduler_state_dict(scheduler: Optional[_LRScheduler], state: Mapping[str, Any]) -> None:
    """
    Load scheduler state from a checkpoint.

    Parameters
    ----------
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Scheduler instance or None.
    state : Mapping[str, Any]
        Serialized state (typically produced by scheduler_state_dict()).

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
def _require_total_steps(total_steps: int, name: str) -> None:
    """
    Validate that total_steps is specified for progress-based schedulers.

    Parameters
    ----------
    total_steps : int
        Global training horizon in optimizer steps.
    name : str
        Scheduler name (for error messages).

    Raises
    ------
    ValueError
        If total_steps <= 0.
    """
    if int(total_steps) <= 0:
        raise ValueError(f"{name} scheduler requires total_steps > 0")


def _lr_lambda_linear(*, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """
    Linear warmup (optional) + linear decay to min_lr_ratio.

    Returns a multiplicative factor f(step) used by LambdaLR.

    Notes
    -----
    - Warmup: f ramps from ~0 to 1 across warmup_steps.
    - Decay:  f linearly decays from 1 to min_lr_ratio over remaining steps.
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
        return (1.0 - t) + t * min_lr_ratio

    return f


def _lr_lambda_cosine(*, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """
    Linear warmup (optional) + cosine decay to min_lr_ratio.

    Returns a multiplicative factor f(step) used by LambdaLR.
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
    Linear warmup (optional) + polynomial decay to min_lr_ratio.

    Returns a multiplicative factor f(step) used by LambdaLR.
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
    Resolve OneCycleLR max_lr argument.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer with param_groups.
    max_lr : Optional[float | Sequence[float]]
        User-provided max_lr.

    Returns
    -------
    resolved : float | Sequence[float]
        Valid max_lr argument for OneCycleLR.

    Raises
    ------
    ValueError
        If a provided sequence length does not match param_groups.
    """
    if max_lr is None:
        # Backward-compatible: treat current group lrs as max_lr.
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