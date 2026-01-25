from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import (
    safe_int_attr,
    IntervalGate,
    coerce_scalar_mapping,
    to_finite_float,
)
from ..utils.log_utils import log


class LRLoggingCallback(BaseCallback):
    """
    Log learning rates from algorithm optimizers/schedulers (best-effort).

    Rationale
    ---------
    Learning-rate schedules are a common source of training instability or
    unexpected performance changes. This callback periodically extracts current
    learning rates from the algorithm and logs them.

    Discovery order
    --------------
    1) algo.get_lr_dict() -> Mapping[str, scalar]
       - Preferred/authoritative source if the algorithm provides it.
       - Expected to already be in "final" logging form (or at least scalar values).

    2) algo.optimizers: Mapping[name, optimizer]
       - Extract learning rate(s) from optimizer.param_groups[*]["lr"].

    3) algo.optimizer: single optimizer fallback

    4) algo.schedulers: Mapping[name, scheduler] (optional)
       - Prefer scheduler.get_last_lr() when available.
       - Otherwise fall back to scheduler.optimizer.param_groups[*]["lr"].

    5) algo.scheduler: single scheduler fallback

    Scheduling
    ----------
    - Triggers on update steps via IntervalGate(mode="mod"), i.e. every N updates.
    - If the trainer does not expose a valid update counter (upd <= 0), this callback
      does nothing (no internal call-count fallback).

    Logged keys
    ----------
    - For single-group optimizer/scheduler:
        lr/<name>
    - For multi-group optimizer/scheduler:
        lr/<name>_g<i>   where i is the param_group index

    Notes
    -----
    - All extraction is best-effort; failures are swallowed to avoid disrupting training.
    - Values are filtered through `to_finite_float` to ignore NaN/Inf/non-numeric values.
    """

    def __init__(self, *, log_every_updates: int = 200, log_prefix: str = "train/") -> None:
        # User configuration
        self.log_every_updates = int(log_every_updates)
        self.log_prefix = str(log_prefix)

        # Gate for periodic update-step logging (mod-based trigger).
        self._gate = IntervalGate(every=self.log_every_updates, mode="mod")

    # =========================================================================
    # Internal extraction helpers
    # =========================================================================
    def _extract_lr_from_optimizer(self, opt: Any, *, name: str) -> Dict[str, float]:
        """
        Extract learning rate(s) from an optimizer-like object.

        Expected optimizer interface
        ----------------------------
        - opt.param_groups: List[Dict] where each group may contain key "lr".

        Output keys
        ----------
        - lr/<name>        : if a single param_group exists
        - lr/<name>_g<i>   : if multiple param_groups exist (i = group index)

        Returns
        -------
        Dict[str, float]
            Empty if optimizer is missing param_groups or no finite lr values exist.
        """
        out: Dict[str, float] = {}
        try:
            groups = getattr(opt, "param_groups", None)
            if not isinstance(groups, list) or not groups:
                return out

            # If there are multiple param groups, log group-wise learning rates.
            multi = len(groups) > 1
            for i, g in enumerate(groups):
                try:
                    lr = g.get("lr", None)
                except Exception:
                    lr = None

                # Only log finite scalar values.
                fv = to_finite_float(lr)
                if fv is None:
                    continue

                key = f"lr/{name}_g{i}" if multi else f"lr/{name}"
                out[key] = fv

        except Exception:
            # Best-effort: return empty on any extraction error.
            return {}
        return out

    def _extract_lr_from_scheduler(self, sch: Any, *, name: str) -> Dict[str, float]:
        """
        Extract learning rate(s) from a scheduler-like object.

        Preferred path
        --------------
        - sch.get_last_lr() -> list/tuple of lrs (PyTorch schedulers commonly implement this)

        Fallback path
        -------------
        - sch.optimizer.param_groups[*]["lr"] (if scheduler exposes .optimizer)

        Returns
        -------
        Dict[str, float]
            Keys follow the same convention as optimizer extraction:
              lr/<name> or lr/<name>_g<i>
        """
        out: Dict[str, float] = {}

        # Prefer get_last_lr() when available.
        fn = getattr(sch, "get_last_lr", None)
        if callable(fn):
            try:
                lrs = fn()
                if isinstance(lrs, (list, tuple)) and lrs:
                    multi = len(lrs) > 1
                    for i, lr in enumerate(lrs):
                        fv = to_finite_float(lr)
                        if fv is None:
                            continue
                        key = f"lr/{name}_g{i}" if multi else f"lr/{name}"
                        out[key] = fv
                    return out
            except Exception:
                # If get_last_lr fails, fall back to optimizer-based extraction below.
                pass

        # Fallback: read from attached optimizer if present.
        opt = getattr(sch, "optimizer", None)
        if opt is not None:
            out.update(self._extract_lr_from_optimizer(opt, name=name))
        return out

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after an update step by the trainer.

        Flow
        ----
        1) Validate schedule (enabled + valid upd counter + gate ready)
        2) Discover algo
        3) Try preferred algo.get_lr_dict()
        4) Else extract from optimizers/optimizer
        5) Optionally extract from schedulers/scheduler
        6) Log if payload is non-empty
        """
        # Disabled mode.
        if self.log_every_updates <= 0:
            return True

        # Require a positive update counter; no call-count fallback by design.
        upd = safe_int_attr(trainer)
        if upd <= 0:
            return True

        # Periodic gate check.
        if not self._gate.ready(upd):
            return True

        # Trainer must expose algo.
        algo = getattr(trainer, "algo", None)
        if algo is None:
            return True

        # ---------------------------------------------------------------------
        # Preferred: algo.get_lr_dict()
        # ---------------------------------------------------------------------
        fn = getattr(algo, "get_lr_dict", None)
        if callable(fn):
            try:
                lr_dict = fn()
                if isinstance(lr_dict, Mapping) and lr_dict:
                    # Coerce values to finite scalars for safe logging.
                    payload = coerce_scalar_mapping(lr_dict)
                    if payload:
                        log(trainer, payload, step=upd, prefix=self.log_prefix)
                        return True
            except Exception:
                # Best-effort: ignore and fall back to extracting from optimizers/schedulers.
                pass

        payload: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # Optimizers extraction
        # ---------------------------------------------------------------------
        opts = getattr(algo, "optimizers", None)
        if isinstance(opts, Mapping):
            for k, opt in opts.items():
                payload.update(self._extract_lr_from_optimizer(opt, name=str(k)))

        # Fallback to single optimizer if mapping did not yield anything.
        if not payload:
            opt = getattr(algo, "optimizer", None)
            if opt is not None:
                payload.update(self._extract_lr_from_optimizer(opt, name="optimizer"))

        # ---------------------------------------------------------------------
        # Schedulers extraction (optional)
        # ---------------------------------------------------------------------
        scheds = getattr(algo, "schedulers", None)
        if isinstance(scheds, Mapping):
            for k, sch in scheds.items():
                payload.update(self._extract_lr_from_scheduler(sch, name=str(k)))

        # If no schedulers mapping, fall back to single scheduler.
        if not scheds:
            sch = getattr(algo, "scheduler", None)
            if sch is not None:
                payload.update(self._extract_lr_from_scheduler(sch, name="scheduler"))

        # Emit logs only if we collected any learning rates.
        if payload:
            log(trainer, payload, step=upd, prefix=self.log_prefix)

        return True
