from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import (
    IntervalGate,
    _coerce_scalar_mapping,
    _safe_update_step,
    _to_finite_float,
)


class LRLoggingCallback(BaseCallback):
    """
    Periodically log learning rates from algorithm optimizers and schedulers (best-effort).

    Rationale
    ---------
    Learning-rate schedules are a common source of training instability or unexpected
    performance changes. This callback periodically extracts the current learning rate(s)
    from the algorithm and logs them for debugging and monitoring.

    Discovery order
    --------------
    Extraction proceeds in the following order (first successful path wins, where applicable):

    1. ``algo.get_lr_dict() -> Mapping[str, scalar]`` (preferred)
       - Intended to be the authoritative source if an algorithm exposes it.
       - Values are coerced to finite scalars via :func:`coerce_scalar_mapping`.

    2. ``algo.optimizers: Mapping[str, optimizer]`` (optional)
       - Reads optimizer ``param_groups[*]["lr"]`` values.

    3. ``algo.optimizer`` (single optimizer fallback)

    4. ``algo.schedulers: Mapping[str, scheduler]`` (optional)
       - Prefer ``scheduler.get_last_lr()`` when available (PyTorch convention).
       - Otherwise fall back to reading ``scheduler.optimizer.param_groups[*]["lr"]``.

    5. ``algo.scheduler`` (single scheduler fallback)

    Scheduling
    ----------
    Logging is triggered on *update steps* using ``IntervalGate(mode="mod")``, i.e. every
    ``log_every_updates`` updates.

    Unlike some other callbacks, this one **does not** fall back to an internal call counter
    if the trainer does not expose a valid update index. If ``safe_update_step(trainer) <= 0``,
    the callback becomes a no-op for that invocation.

    Logged keys
    ----------
    For a given component name ``name`` (optimizer/scheduler key):

    - Single param group:
        ``lr/<name>``
    - Multiple param groups:
        ``lr/<name>_g<i>`` where ``i`` is the param-group index

    Parameters
    ----------
    log_every_updates:
        Emit logs every N update steps. If ``<= 0``, logging is disabled (no-op).
    log_prefix:
        Prefix passed to :meth:`BaseCallback.log` so metrics are namespaced (e.g. ``"train/"``).

    Notes
    -----
    - All extraction is best-effort; errors are swallowed so training is not disrupted.
    - Values are filtered through :func:`to_finite_float` to ignore NaN/Inf/non-numeric values.
    """

    def __init__(self, *, log_every_updates: int = 200, log_prefix: str = "train/") -> None:
        self.log_every_updates = int(log_every_updates)
        self.log_prefix = str(log_prefix)

        # Periodic trigger based on update index (upd % every == 0).
        self._gate = IntervalGate(every=self.log_every_updates, mode="mod")

    # =========================================================================
    # Internal extraction helpers
    # =========================================================================
    def _extract_lr_from_optimizer(self, opt: Any, *, name: str) -> Dict[str, float]:
        """
        Extract learning rate(s) from an optimizer-like object.

        Parameters
        ----------
        opt:
            Optimizer-like object expected to expose ``param_groups``.
            Each param group is expected to be a dict-like object with key ``"lr"``.
        name:
            Logical name used to construct output logging keys.

        Returns
        -------
        Dict[str, float]
            Mapping of logging keys to finite learning-rate values.
            Returns an empty dict if no usable learning rates are found.

        Notes
        -----
        Output key conventions:
        - Single param group:
            ``lr/<name>``
        - Multiple param groups:
            ``lr/<name>_g<i>`` for group index i

        Robustness:
        - Missing/invalid ``param_groups`` => empty result.
        - Non-finite lr values (NaN/Inf) are dropped.
        - Any exception => empty result.
        """
        out: Dict[str, float] = {}
        try:
            groups = getattr(opt, "param_groups", None)
            if not isinstance(groups, list) or not groups:
                return out

            multi = len(groups) > 1
            for i, g in enumerate(groups):
                try:
                    lr = g.get("lr", None)
                except Exception:
                    lr = None

                fv = _to_finite_float(lr)
                if fv is None:
                    continue

                key = f"lr/{name}_g{i}" if multi else f"lr/{name}"
                out[key] = fv

        except Exception:
            return {}
        return out

    def _extract_lr_from_scheduler(self, sch: Any, *, name: str) -> Dict[str, float]:
        """
        Extract learning rate(s) from a scheduler-like object.

        Parameters
        ----------
        sch:
            Scheduler-like object. Preferred interface:
            - ``get_last_lr() -> Sequence[float]`` (PyTorch LR scheduler convention)
            Fallback interface:
            - ``sch.optimizer.param_groups[*]["lr"]``
        name:
            Logical name used to construct output logging keys.

        Returns
        -------
        Dict[str, float]
            Mapping of logging keys to finite learning-rate values.
            Returns an empty dict if no usable learning rates are found.

        Notes
        -----
        Preferred extraction:
        - If ``sch.get_last_lr`` exists and returns a non-empty list/tuple, use it.

        Fallback extraction:
        - If ``sch.optimizer`` exists, read optimizer param group LRs via
          :meth:`_extract_lr_from_optimizer`.

        Output keys follow the same conventions as optimizer extraction:
        - ``lr/<name>`` or ``lr/<name>_g<i>``
        """
        out: Dict[str, float] = {}

        fn = getattr(sch, "get_last_lr", None)
        if callable(fn):
            try:
                lrs = fn()
                if isinstance(lrs, (list, tuple)) and lrs:
                    multi = len(lrs) > 1
                    for i, lr in enumerate(lrs):
                        fv = _to_finite_float(lr)
                        if fv is None:
                            continue
                        key = f"lr/{name}_g{i}" if multi else f"lr/{name}"
                        out[key] = fv
                    return out
            except Exception:
                pass

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

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Expected to expose:
            - update counter (via :func:`safe_update_step`)
            - ``trainer.algo`` (algorithm object)
            - logger backend used by :meth:`BaseCallback.log` (optional)
        metrics:
            Optional training metrics dict (unused; accepted for hook compatibility).

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Control flow
        ------------
        1. Validate schedule: enabled + valid update counter + gate ready
        2. Discover algorithm object
        3. Try preferred ``algo.get_lr_dict()``
        4. Else extract from optimizers / optimizer
        5. Optionally extract from schedulers / scheduler
        6. Log if payload is non-empty
        """
        if self.log_every_updates <= 0:
            return True

        upd = _safe_update_step(trainer)
        if upd <= 0:
            return True

        if not self._gate.ready(upd):
            return True

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
                    payload = _coerce_scalar_mapping(lr_dict)
                    if payload:
                        self.log(trainer, payload, step=upd, prefix=self.log_prefix)
                        return True
            except Exception:
                pass

        payload: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # Optimizers extraction
        # ---------------------------------------------------------------------
        opts = getattr(algo, "optimizers", None)
        if isinstance(opts, Mapping):
            for k, opt in opts.items():
                payload.update(self._extract_lr_from_optimizer(opt, name=str(k)))

        # Single optimizer fallback if mapping is absent or yielded nothing.
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
        else:
            sch = getattr(algo, "scheduler", None)
            if sch is not None:
                payload.update(self._extract_lr_from_scheduler(sch, name="scheduler"))

        if payload:
            self.log(trainer, payload, step=upd, prefix=self.log_prefix)

        return True
