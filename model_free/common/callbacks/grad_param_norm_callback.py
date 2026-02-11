from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch as th
import torch.nn as nn

from .base_callback import BaseCallback
from ..utils.callback_utils import IntervalGate, _safe_update_step


class GradParamNormCallback(BaseCallback):
    """
    Log parameter p-norm and gradient p-norm for debugging and troubleshooting.

    Motivation
    ----------
    When training becomes unstable (exploding/vanishing gradients, drifting weights,
    sudden divergence), logging parameter norms and gradient norms is a cheap and
    effective diagnostic signal.

    What it logs
    ------------
    On a periodic *update-step* schedule, it computes:

    - Parameter p-norm: ``||θ||_p`` over selected modules' parameters
    - Gradient p-norm: ``||∇θ||_p`` over selected modules' gradients (only parameters
      that currently have ``grad != None``)

    Logging modes
    -------------
    **Global mode (per_module=False)**:
        Aggregates norms across all selected modules into:
        - ``{log_prefix}param_norm``
        - ``{log_prefix}grad_norm``

    **Per-module mode (per_module=True)**:
        Logs norms separately per module:
        - ``{log_prefix}{module_name}/param_norm``
        - ``{log_prefix}{module_name}/grad_norm``

    Module selection
    ----------------
    1) Preferred:
        ``algo.get_modules_for_logging() -> Mapping[str, nn.Module]``

    2) Fallback (if above is absent or invalid):
        - ``algo.policy.head`` (if an ``nn.Module``)
        - ``algo.head``        (if an ``nn.Module``)
      The fallback module is logged under name ``"head"``.

    Scheduling
    ----------
    Uses ``IntervalGate(mode="mod")`` with ``every=log_every_updates`` so the callback
    triggers when::

        update_idx % log_every_updates == 0

    If a trainer does not expose a usable update counter (``safe_update_step(trainer) <= 0``),
    the callback falls back to an internal counter to avoid "never triggers" behavior.

    Parameters
    ----------
    log_every_updates:
        Emit logs every N updates. If ``<= 0``, logging is disabled (no-op).
    log_prefix:
        Prefix applied to all logged keys.
    include_param_norm:
        Whether to compute parameter norms.
    include_grad_norm:
        Whether to compute gradient norms (only for parameters with ``grad != None``).
    norm_type:
        The p in p-norm. Must be ``> 0``. Typical choices: 2.0 (L2), 1.0 (L1).
        (``inf`` is not explicitly handled here.)
    per_module:
        If True, log per-module norms. Otherwise log global aggregated norms.

    Notes
    -----
    - This callback is best-effort and should not crash training due to logging.
      Individual tensor norm computation is wrapped defensively.
    - Norm composition across tensors uses the mathematically correct p-norm
      aggregation for disjoint blocks:
        ``(Σ ||b_i||_p^p)^(1/p)``.
    """

    def __init__(
        self,
        *,
        log_every_updates: int = 200,
        log_prefix: str = "debug/",
        include_param_norm: bool = True,
        include_grad_norm: bool = True,
        norm_type: float = 2.0,
        per_module: bool = False,
    ) -> None:
        self.log_every_updates = int(log_every_updates)
        self.log_prefix = str(log_prefix)
        self.include_param_norm = bool(include_param_norm)
        self.include_grad_norm = bool(include_grad_norm)
        self.norm_type = float(norm_type)
        self.per_module = bool(per_module)

        if self.norm_type <= 0.0:
            raise ValueError(f"norm_type must be > 0, got: {self.norm_type}")

        # Periodic trigger based on update index.
        self._gate = IntervalGate(every=self.log_every_updates, mode="mod")

        # Used only when trainer update counter is unavailable.
        self._calls: int = 0

    # =========================================================================
    # Tensor iteration helpers
    # =========================================================================
    @staticmethod
    def _iter_param_tensors(module: nn.Module) -> Iterable[th.Tensor]:
        """
        Iterate detached parameter tensors of a module.

        Parameters
        ----------
        module:
            PyTorch module whose parameters will be iterated (recursively).

        Yields
        ------
        torch.Tensor
            Detached parameter tensor. Tensors are not cloned.

        Notes
        -----
        - Using ``detach()`` avoids keeping autograd references.
        - This does not allocate new storage (cheap).
        """
        for p in module.parameters(recurse=True):
            if p is None:
                continue
            yield p.detach()

    @staticmethod
    def _iter_grad_tensors(module: nn.Module) -> Iterable[th.Tensor]:
        """
        Iterate detached gradient tensors for parameters that currently have gradients.

        Parameters
        ----------
        module:
            PyTorch module whose parameter gradients will be iterated (recursively).

        Yields
        ------
        torch.Tensor
            Detached gradient tensor.

        Notes
        -----
        - Skips parameters with ``p.grad is None`` (e.g., frozen parameters or before backward).
        - ``detach()`` avoids autograd graph interactions.
        """
        for p in module.parameters(recurse=True):
            if p is None or p.grad is None:
                continue
            yield p.grad.detach()

    @staticmethod
    def _tensor_pnorm(t: th.Tensor, p: float) -> float:
        """
        Compute the p-norm of a tensor (best-effort).

        Parameters
        ----------
        t:
            Tensor whose norm will be computed.
        p:
            Norm order (p > 0).

        Returns
        -------
        float
            The p-norm as a Python float. Returns 0.0 if the tensor is empty or
            if any error occurs.

        Notes
        -----
        - Uses ``torch.linalg.vector_norm`` on a flattened tensor.
        - For sparse tensors, computes the norm over stored values only.
        """
        if t is None or t.numel() == 0:
            return 0.0
        try:
            if t.is_sparse:
                v = t.coalesce().values()
                if v.numel() == 0:
                    return 0.0
                return float(th.linalg.vector_norm(v.reshape(-1), ord=p).cpu().item())

            return float(th.linalg.vector_norm(t.reshape(-1), ord=p).cpu().item())
        except Exception:
            return 0.0

    @staticmethod
    def _aggregate_pnorm(block_norms: List[float], p: float) -> float:
        """
        Combine p-norms from disjoint blocks into a single global p-norm.

        If each block i has p-norm ``||b_i||_p``, the combined global p-norm is::

            ( Σ_i (||b_i||_p)^p )^(1/p)

        Parameters
        ----------
        block_norms:
            List of non-negative p-norm values, one per disjoint block.
        p:
            Norm order (p > 0).

        Returns
        -------
        float
            Combined global p-norm. Returns 0.0 if the input is empty or invalid.

        Notes
        -----
        - Filters out non-finite and negative inputs defensively.
        """
        if not block_norms:
            return 0.0

        acc = 0.0
        for n in block_norms:
            try:
                fn = float(n)
            except Exception:
                continue
            if not math.isfinite(fn) or fn < 0.0:
                continue
            acc += fn**p

        if acc <= 0.0:
            return 0.0
        return float(acc ** (1.0 / p))

    # =========================================================================
    # Module selection
    # =========================================================================
    @staticmethod
    def _fallback_head_module(algo: Any) -> Optional[nn.Module]:
        """
        Fallback module discovery when algo does not provide a logging mapping.

        Parameters
        ----------
        algo:
            Algorithm-like object (duck-typed).

        Returns
        -------
        Optional[nn.Module]
            The discovered module if present, otherwise None.

        Notes
        -----
        Discovery order:
        1. ``algo.policy.head`` (if exists and is an nn.Module)
        2. ``algo.head``        (if exists and is an nn.Module)
        """
        head = getattr(getattr(algo, "policy", None), "head", None)
        if head is None:
            head = getattr(algo, "head", None)
        return head if isinstance(head, nn.Module) else None

    @staticmethod
    def _modules_for_logging(algo: Any) -> Dict[str, nn.Module]:
        """
        Discover modules to log norms for.

        Parameters
        ----------
        algo:
            Algorithm-like object (duck-typed).

        Returns
        -------
        Dict[str, nn.Module]
            Mapping from module name to module instance. Empty if none found.

        Notes
        -----
        Preferred contract:
            ``algo.get_modules_for_logging() -> Mapping[str, nn.Module]``

        Fallback:
            ``{"head": <module>}`` if a head module can be discovered.
        """
        fn = getattr(algo, "get_modules_for_logging", None)
        if callable(fn):
            try:
                mods = fn()
                if isinstance(mods, Mapping):
                    out = {str(k): v for k, v in mods.items() if isinstance(v, nn.Module)}
                    if out:
                        return out
            except Exception:
                pass

        head = GradParamNormCallback._fallback_head_module(algo)
        if head is not None:
            return {"head": head}

        return {}

    # =========================================================================
    # Norm computations
    # =========================================================================
    def _module_param_norm(self, module: nn.Module, p: float) -> float:
        """
        Compute p-norm of all parameters in a module.

        Parameters
        ----------
        module:
            Module whose parameters are included.
        p:
            Norm order.

        Returns
        -------
        float
            Global p-norm over all parameters in the module.
        """
        norms = [self._tensor_pnorm(t, p) for t in self._iter_param_tensors(module)]
        return self._aggregate_pnorm(norms, p)

    def _module_grad_norm(self, module: nn.Module, p: float) -> float:
        """
        Compute p-norm of all gradients in a module.

        Parameters
        ----------
        module:
            Module whose gradients are included (only params with grad).
        p:
            Norm order.

        Returns
        -------
        float
            Global p-norm over all gradients in the module.
        """
        norms = [self._tensor_pnorm(g, p) for g in self._iter_grad_tensors(module)]
        return self._aggregate_pnorm(norms, p)

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after an optimizer update (or update-like event).

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Expected to expose:
            - update counter (via :func:`safe_update_step`) or none/invalid
            - ``trainer.algo`` for module selection
            - optional logger backend used by :meth:`BaseCallback.log`
        metrics:
            Optional training metrics dict (unused; accepted for hook compatibility).

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Scheduling logic:
        - Prefer trainer-provided update index: ``upd = safe_update_step(trainer)``.
        - If unavailable (``upd <= 0``), fall back to an internal call counter.
        - Triggering is controlled by ``IntervalGate(mode="mod")``.
        """
        if self.log_every_updates <= 0:
            return True

        upd = _safe_update_step(trainer)
        if upd <= 0:
            self._calls += 1
            upd = self._calls

        if not self._gate.ready(upd):
            return True

        algo = getattr(trainer, "algo", None)
        if algo is None:
            return True

        mods = self._modules_for_logging(algo)
        if not mods:
            return True

        out: Dict[str, Any] = {}

        # --------------------------------------------------------------
        # Per-module logging
        # --------------------------------------------------------------
        if self.per_module:
            for name, m in mods.items():
                if self.include_param_norm:
                    out[f"{name}/param_norm"] = self._module_param_norm(m, self.norm_type)
                if self.include_grad_norm:
                    out[f"{name}/grad_norm"] = self._module_grad_norm(m, self.norm_type)

            self.log(trainer, out, step=upd, prefix=self.log_prefix)
            return True

        # --------------------------------------------------------------
        # Global logging: aggregate across all parameter/gradient tensors
        # --------------------------------------------------------------
        param_block_norms: List[float] = []
        grad_block_norms: List[float] = []

        if self.include_param_norm:
            for m in mods.values():
                for t in self._iter_param_tensors(m):
                    param_block_norms.append(self._tensor_pnorm(t, self.norm_type))

        if self.include_grad_norm:
            for m in mods.values():
                for g in self._iter_grad_tensors(m):
                    grad_block_norms.append(self._tensor_pnorm(g, self.norm_type))

        if self.include_param_norm:
            out["param_norm"] = self._aggregate_pnorm(param_block_norms, self.norm_type)
        if self.include_grad_norm:
            out["grad_norm"] = self._aggregate_pnorm(grad_block_norms, self.norm_type)

        self.log(trainer, out, step=upd, prefix=self.log_prefix)
        return True
