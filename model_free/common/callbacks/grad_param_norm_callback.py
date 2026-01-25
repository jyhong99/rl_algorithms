from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, List
import math

import torch as th
import torch.nn as nn

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, IntervalGate
from ..utils.log_utils import log


class GradParamNormCallback(BaseCallback):
    """
    Log parameter p-norm and gradient p-norm for debugging/troubleshooting.

    Motivation
    ----------
    When training becomes unstable (exploding/vanishing gradients, drifting weights,
    sudden divergence), logging parameter norms and gradient norms is a cheap and
    effective diagnostic signal.

    What it does
    ------------
    On a periodic *update-step* schedule, it computes and logs:
      - Parameter p-norm (||theta||_p)
      - Gradient p-norm (||grad||_p), only for parameters that currently have grads

    Logging modes
    -------------
    - per_module=False (default):
        Aggregates norms across all selected modules into global values:
          * {log_prefix}param_norm
          * {log_prefix}grad_norm

    - per_module=True:
        Logs norms for each selected module separately:
          * {log_prefix}{module}/param_norm
          * {log_prefix}{module}/grad_norm

    Module selection
    ----------------
    1) Preferred:
         algo.get_modules_for_logging() -> Mapping[str, nn.Module]
       This lets each algorithm define exactly which modules matter for norm logging.

    2) Fallback:
         - algo.policy.head (if nn.Module)
         - algo.head        (if nn.Module)
       In this case, the module is logged under name "head".

    Scheduling
    ----------
    - Uses IntervalGate(mode="mod") with `every=log_every_updates`.
      This triggers on update indices like:
         upd % log_every_updates == 0

    - If the trainer does not expose a usable update counter (upd <= 0),
      it falls back to an internal call counter to avoid "never triggers" behavior.

    Parameters
    ----------
    log_every_updates : int, default=200
        Emit logs every N updates. If <= 0, logging is disabled.
    log_prefix : str, default="debug/"
        Prefix applied to all logged keys.
    include_param_norm : bool, default=True
        Whether to compute parameter norms.
    include_grad_norm : bool, default=True
        Whether to compute gradient norms (only for params with grad != None).
    norm_type : float, default=2.0
        The p in p-norm. Must be > 0.
        Typical choices: 2.0 (L2), 1.0 (L1). (inf is not explicitly handled here.)
    per_module : bool, default=False
        If True, log per-module norms. Otherwise log global aggregated norms.
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
        # User configuration
        self.log_every_updates = int(log_every_updates)
        self.log_prefix = str(log_prefix)
        self.include_param_norm = bool(include_param_norm)
        self.include_grad_norm = bool(include_grad_norm)
        self.norm_type = float(norm_type)
        self.per_module = bool(per_module)

        # Validate norm type (p must be positive).
        if self.norm_type <= 0:
            raise ValueError(f"norm_type must be > 0, got: {self.norm_type}")

        # Gate used to trigger logging periodically based on update index.
        self._gate = IntervalGate(every=self.log_every_updates, mode="mod")

        # Fallback counter used only when trainer update step is unavailable.
        self._calls: int = 0

    # =========================================================================
    # Tensor iteration helpers
    # =========================================================================
    @staticmethod
    def _iter_param_tensors(module: nn.Module) -> Iterable[th.Tensor]:
        """
        Yield detached parameter tensors of a module (recurse=True).

        Notes
        -----
        - Using .detach() avoids accidentally creating autograd references.
        - This does not clone tensors; it is cheap.
        """
        for p in module.parameters(recurse=True):
            if p is None:
                continue
            yield p.detach()

    @staticmethod
    def _iter_grad_tensors(module: nn.Module) -> Iterable[th.Tensor]:
        """
        Yield detached gradient tensors for parameters that currently have gradients.

        Notes
        -----
        - Skips parameters with p.grad is None (e.g., frozen parameters or before backward).
        - Using .detach() avoids autograd graph interactions.
        """
        for p in module.parameters(recurse=True):
            if p is None or p.grad is None:
                continue
            yield p.grad.detach()

    @staticmethod
    def _tensor_pnorm(t: th.Tensor, p: float) -> float:
        """
        Compute the p-norm of a tensor in a robust, best-effort way.

        Implementation
        --------------
        - Uses torch.linalg.vector_norm on a flattened tensor.
        - Handles sparse tensors by measuring the p-norm of their stored values.
        - Returns 0.0 for empty tensors.
        - Any exception returns 0.0 (best-effort diagnostics should not crash training).
        """
        if t is None or t.numel() == 0:
            return 0.0
        try:
            # Sparse tensors: only norm over stored values.
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
        Compose a global p-norm from a list of block p-norms.

        If each block i has ||b_i||_p, then the combined global norm is:
            ( sum_i (||b_i||_p)^p )^(1/p)

        Notes
        -----
        - This is the correct way to combine p-norms across disjoint blocks.
        - Filters non-finite or negative values defensively.
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
            acc += fn ** p

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

        Priority:
        - algo.policy.head
        - algo.head

        Returns
        -------
        nn.Module or None
            The discovered module if it exists and is an nn.Module.
        """
        head = getattr(getattr(algo, "policy", None), "head", None)
        if head is None:
            head = getattr(algo, "head", None)
        return head if isinstance(head, nn.Module) else None

    @staticmethod
    def _modules_for_logging(algo: Any) -> Dict[str, nn.Module]:
        """
        Discover modules to log norms for.

        Preferred:
            algo.get_modules_for_logging() -> Mapping[str, nn.Module]

        Fallback:
            {"head": <nn.Module>} if algo.policy.head or algo.head is available.

        Returns
        -------
        Dict[str, nn.Module]
            A mapping from module name to nn.Module. Empty if none found.
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
                # Best-effort: ignore and fall back below.
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
        Compute p-norm of all parameters in a module (global within that module).
        """
        norms = [self._tensor_pnorm(t, p) for t in self._iter_param_tensors(module)]
        return self._aggregate_pnorm(norms, p)

    def _module_grad_norm(self, module: nn.Module, p: float) -> float:
        """
        Compute p-norm of all gradients in a module (only params with grad).
        """
        norms = [self._tensor_pnorm(g, p) for g in self._iter_grad_tensors(module)]
        return self._aggregate_pnorm(norms, p)

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after an optimizer update (or update-like event) by the Trainer.

        Scheduling logic
        --------------
        - Uses trainer update counter if available (safe_int_attr(trainer)).
        - If unavailable (<= 0), uses internal call counter so the gate still triggers.
        """
        # Disabled mode.
        if self.log_every_updates <= 0:
            return True

        # Prefer trainer-provided update index (if implemented).
        upd = safe_int_attr(trainer)

        # If update step isn't available (upd <= 0), fall back to call-count gating.
        # This prevents IntervalGate(mode="mod") from never triggering.
        if upd <= 0:
            self._calls += 1
            upd = self._calls

        # Check periodic gate.
        if not self._gate.ready(upd):
            return True

        # Trainer must expose algo to discover modules.
        algo = getattr(trainer, "algo", None)
        if algo is None:
            return True

        # Select modules to log.
        mods = self._modules_for_logging(algo)
        if not mods:
            return True

        out: Dict[str, Any] = {}

        # --------------------------------------------------------------
        # Per-module logging: one param/grad norm per module
        # --------------------------------------------------------------
        if self.per_module:
            for name, m in mods.items():
                if self.include_param_norm:
                    out[f"{name}/param_norm"] = self._module_param_norm(m, self.norm_type)
                if self.include_grad_norm:
                    out[f"{name}/grad_norm"] = self._module_grad_norm(m, self.norm_type)

            log(trainer, out, step=upd, prefix=self.log_prefix)
            return True

        # --------------------------------------------------------------
        # Global logging: aggregate across all parameter tensors
        #
        # We compute block norms for each tensor and then combine them
        # into one global p-norm using _aggregate_pnorm.
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

        log(trainer, out, step=upd, prefix=self.log_prefix)
        return True
