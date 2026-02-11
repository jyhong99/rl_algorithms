from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch as th
import torch.nn as nn
from torch.optim import Optimizer


class Lion(Optimizer):
    r"""
    Lion optimizer (sign-based momentum descent).

    Lion is a lightweight, sign-based optimizer that maintains an exponential
    moving average (EMA) of gradients and updates parameters using the **sign**
    of a blended direction.

    Compared to Adam/AdamW:
    - No second-moment (variance) accumulator.
    - The update direction is sign(·), which makes step magnitudes largely
      independent of gradient scale (but still sensitive to learning rate).
    - Weight decay is applied in a **decoupled** manner (AdamW-style shrinkage).

    Conceptual update
    -----------------
    Let `g_t` be the gradient at step `t`, and `m_t` the EMA of gradients.

    1) EMA update (momentum-like):
        m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

    2) Blended direction:
        u_t = beta1 * m_t + (1 - beta1) * g_t

    3) Decoupled weight decay + sign update:
        p_t = (1 - lr * wd) * p_{t-1} - lr * sign(u_t)

    Here:
    - `beta2` primarily controls smoothing of the EMA (`m_t`).
    - `beta1` controls how much we trust the EMA vs current gradient in `u_t`.

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter] or Iterable[Dict[str, Any]]
        Iterable of parameters or parameter-group dicts (standard PyTorch format).
    lr : float, default=1e-4
        Learning rate. Must be strictly positive.
    betas : Tuple[float, float], default=(0.9, 0.99)
        Coefficients for the update:
        - beta1 : float
            Blend factor between EMA direction and raw gradient in `u_t`.
        - beta2 : float
            EMA coefficient for the gradient accumulator `m_t` (stored as `exp_avg`).
        Both must lie in [0, 1).
    weight_decay : float, default=0.0
        Decoupled weight decay coefficient (AdamW-style). Must be >= 0.

    State
    -----
    For each parameter tensor `p`, the optimizer maintains:
    - exp_avg : torch.Tensor
        EMA of gradients, same shape/dtype/device as `p`.

    Notes
    -----
    - Sparse gradients are not supported.
    - Mixed precision (AMP):
      If you use `torch.cuda.amp.GradScaler`, make sure gradients are unscaled
      before calling `step()` (e.g., `scaler.unscale_(optimizer)`), otherwise
      the sign direction may be distorted.
    - This implementation uses `grad = p.grad.detach()` to avoid tracking graph.
      It assumes gradients are already computed and (optionally) unscaled.

    References
    ----------
    Lion was proposed by Google researchers as a memory-efficient optimizer.
    Many community implementations follow the same sign-based update pattern.

    See also
    --------
    torch.optim.AdamW : decoupled weight decay baseline
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        lr = float(lr)
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got: {lr}")

        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got: {(beta1, beta2)}")

        wd = float(weight_decay)
        if wd < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got: {wd}")

        defaults = dict(lr=lr, betas=(beta1, beta2), weight_decay=wd)
        super().__init__(params, defaults)

    @staticmethod
    def _init_state(state: Dict[str, Any], p: th.Tensor) -> th.Tensor:
        """
        Initialize per-parameter state if missing.

        Parameters
        ----------
        state : Dict[str, Any]
            Optimizer state dict for parameter `p`.
        p : torch.Tensor
            Parameter tensor.

        Returns
        -------
        exp_avg : torch.Tensor
            The EMA buffer for gradients.
        """
        exp_avg = state.get("exp_avg", None)
        if exp_avg is None:
            exp_avg = th.zeros_like(p, memory_format=th.preserve_format)
            state["exp_avg"] = exp_avg
        return exp_avg

    @th.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : Optional[Callable[[], Any]], optional
            A closure that re-evaluates the model and returns the loss.
            This is included for PyTorch optimizer API compatibility.
            If provided, it is executed under `torch.enable_grad()`.

        Returns
        -------
        loss : Optional[float]
            The loss returned by `closure` if provided; otherwise None.

        Raises
        ------
        RuntimeError
            If a sparse gradient is encountered (Lion does not support it).
        """
        loss: Optional[float] = None
        if closure is not None:
            with th.enable_grad():
                loss_t = closure()
            loss = float(loss_t.detach().cpu().item()) if th.is_tensor(loss_t) else float(loss_t)

        for group in self.param_groups:
            lr: float = float(group["lr"])
            wd: float = float(group.get("weight_decay", 0.0))
            beta1, beta2 = group["betas"]
            beta1 = float(beta1)
            beta2 = float(beta2)

            # Precompute for minor efficiency/readability
            one_minus_beta1 = 1.0 - beta1
            one_minus_beta2 = 1.0 - beta2

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                grad = p.grad.detach()

                # ------------------------------------------------------------
                # 1) Decoupled weight decay (AdamW-style parameter shrinkage)
                #    p <- (1 - lr * wd) * p
                # ------------------------------------------------------------
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ------------------------------------------------------------
                # 2) EMA update:
                #    exp_avg <- beta2 * exp_avg + (1 - beta2) * grad
                # ------------------------------------------------------------
                state = self.state[p]
                exp_avg = self._init_state(state, p)
                exp_avg.mul_(beta2).add_(grad, alpha=one_minus_beta2)

                # ------------------------------------------------------------
                # 3) Blended direction:
                #    u <- beta1 * exp_avg + (1 - beta1) * grad
                #
                #    Note: `update` is a temporary tensor (not in-place on exp_avg)
                #    so exp_avg remains the EMA buffer.
                # ------------------------------------------------------------
                update = exp_avg.mul(beta1).add(grad, alpha=one_minus_beta1)

                # ------------------------------------------------------------
                # 4) Sign-based parameter update:
                #    p <- p - lr * sign(u)
                # ------------------------------------------------------------
                p.add_(update.sign(), alpha=-lr)

        return loss
