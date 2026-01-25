from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union, Callable

import torch as th
import torch.nn as nn
from torch.optim import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer (sign-based momentum descent).

    Lion maintains an exponential moving average (EMA) of gradients and updates
    parameters using the sign of a blended direction. Weight decay is applied
    in a decoupled manner (AdamW-style shrinkage).

    Parameters
    ----------
    params : Iterable[nn.Parameter] or Iterable[Dict[str, Any]]
        Parameters to optimize (PyTorch optimizer format).
    lr : float, optional
        Learning rate (must be positive), by default 1e-4.
    betas : Tuple[float, float], optional
        (beta1, beta2) where:
          - beta2 controls EMA smoothing for `exp_avg`
          - beta1 controls blending between EMA and current gradient
        by default (0.9, 0.99).
    weight_decay : float, optional
        Decoupled weight decay coefficient, by default 0.0.

    Notes
    -----
    Conceptual update:

      exp_avg <- beta2 * exp_avg + (1 - beta2) * grad
      u       <- beta1 * exp_avg + (1 - beta1) * grad
      p       <- (1 - lr * wd) * p - lr * sign(u)

    Limitations
    -----------
    - Sparse gradients are not supported.
    - This optimizer assumes `p.grad` is already scaled correctly (e.g., if using AMP,
      call scaler.unscale_(optimizer) before step()).
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

        b1, b2 = float(betas[0]), float(betas[1])
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got: {(b1, b2)}")

        wd = float(weight_decay)
        if wd < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got: {wd}")

        defaults = dict(lr=lr, betas=(b1, b2), weight_decay=wd)
        super().__init__(params, defaults)

    @th.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : Optional[Callable[[], Any]], optional
            A closure that re-evaluates the model and returns the loss.
            Included for API compatibility.

        Returns
        -------
        loss : Optional[float]
            Closure loss if provided; otherwise None.

        Raises
        ------
        RuntimeError
            If a sparse gradient is encountered.
        """
        loss: Optional[float] = None
        if closure is not None:
            with th.enable_grad():
                loss_t = closure()
            if th.is_tensor(loss_t):
                loss = float(loss_t.detach().cpu().item())
            else:
                loss = float(loss_t)

        for group in self.param_groups:
            lr: float = float(group["lr"])
            wd: float = float(group.get("weight_decay", 0.0))
            beta1, beta2 = group["betas"]
            beta1 = float(beta1)
            beta2 = float(beta2)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                grad = p.grad.detach()

                # Decoupled weight decay (AdamW-style)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                exp_avg = state.get("exp_avg", None)
                if exp_avg is None:
                    exp_avg = th.zeros_like(p, memory_format=th.preserve_format)
                    state["exp_avg"] = exp_avg

                # exp_avg <- beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=(1.0 - beta2))

                # u <- beta1 * exp_avg + (1 - beta1) * grad
                # We avoid building multiple temporaries where practical.
                update = exp_avg.mul(beta1).add(grad, alpha=(1.0 - beta1))

                # p <- p - lr * sign(u)
                p.add_(update.sign(), alpha=-lr)

        return loss