from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import math

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from ..utils.common_utils import img2col, ema_update


class KFAC(Optimizer):
    """
    Kronecker-Factored Approximate Curvature (KFAC) Optimizer.

    This optimizer collects per-layer covariance statistics for supported layers
    (nn.Linear, nn.Conv2d) using hooks:

      - aa_hat: covariance of layer inputs / activations
      - gg_hat: covariance of output gradients ("gradient signals")

    Periodically, it forms an approximate inverse Fisher block via eigendecomposition:
      (G ⊗ A)^{-1} ≈ (Q_g Λ_g Q_g^T ⊗ Q_a Λ_a Q_a^T)^{-1}

    Then it preconditions the gradients and applies an internal SGD step.

    Parameters
    ----------
    model : nn.Module
        Model to optimize. Required because KFAC attaches hooks to modules.
    lr : float, optional
        Base learning rate (before trust-region scaling), by default 0.25.
    weight_decay : float, optional
        Coupled L2 penalty (adds wd * p to p.grad), by default 0.0.
        Note: This is *coupled* decay, not decoupled AdamW-style.
    damping : float, optional
        Damping term added to curvature approximation, by default 1e-2.
    momentum : float, optional
        Momentum used by internal SGD, by default 0.9.
    eps : float, optional
        EMA coefficient for running covariances (Polyak averaging), by default 0.95.
        Higher => slower update of running covariances.
    Ts : int, optional
        Statistics collection interval (in optimizer steps), by default 1.
    Tf : int, optional
        Inverse (eigendecomposition) update interval (in optimizer steps), by default 10.
    max_lr : float, optional
        Upper bound for trust-region scaling factor nu, by default 1.0.
    trust_region : float, optional
        Trust-region radius used to compute nu, by default 2e-3.

    Notes
    -----
    1) Supported layers: nn.Linear and nn.Conv2d only.

    2) Backward hook:
       This implementation uses `register_full_backward_hook`.
       In PyTorch, module-level backward hooks can be subtle with re-entrant graphs
       or certain compiled/optimized modes. If you see missing stats, consider:
       - capturing gradients via Tensor hooks instead, or
       - ensuring the layer participates in the backward graph as expected.

    3) Fisher stats toggle:
       gg_hat collection is gated by `self.fisher_backprop`.
       Typical usage is to run a dedicated Fisher-backprop pass (or enable it
       for policy loss only).

    4) Checkpointing:
       State is serialized as layer-order lists (aligned to `_trainable_layers`)
       rather than dict keyed by module object identity.
       This assumes identical model structure and identical traversal order.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.25,
        weight_decay: float = 0.0,
        damping: float = 1e-2,
        momentum: float = 0.9,
        eps: float = 0.95,
        Ts: int = 1,
        Tf: int = 10,
        max_lr: float = 1.0,
        trust_region: float = 2e-3,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got: {lr}")
        if not (0.0 <= weight_decay):
            raise ValueError(f"weight_decay must be >= 0, got: {weight_decay}")
        if damping < 0:
            raise ValueError(f"damping must be >= 0, got: {damping}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got: {momentum}")
        if not (0.0 < eps < 1.0):
            raise ValueError(f"eps must be in (0, 1), got: {eps}")
        if Ts <= 0 or Tf <= 0:
            raise ValueError(f"Ts and Tf must be > 0, got Ts={Ts}, Tf={Tf}")
        if max_lr <= 0:
            raise ValueError(f"max_lr must be > 0, got: {max_lr}")
        if trust_region <= 0:
            raise ValueError(f"trust_region must be > 0, got: {trust_region}")

        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.damping = float(damping)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.Ts = int(Ts)
        self.Tf = int(Tf)
        self.max_lr = float(max_lr)
        self.trust_region = float(trust_region)

        # Optimizer step counter
        self._k: int = 0

        # Toggle to collect gg_hat in backward hook
        self.fisher_backprop: bool = False

        self.acceptable_layer_types = (nn.Linear, nn.Conv2d)
        self._trainable_layers: List[nn.Module] = []

        # Running covariances
        self._aa_hat: Dict[nn.Module, th.Tensor] = {}
        self._gg_hat: Dict[nn.Module, th.Tensor] = {}

        # Cached eigendecompositions
        self._eig_a: Dict[nn.Module, th.Tensor] = {}
        self._Q_a: Dict[nn.Module, th.Tensor] = {}
        self._eig_g: Dict[nn.Module, th.Tensor] = {}
        self._Q_g: Dict[nn.Module, th.Tensor] = {}

        # Keep hook handles to avoid accidental double-registration
        self._hook_handles: List[Any] = []

        # Internal SGD (many KFAC refs use lr*(1-momentum) for the "base" step)
        self._sgd = optim.SGD(
            self.model.parameters(),
            lr=self.lr * (1.0 - self.momentum),
            momentum=self.momentum,
        )

        # Initialize Optimizer base class (required by torch API)
        defaults = dict(
            lr=self.lr,
            weight_decay=self.weight_decay,
            damping=self.damping,
            momentum=self.momentum,
            eps=self.eps,
            Ts=self.Ts,
            Tf=self.Tf,
            max_lr=self.max_lr,
            trust_region=self.trust_region,
        )
        super().__init__(self.model.parameters(), defaults)

        self._register_hooks()

    # ---------------------------------------------------------------------
    # Public controls
    # ---------------------------------------------------------------------
    def set_fisher_backprop(self, enabled: bool) -> None:
        """
        Enable/disable gg_hat collection in backward hook.

        Parameters
        ----------
        enabled : bool
            If True, collects gradient-signal statistics in backward hook.
        """
        self.fisher_backprop = bool(enabled)

    # ---------------------------------------------------------------------
    # Hook registration
    # ---------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """
        Register hooks for supported layer types.

        Notes
        -----
        - Forward-pre-hook collects activation stats (aa_hat).
        - Full backward hook collects grad-output stats (gg_hat) when enabled.
        """
        # Prevent accidental multiple registrations
        if self._hook_handles:
            return

        for m in self.model.modules():
            if isinstance(m, self.acceptable_layer_types):
                h1 = m.register_forward_pre_hook(self._save_aa)
                h2 = m.register_full_backward_hook(self._save_gg)
                self._hook_handles.extend([h1, h2])
                self._trainable_layers.append(m)

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.

        Notes
        -----
        Useful when you want to reuse the optimizer object across different models
        (generally not recommended) or to avoid hook side-effects.
        """
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    # ---------------------------------------------------------------------
    # Statistics collection
    # ---------------------------------------------------------------------
    @th.no_grad()
    def _save_aa(self, layer: nn.Module, layer_input: Tuple[th.Tensor, ...]) -> None:
        """
        Update running activation covariance aa_hat for a layer.

        Parameters
        ----------
        layer : nn.Module
            nn.Linear or nn.Conv2d.
        layer_input : Tuple[torch.Tensor, ...]
            Hook input tuple; uses layer_input[0].

        Notes
        -----
        For Linear:
            a has shape (B, in_features)
            aa = (a^T a) / B

        For Conv2d:
            We convert input to im2col:
              a_col has shape (B*OH*OW, in_channels * KH * KW)
            Then:
              aa = (a_col^T a_col) / (B*OH*OW)

        Bias handling:
            If layer has bias, append 1s column to activations so that bias
            is included in the KFAC block.
        """
        if (self._k % self.Ts) != 0:
            return
        if not layer_input or layer_input[0] is None:
            return

        a = layer_input[0].detach()

        if isinstance(layer, nn.Conv2d):
            a = img2col(a, layer.kernel_size, layer.stride, layer.padding)
        else:
            a = a.view(a.size(0), -1)

        n = a.size(0)

        if getattr(layer, "bias", None) is not None:
            a = th.cat([a, a.new_ones(n, 1)], dim=1)

        aa = (a.t() @ a) / float(max(n, 1))

        prev = self._aa_hat.get(layer, None)
        if prev is None:
            self._aa_hat[layer] = aa.clone()
        else:
            # running update: prev <- beta*prev + (1-beta)*aa
            ema_update(prev, aa, beta=self.eps)

    @th.no_grad()
    def _save_gg(
        self,
        layer: nn.Module,
        grad_input: Tuple[Optional[th.Tensor], ...],
        grad_output: Tuple[Optional[th.Tensor], ...],
    ) -> None:
        """
        Update running gradient-signal covariance gg_hat for a layer.

        Parameters
        ----------
        layer : nn.Module
            nn.Linear or nn.Conv2d.
        grad_input : Tuple[Optional[torch.Tensor], ...]
            Unused (hook signature).
        grad_output : Tuple[Optional[torch.Tensor], ...]
            Uses grad_output[0] as output gradient.

        Notes
        -----
        This is gated by `self.fisher_backprop`.

        For Linear:
            ds has shape (B, out_features)
            gg = (ds^T ds) / B

        For Conv2d:
            ds original shape: (B, out_channels, OH, OW)
            reshape to (B*OH*OW, out_channels)
            gg = (ds^T ds) / (B*OH*OW)

        Important:
        Some older code multiplies ds by batch_size before forming gg.
        That scaling typically does NOT match the standard Fisher block estimate
        and can destabilize trust-region scaling, so it is omitted here.
        """
        if not self.fisher_backprop:
            return
        if (self._k % self.Ts) != 0:
            return
        if not grad_output or grad_output[0] is None:
            return

        ds = grad_output[0].detach()

        if isinstance(layer, nn.Conv2d):
            ds = ds.permute(0, 2, 3, 1).contiguous().view(-1, ds.size(1))
        else:
            ds = ds.view(ds.size(0), -1)

        n = ds.size(0)
        gg = (ds.t() @ ds) / float(max(n, 1))

        prev = self._gg_hat.get(layer, None)
        if prev is None:
            self._gg_hat[layer] = gg.clone()
        else:
            # running update: prev <- beta*prev + (1-beta)*gg
            ema_update(prev, gg, beta=self.eps)

    @th.no_grad()
    def _update_inverses(self, layer: nn.Module) -> None:
        """
        Compute and cache eigendecompositions for aa_hat and gg_hat.

        Parameters
        ----------
        layer : nn.Module
            Target layer.

        Notes
        -----
        Uses symmetric eigendecomposition (eigh). Small eigenvalues are floored
        to improve numerical stability.
        """
        aa = self._aa_hat[layer]
        gg = self._gg_hat[layer]

        eig_a, Q_a = th.linalg.eigh(aa, UPLO="U")
        eig_g, Q_g = th.linalg.eigh(gg, UPLO="U")

        # Floor tiny eigenvalues (avoid exploding inverse)
        floor = 1e-6
        eig_a = th.clamp(eig_a, min=floor)
        eig_g = th.clamp(eig_g, min=floor)

        self._eig_a[layer], self._Q_a[layer] = eig_a, Q_a
        self._eig_g[layer], self._Q_g[layer] = eig_g, Q_g

    # ---------------------------------------------------------------------
    # Optimization step
    # ---------------------------------------------------------------------
    @th.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """
        Perform one KFAC step (precondition + internal SGD update).

        Parameters
        ----------
        closure : optional
            A closure that re-evaluates the model and returns the loss.
            Kept for API compatibility; usually unused.

        Returns
        -------
        loss : Optional[float]
            Closure loss if provided; otherwise None.
        """
        loss: Optional[float] = None
        if closure is not None:
            with th.enable_grad():
                loss_t = closure()
            if th.is_tensor(loss_t):
                loss = float(loss_t.detach().cpu().item())
            else:
                loss = float(loss_t)

        # Coupled L2 penalty (applied to raw gradients)
        if self.weight_decay > 0.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.add_(p.data, alpha=self.weight_decay)

        # Build preconditioned updates per layer
        updates: Dict[nn.Module, List[th.Tensor]] = {}

        for layer in self._trainable_layers:
            if layer not in self._aa_hat or layer not in self._gg_hat:
                continue
            if getattr(layer, "weight", None) is None or layer.weight.grad is None:
                continue

            if (self._k % self.Tf == 0) or (layer not in self._eig_a) or (layer not in self._eig_g):
                self._update_inverses(layer)

            grad_w = layer.weight.grad.detach()
            if isinstance(layer, nn.Conv2d):
                grad_w_mat = grad_w.view(grad_w.size(0), -1)
            else:
                grad_w_mat = grad_w.view(grad_w.size(0), -1)

            has_bias = getattr(layer, "bias", None) is not None and layer.bias is not None and layer.bias.grad is not None
            if has_bias:
                grad_b = layer.bias.grad.detach().view(-1, 1)
                grad = th.cat([grad_w_mat, grad_b], dim=1)
            else:
                grad = grad_w_mat

            # Precondition: inv(G) * grad * inv(A)
            Qg = self._Q_g[layer]
            Qa = self._Q_a[layer]
            eg = self._eig_g[layer]
            ea = self._eig_a[layer]

            V1 = Qg.t() @ grad @ Qa
            denom = (eg.unsqueeze(-1) @ ea.unsqueeze(0)) + (self.damping + self.weight_decay)
            V2 = V1 / denom
            delta = Qg @ V2 @ Qa.t()

            if has_bias:
                delta_w = delta[:, :-1].contiguous().view_as(layer.weight.grad)
                delta_b = delta[:, -1:].contiguous().view_as(layer.bias.grad)
                updates[layer] = [delta_w, delta_b]
            else:
                updates[layer] = [delta.contiguous().view_as(layer.weight.grad)]

        # Trust-region scaling (nu)
        #
        # Many implementations approximate:
        #   nu = sqrt(2 * trust_region / (g^T F^{-1} g))
        #
        # Here, we approximate the denominator using the preconditioned direction
        # and current gradients. This is a heuristic; stability relies on damping.
        second_term = 0.0
        lr2 = self.lr ** 2

        for layer, v in updates.items():
            g_w = layer.weight.grad.detach()
            second_term += float((v[0] * g_w * lr2).sum().item())
            if len(v) > 1 and getattr(layer, "bias", None) is not None and layer.bias is not None:
                if layer.bias.grad is not None:
                    second_term += float((v[1] * layer.bias.grad.detach() * lr2).sum().item())

        denom = max(second_term, 1e-12)
        nu = math.sqrt(2.0 * self.trust_region / denom)
        nu = min(self.max_lr, nu)

        # Replace raw grads with KFAC-preconditioned grads (scaled by nu)
        for layer, v in updates.items():
            layer.weight.grad.copy_(v[0]).mul_(nu)
            if len(v) > 1 and getattr(layer, "bias", None) is not None and layer.bias is not None:
                if layer.bias.grad is not None:
                    layer.bias.grad.copy_(v[1]).mul_(nu)

        self._sgd.step()
        self._k += 1
        return loss

    # ---------------------------------------------------------------------
    # Checkpointing (layer-order lists)
    # ---------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize KFAC state for checkpointing.

        Returns
        -------
        state : Dict[str, Any]
            Contains:
              - sgd_state_dict : dict
              - k : int
              - aa_hat_list, gg_hat_list : list[Tensor|None]
              - eig_a_list, eig_g_list : list[Tensor|None]
              - Q_a_list, Q_g_list : list[Tensor|None]

        Notes
        -----
        This format assumes the same model structure and same `_trainable_layers` order.
        """
        aa_list: List[Optional[th.Tensor]] = []
        gg_list: List[Optional[th.Tensor]] = []
        eig_a_list: List[Optional[th.Tensor]] = []
        eig_g_list: List[Optional[th.Tensor]] = []
        Q_a_list: List[Optional[th.Tensor]] = []
        Q_g_list: List[Optional[th.Tensor]] = []

        for layer in self._trainable_layers:
            aa_list.append(self._aa_hat.get(layer))
            gg_list.append(self._gg_hat.get(layer))
            eig_a_list.append(self._eig_a.get(layer))
            eig_g_list.append(self._eig_g.get(layer))
            Q_a_list.append(self._Q_a.get(layer))
            Q_g_list.append(self._Q_g.get(layer))

        return {
            "sgd_state_dict": self._sgd.state_dict(),
            "k": int(self._k),
            "aa_hat_list": aa_list,
            "gg_hat_list": gg_list,
            "eig_a_list": eig_a_list,
            "eig_g_list": eig_g_list,
            "Q_a_list": Q_a_list,
            "Q_g_list": Q_g_list,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Restore KFAC state from a checkpoint.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            A dict produced by `KFAC.state_dict()`.

        Raises
        ------
        ValueError
            If checkpoint list lengths do not match current `_trainable_layers`.
        """
        sd = dict(state_dict)

        self._sgd.load_state_dict(sd["sgd_state_dict"])
        self._k = int(sd["k"])

        aa_list = list(sd["aa_hat_list"])
        gg_list = list(sd["gg_hat_list"])
        eig_a_list = list(sd["eig_a_list"])
        eig_g_list = list(sd["eig_g_list"])
        Q_a_list = list(sd["Q_a_list"])
        Q_g_list = list(sd["Q_g_list"])

        n_layers = len(self._trainable_layers)
        lists: Sequence[List[Any]] = [aa_list, gg_list, eig_a_list, eig_g_list, Q_a_list, Q_g_list]
        if not all(len(x) == n_layers for x in lists):
            raise ValueError("Invalid KFAC checkpoint: list lengths do not match current trainable layers.")

        # Clear then restore (module-keyed dicts)
        self._aa_hat.clear()
        self._gg_hat.clear()
        self._eig_a.clear()
        self._eig_g.clear()
        self._Q_a.clear()
        self._Q_g.clear()

        for layer, a, g, ea, eg, qa, qg in zip(
            self._trainable_layers,
            aa_list,
            gg_list,
            eig_a_list,
            eig_g_list,
            Q_a_list,
            Q_g_list,
        ):
            if a is not None:
                self._aa_hat[layer] = a
            if g is not None:
                self._gg_hat[layer] = g
            if ea is not None:
                self._eig_a[layer] = ea
            if eg is not None:
                self._eig_g[layer] = eg
            if qa is not None:
                self._Q_a[layer] = qa
            if qg is not None:
                self._Q_g[layer] = qg