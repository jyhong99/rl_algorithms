from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import math

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from ..utils.common_utils import _img2col, _ema_update


class KFAC(Optimizer):
    r"""
    Kronecker-Factored Approximate Curvature (KFAC) Optimizer.

    KFAC approximates the Fisher information matrix (or Gauss-Newton) block
    for each supported layer as a Kronecker product:

        F ≈ G ⊗ A

    where:
        - A is the covariance of layer inputs (activations),
        - G is the covariance of layer output gradients ("gradient signals").

    This yields a cheap approximate inverse:

        (G ⊗ A)^{-1} ≈ (Q_g Λ_g Q_g^T ⊗ Q_a Λ_a Q_a^T)^{-1}

    so the preconditioned direction for a layer gradient `∇W` becomes:

        ΔW = G^{-1} ∇W A^{-1}

    In this implementation:
        - A and G are tracked as running EMA covariances (aa_hat, gg_hat).
        - Eigendecompositions are refreshed every `Tf` steps (or on-demand).
        - A simple trust-region scaling computes a scalar step multiplier `nu`.
        - The final update is applied via an internal SGD optimizer.

    Supported layers
    ----------------
    - nn.Linear
    - nn.Conv2d

    Notes on hooks
    --------------
    - Forward pre-hook captures inputs to build A.
    - Full backward hook captures grad_output to build G, gated by
      `self.fisher_backprop`.

    PyTorch module-level backward hooks can be brittle in some edge cases
    (e.g., re-entrant backward, compilation, graph breaks). If you observe
    missing statistics, consider switching to Tensor hooks.

    Parameters
    ----------
    model : nn.Module
        Model to optimize. KFAC attaches hooks to modules within `model`.
    lr : float, default=0.25
        Base learning rate used in trust-region scaling and for the internal
        SGD step (internally scaled by (1 - momentum) following common KFAC refs).
    weight_decay : float, default=0.0
        Coupled L2 penalty: adds `weight_decay * p` to `p.grad` prior to
        preconditioning. This is *not* AdamW-style decoupled decay.
    damping : float, default=1e-2
        Damping value added to the curvature eigenvalues (stabilizes inverse).
        Acts like Tikhonov regularization.
    momentum : float, default=0.9
        Momentum for internal SGD.
    eps : float, default=0.95
        Exponential moving average coefficient for running covariances.
        Update rule: C <- eps*C + (1-eps)*C_new
    Ts : int, default=1
        Statistics collection period in optimizer steps. If Ts=1, collect every step.
    Tf : int, default=10
        Frequency of inverse (eigendecomposition) refresh in optimizer steps.
    max_lr : float, default=1.0
        Upper bound on trust-region scaling factor `nu`.
    trust_region : float, default=2e-3
        Trust-region radius used to compute scaling factor `nu`.

    Attributes
    ----------
    fisher_backprop : bool
        When True, backward hook collects gg_hat. You typically enable this
        during a Fisher-estimation pass (or for specific losses only).

    Warnings
    --------
    This KFAC implementation stores per-layer state keyed by module objects and
    serializes them as lists aligned with `_trainable_layers` order. This implies:
      - You must restore into an identical model structure,
      - The module traversal order must match.

    Examples
    --------
    >>> opt = KFAC(model, lr=0.25, damping=1e-2)
    >>> opt.set_fisher_backprop(True)
    >>> loss.backward()
    >>> opt.step()
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
        # -------------------------
        # Argument validation
        # -------------------------
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got: {lr}")
        if weight_decay < 0:
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

        # Step counter
        self._k: int = 0

        # Backward-statistics toggle (collect G only if enabled)
        self.fisher_backprop: bool = False

        # Layer filtering
        self.acceptable_layer_types = (nn.Linear, nn.Conv2d)
        self._trainable_layers: List[nn.Module] = []

        # Running covariances: module -> tensor
        self._aa_hat: Dict[nn.Module, th.Tensor] = {}
        self._gg_hat: Dict[nn.Module, th.Tensor] = {}

        # Cached eigendecompositions (module -> tensor)
        self._eig_a: Dict[nn.Module, th.Tensor] = {}
        self._Q_a: Dict[nn.Module, th.Tensor] = {}
        self._eig_g: Dict[nn.Module, th.Tensor] = {}
        self._Q_g: Dict[nn.Module, th.Tensor] = {}

        # Hook handles (avoid double registration)
        self._hook_handles: List[Any] = []

        # Internal SGD: common choice is lr*(1-momentum) to match "effective lr"
        self._sgd = optim.SGD(
            self.model.parameters(),
            lr=self.lr * (1.0 - self.momentum),
            momentum=self.momentum,
        )

        # Initialize torch Optimizer base
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
        Enable/disable collection of gg_hat during backward.

        Parameters
        ----------
        enabled : bool
            If True, `_save_gg` updates running gradient-signal covariance (G).
        """
        self.fisher_backprop = bool(enabled)

    # ---------------------------------------------------------------------
    # Hook registration / removal
    # ---------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """
        Register hooks for supported layer types.

        Notes
        -----
        - Forward pre-hook: saves activation covariance (A).
        - Full backward hook: saves gradient-signal covariance (G), gated by
          `self.fisher_backprop`.

        This method is idempotent: if hooks exist, it will not register again.
        """
        if self._hook_handles:
            return

        for m in self.model.modules():
            if isinstance(m, self.acceptable_layer_types):
                self._trainable_layers.append(m)
                h1 = m.register_forward_pre_hook(self._save_aa)
                h2 = m.register_full_backward_hook(self._save_gg)
                self._hook_handles.extend([h1, h2])

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.

        Notes
        -----
        Useful if you need to tear down the optimizer cleanly or avoid hook
        side-effects when reusing a model object.
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
    @staticmethod
    def _has_bias(layer: nn.Module) -> bool:
        """Return True if layer has a trainable bias with an existing grad slot."""
        b = getattr(layer, "bias", None)
        return (b is not None) and (isinstance(b, th.Tensor))

    @th.no_grad()
    def _save_aa(self, layer: nn.Module, layer_input: Tuple[th.Tensor, ...]) -> None:
        """
        Update running activation covariance A for the given layer.

        Parameters
        ----------
        layer : nn.Module
            Target module. Must be nn.Linear or nn.Conv2d.
        layer_input : tuple[Tensor, ...]
            Hook input tuple. Uses `layer_input[0]` as the layer input activation.

        Notes
        -----
        Linear:
            a: (B, in_features)
            A = a^T a / B

        Conv2d:
            a: (B, Cin, H, W)
            Convert to im2col:
                a_col: (B*OH*OW, Cin*KH*KW)
            A = a_col^T a_col / (B*OH*OW)

        Bias handling:
            If bias exists, append a column of ones to `a` so that bias is
            included in the Kronecker block.
        """
        if (self._k % self.Ts) != 0:
            return
        if not layer_input or layer_input[0] is None:
            return

        a = layer_input[0].detach()

        if isinstance(layer, nn.Conv2d):
            a = _img2col(a, layer.kernel_size, layer.stride, layer.padding)
        else:
            a = a.view(a.size(0), -1)

        n = a.size(0)
        if self._has_bias(layer):
            a = th.cat([a, a.new_ones(n, 1)], dim=1)

        aa = (a.t() @ a) / float(max(n, 1))

        prev = self._aa_hat.get(layer)
        if prev is None:
            self._aa_hat[layer] = aa.clone()
        else:
            _ema_update(prev, aa, beta=self.eps)

    @th.no_grad()
    def _save_gg(
        self,
        layer: nn.Module,
        grad_input: Tuple[Optional[th.Tensor], ...],
        grad_output: Tuple[Optional[th.Tensor], ...],
    ) -> None:
        """
        Update running gradient-signal covariance G for the given layer.

        Parameters
        ----------
        layer : nn.Module
            Target module. Must be nn.Linear or nn.Conv2d.
        grad_input : tuple[Optional[Tensor], ...]
            Unused (required by hook signature).
        grad_output : tuple[Optional[Tensor], ...]
            Hook output gradients. Uses `grad_output[0]`.

        Notes
        -----
        This statistic is typically associated with the Fisher factor for the
        layer outputs. We gate this collection via `self.fisher_backprop` so
        you can choose when to estimate Fisher statistics.

        Linear:
            ds: (B, out_features)
            G = ds^T ds / B

        Conv2d:
            ds: (B, Cout, OH, OW)
            reshape to (B*OH*OW, Cout)
            G = ds^T ds / (B*OH*OW)

        Scaling caveat:
            Some codebases scale ds by batch_size before forming G, but that
            changes the Fisher estimate and can destabilize trust-region scaling.
            This implementation does not apply that scaling.
        """
        if not self.fisher_backprop:
            return
        if (self._k % self.Ts) != 0:
            return
        if not grad_output or grad_output[0] is None:
            return

        ds = grad_output[0].detach()

        if isinstance(layer, nn.Conv2d):
            # (B, Cout, OH, OW) -> (B*OH*OW, Cout)
            ds = ds.permute(0, 2, 3, 1).contiguous().view(-1, ds.size(1))
        else:
            ds = ds.view(ds.size(0), -1)

        n = ds.size(0)
        gg = (ds.t() @ ds) / float(max(n, 1))

        prev = self._gg_hat.get(layer)
        if prev is None:
            self._gg_hat[layer] = gg.clone()
        else:
            _ema_update(prev, gg, beta=self.eps)

    @th.no_grad()
    def _update_inverses(self, layer: nn.Module) -> None:
        """
        Compute and cache eigendecompositions for A and G.

        Parameters
        ----------
        layer : nn.Module
            Target layer.

        Notes
        -----
        We use symmetric eigendecomposition (eigh) for numerical stability:

            A = Q_a diag(eig_a) Q_a^T
            G = Q_g diag(eig_g) Q_g^T

        To avoid exploding inverse factors, we clamp eigenvalues from below.
        """
        aa = self._aa_hat[layer]
        gg = self._gg_hat[layer]

        eig_a, Q_a = th.linalg.eigh(aa, UPLO="U")
        eig_g, Q_g = th.linalg.eigh(gg, UPLO="U")

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
        Perform one optimization step.

        The step consists of:
        1) Optional closure evaluation (API compatibility).
        2) Coupled L2 penalty (if enabled).
        3) For each supported layer with stats:
           - refresh eigendecompositions if needed
           - form preconditioned direction Δ
        4) Compute trust-region scalar `nu`.
        5) Replace raw gradients with preconditioned gradients scaled by `nu`.
        6) Apply internal SGD step.

        Parameters
        ----------
        closure : callable, optional
            If provided, reevaluates the model and returns a loss.

        Returns
        -------
        loss : float or None
            Loss from closure if provided; otherwise None.
        """
        loss: Optional[float] = None
        if closure is not None:
            with th.enable_grad():
                loss_t = closure()
            loss = float(loss_t.detach().cpu().item()) if th.is_tensor(loss_t) else float(loss_t)

        # -------------------------
        # Coupled weight decay
        # -------------------------
        if self.weight_decay > 0.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.add_(p.data, alpha=self.weight_decay)

        # -------------------------
        # Build per-layer preconditioned directions
        # -------------------------
        updates: Dict[nn.Module, List[th.Tensor]] = {}

        for layer in self._trainable_layers:
            if layer not in self._aa_hat or layer not in self._gg_hat:
                continue
            if getattr(layer, "weight", None) is None or layer.weight.grad is None:
                continue

            need_refresh = (self._k % self.Tf == 0) or (layer not in self._eig_a) or (layer not in self._eig_g)
            if need_refresh:
                self._update_inverses(layer)

            # Weight grad as 2D: (out_features, in_features [+ bias])
            grad_w = layer.weight.grad.detach()
            grad_w_mat = grad_w.view(grad_w.size(0), -1)

            has_bias = (
                getattr(layer, "bias", None) is not None
                and layer.bias is not None
                and layer.bias.grad is not None
            )
            if has_bias:
                grad_b = layer.bias.grad.detach().view(-1, 1)
                grad = th.cat([grad_w_mat, grad_b], dim=1)
            else:
                grad = grad_w_mat

            Qg, eg = self._Q_g[layer], self._eig_g[layer]
            Qa, ea = self._Q_a[layer], self._eig_a[layer]

            # Δ = G^{-1} grad A^{-1} using eig factors
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

        # -------------------------
        # Trust-region scaling (scalar nu)
        # -------------------------
        # Heuristic:
        #   nu = sqrt( 2 * trust_region / (g^T F^{-1} g) )
        #
        # We approximate g^T F^{-1} g using elementwise product between the
        # preconditioned direction and the current gradient (per parameter),
        # plus an lr^2 scaling consistent with several practical KFAC codebases.
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

        # -------------------------
        # Overwrite grads with preconditioned grads
        # -------------------------
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
        Serialize KFAC internal state.

        Returns
        -------
        state : dict
            Dictionary with:
            - sgd_state_dict : dict
                Internal SGD optimizer state.
            - k : int
                Step counter.
            - aa_hat_list : list[Tensor | None]
                Running activation covariances aligned to `_trainable_layers`.
            - gg_hat_list : list[Tensor | None]
                Running gradient-signal covariances aligned to `_trainable_layers`.
            - eig_a_list, eig_g_list : list[Tensor | None]
                Cached eigenvalues aligned to `_trainable_layers`.
            - Q_a_list, Q_g_list : list[Tensor | None]
                Cached eigenvectors aligned to `_trainable_layers`.

        Notes
        -----
        This format assumes:
        - identical model structure at restore time,
        - identical module traversal order.
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
        Restore KFAC internal state.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            A dictionary produced by `KFAC.state_dict()`.

        Raises
        ------
        ValueError
            If the checkpoint list lengths do not match current `_trainable_layers`.
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

        # Clear then restore module-keyed dicts
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
