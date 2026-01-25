from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Callable, List

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import BaseCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class TRPOCore(BaseCore):
    """
    TRPO Update Engine (config-free)  [continuous-only]

    Features
    --------
    - One update per call (on-policy batch)
    - Natural gradient via conjugate gradient (CG) using Fisher-vector product (FVP)
    - KL constraint via backtracking line search
    - Critic updated with standard regression optimizer/scheduler

    Expected head interface
    -----------------------
    head.actor: policy network (must implement get_dist(obs))
    head.critic: value network V(s)
    head.device: torch.device (or string)
    """

    def __init__(
        self,
        *,
        head: Any,
        # KL / natural gradient
        max_kl: float = 1e-2,
        cg_iters: int = 10,
        cg_damping: float = 1e-2,
        # line search
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        accept_ratio: float = 0.1,
        # critic
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        critic_sched_name: str = "none",
        # sched shared knobs (optional; keep parity with other cores)
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # misc
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
    ) -> None:
        super().__init__(head=head, use_amp=use_amp)

        self.max_kl = float(max_kl)
        self.cg_iters = int(cg_iters)
        self.cg_damping = float(cg_damping)

        self.backtrack_iters = int(backtrack_iters)
        self.backtrack_coeff = float(backtrack_coeff)
        self.accept_ratio = float(accept_ratio)

        self.max_grad_norm = float(max_grad_norm)

        # -------------------------
        # Basic argument validation
        # -------------------------
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.max_kl <= 0.0:
            raise ValueError(f"max_kl must be > 0, got {self.max_kl}")
        if self.cg_iters <= 0:
            raise ValueError(f"cg_iters must be > 0, got {self.cg_iters}")
        if self.cg_damping < 0.0:
            raise ValueError(f"cg_damping must be >= 0, got {self.cg_damping}")
        if not (0.0 < self.backtrack_coeff < 1.0):
            raise ValueError(f"backtrack_coeff must be in (0,1), got {self.backtrack_coeff}")
        if self.backtrack_iters <= 0:
            raise ValueError(f"backtrack_iters must be > 0, got {self.backtrack_iters}")
        if self.accept_ratio < 0.0:
            raise ValueError(f"accept_ratio must be >= 0, got {self.accept_ratio}")

        # --------------------------------------
        # Critic optimizer / scheduler (baseline)
        # --------------------------------------
        self.critic_opt = build_optimizer(
            self.head.critic.parameters(),
            name=str(critic_optim_name),
            lr=float(critic_lr),
            weight_decay=float(critic_weight_decay),
        )
        self.critic_sched = build_scheduler(
            self.critic_opt,
            name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

    def _step_scheds(self) -> None:
        """Step actor/critic schedulers if they exist."""
        if self.critic_sched is not None:
            self.critic_sched.step()

    # ============================================================
    # Flat parameter utilities
    # ============================================================
    @staticmethod
    def _flat_params(module: th.nn.Module) -> th.Tensor:
        """
        Return a single flat vector containing all parameters (in module order).
        Uses .data for speed; TRPO line search explicitly assigns candidate params.
        """
        return th.cat([p.data.view(-1) for p in module.parameters()])

    @staticmethod
    def _assign_flat_params(module: th.nn.Module, flat: th.Tensor) -> None:
        """Assign parameters from a flat vector (must match total parameter count)."""
        i = 0
        for p in module.parameters():
            n = p.numel()
            p.data.copy_(flat[i : i + n].view_as(p))
            i += n

    @staticmethod
    def _flat_grad_like_params(
        params: List[th.nn.Parameter],
        grads: Sequence[th.Tensor | None],
    ) -> th.Tensor:
        """
        Flatten gradients into a single vector, matching the parameter vector length.

        Unlike a naive "skip None grads" approach, this fills missing grads with zeros,
        ensuring shape consistency with the flat parameter vector. This is critical for
        TRPO/CG where vector lengths must match exactly.
        """
        out = []
        for p, g in zip(params, grads):
            if g is None:
                out.append(th.zeros_like(p).contiguous().view(-1))
            else:
                out.append(g.contiguous().view(-1))
        return th.cat(out) if len(out) > 0 else th.zeros(0)

    # ============================================================
    # KL + Fisher-vector product
    # ============================================================
    def _mean_kl(self, obs: th.Tensor, old_dist: Any) -> th.Tensor:
        """
        Compute mean KL(new || old) over the batch.

        Requires `new_dist.kl(old_dist)` to exist and be differentiable w.r.t.
        the new policy parameters (actor parameters).
        """
        new_dist = self.head.actor.get_dist(obs)

        # Prefer analytic KL if available (your dist implementation must provide it)
        if hasattr(new_dist, "kl") and callable(getattr(new_dist, "kl")):
            return new_dist.kl(old_dist).mean()

        raise RuntimeError("TRPO requires distribution.kl(old_dist).")

    def _fvp(self, obs: th.Tensor, old_dist: Any, v: th.Tensor) -> th.Tensor:
        """
        Fisher-vector product: (F + damping*I) v, approximated by Hessian of KL.

        IMPORTANT: Do NOT wrap in no_grad; this needs second-order gradients.
        """
        kl = self._mean_kl(obs, old_dist)

        params = list(self.head.actor.parameters())

        grads = th.autograd.grad(
            kl,
            params,
            create_graph=True,
            retain_graph=True,
        )
        flat_grad_kl = self._flat_grad_like_params(params, grads)

        grad_v = (flat_grad_kl * v).sum()
        hvp = th.autograd.grad(grad_v, params, retain_graph=False)
        flat_hvp = self._flat_grad_like_params(params, hvp)

        return flat_hvp + self.cg_damping * v

    @staticmethod
    def _conjugate_gradient(
        Avp: Callable[[th.Tensor], th.Tensor],
        b: th.Tensor,
        iters: int,
    ) -> th.Tensor:
        """
        Solve A x = b for x using conjugate gradient, where Avp computes A v.

        Notes
        -----
        - A is assumed symmetric positive definite (approximately true for Fisher).
        - Stops early if residual becomes sufficiently small.
        """
        x = th.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = th.dot(r, r)

        for _ in range(int(iters)):
            Avp_p = Avp(p)
            denom = th.dot(p, Avp_p) + 1e-8
            alpha = rdotr / denom
            x = x + alpha * p
            r = r - alpha * Avp_p
            new_rdotr = th.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr

        return x

    # ============================================================
    # Main update
    # ============================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one TRPO-style update from an on-policy batch.

        Batch requirements (typical)
        ----------------------------
        batch.observations : (B, obs_dim)
        batch.actions      : (B, act_dim)  (or (act_dim,) for B=1)
        batch.returns      : (B,) or (B,1)
        batch.advantages   : (B,) or (B,1)
        batch.log_probs    : (B,) or (B,1)  old log-prob under behavior policy
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        ret = to_column(batch.returns.to(self.device))
        adv = to_column(batch.advantages.to(self.device))
        old_logp = to_column(batch.log_probs.to(self.device))

        # ------------------------------------------------------------
        # Critic update (regression on returns)
        # ------------------------------------------------------------
        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                v = to_column(self.head.critic(obs))
                value_loss = 0.5 * F.mse_loss(v, ret)

            self.scaler.scale(value_loss).backward()
            self._clip_params(
                list(self.head.critic.parameters()),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            v = to_column(self.head.critic(obs))
            value_loss = 0.5 * F.mse_loss(v, ret)

            value_loss.backward()
            self._clip_params(
                list(self.head.critic.parameters()),
                max_grad_norm=self.max_grad_norm,
                optimizer=None,
            )
            self.critic_opt.step()

        # ------------------------------------------------------------
        # Policy update (TRPO)
        # ------------------------------------------------------------
        # Freeze the "old" distribution as a fixed reference.
        with th.no_grad():
            old_dist = self.head.actor.get_dist(obs)

        def surrogate(with_grad: bool) -> th.Tensor:
            """
            Standard importance-weighted surrogate objective:
                E[ exp(logp - old_logp) * adv ]
            """
            if with_grad:
                dist = self.head.actor.get_dist(obs)
                logp = to_column(dist.log_prob(act))
                ratio = th.exp(logp - old_logp)
                return (ratio * adv).mean()

            with th.no_grad():
                dist = self.head.actor.get_dist(obs)
                logp = to_column(dist.log_prob(act))
                ratio = th.exp(logp - old_logp)
                return (ratio * adv).mean()

        # We maximize surrogate, but autograd is a minimizer mindset.
        # Use surr gradient (NOT -surr) as b in CG.
        surr = surrogate(with_grad=True)  # graph needed for policy gradient
        params = list(self.head.actor.parameters())

        grads_surr = th.autograd.grad(surr, params, retain_graph=False)
        g = self._flat_grad_like_params(params, grads_surr).detach()  # b = ∇surr

        # Solve (F + damping I) x = g
        step_dir = self._conjugate_gradient(
            lambda v_: self._fvp(obs, old_dist, v_),
            g,
            self.cg_iters,
        )

        # Scale step so that 0.5 * step^T F step <= max_kl
        shs = 0.5 * (step_dir * self._fvp(obs, old_dist, step_dir)).sum()
        step_size = th.sqrt(self.max_kl / (shs + 1e-12))
        full_step = step_size * step_dir

        old_params = self._flat_params(self.head.actor).clone()

        # Expected improvement under linear model: g^T step
        expected_improve = float((g * full_step).sum().item())

        surr_old = float(surr.detach().cpu().item())
        accepted = False
        step_frac = 1.0
        best_kl = 0.0

        # Backtracking line search to satisfy KL constraint and sufficient improvement
        for _ in range(self.backtrack_iters):
            step = step_frac * full_step
            self._assign_flat_params(self.head.actor, old_params + step)

            new_surr = float(surrogate(with_grad=False).cpu().item())
            kl = float(self._mean_kl(obs, old_dist).detach().cpu().item())

            improve = new_surr - surr_old
            expected = expected_improve * step_frac
            ratio = improve / (expected + 1e-8)

            if (improve > 0.0) and (kl <= self.max_kl) and (ratio >= self.accept_ratio):
                accepted = True
                best_kl = kl
                break

            step_frac *= self.backtrack_coeff

        if not accepted:
            # Revert parameters if no acceptable step was found
            self._assign_flat_params(self.head.actor, old_params)
            step_frac = 0.0
            best_kl = 0.0

        return {
            "loss/value": float(to_scalar(value_loss)),
            "stats/surr": float(surr_old),
            "stats/kl": float(best_kl),
            "stats/step_frac": float(step_frac),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
        }

    # ============================================================
    # Persistence
    # ============================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        Notes
        -----
        - `super().state_dict()` is assumed to include generic BaseCore fields.
        - Critic optimizer/scheduler state is stored using BaseCore helpers.
        """
        s = super().state_dict()
        s.update(
            {
                "critic": self._save_opt_sched(self.critic_opt, self.critic_sched),
                # Store TRPO hyperparameters at the root level (keep backward compatibility)
                "max_kl": float(self.max_kl),
                "cg_iters": int(self.cg_iters),
                "cg_damping": float(self.cg_damping),
                "backtrack_iters": int(self.backtrack_iters),
                "backtrack_coeff": float(self.backtrack_coeff),
                "accept_ratio": float(self.accept_ratio),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state.

        Fix
        ---
        The previous implementation read TRPO config from `state["trpo"]`,
        but `state_dict()` stores keys at the root. This version restores
        from the root keys to match the saved format.
        """
        super().load_state_dict(state)

        if "critic" in state:
            self._load_opt_sched(self.critic_opt, self.critic_sched, state["critic"])

        # Restore TRPO hyperparameters from root keys (matches state_dict()).
        if "max_kl" in state:
            self.max_kl = float(state["max_kl"])
        if "cg_iters" in state:
            self.cg_iters = int(state["cg_iters"])
        if "cg_damping" in state:
            self.cg_damping = float(state["cg_damping"])
        if "backtrack_iters" in state:
            self.backtrack_iters = int(state["backtrack_iters"])
        if "backtrack_coeff" in state:
            self.backtrack_coeff = float(state["backtrack_coeff"])
        if "accept_ratio" in state:
            self.accept_ratio = float(state["accept_ratio"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
