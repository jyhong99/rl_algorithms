from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Callable, List, Optional

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.policies.base_core import BaseCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class TRPOCore(BaseCore):
    """
    Trust Region Policy Optimization (TRPO) update engine.

    This core performs a single TRPO update per call using an on-policy batch:
    - Critic: supervised regression on returns.
    - Actor: natural-gradient step computed by conjugate gradient (CG) on the
      Fisher information matrix (approximated via the Hessian of the mean KL),
      followed by backtracking line search to satisfy a KL trust region.

    Notes
    -----
    - This implementation is intended for *continuous* action spaces where the
      policy distribution provides a differentiable log_prob and KL.
    - The "old" policy distribution is treated as a fixed reference during the
      actor update.
    - The actor step is computed by solving (F + damping I) x = g, where:
        * F is the Fisher matrix approximation,
        * damping stabilizes CG and improves conditioning,
        * g is the policy gradient of the surrogate objective.
    - The final step is scaled to satisfy: 0.5 * step^T F step <= max_kl.

    Expected `head` interface (duck-typed)
    --------------------------------------
    - head.actor: torch.nn.Module
        Must implement get_dist(obs) -> dist, where dist provides:
        * dist.log_prob(action) -> Tensor
        * dist.kl(old_dist) -> Tensor   (analytic KL preferred)
    - head.critic: torch.nn.Module
        Must implement forward(obs) -> value prediction V(s)
    - head.device: torch.device or str
        Device where computation happens.

    Parameters
    ----------
    head:
        Container that exposes actor/critic and device.
    max_kl:
        Maximum allowed mean KL divergence for the TRPO trust region.
    cg_iters:
        Number of conjugate gradient iterations for solving the natural gradient system.
    cg_damping:
        Damping coefficient added to Fisher-vector product: (F + damping I) v.
    backtrack_iters:
        Maximum number of backtracking steps during line search.
    backtrack_coeff:
        Multiplicative factor applied to step fraction each backtracking iteration.
        Must be in (0, 1).
    accept_ratio:
        Minimum ratio of actual improvement to expected improvement required to accept
        a line-search step. Larger values are stricter.
    critic_optim_name:
        Name of critic optimizer (via `build_optimizer`).
    critic_lr:
        Critic learning rate.
    critic_weight_decay:
        Critic weight decay.
    critic_sched_name:
        Name of critic scheduler (via `build_scheduler`). Use "none" to disable.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones:
        Generic scheduler knobs passed through to `build_scheduler` for parity with other cores.
    max_grad_norm:
        Global norm clipping threshold for critic gradients. Set to 0 to disable clipping.
    use_amp:
        Enable mixed precision for critic update (actor update uses higher-order grads and
        is not run under autocast).

    Raises
    ------
    ValueError
        If hyperparameters are invalid (e.g., non-positive max_kl, invalid backtrack_coeff).
    RuntimeError
        If the policy distribution does not implement a differentiable KL method.
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
        # sched shared knobs
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

        # -------------------------
        # Hyperparameters
        # -------------------------
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
            raise ValueError(
                f"backtrack_coeff must be in (0,1), got {self.backtrack_coeff}"
            )
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

    # ============================================================
    # Schedulers
    # ============================================================
    def _step_scheds(self) -> None:
        """
        Step learning-rate schedulers if configured.

        Notes
        -----
        - TRPO actor update does not use an optimizer/scheduler; only the critic has one.
        - This is called by the outer algorithm driver (via BaseCore conventions).
        """
        if self.critic_sched is not None:
            self.critic_sched.step()

    # ============================================================
    # Flat parameter utilities
    # ============================================================
    @staticmethod
    def _flat_params(module: th.nn.Module) -> th.Tensor:
        """
        Flatten module parameters into a single 1-D tensor.

        Parameters
        ----------
        module:
            Torch module whose parameters will be concatenated in iteration order.

        Returns
        -------
        flat:
            1-D tensor containing all parameters concatenated.

        Notes
        -----
        - Uses `p.data` for speed because TRPO line search explicitly assigns parameter
          candidates. This bypasses autograd tracking intentionally.
        """
        return th.cat([p.data.view(-1) for p in module.parameters()])

    @staticmethod
    def _assign_flat_params(module: th.nn.Module, flat: th.Tensor) -> None:
        """
        Assign module parameters from a single 1-D tensor.

        Parameters
        ----------
        module:
            Module to be updated in-place.
        flat:
            Flat parameter vector. Total length must match the sum of module parameter sizes.

        Notes
        -----
        - Uses `p.data.copy_` to overwrite parameters without building a graph.
        - This is required for backtracking line search (try candidate steps efficiently).
        """
        i = 0
        for p in module.parameters():
            n = p.numel()
            p.data.copy_(flat[i : i + n].view_as(p))
            i += n

    @staticmethod
    def _flat_grad_like_params(
        params: List[th.nn.Parameter],
        grads: Sequence[Optional[th.Tensor]],
    ) -> th.Tensor:
        """
        Flatten gradients into a single vector with parameter-aligned shape.

        Parameters
        ----------
        params:
            Parameter list defining the target flattening layout.
        grads:
            Gradients returned by `torch.autograd.grad`. Some entries may be None.

        Returns
        -------
        flat_grad:
            1-D tensor of the same length as the flattened parameter vector.

        Notes
        -----
        - None gradients are replaced with zeros to preserve vector length.
          This is critical for TRPO/CG where vector dimensions must match exactly.
        """
        out: List[th.Tensor] = []
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
        Compute the mean KL divergence KL(new || old) over a batch.

        Parameters
        ----------
        obs:
            Observation batch, shape (B, obs_dim).
        old_dist:
            Distribution produced by the behavior policy (treated as constant reference).

        Returns
        -------
        mean_kl:
            Scalar tensor: mean KL over the batch.

        Raises
        ------
        RuntimeError
            If the distribution does not implement `kl(old_dist)`.

        Notes
        -----
        - KL must be differentiable w.r.t. *new* policy parameters (actor params).
        - This value defines the trust region constraint used in line search.
        """
        new_dist = self.head.actor.get_dist(obs)

        if hasattr(new_dist, "kl") and callable(getattr(new_dist, "kl")):
            return new_dist.kl(old_dist).mean()

        raise RuntimeError("TRPO requires distribution.kl(old_dist).")

    def _fvp(self, obs: th.Tensor, old_dist: Any, v: th.Tensor) -> th.Tensor:
        """
        Compute Fisher-vector product (F + damping I) v.

        Parameters
        ----------
        obs:
            Observation batch.
        old_dist:
            Frozen old policy distribution (reference for KL).
        v:
            Vector to multiply, shape (P,) where P is number of actor parameters.

        Returns
        -------
        fvp:
            Vector (P,) representing (F + damping I) v.

        Notes
        -----
        - Fisher is approximated via the Hessian of mean KL(new || old) w.r.t.
          actor parameters.
        - Requires second-order gradients; do NOT wrap this function in no_grad.
        - Damping improves conditioning and stabilizes CG.
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
        Solve A x = b using conjugate gradient (CG), given a matrix-vector product.

        Parameters
        ----------
        Avp:
            Callable computing A @ v (matrix-vector product).
        b:
            Right-hand side vector.
        iters:
            Maximum number of CG iterations.

        Returns
        -------
        x:
            Approximate solution to A x = b.

        Notes
        -----
        - A is assumed symmetric positive definite (SPD). The Fisher approximation
          is typically near-SPD in practice (with damping).
        - Stops early if the residual norm becomes sufficiently small.
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
        Perform one TRPO update using an on-policy batch.

        Parameters
        ----------
        batch:
            Batch container providing (at minimum):
            - observations: Tensor, shape (B, obs_dim)
            - actions: Tensor, shape (B, act_dim) or (act_dim,)
            - returns: Tensor, shape (B,) or (B,1)
            - advantages: Tensor, shape (B,) or (B,1)
            - log_probs: Tensor, shape (B,) or (B,1), behavior-policy log-prob of actions

        Returns
        -------
        metrics:
            Dict of scalars for logging (loss/value, KL, line-search step fraction, lr, etc.).

        Notes
        -----
        Update structure:
        1) Critic: minimize 0.5 * MSE(V(s), returns)
        2) Actor:
           - Define surrogate objective: E[ exp(logp_new - logp_old) * adv ]
           - Compute gradient g = ∇ surrogate
           - Solve (F + damping I) step_dir = g via CG
           - Scale step to satisfy the KL constraint
           - Backtracking line search to ensure:
               * improvement > 0
               * mean KL <= max_kl
               * actual/expected improvement ratio >= accept_ratio
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        ret = _to_column(batch.returns.to(self.device))
        adv = _to_column(batch.advantages.to(self.device))
        old_logp = _to_column(batch.log_probs.to(self.device))

        # ------------------------------------------------------------
        # Critic update (regression on returns)
        # ------------------------------------------------------------
        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                v = _to_column(self.head.critic(obs))
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
            v = _to_column(self.head.critic(obs))
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
        # Freeze the old distribution (behavior policy) as reference.
        with th.no_grad():
            old_dist = self.head.actor.get_dist(obs)

        def surrogate(*, with_grad: bool) -> th.Tensor:
            """
            Compute the TRPO surrogate objective.

            Parameters
            ----------
            with_grad:
                If True, build a computation graph w.r.t. actor parameters.

            Returns
            -------
            surr:
                Scalar tensor equal to mean(ratio * advantages) where
                ratio = exp(logp_new - logp_old).

            Notes
            -----
            - Uses importance sampling to compare new policy to behavior policy.
            - We maximize this surrogate; the CG system uses g = ∇ surrogate.
            """
            if with_grad:
                dist = self.head.actor.get_dist(obs)
                logp = _to_column(dist.log_prob(act))
                ratio = th.exp(logp - old_logp)
                return (ratio * adv).mean()

            with th.no_grad():
                dist = self.head.actor.get_dist(obs)
                logp = _to_column(dist.log_prob(act))
                ratio = th.exp(logp - old_logp)
                return (ratio * adv).mean()

        # Compute policy gradient of surrogate objective.
        surr = surrogate(with_grad=True)
        params = list(self.head.actor.parameters())

        grads_surr = th.autograd.grad(surr, params, retain_graph=False)
        g = self._flat_grad_like_params(params, grads_surr).detach()

        # Solve (F + damping I) x = g via conjugate gradient.
        step_dir = self._conjugate_gradient(
            lambda v_: self._fvp(obs, old_dist, v_),
            g,
            self.cg_iters,
        )

        # Scale step so that 0.5 * step^T F step <= max_kl.
        shs = 0.5 * (step_dir * self._fvp(obs, old_dist, step_dir)).sum()
        step_scale = th.sqrt(self.max_kl / (shs + 1e-12))
        full_step = step_scale * step_dir

        old_params = self._flat_params(self.head.actor).clone()

        # Expected improvement under first-order approximation: g^T step
        expected_improve = float((g * full_step).sum().item())

        surr_old = float(surr.detach().cpu().item())
        accepted = False
        step_frac = 1.0
        best_kl = 0.0

        # Backtracking line search:
        # Accept if we improve, satisfy KL, and the improvement is not too small
        # relative to linearized expectation.
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
            # Revert parameters if no acceptable step was found.
            self._assign_flat_params(self.head.actor, old_params)
            step_frac = 0.0
            best_kl = 0.0

        return {
            "loss/value": float(_to_scalar(value_loss)),
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

        Returns
        -------
        state:
            Serializable state dictionary.

        Notes
        -----
        - `super().state_dict()` is assumed to include BaseCore generic fields.
        - Critic optimizer/scheduler state is stored under "critic".
        - TRPO hyperparameters are stored at the root for backward compatibility.
        """
        s = super().state_dict()
        s.update(
            {
                "critic": self._save_opt_sched(self.critic_opt, self.critic_sched),
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
        Restore core state from a serialized dictionary.

        Parameters
        ----------
        state:
            State dictionary produced by `state_dict()`.

        Notes
        -----
        - Restores critic optimizer/scheduler if present.
        - Restores TRPO hyperparameters from root keys (matching `state_dict()`).
        - Ignores missing keys to allow forward/backward compatibility across versions.
        """
        super().load_state_dict(state)

        if "critic" in state:
            self._load_opt_sched(self.critic_opt, self.critic_sched, state["critic"])

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
