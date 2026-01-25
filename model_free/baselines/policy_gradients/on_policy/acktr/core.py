from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore


class ACKTRCore(ActorCriticCore):
    """
    ACKTR update engine (continuous actions only).

    Overview
    --------
    ACKTR (Actor Critic using Kronecker-Factored Trust Region) typically uses:
      - the same A2C-style objective (policy loss + value loss + entropy bonus)
      - a K-FAC optimizer to approximate natural-gradient updates

    This core assumes a *Gaussian* actor (continuous actions) and a state-value critic.

    Assumptions / contracts
    -----------------------
    - Continuous actions only (Gaussian policy).
    - batch.actions is float and shaped like:
        * (B, action_dim)
        * (action_dim,)  (single sample)
      The actor distribution must accept the provided shape (or be broadcastable).

    Notes
    -----
    - AMP is generally discouraged with K-FAC (due to second-order statistics),
      but is kept as an option for parity with other cores.
    - K-FAC implementations often require registering hooks on the model to collect
      curvature statistics; therefore build_optimizer(...) must support model=...
      and optimizer objects may expose additional flags (e.g., fisher_backprop).
    """

    def __init__(
        self,
        *,
        head: Any,
        # loss coefficients
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        # gradient / AMP
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # optimizer/scheduler
        actor_optim_name: str = "kfac",
        actor_lr: float = 0.25,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "kfac",
        critic_lr: float = 0.25,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # shared scheduler knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # ---------------------------------------------------------------------
        # K-FAC-specific knobs (passed through build_optimizer via *_optim_kwargs)
        # ---------------------------------------------------------------------
        actor_damping: float = 1e-2,
        actor_momentum: float = 0.9,
        actor_eps: float = 0.95,
        actor_Ts: int = 1,
        actor_Tf: int = 10,
        actor_max_lr: float = 1.0,
        actor_trust_region: float = 2e-3,
        critic_damping: float = 1e-2,
        critic_momentum: float = 0.9,
        critic_eps: float = 0.95,
        critic_Ts: int = 1,
        critic_Tf: int = 10,
        critic_max_lr: float = 1.0,
        critic_trust_region: float = 2e-3,
    ) -> None:
        # Pass KFAC extras through ActorCriticCore so it can build optimizers once
        # (and auto-inject model=... when name=="kfac").
        super().__init__(
            head=head,
            use_amp=use_amp,
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            actor_optim_kwargs={
                "damping": float(actor_damping),
                "momentum": float(actor_momentum),
                "kfac_eps": float(actor_eps),
                "Ts": int(actor_Ts),
                "Tf": int(actor_Tf),
                "max_lr": float(actor_max_lr),
                "trust_region": float(actor_trust_region),
                # "model": self.head.actor  # optional; base auto-injects if missing
            },
            critic_optim_kwargs={
                "damping": float(critic_damping),
                "momentum": float(critic_momentum),
                "kfac_eps": float(critic_eps),
                "Ts": int(critic_Ts),
                "Tf": int(critic_Tf),
                "max_lr": float(critic_max_lr),
                "trust_region": float(critic_trust_region),
                # "model": self.head.critic # optional; base auto-injects if missing
            },
            actor_sched_name=str(actor_sched_name),
            critic_sched_name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            sched_gamma=float(sched_gamma),
            milestones=milestones,
        )

        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

    # =============================================================================
    # Internal helpers: K-FAC fisher_backprop toggle (best-effort)
    # =============================================================================
    @staticmethod
    def _maybe_set_fisher_backprop(opt: Any, enabled: bool) -> None:
        """
        Enable/disable Fisher-statistics backprop if the optimizer supports it.

        Some K-FAC implementations expose a boolean flag (e.g., fisher_backprop)
        that controls whether to accumulate curvature statistics during backward.
        This helper is best-effort: if the attribute does not exist, it does nothing.
        """
        if hasattr(opt, "fisher_backprop"):
            setattr(opt, "fisher_backprop", bool(enabled))

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """
        Return a readable learning rate for logging.

        K-FAC wrappers sometimes wrap a torch optimizer as `opt.optim`.
        This handles both:
          - opt.optim.param_groups[0]["lr"]
          - opt.param_groups[0]["lr"]
        """
        if hasattr(opt, "optim") and hasattr(opt.optim, "param_groups"):
            return float(opt.optim.param_groups[0]["lr"])
        if hasattr(opt, "param_groups"):
            return float(opt.param_groups[0]["lr"])
        return float("nan")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one ACKTR update step.

        Losses (A2C-style)
        ------------------
        - policy_loss = -E[ log π(a|s) * advantage ]
        - value_loss  = 0.5 * MSE(V(s), returns)
        - ent_loss    = -E[ entropy(π(.|s)) ]
        - total_loss  = policy_loss + vf_coef * value_loss + ent_coef * ent_loss
        """
        self._bump()

        # Move batch tensors to device
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        ret = to_column(batch.returns.to(self.device))
        adv = to_column(batch.advantages.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            # Actor distribution π(.|s)
            dist = self.head.actor.get_dist(obs)

            # log_prob/entropy might be (B,) or (B, action_dim) depending on dist;
            # your to_column expects a vector, but if (B, action_dim) appears,
            # you may want to sum over last dim before to_column.
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            # If distribution returns per-dimension values, reduce to joint scalar per sample.
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            logp = to_column(logp)       # (B,1)
            entropy = to_column(entropy) # (B,1)

            # Critic value V(s)
            v = to_column(self.head.critic(obs))

            policy_loss = -(logp * adv.detach()).mean()
            value_loss = 0.5 * F.mse_loss(v, ret)
            ent_loss = -entropy.mean()

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v.mean()

        # Clear gradients
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        # Enable Fisher-statistics collection during backward (if supported)
        self._maybe_set_fisher_backprop(self.actor_opt, True)
        self._maybe_set_fisher_backprop(self.critic_opt, True)

        if self.use_amp:
            # AMP path (generally not recommended for K-FAC, but supported)
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Try unscale for safe clipping (may fail for non-torch optim wrappers)
            try:
                self.scaler.unscale_(self.actor_opt)
            except Exception:
                pass
            try:
                self.scaler.unscale_(self.critic_opt)
            except Exception:
                pass

            # Global grad clipping across actor+critic
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            # Step both optimizers
            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            # Standard fp32 path
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # Disable Fisher-statistics collection after the step (if supported)
        self._maybe_set_fisher_backprop(self.actor_opt, False)
        self._maybe_set_fisher_backprop(self.critic_opt, False)

        # Step LR schedulers (if present)
        self._step_scheds()

        # Return metrics for logging
        return {
            "loss/policy": float(to_scalar(policy_loss)),
            "loss/value": float(to_scalar(value_loss)),
            "loss/entropy": float(to_scalar(ent_loss)),
            "loss/total": float(to_scalar(total_loss)),
            "stats/entropy": float(to_scalar(ent_mean)),
            "stats/value_mean": float(to_scalar(v_mean)),
            "lr/actor": float(self._get_optimizer_lr(self.actor_opt)),
            "lr/critic": float(self._get_optimizer_lr(self.critic_opt)),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend the base ActorCriticCore state_dict with ACKTR-specific scalars.
        """
        s = super().state_dict()
        s.update(
            {
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base state (optimizers/schedulers/scaler) and ACKTR scalars.

        NOTE:
        Your code stores ACKTR scalars under state["acktr"] (nested mapping).
        This matches your existing convention, but ensure state_dict() uses the
        same nesting if you want round-trip symmetry.
        """
        super().load_state_dict(state)

        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])