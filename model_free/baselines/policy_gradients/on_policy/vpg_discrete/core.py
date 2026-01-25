from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import BaseCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class VPGDiscreteCore(BaseCore):
    """
    VPG Update Engine (DISCRETE)

    This core implements Vanilla Policy Gradient updates for discrete action spaces.

    Overview
    --------
    - Actor update:
        Maximizes E[ log π(a|s) * A(s,a) ] with optional entropy regularization.
    - Critic update (optional baseline):
        Fits V(s) to Monte-Carlo returns using MSE regression.

    Baseline policy (important)
    ---------------------------
    This core does NOT own a separate "use_baseline" configuration.
    Instead, it follows the head configuration:

    - If `head.use_baseline` exists, that value is used.
    - Otherwise, it infers baseline usage by checking whether `head.critic` exists.

    As a result:
    - baseline OFF => actor-only REINFORCE-style updates
    - baseline ON  => actor + critic updates
    """

    def __init__(
        self,
        *,
        head: Any,
        # -----------------------------
        # Loss coefficients
        # -----------------------------
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        # -----------------------------
        # Optimizers
        # -----------------------------
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # -----------------------------
        # Schedulers (optional)
        # -----------------------------
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # -----------------------------
        # Misc
        # -----------------------------
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Head object that must provide:
              - head.actor.get_dist(obs) returning a Categorical-like distribution
              - (optional) head.critic(obs) returning V(s) when baseline is enabled
        vf_coef : float
            Weight for the value loss when baseline is enabled.
        ent_coef : float
            Weight for entropy bonus (entropy regularization).
        actor_optim_name / actor_lr / actor_weight_decay
            Actor optimizer configuration.
        critic_optim_name / critic_lr / critic_weight_decay
            Critic optimizer configuration (only used if baseline is enabled).
        actor_sched_name / critic_sched_name, plus scheduler knobs
            Optional learning-rate scheduler configuration.
        max_grad_norm : float
            Global gradient clipping threshold.
        use_amp : bool
            Enable torch AMP (mixed precision).
        """
        super().__init__(head=head, use_amp=use_amp)

        # Store scalar coefficients
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        # Gradient clipping threshold
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # ------------------------------------------------------------------
        # Baseline configuration is dictated by head
        # ------------------------------------------------------------------
        # head_has_critic: actual structural availability of critic module
        head_has_critic = getattr(self.head, "critic", None) is not None

        # use_baseline: semantic flag (preferred if exposed by head)
        if hasattr(self.head, "use_baseline"):
            self.use_baseline = bool(getattr(self.head, "use_baseline"))
        else:
            self.use_baseline = bool(head_has_critic)

        self._has_critic = bool(head_has_critic)

        # Sanity check: if head says baseline ON, critic must exist.
        if self.use_baseline and not self._has_critic:
            raise ValueError("Head indicates baseline enabled (use_baseline=True) but head.critic is None.")

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------
        # Actor optimizer is always constructed.
        self.actor_opt = build_optimizer(
            self.head.actor.parameters(),
            name=str(actor_optim_name),
            lr=float(actor_lr),
            weight_decay=float(actor_weight_decay),
        )

        # Critic optimizer only exists when baseline is enabled.
        self.critic_opt = None
        if self.use_baseline:
            self.critic_opt = build_optimizer(
                self.head.critic.parameters(),  # type: ignore[attr-defined]
                name=str(critic_optim_name),
                lr=float(critic_lr),
                weight_decay=float(critic_weight_decay),
            )

        # ------------------------------------------------------------------
        # Schedulers (optional)
        # ------------------------------------------------------------------
        self.actor_sched = build_scheduler(
            self.actor_opt,
            name=str(actor_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        self.critic_sched = None
        if self.critic_opt is not None:
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
        """
        Step schedulers (best-effort).

        This helper is optional; you currently step schedulers inline in update().
        Keeping it here makes the API consistent with other cores.
        """
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one VPG update from an on-policy batch.

        Expected batch fields (typical)
        --------------------------------
        batch.observations : (B, obs_dim)
        batch.actions      : (B,) or (B,1) or scalar for B=1
        batch.returns      : (B,) or (B,1)
        batch.advantages   : optional (B,) or (B,1)
            - If absent, returns are used as REINFORCE advantage fallback.

        Returns
        -------
        metrics : Dict[str, float]
            Common training metrics for logging/monitoring.
        """
        self._bump()

        # ------------------------------------------------------------
        # Move batch tensors to device and normalize discrete actions
        # ------------------------------------------------------------
        obs = batch.observations.to(self.device)
        act_raw = batch.actions.to(self.device)
        act = self.head._normalize_discrete_action(act_raw)

        ret = to_column(batch.returns.to(self.device))
        adv = getattr(batch, "advantages", None)
        if adv is None:
            adv = ret
        else:
            adv = to_column(adv.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute losses for actor and critic.

            Returns
            -------
            total_loss : scalar tensor
            policy_loss: scalar tensor
            value_loss : scalar tensor (0 if baseline disabled)
            ent_loss   : scalar tensor
            ent_mean   : scalar tensor (mean entropy)
            v_mean     : scalar tensor (mean value prediction, 0 if baseline disabled)
            """
            dist = self.head.actor.get_dist(obs)

            # log_prob / entropy: categorical -> typically returns shape (B,)
            logp = dist.log_prob(act)
            ent = dist.entropy()

            # Defensive reduction in case implementation returns (B, A) or similar
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if ent.dim() > 1:
                ent = ent.sum(dim=-1)

            # Force (B,1)
            logp = to_column(logp)
            ent = to_column(ent)

            # Policy gradient objective:
            # maximize E[ log π(a|s) * adv ]
            # => minimize negative of it
            policy_loss = -(logp * adv.detach()).mean()

            # Entropy bonus:
            # maximize entropy => minimize (-entropy)
            ent_loss = -ent.mean()

            # Value regression loss (only if baseline enabled)
            value_loss = th.zeros((), device=self.device)
            v_mean = th.zeros((), device=self.device)

            if self.use_baseline:
                v = to_column(self.head.critic(obs))  # type: ignore[attr-defined]
                value_loss = 0.5 * F.mse_loss(v, ret)
                v_mean = v.mean()

            # Total loss combines actor (policy + entropy) and optional critic term
            total_loss = policy_loss + self.ent_coef * ent_loss
            if self.use_baseline:
                total_loss = total_loss + self.vf_coef * value_loss

            return total_loss, policy_loss, value_loss, ent_loss, ent.mean(), v_mean

        # ------------------------------------------------------------
        # Zero gradients
        # ------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        if self.critic_opt is not None:
            self.critic_opt.zero_grad(set_to_none=True)

        # ------------------------------------------------------------
        # Backprop + optimizer steps
        # ------------------------------------------------------------
        if self.use_amp:
            # Mixed precision path
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Clip actor (+ critic) gradients
            params = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            # Step optimizers under AMP
            self.scaler.step(self.actor_opt)
            if self.critic_opt is not None:
                self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            # Standard FP32 path
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.actor_opt.step()
            if self.critic_opt is not None:
                self.critic_opt.step()

        # ------------------------------------------------------------
        # Step schedulers (if configured)
        # ------------------------------------------------------------
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        out: Dict[str, float] = {
            "loss/policy": float(to_scalar(policy_loss)),
            "loss/entropy": float(to_scalar(ent_loss)),
            "loss/total": float(to_scalar(total_loss)),
            "stats/entropy": float(to_scalar(ent_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
        }

        # Baseline metrics only when critic exists
        if self.use_baseline:
            out["loss/value"] = float(to_scalar(value_loss))
            out["stats/value_mean"] = float(to_scalar(v_mean))
            out["lr/critic"] = float(self.critic_opt.param_groups[0]["lr"]) if self.critic_opt is not None else 0.0

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        Includes
        --------
        - actor optimizer/scheduler state
        - critic optimizer/scheduler state (or None if baseline disabled)
        - scalar hyperparameters (vf_coef, ent_coef, etc.)
        """
        s = super().state_dict()
        s.update(
            {
                "actor": self._save_opt_sched(self.actor_opt, self.actor_sched),
                "critic": None if self.critic_opt is None else self._save_opt_sched(self.critic_opt, self.critic_sched),
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "max_grad_norm": float(self.max_grad_norm),
                "has_baseline": bool(self.use_baseline),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state.

        Notes
        -----
        This enforces baseline compatibility:
        - baseline enabled core must load a checkpoint containing critic optimizer state
        - baseline disabled core must not load critic optimizer state
        """
        super().load_state_dict(state)

        if "actor" in state:
            self._load_opt_sched(self.actor_opt, self.actor_sched, state["actor"])

        ckpt_critic = state.get("critic", None)

        # Checkpoint <-> current baseline compatibility checks
        if self.use_baseline:
            if ckpt_critic is None:
                raise ValueError("Checkpoint has no critic optimizer state but head baseline is enabled.")
            if self.critic_opt is None:
                raise ValueError("Head baseline enabled but critic optimizer is None (internal inconsistency).")
            self._load_opt_sched(self.critic_opt, self.critic_sched, ckpt_critic)
        else:
            if ckpt_critic is not None:
                raise ValueError("Checkpoint contains critic optimizer state but head baseline is disabled.")
