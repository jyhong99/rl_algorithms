from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import BaseCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class VPGCore(BaseCore):
    """
    Vanilla Policy Gradient (VPG) update engine for continuous control.

    This core performs a single on-policy gradient update per call using:
      - Policy loss:   -E[ log π(a|s) * A(s,a) ]
      - Entropy bonus:  -E[ H(π(.|s)) ]  (optional; via ent_coef)
      - Value loss:     0.5 * MSE(V(s), R) (optional baseline; via vf_coef)

    Baseline policy (critic usage)
    ------------------------------
    Baseline usage is dictated by the head:
      - If head has attribute `use_baseline`, follow it strictly.
      - Else, infer baseline usage as (head.critic is not None).

    Notes
    -----
    - If baseline is disabled: critic optimizer/scheduler are not created and
      value_loss is a constant zero scalar.
    - Advantage normalization is intentionally NOT handled here (should be done
      by buffer/algorithm).
    """

    def __init__(
        self,
        *,
        head: Any,
        # coefficients
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        # optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # schedulers
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # (optional scheduler knobs)
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
        """
        Parameters
        ----------
        head : Any
            Policy head providing:
              - head.actor.get_dist(obs)  -> distribution with log_prob / entropy
              - head.critic(obs)          -> V(s) (only if baseline enabled)
              - head.device               -> device placement
        vf_coef : float
            Coefficient for value loss when baseline is enabled.
        ent_coef : float
            Coefficient for entropy bonus term (encourages exploration).
        actor_optim_name / actor_lr / actor_weight_decay
            Optimizer config for actor parameters.
        critic_optim_name / critic_lr / critic_weight_decay
            Optimizer config for critic parameters (only if baseline enabled).
        actor_sched_name / critic_sched_name
            Learning-rate scheduler names (optional).
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Common scheduler knobs used by build_scheduler().
        max_grad_norm : float
            Gradient clipping norm (global norm).
        use_amp : bool
            Enable mixed precision (AMP) training if supported by BaseCore.
        """
        super().__init__(head=head, use_amp=use_amp)

        # Coefficients for composing the total loss
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        # Gradient clipping knob
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # ------------------------------------------------------------------
        # Baseline usage is dictated by the head
        # ------------------------------------------------------------------
        # We support both styles:
        #  - explicit: head.use_baseline (preferred)
        #  - implicit: critic exists -> baseline enabled
        head_has_critic = getattr(self.head, "critic", None) is not None
        if hasattr(self.head, "use_baseline"):
            self.use_baseline = bool(getattr(self.head, "use_baseline"))
        else:
            self.use_baseline = bool(head_has_critic)

        # Track whether a critic module is actually present (internal consistency)
        self._has_critic = bool(head_has_critic)
        if self.use_baseline and not self._has_critic:
            raise ValueError("Head indicates baseline enabled (use_baseline=True) but head.critic is None.")

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------
        # Actor optimizer always exists.
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
        # Schedulers
        # ------------------------------------------------------------------
        # Scheduler construction is best-effort; build_scheduler may return None.
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
        """Step actor/critic schedulers if they exist."""
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
        -------------------------------
        batch.observations : (B, obs_dim)
        batch.actions      : (B, act_dim) or (act_dim,) when B=1
        batch.returns      : (B,) or (B,1)
        batch.advantages   : (B,) or (B,1) (optional; if missing, uses returns)

        Notes
        -----
        - This core does NOT compute advantages (except a REINFORCE fallback).
        - If advantages are present, they are treated as fixed targets for the policy
          gradient (detached).
        """
        self._bump()  # increment internal update counters (BaseCore)

        # Move batch tensors to device
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        ret = to_column(batch.returns.to(self.device))

        # Advantage:
        # - If batch provides advantages, use them.
        # - Else fall back to returns (REINFORCE-style). This is not variance-optimal,
        #   but allows the core to run without a separate advantage computation stage.
        adv = getattr(batch, "advantages", None)
        if adv is None:
            adv = ret
        else:
            adv = to_column(adv.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute the total loss and its components.

            Returns
            -------
            total_loss : torch.Tensor (scalar)
            policy_loss: torch.Tensor (scalar)
            value_loss : torch.Tensor (scalar, 0 if baseline disabled)
            ent_loss   : torch.Tensor (scalar)
            ent_mean   : torch.Tensor (scalar) mean entropy
            v_mean     : torch.Tensor (scalar) mean value prediction (0 if baseline disabled)
            """
            # Distribution parameterized by current policy
            dist = self.head.actor.get_dist(obs)

            # Log-prob and entropy may have shape (B,) or (B, act_dim) depending on dist implementation.
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            # Normalize to (B,1) so downstream reductions are consistent.
            logp = to_column(logp)
            entropy = to_column(entropy)

            # Policy gradient loss: maximize E[logp * adv] -> minimize -(logp * adv)
            # adv is detached to avoid backprop through advantage estimation.
            policy_loss = -(logp * adv.detach()).mean()

            # Entropy loss: typical convention is to add +ent_coef * entropy to objective.
            # Here we write ent_loss = -entropy.mean(), so total adds ent_coef * ent_loss.
            ent_loss = -entropy.mean()

            # Value loss: only when baseline enabled
            value_loss = th.zeros((), device=self.device)
            v_mean = th.zeros((), device=self.device)
            if self.use_baseline:
                v = to_column(self.head.critic(obs))  # type: ignore[attr-defined]
                value_loss = 0.5 * F.mse_loss(v, ret)
                v_mean = v.mean()

            # Compose total loss
            total_loss = policy_loss + self.ent_coef * ent_loss
            if self.use_baseline:
                total_loss = total_loss + self.vf_coef * value_loss

            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v_mean

        # ------------------------------------------------------------------
        # Zero gradients
        # ------------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        if self.critic_opt is not None:
            self.critic_opt.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------
        # Backward + optimizer steps (optionally AMP)
        # ------------------------------------------------------------------
        if self.use_amp:
            # AMP path: loss scaling is handled by BaseCore's scaler
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Clip gradients across all updated parameters
            params = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            # Optimizer steps under AMP
            self.scaler.step(self.actor_opt)
            if self.critic_opt is not None:
                self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            # FP32 path
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.actor_opt.step()
            if self.critic_opt is not None:
                self.critic_opt.step()

        # ------------------------------------------------------------------
        # Scheduler steps (if enabled)
        # ------------------------------------------------------------------
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        out: Dict[str, float] = {
            "loss/policy": float(to_scalar(policy_loss)),
            "loss/entropy": float(to_scalar(ent_loss)),
            "loss/total": float(to_scalar(total_loss)),
            "stats/entropy": float(to_scalar(ent_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
        }
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

        Stored fields
        -------------
        - actor optimizer/scheduler state
        - critic optimizer/scheduler state (or None if no baseline)
        - hyperparameters needed to resume training consistently
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

        Compatibility rules
        -------------------
        - If current head baseline is enabled, checkpoint must contain critic optimizer state.
        - If current head baseline is disabled, checkpoint must NOT contain critic optimizer state.
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
