from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column, reduce_joint
from model_free.common.policies.base_core import ActorCriticCore


class PPOCore(ActorCriticCore):
    """
    PPO update engine (single-minibatch update per call).

    Design
    ------
    This core reuses ActorCriticCore to manage:
      - actor/critic optimizers + schedulers
      - AMP scaler (optional) and update counter
      - gradient clipping helper (_clip_params)

    Batch contract (expected fields)
    --------------------------------
    The minibatch must provide:
      - observations : (B, obs_dim)
      - actions      : (B, action_dim) for continuous actions
      - log_probs    : old log π_old(a|s), shape (B,) or (B,1)
      - values       : old V_old(s), shape (B,) or (B,1)
      - returns      : target returns, shape (B,) or (B,1)
      - advantages   : advantages, shape (B,) or (B,1)

    Notes on distributions
    ----------------------
    Depending on how your policy distribution is implemented:
      - dist.log_prob(action) may return (B,)  (already joint log-prob)
      - or (B, action_dim) (per-dimension log-prob)

    PPO needs *joint* log_prob per sample, so if (B, action_dim) appears,
    we reduce via sum(dim=-1) before converting to (B,1) with to_column().

    Early stopping
    --------------
    - If target_kl is set, we compute an approximate KL per minibatch
      and return an "early_stop" flag (1.0/0.0).
    - The outer trainer/algorithm can stop further epochs when early_stop=1.0.
    """

    def __init__(
        self,
        *,
        head: Any,
        # PPO hyperparameters
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        clip_vloss: bool = True,
        # target_kl (optional)
        target_kl: Optional[float] = None,
        kl_stop_multiplier: float = 1.0,
        # grad / amp
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        # actor/critic opt/sched (inherited)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # sched shared knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        # Let ActorCriticCore wire:
        # - device + amp scaler
        # - actor/critic optimizers
        # - actor/critic schedulers
        super().__init__(
            head=head,
            use_amp=use_amp,
            actor_optim_name=actor_optim_name,
            actor_lr=actor_lr,
            actor_weight_decay=actor_weight_decay,
            critic_optim_name=critic_optim_name,
            critic_lr=critic_lr,
            critic_weight_decay=critic_weight_decay,
            actor_sched_name=actor_sched_name,
            critic_sched_name=critic_sched_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            poly_power=poly_power,
            step_size=step_size,
            sched_gamma=sched_gamma,
            milestones=milestones,
        )

        # Store PPO hyperparameters
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.clip_vloss = bool(clip_vloss)

        # KL early-stop configuration
        self.target_kl = None if target_kl is None else float(target_kl)
        self.kl_stop_multiplier = float(kl_stop_multiplier)

        # Gradient clipping
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError(f"target_kl must be > 0 when provided, got {self.target_kl}")
        if self.kl_stop_multiplier <= 0.0:
            raise ValueError(f"kl_stop_multiplier must be > 0, got {self.kl_stop_multiplier}")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform ONE PPO minibatch update.

        Returns a metrics dict including:
          - losses: policy/value/entropy/total
          - stats: approx_kl, clip_frac, entropy, value_mean
          - train: early_stop flag if target_kl is configured
        """
        self._bump()

        # Move data to device
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        # Old (behavior) logp/value stored at rollout time
        old_logp = to_column(batch.log_probs.to(self.device))
        old_v = to_column(batch.values.to(self.device))

        # Targets
        ret = to_column(batch.returns.to(self.device))
        adv = to_column(batch.advantages.to(self.device))

        clip_eps = self.clip_range

        def _forward_losses() -> Tuple[
            th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor
        ]:
            # Current policy distribution π(.|s)
            dist = self.head.actor.get_dist(obs)

            # New log-prob under current policy
            new_logp = dist.log_prob(act)
            entropy = dist.entropy()

            # Ensure (B,) then standardize to (B,1)
            new_logp = to_column(reduce_joint(new_logp))
            entropy = to_column(reduce_joint(entropy))

            # Current critic value
            v = to_column(self.head.critic(obs))

            # Policy ratio: exp(log π - log π_old)
            log_ratio = new_logp - old_logp
            ratio = th.exp(log_ratio)

            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = th.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -th.min(surr1, surr2).mean()

            # Diagnostics
            # (This "approx_kl" form is a common cheap approximation used in PPO impls.)
            approx_kl = (ratio - 1.0 - log_ratio).mean()
            clip_frac = (th.abs(ratio - 1.0) > clip_eps).float().mean()

            # Value loss (optionally clipped)
            if self.clip_vloss:
                v_clipped = old_v + th.clamp(v - old_v, -clip_eps, clip_eps)
                v_loss_unclipped = (v - ret).pow(2)
                v_loss_clipped = (v_clipped - ret).pow(2)
                value_loss = 0.5 * th.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(v, ret)

            # Entropy bonus (as loss term: negative sign)
            ent_loss = -entropy.mean()

            # Total loss
            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss

            return (
                total_loss,
                policy_loss,
                value_loss,
                ent_loss,
                approx_kl,
                clip_frac,
                entropy.mean(),
                v.mean(),
            )

        # ------------------------------------------------------------------
        # Backward + optimizer step
        # ------------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            # AMP path (mixed precision)
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_frac, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # One global clip across actor+critic (common PPO practice)
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            # Standard fp32 path
            total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_frac, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # Step schedulers if present
        self._step_scheds()

        # ------------------------------------------------------------------
        # target_kl early-stop signal (per minibatch)
        # ------------------------------------------------------------------
        early_stop = 0.0
        if self.target_kl is not None:
            if float(to_scalar(approx_kl)) > self.kl_stop_multiplier * self.target_kl:
                early_stop = 1.0

        # ------------------------------------------------------------------
        # Return metrics for logging
        # ------------------------------------------------------------------
        return {
            "loss/policy": float(to_scalar(policy_loss)),
            "loss/value": float(to_scalar(value_loss)),
            "loss/entropy": float(to_scalar(ent_loss)),
            "loss/total": float(to_scalar(total_loss)),
            "stats/approx_kl": float(to_scalar(approx_kl)),
            "stats/clip_frac": float(to_scalar(clip_frac)),
            "stats/entropy": float(to_scalar(ent_mean)),
            "stats/value_mean": float(to_scalar(v_mean)),
            "train/early_stop": float(early_stop),
            "train/target_kl": float(self.target_kl) if self.target_kl is not None else 0.0,
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend ActorCriticCore.state_dict() with PPO-specific hyperparameters.

        ActorCriticCore already includes:
          - update_calls
          - actor optimizer/scheduler state
          - critic optimizer/scheduler state
          - AMP scaler state (if enabled)

        We add a nested "ppo" dict for PPO knobs.
        """
        s = super().state_dict()
        s.update(
            {
                "clip_range": float(self.clip_range),
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "clip_vloss": bool(self.clip_vloss),
                "target_kl": None if self.target_kl is None else float(self.target_kl),
                "kl_stop_multiplier": float(self.kl_stop_multiplier),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base state and (optionally) PPO hyperparameters.

        Notes
        -----
        - Base class restores optimizer/scheduler/scaler states.
        - You can choose whether PPO hyperparameters should be overwritten
          from checkpoints (some codebases treat them as "static config").
        """
        super().load_state_dict(state)

        if "clip_range" in state:
            self.clip_range = float(state["clip_range"])
        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "clip_vloss" in state:
            self.clip_vloss = bool(state["clip_vloss"])
        if "target_kl" in state:
            self.target_kl = None if state["target_kl"] is None else float(state["target_kl"])
        if "kl_stop_multiplier" in state:
            self.kl_stop_multiplier = float(state["kl_stop_multiplier"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
