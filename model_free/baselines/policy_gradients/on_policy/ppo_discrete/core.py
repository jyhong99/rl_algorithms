from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore


class PPODiscreteCore(ActorCriticCore):
    """
    PPO Discrete Update Engine (PPODiscreteCore)
    --------------------------------------------
    Implements PPO's clipped surrogate objective for DISCRETE action spaces.

    Design
    ------
    - This core performs **exactly one minibatch update** per `update_from_batch()` call.
    - Higher-level training logic (e.g., looping over epochs/minibatches) is handled
      by `OnPolicyAlgorithm`, not here.
    - If `target_kl` is enabled, this core computes an "early stop" flag per minibatch.

    Discrete-only assumptions
    -------------------------
    - `head.actor.get_dist(obs)` returns a **Categorical-like** distribution.
    - `dist.log_prob(action)` expects a tensor of dtype long and shape (B,).
    - `dist.entropy()` typically returns (B,).

    Expected batch fields
    ---------------------
    batch should expose (at minimum):
      - observations : (B, obs_dim)
      - actions      : scalar / (B,) / (B,1) discrete action indices
      - log_probs    : (B,) or (B,1) old action log-prob from behavior policy
      - values       : (B,) or (B,1) old value estimates (baseline at rollout time)
      - returns      : (B,) or (B,1) computed returns
      - advantages   : (B,) or (B,1) computed advantages

    Notes
    -----
    - PPO requires `old_logp` and `old_v` saved from rollout time to compute:
        * ratio = exp(new_logp - old_logp)
        * optionally value clipping using old_v
    - This implementation standardizes vectors to (B,1) via `to_column()`.
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
        """
        Initialize PPODiscreteCore.

        Parameters
        ----------
        head : Any
            Actor-critic head that must provide:
              - head.actor: discrete policy network
              - head.critic: state-value network
        clip_range : float
            PPO clip epsilon (commonly 0.1~0.3).
        vf_coef : float
            Weight for value-function loss term in total loss.
        ent_coef : float
            Weight for entropy bonus term in total loss.
        clip_vloss : bool
            If True, use PPO-style value clipping with old values.
        target_kl : Optional[float]
            If set, compute approx KL divergence and mark minibatch early-stop.
        kl_stop_multiplier : float
            Stop threshold multiplier: stop if approx_kl > multiplier * target_kl.
        max_grad_norm : float
            Gradient clipping max norm. Must be >= 0.
        use_amp : bool
            Enable automatic mixed precision (cuda amp).
        actor_optim_name / critic_optim_name : str
            Optimizer names passed into your build_optimizer().
        actor_lr / critic_lr : float
            Learning rates.
        actor_weight_decay / critic_weight_decay : float
            Weight decay.
        actor_sched_name / critic_sched_name : str
            Scheduler names.
        total_steps / warmup_steps / min_lr_ratio / poly_power / step_size / sched_gamma / milestones
            Scheduler parameters forwarded to build_scheduler().
        """
        # ActorCriticCore will:
        # - set device
        # - create actor_opt / critic_opt
        # - create actor_sched / critic_sched
        # - create AMP scaler if enabled
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

        # PPO hyperparameters
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.clip_vloss = bool(clip_vloss)

        # Target KL control (optional)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.kl_stop_multiplier = float(kl_stop_multiplier)

        # Gradient clipping
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # Safety checks for KL early-stopping
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError(f"target_kl must be > 0 when provided, got: {self.target_kl}")
        if self.kl_stop_multiplier <= 0.0:
            raise ValueError(f"kl_stop_multiplier must be > 0, got {self.kl_stop_multiplier}")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one PPO minibatch update (DISCRETE).

        Core PPO computation
        --------------------
        ratio = exp(new_logp - old_logp)
        policy_loss = -E[min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)]

        Value loss
        ----------
        - If clip_vloss=True:
            V_clipped = old_v + clip(V - old_v, -eps, eps)
            value_loss = 0.5 * mean(max((V-ret)^2, (V_clipped-ret)^2))
        - Else:
            value_loss = 0.5 * MSE(V, ret)

        Entropy loss
        ------------
        ent_loss = -mean(entropy)
        total_loss = policy_loss + vf_coef * value_loss + ent_coef * ent_loss
        """
        # Increment update counter, etc. (BaseCore/ActorCriticCore utility)
        self._bump()

        # ------------------------------------------------------------------
        # Move tensors to device
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)
        act_raw = batch.actions.to(self.device)
        act = self.head._normalize_discrete_action(act_raw)

        # Old rollout-time statistics
        old_logp = to_column(batch.log_probs.to(self.device))   # (B,1)
        old_v = to_column(batch.values.to(self.device))         # (B,1)

        # Targets from rollout/GAE
        ret = to_column(batch.returns.to(self.device))          # (B,1)
        adv = to_column(batch.advantages.to(self.device))       # (B,1)

        clip_eps = self.clip_range

        def _forward_losses() -> Tuple[
            th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor
        ]:
            """
            Forward pass + compute PPO losses for one minibatch.

            Returns
            -------
            total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_frac, ent_mean, v_mean
            """
            # Distribution π(.|s) from actor
            dist = self.head.actor.get_dist(obs)

            # New log prob and entropy (convert (B,) -> (B,1))
            new_logp = to_column(dist.log_prob(act))
            entropy = to_column(dist.entropy())

            # Current value estimates V(s)
            v = to_column(self.head.critic(obs))

            # PPO ratio
            log_ratio = new_logp - old_logp
            ratio = th.exp(log_ratio)

            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = th.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -th.min(surr1, surr2).mean()

            # Approx KL divergence estimate (common PPO diagnostic)
            approx_kl = (ratio - 1.0 - log_ratio).mean()

            # Clip fraction (what fraction was clipped)
            clip_frac = (th.abs(ratio - 1.0) > clip_eps).float().mean()

            # Value function loss
            if self.clip_vloss:
                # PPO value clipping uses old values as anchor
                v_clipped = old_v + th.clamp(v - old_v, -clip_eps, clip_eps)
                v_loss_unclipped = (v - ret).pow(2)
                v_loss_clipped = (v_clipped - ret).pow(2)
                value_loss = 0.5 * th.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(v, ret)

            # Entropy bonus (negative sign because we minimize the loss)
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
            # AMP forward/backward
            with th.cuda.amp.autocast(enabled=True):
                (
                    total_loss,
                    policy_loss,
                    value_loss,
                    ent_loss,
                    approx_kl,
                    clip_frac,
                    ent_mean,
                    v_mean,
                ) = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Clip gradients across both actor+critic parameters
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            # Step optimizer via scaler
            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            # Standard fp32 path
            (
                total_loss,
                policy_loss,
                value_loss,
                ent_loss,
                approx_kl,
                clip_frac,
                ent_mean,
                v_mean,
            ) = _forward_losses()

            total_loss.backward()

            # Clip gradients across both networks
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            # Apply parameter updates
            self.actor_opt.step()
            self.critic_opt.step()

        # Step schedulers (if enabled in ActorCriticCore)
        self._step_scheds()

        # ------------------------------------------------------------------
        # target_kl early-stop signal (per minibatch)
        # ------------------------------------------------------------------
        # NOTE:
        # This does not stop inside core. It returns a scalar flag so the outer
        # loop (OnPolicyAlgorithm) can break out of epoch/minibatch iteration.
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
        Extend ActorCriticCore state_dict with PPO-discrete specific hyperparameters.

        Returned dict includes:
          - base (ActorCriticCore) state:
              * update counter
              * actor optimizer/scheduler state
              * critic optimizer/scheduler state
              * AMP scaler (if enabled)
          - PPO-specific config under key "ppo_discrete"
        """
        s = super().state_dict()
        s.update(
            {
                "ppo_discrete": {
                    "clip_range": float(self.clip_range),
                    "vf_coef": float(self.vf_coef),
                    "ent_coef": float(self.ent_coef),
                    "clip_vloss": bool(self.clip_vloss),
                    "target_kl": None if self.target_kl is None else float(self.target_kl),
                    "kl_stop_multiplier": float(self.kl_stop_multiplier),
                    "max_grad_norm": float(self.max_grad_norm),
                }
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore optimizer/scheduler state AND PPO-discrete hyperparameters.

        Notes
        -----
        - Optimizer/scheduler/scaler restore is handled by ActorCriticCore.
        - PPO hyperparameters are restored from state["ppo_discrete"] if present.
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