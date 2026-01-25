from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore


class A2CDiscreteCore(ActorCriticCore):
    """
    A2C update engine for DISCRETE actions only.

    What this core does
    -------------------
    - Implements a single A2C update step for categorical policies.
    - Computes:
        * policy loss     :  -E[ log π(a|s) * A(s,a) ]
        * value loss      :  0.5 * MSE( V(s), return )
        * entropy bonus   :  -E[ H(π(.|s)) ]  (negative sign so adding ent_coef encourages exploration)
      and then performs an optimizer step on actor and critic.

    Batch contract (rollout minibatch)
    ----------------------------------
    Required fields (duck-typed):
      - observations : (B, obs_dim)
      - actions      : scalar, (B,), or (B,1)          (discrete action indices)
      - returns      : (B,) or (B,1)
      - advantages   : (B,) or (B,1)

    Distribution contract
    ---------------------
    - dist = head.actor.get_dist(obs) must be categorical-like.
    - dist.log_prob(action_idx) expects LongTensor shaped (B,) (typical).
    - dist.entropy() returns (B,) (typical).

    Shape conventions
    -----------------
    This core standardizes:
      - log_prob  -> (B,1) via to_column()
      - entropy   -> (B,1) via to_column()
      - value     -> (B,1) via to_column()
      - returns   -> (B,1) via to_column()
      - adv       -> (B,1) via to_column()
    """

    def __init__(
        self,
        *,
        head: Any,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        # Optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 7e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 7e-4,
        critic_weight_decay: float = 0.0,
        # (Optional) LR schedulers
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Actor-critic head providing:
              - head.actor.get_dist(obs) -> categorical-like dist
              - head.critic(obs) -> V(s)
              - head.device (used by base core)
        vf_coef : float
            Coefficient for value loss term.
        ent_coef : float
            Coefficient for entropy bonus (encourages exploration if > 0).
        max_grad_norm : float
            Global gradient clipping threshold (>= 0).
        use_amp : bool
            Enable mixed-precision training with GradScaler.
        Optimizer/scheduler args
            Forwarded to ActorCriticCore which builds optimizers and schedulers.
        """
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

        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one A2C update step (discrete actions).

        Returns
        -------
        metrics : Dict[str, float]
            Standard training metrics (losses, stats, lrs).
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch to device and normalize shapes/dtypes.
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)
        act_raw = batch.actions.to(self.device)
        act = self.head._normalize_discrete_action(act_raw)

        ret = to_column(batch.returns.to(self.device))
        adv = to_column(batch.advantages.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            # -----------------------------------------------------------------
            # 1) Policy distribution
            # -----------------------------------------------------------------
            dist = self.head.actor.get_dist(obs)

            # log_prob/entropy are typically (B,) for categorical distributions.
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            # If a custom dist returns extra dims (rare for categorical),
            # reduce to joint scalar per sample.
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            logp = to_column(logp)
            entropy = to_column(entropy)

            # -----------------------------------------------------------------
            # 2) Critic
            # -----------------------------------------------------------------
            v = to_column(self.head.critic(obs))

            # -----------------------------------------------------------------
            # 3) Losses
            # -----------------------------------------------------------------
            policy_loss = -(logp * adv.detach()).mean()
            value_loss = 0.5 * F.mse_loss(v, ret)
            ent_loss = -entropy.mean()  # negative so +ent_coef encourages higher entropy

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v.mean()

        # ---------------------------------------------------------------------
        # Backprop + optimizer step
        # ---------------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Clip gradients across both actor and critic for stability.
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # Step learning-rate schedulers (if enabled).
        self._step_scheds()

        return {
            "loss/policy": float(to_scalar(policy_loss)),
            "loss/value": float(to_scalar(value_loss)),
            "loss/entropy": float(to_scalar(ent_loss)),
            "loss/total": float(to_scalar(total_loss)),
            "stats/entropy": float(to_scalar(ent_mean)),
            "stats/value_mean": float(to_scalar(v_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend base core state with A2C-specific hyperparameters.
        Optimizer/scheduler state is handled by ActorCriticCore.
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
        Restore base core state and A2C-specific hyperparameters.
        """
        super().load_state_dict(state)
        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
