from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore


class A2CCore(ActorCriticCore):
    """
    A2C update engine for CONTINUOUS action spaces.

    What this core does
    -------------------
    - Performs a single actor-critic update per call using a rollout minibatch.
    - Uses the standard A2C objective:
        * Policy loss   : -E[ log pi(a|s) * A(s,a) ]
        * Value loss    :  0.5 * MSE( V(s), R )
        * Entropy bonus : -E[ H(pi(.|s)) ]   (scaled by ent_coef)

      Total loss:
        L = policy_loss + vf_coef * value_loss + ent_coef * ent_loss

    Batch contract (rollout minibatch)
    ----------------------------------
    Required fields on `batch`:
      - observations : Tensor, shape (B, obs_dim)
      - actions      : Tensor, shape (B, act_dim) OR (act_dim,) OR scalar when act_dim==1
      - returns      : Tensor, shape (B,) or (B, 1)
      - advantages   : Tensor, shape (B,) or (B, 1)

    Distribution contract (continuous)
    ----------------------------------
    - dist = head.actor.get_dist(obs) must be Gaussian-like.
    - dist.log_prob(action) may return:
        * (B,)           if the distribution already reduces action dims
          (e.g., Independent(Normal(...), 1))
        * (B, act_dim)   if per-dimension log-prob is returned (e.g., Normal without Independent)
    - dist.entropy() may similarly return (B,) or (B, act_dim)

    Implementation convention
    -------------------------
    - We reduce log_prob/entropy to one scalar per sample by summing over the last
      dimension if needed, then enforce a column shape (B, 1) via `to_column()`.
    - Advantages are treated as constants for the policy gradient term (detach()).
    - Gradient clipping is applied globally across actor + critic parameters.

    Notes
    -----
    - This core assumes the head always has both `actor` and `critic`.
      (A2C is inherently actor-critic; no baseline toggle here.)
    - Advantage normalization (if desired) should happen upstream (buffer/algorithm).
    """

    def __init__(
        self,
        *,
        head: Any,
        # loss coefficients
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        # gradient / AMP
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        # optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 7e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 7e-4,
        critic_weight_decay: float = 0.0,
        # schedulers (optional)
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
            Actor-critic head with:
              - head.actor.get_dist(obs) -> distribution
              - head.critic(obs) -> V(s)
              - head.device
        vf_coef : float
            Value loss coefficient.
        ent_coef : float
            Entropy loss coefficient (note: ent_loss is negative entropy).
        max_grad_norm : float
            Global gradient norm clip. Use 0 to disable clipping.
        use_amp : bool
            Enable CUDA AMP for forward/backward (best-effort).
        actor_optim_name, critic_optim_name : str
            Optimizer names for your optimizer builder.
        actor_lr, critic_lr : float
            Learning rates.
        actor_weight_decay, critic_weight_decay : float
            Weight decay (L2).
        actor_sched_name, critic_sched_name : str
            Scheduler names for your scheduler builder.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler parameters passed through the base class.
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
        Perform one A2C update step (continuous actions).

        Returns
        -------
        Dict[str, float]
            Logging metrics (losses, stats, lrs).
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        ret = to_column(batch.returns.to(self.device))
        adv = to_column(batch.advantages.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            # Actor distribution
            dist = self.head.actor.get_dist(obs)

            # log pi(a|s) and entropy
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            # Reduce to one scalar per sample if distribution returns per-dimension values
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            logp = to_column(logp)           # (B,1)
            entropy = to_column(entropy)     # (B,1)

            # Critic value
            v = to_column(self.head.critic(obs))  # (B,1)

            # Loss terms
            policy_loss = -(logp * adv.detach()).mean()
            value_loss = 0.5 * F.mse_loss(v, ret)
            ent_loss = -entropy.mean()

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v.mean()

        # Reset grads (set_to_none=True is usually a bit faster/cleaner)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        # AMP path (CUDA only, best-effort)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Global clip across actor + critic
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

        # Step schedulers (if enabled by the base class)
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
        Extend base state_dict with A2C-specific hyperparameters.

        Notes
        -----
        - Optimizer/scheduler states are handled by the base class.
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
        Load optimizer/scheduler state (base) and restore A2C-specific hyperparameters.
        """
        super().load_state_dict(state)
        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])