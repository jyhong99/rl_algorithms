from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights
from model_free.common.utils.common_utils import to_scalar, to_column


class DDPGCore(ActorCriticCore):
    """
    DDPG update engine built on ActorCriticCore infrastructure (config-free).

    Key design choice
    -----------------
    This core does NOT call `head.critic(...)` / `head.critic_target(...)` directly
    for Q evaluation. Instead, it uses the head's explicit Q interfaces:

      - head.q_values(obs, action) -> Q(s,a)
      - head.q_values_target(obs, action) -> Q_target(s,a)

    This keeps the core independent from the head's internal critic naming/layout,
    while still allowing ActorCriticCore to discover and optimize actor/critic params.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes (ActorCriticCore expects these):
      - head.actor: nn.Module
      - head.critic: nn.Module
      - head.actor_target: nn.Module
      - head.critic_target: nn.Module
      - head.device: torch.device (or str)

    Required methods (for this core's Q access):
      - head.q_values(obs, action) -> torch.Tensor          (B,1) or (B,)
      - head.q_values_target(obs, action) -> torch.Tensor   (B,1) or (B,)

    Batch contract
    --------------
    update_from_batch(batch) expects:
      - batch.observations:      (B, obs_dim) tensor
      - batch.actions:           (B, action_dim) tensor
      - batch.rewards:           (B,) or (B,1) tensor
      - batch.next_observations: (B, obs_dim) tensor
      - batch.dones:             (B,) or (B,1) tensor

    PER contract (optional)
    -----------------------
    If batch provides importance weights, critic loss is IS-weighted:
      - batch.weights: (B,) or (B,1)

    Priority feedback (optional)
    ----------------------------
    Returns "per/td_errors": np.ndarray of shape (B,)
    """

    def __init__(
        self,
        *,
        head: Any,
        # DDPG core hparams
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        # optim
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # sched
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # grad / amp
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__(
            head=head,
            use_amp=use_amp,
            # optim
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            # sched
            actor_sched_name=str(actor_sched_name),
            critic_sched_name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            sched_gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        # Minimal sanity checks
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # Ensure target modules are frozen (core owns freeze responsibility).
        at = getattr(self.head, "actor_target", None)
        ct = getattr(self.head, "critic_target", None)
        if at is not None:
            self._freeze_target(at)
        if ct is not None:
            self._freeze_target(ct)

        # Validate the head exposes the Q interfaces we rely on.
        if not callable(getattr(self.head, "q_values", None)):
            raise ValueError("DDPGCore requires head.q_values(obs, action).")
        if not callable(getattr(self.head, "q_values_target", None)):
            raise ValueError("DDPGCore requires head.q_values_target(obs, action).")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Run one DDPG update from a replay batch.

        Returns
        -------
        out : Dict[str, float]
            Scalar metrics + optional PER feedback:
              - out["per/td_errors"] : np.ndarray (B,)
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = to_column(batch.dones.to(self.device))

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ---------------------------------------------------------------------
        # Target Q: y = r + gamma * (1-done) * Q_target(s', a'_target)
        # ---------------------------------------------------------------------
        with th.no_grad():
            # Prefer actor_target.act(...) if available; else forward()
            act_fn = getattr(self.head.actor_target, "act", None)
            if callable(act_fn):
                next_a, _ = act_fn(nxt, deterministic=True)
            else:
                next_a = self.head.actor_target(nxt)

            q_t = self.head.q_values_target(nxt, next_a)
            q_t = to_column(q_t)

            target_q = rew + self.gamma * (1.0 - done) * q_t

        # ---------------------------------------------------------------------
        # Critic update (PER-weighted)
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            q = self.head.q_values(obs, act)
            q = to_column(q)

            # Per-sample MSE so we can apply PER weights
            per_sample = F.mse_loss(q, target_q, reduction="none")  # (B,1)

            # TD error used for PER priorities: |Q - y|
            td = (q - target_q).detach().squeeze(1)  # (B,)
            td_abs = td.abs()

            if w is None:
                loss = per_sample.mean()
            else:
                loss = (w * per_sample).mean()

            return loss, td_abs, q

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs, q_now = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()

            # Clip critic grads (use actual critic params for clipping/optimizer)
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )

            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs, q_now = _critic_loss_and_td()
            critic_loss.backward()

            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)

            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update: maximize Q(s, pi(s))  <=> minimize -Q(s, pi(s))
        # ---------------------------------------------------------------------
        def _actor_loss() -> th.Tensor:
            act_fn = getattr(self.head.actor, "act", None)
            if callable(act_fn):
                pi, _ = act_fn(obs, deterministic=True)
            else:
                pi = self.head.actor(obs)

            q_pi = self.head.q_values(obs, pi)
            q_pi = to_column(q_pi)
            return (-q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss = _actor_loss()
            self.scaler.scale(actor_loss).backward()

            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )

            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss = _actor_loss()
            actor_loss.backward()

            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Target update gate (core owns update + freeze)
        # ---------------------------------------------------------------------
        did_target = (self.target_update_interval > 0) and (self.update_calls % self.target_update_interval == 0)
        if did_target:
            self._maybe_update_target(
                target=getattr(self.head, "actor_target", None),
                source=self.head.actor,
                interval=self.target_update_interval,
                tau=self.tau,
            )
            self._maybe_update_target(
                target=getattr(self.head, "critic_target", None),
                source=self.head.critic,
                interval=self.target_update_interval,
                tau=self.tau,
            )

        out: Dict[str, float] = {
            "loss/critic": float(to_scalar(critic_loss)),
            "loss/actor": float(to_scalar(actor_loss)),
            "q/q_mean": float(to_scalar(q_now.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "ddpg/did_target_update": float(1.0 if did_target else 0.0),
        }

        # Non-scalar PER feedback (OffPolicyAlgorithm will consume this for priorities)
        out["per/td_errors"] = td_abs.detach().cpu().numpy()  # type: ignore[assignment]
        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        # Hyperparameters are ctor-owned; do not override silently.
        return