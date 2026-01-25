from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple  # FIX: add Tuple (used below)

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights
from model_free.common.utils.common_utils import to_scalar, to_column


class TD3Core(ActorCriticCore):
    """
    TD3 Update Engine (TD3Core) built on ActorCriticCore infrastructure (NO config dataclass).

    Expected head interface (duck-typed)
    -----------------------------------
    Required attributes:
      - head.actor: nn.Module
      - head.critic: nn.Module                 (twin Q; forward -> (q1, q2))
      - head.actor_target: nn.Module
      - head.critic_target: nn.Module
      - head.device: torch.device (or str)

    Required methods:
      - head.target_action(next_obs, noise_std, noise_clip) -> torch.Tensor

    Optional methods (preferred if present):
      - head.soft_update(target, source, tau)  OR head.soft_update_target(tau=...)
      - head.freeze_target(module)

    Batch contract
    --------------
    update_from_batch(batch) expects:
      - batch.observations:      (B, obs_dim)
      - batch.actions:           (B, action_dim)
      - batch.rewards:           (B,) or (B, 1)
      - batch.next_observations: (B, obs_dim)
      - batch.dones:             (B,) or (B, 1)
      - (optional) batch.weights: (B,) or (B,1) for PER
    """

    def __init__(
        self,
        *,
        head: Any,
        # TD3 core hparams
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        target_update_interval: int = 1,
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

        # --- core scalars ---
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)

        self.policy_delay = int(policy_delay)
        self.target_update_interval = int(target_update_interval)

        self.max_grad_norm = float(max_grad_norm)

        # (Optional but recommended) basic validation to fail fast on bad configs.
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.policy_delay < 0:
            raise ValueError(f"policy_delay must be >= 0, got {self.policy_delay}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        self.head.freeze_target(self.head.actor_target)
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one TD3 update using a replay batch.

        Steps
        -----
        1) Compute TD target using target networks and policy smoothing.
        2) Update critic(s) every call.
        3) Update actor only every `policy_delay` calls.
        4) When actor updates, also update target networks (polyak / hard update)
           at frequency `target_update_interval`.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar metrics. PER td_errors is returned as numpy array under
            key "per/td_errors" for replay priority updates.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = to_column(batch.dones.to(self.device))

        # (B,1) weights for PER, or None
        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)

        # ---------------------------------------------------------------------
        # Target Q (no grad)
        # ---------------------------------------------------------------------
        with th.no_grad():
            # TD3 target policy smoothing: a' = π'(s') + clipped noise, then clamp
            next_a = self.head.target_action(
                nxt,
                noise_std=float(self.policy_noise),
                noise_clip=float(self.noise_clip),
            )
            q1_t, q2_t = self.head.q_values_target(nxt, next_a)  # (B,1), (B,1)
            q_t = th.min(q1_t, q2_t)
            target_q = rew + self.gamma * (1.0 - done) * q_t  # (B,1)

        # ---------------------------------------------------------------------
        # Critic update (PER-weighted)
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            q1, q2 = self.head.q_values(obs, act)  # each (B,1)

            # per-sample MSE (keepdim so PER multiply works with (B,1))
            l1 = F.mse_loss(q1, target_q, reduction="none")
            l2 = F.mse_loss(q2, target_q, reduction="none")
            per_sample = l1 + l2  # (B,1)

            # TD error for PER priorities: use min(Q1,Q2) - target (common choice)
            td = th.min(q1, q2) - target_q  # (B,1)
            td_abs = td.abs().detach().squeeze(1)  # (B,)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
            return loss, td_abs, q1, q2

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.critic_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update (delayed)
        # ---------------------------------------------------------------------
        did_actor = (self.policy_delay > 0) and (self.update_calls % self.policy_delay == 0)
        actor_loss_scalar = 0.0

        if did_actor:

            def _actor_loss() -> th.Tensor:
                # TD3 actor objective: maximize Q(s, π(s)) (use Q1 by convention).
                # Prefer actor.act if it exists (common in your policy nets), else forward().
                act_fn = getattr(self.head.actor, "act", None)
                if callable(act_fn):
                    pi, _ = act_fn(obs, deterministic=True)
                else:
                    pi = self.head.actor(obs)

                q1_pi, _q2_pi = self.head.q_values(obs, pi)
                return (-q1_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)

            if self.use_amp:
                with th.cuda.amp.autocast(enabled=True):
                    actor_loss = _actor_loss()
                self.scaler.scale(actor_loss).backward()
                self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.actor_opt)
                self.scaler.step(self.actor_opt)
                self.scaler.update()
            else:
                actor_loss = _actor_loss()
                actor_loss.backward()
                self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
                self.actor_opt.step()

            if self.actor_sched is not None:
                self.actor_sched.step()

            actor_loss_scalar = float(to_scalar(actor_loss))

            # -----------------------------------------------------------------
            # Target updates: actor_target + critic_target
            # -----------------------------------------------------------------
            # NOTE:
            # - _maybe_update_target() handles interval gating and freeze.
            # - TD3 updates targets typically when actor updates; you also gate by
            #   target_update_interval for flexibility.
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

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        out: Dict[str, float] = {
            "loss/critic": float(to_scalar(critic_loss)),
            "loss/actor": float(actor_loss_scalar),
            "q/q1_mean": float(to_scalar(q1_now.mean())),
            "q/q2_mean": float(to_scalar(q2_now.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "td3/did_actor_update": float(1.0 if did_actor else 0.0),
        }
        # PER feedback (non-scalar). Keep as numpy for buffer priority update.
        out["per/td_errors"] = td_abs.detach().cpu().numpy()  # type: ignore[assignment]
        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend ActorCriticCore state with TD3 hyperparameters (repro/debug).
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "policy_noise": float(self.policy_noise),
                "noise_clip": float(self.noise_clip),
                "policy_delay": int(self.policy_delay),
                "target_update_interval": int(self.target_update_interval),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base core (opts/scheds/update_calls) and keep hparams if present.

        Note
        ----
        This method restores optimizer/scheduler state via ActorCriticCore.
        Hyperparameters are kept as the instance's current values unless you
        explicitly want to overwrite them from state (not done here to avoid
        surprising behavior when loading across configs).
        """
        super().load_state_dict(state)

        self.head.freeze_target(self.head.actor_target)
        self.head.freeze_target(self.head.critic_target)
