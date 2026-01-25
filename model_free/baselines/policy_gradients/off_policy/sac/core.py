from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class SACCore(ActorCriticCore):
    """
    Soft Actor-Critic update engine.

    This core implements the SAC update rules while reusing the shared
    ActorCriticCore infrastructure for:
      - actor/critic optimizers + schedulers
      - AMP GradScaler (optional)
      - update counters / common persistence helpers
      - target-network Polyak updates via core helper methods

    Expected head (duck-typed)
    --------------------------
    Required:
      - head.actor: nn.Module
      - head.critic: nn.Module, signature: critic(obs, act) -> (q1, q2) each (B,1)
      - head.critic_target: nn.Module, same signature
      - head.sample_action_and_logp(obs) -> (action, logp), where:
          * action: (B, action_dim)
          * logp: (B,1) preferred (core will normalize with to_column)
      - head.device: torch.device

    Optional (if provided by BaseHead)
    ---------------------------------
      - head.hard_update / head.soft_update / head.freeze_target
        (core can still freeze / update targets via its own helpers)

    Notes on shapes
    ---------------
    - Rewards and dones are normalized to (B,1) using to_column()
    - log-probabilities are normalized to (B,1) using to_column()
      (your updated head.sample_action_and_logp enforces this already)
    """

    def __init__(
        self,
        *,
        head: Any,
        # SAC hyperparameters
        gamma: float = 0.99,
        # target update
        tau: float = 0.005,
        target_update_interval: int = 1,
        # entropy temperature
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # alpha optimizer/scheduler
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        alpha_sched_name: str = "none",
        # sched shared knobs
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
        # actor/critic opt/sched (inherited)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
    ) -> None:
        # Build actor/critic optimizer+scheduler via ActorCriticCore.
        # (alpha is managed separately in this class)
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

        # -----------------------------
        # Hyperparameters / validation
        # -----------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # -----------------------------
        # Target entropy
        # -----------------------------
        # Common SAC heuristic: target_entropy = -|A|
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # -----------------------------
        # Temperature parameter alpha
        # -----------------------------
        # Optimize log(alpha) for numerical stability.
        self.auto_alpha = bool(auto_alpha)
        log_alpha_init = math.log(float(alpha_init))
        self.log_alpha = th.tensor(
            float(log_alpha_init),
            device=self.device,
            requires_grad=self.auto_alpha,
        )

        # alpha optimizer/scheduler (separate from ActorCriticCore)
        self.alpha_opt = None
        if self.auto_alpha:
            self.alpha_opt = build_optimizer(
                [self.log_alpha],
                name=str(alpha_optim_name),
                lr=float(alpha_lr),
                weight_decay=float(alpha_weight_decay),
            )

        self.alpha_sched = None
        if self.alpha_opt is not None:
            self.alpha_sched = build_scheduler(
                self.alpha_opt,
                name=str(alpha_sched_name),
                total_steps=int(total_steps),
                warmup_steps=int(warmup_steps),
                min_lr_ratio=float(min_lr_ratio),
                poly_power=float(poly_power),
                step_size=int(step_size),
                gamma=float(sched_gamma),
                milestones=tuple(int(m) for m in milestones),
            )

        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """Entropy temperature alpha = exp(log_alpha)."""
        return self.log_alpha.exp()

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one SAC update from a sampled replay batch.

        Update steps
        ------------
        1) Compute TD target using target critics (no grad):
             y = r + gamma * (1-d) * (min(Q1_t, Q2_t) - alpha * logπ(a'|s'))
        2) Critic update: MSE(Q_i(s,a), y) for i=1,2 (PER-weighted if enabled)
        3) Actor update:
             J_pi = E[ alpha * logπ(a|s) - min(Q1,Q2)(s,a) ]
        4) (Optional) Alpha update:
             J_alpha = -E[ log_alpha * (logπ(a|s) + target_entropy) ]
        5) Target critic Polyak update at configured interval.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar metrics plus PER td_errors array for priority updates.
        """
        self._bump()

        # -----------------------------
        # Move batch to device + normalize shapes
        # -----------------------------
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = to_column(batch.rewards.to(self.device))            # (B,1)
        nxt = batch.next_observations.to(self.device)
        done = to_column(batch.dones.to(self.device))            # (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)        # (B,1) or None

        # ------------------------------------------------------------------
        # Target Q computation (no grad)
        # ------------------------------------------------------------------
        with th.no_grad():
            # Next action from current policy + its log-probability
            nxt_a, nxt_logp = self.head.sample_action_and_logp(nxt)
            nxt_logp = to_column(nxt_logp)  # robust: (B,) -> (B,1)

            # Target critics evaluate Q(s', a')
            q1_t, q2_t = self.head.q_values_target(nxt, nxt_a)     # each (B,1)
            q_min_t = th.min(q1_t, q2_t)                         # (B,1)

            # Soft TD target
            target_q = rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * nxt_logp)  # (B,1)

        # ------------------------------------------------------------------
        # Critic update (PER-weighted)
        # ------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute critic loss and per-sample TD-error magnitude for PER.

            Returns
            -------
            loss : torch.Tensor
                Scalar loss.
            td_abs : torch.Tensor
                Shape (B,) absolute TD error proxy used for PER priorities.
            """
            q1, q2 = self.head.q_values(obs, act)  # each (B,1)

            # Per-sample squared error (B,1)
            l1 = F.mse_loss(q1, target_q, reduction="none")
            l2 = F.mse_loss(q2, target_q, reduction="none")
            per_sample = l1 + l2

            # TD magnitude proxy for PER (use min(Q1,Q2) like target)
            td = th.min(q1, q2) - target_q
            td_abs = td.abs().detach().squeeze(1)  # (B,)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
            return loss, td_abs

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.critic_opt)
            self.scaler.step(self.critic_opt)
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ------------------------------------------------------------------
        # Actor update
        # ------------------------------------------------------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute actor loss.

            Returns
            -------
            actor_loss : torch.Tensor
                Scalar loss.
            logp : torch.Tensor
                Shape (B,1) log π(a|s) for sampled actions.
            q_pi : torch.Tensor
                Shape (B,1) min(Q1,Q2)(s,a) for sampled actions.
            """
            new_a, logp = self.head.sample_action_and_logp(obs)
            logp = to_column(logp)  # (B,1)

            q1_pi, q2_pi = self.head.q_values(obs, new_a)
            q_pi = th.min(q1_pi, q2_pi)  # (B,1)

            # SAC actor objective: minimize E[ alpha*logp - Q ]
            actor_loss = (self.alpha * logp - q_pi).mean()
            return actor_loss, logp, q_pi

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, logp, q_pi = _actor_loss()
            self.scaler.scale(actor_loss).backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.actor_opt)
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss, logp, q_pi = _actor_loss()
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ------------------------------------------------------------------
        # Alpha update (optional)
        # ------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            # Gradient should not flow into actor from alpha loss, so detach logp.
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(to_scalar(alpha_loss))

        # ------------------------------------------------------------------
        # Target update (Polyak)
        # ------------------------------------------------------------------
        # Uses core helper which:
        #  - applies soft update at given interval
        #  - re-freezes targets (requires_grad False)
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ------------------------------------------------------------------
        # Metrics / logging
        # ------------------------------------------------------------------
        with th.no_grad():
            q1_b, q2_b = self.head.critic(obs, act)

        out: Dict[str, Any] = {
            "loss/critic": float(to_scalar(critic_loss)),
            "loss/actor": float(to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "alpha": float(to_scalar(self.alpha)),
            "q/q1_mean": float(to_scalar(q1_b.mean())),
            "q/q2_mean": float(to_scalar(q2_b.mean())),
            "q/pi_min_mean": float(to_scalar(q_pi.mean())),
            "logp_mean": float(to_scalar(logp.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state including actor/critic (via super) plus alpha state.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "auto_alpha": bool(self.auto_alpha),
                "target_entropy": float(self.target_entropy),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state including alpha (log_alpha + optimizer/scheduler).
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        # auto_alpha/target_entropy are configuration; restore if present (best-effort)
        if "auto_alpha" in state:
            self.auto_alpha = bool(state["auto_alpha"])
            # NOTE: we do not rebuild alpha_opt here; assume constructor config matches.
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])

        if self.alpha_opt is not None and state.get("alpha") is not None:
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, state["alpha"])
