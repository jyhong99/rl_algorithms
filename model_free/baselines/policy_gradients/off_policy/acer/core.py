from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights


class ACERCore(ActorCriticCore):
    """
    ACER Update Engine (1-step, discrete) implemented on top of ActorCriticCore.

    Summary
    -------
    This core performs a single-step off-policy actor-critic update for discrete
    actions in the ACER style:
      - Critic: TD(0) regression toward V_pi(s') computed via target Q and current policy
      - Actor: truncated importance sampling (IS) policy gradient
      - Optional: bias-correction term if behavior action probabilities are available
      - Optional: entropy regularization

    Expected head interface (duck-typed)
    ------------------------------------
    Attributes:
      - head.actor: nn.Module
          Discrete policy network. Typically supports:
            - actor(obs) -> logits, or
            - actor.get_dist(obs) -> categorical distribution
      - head.q: nn.Module
          Q-network for Q(s, a). Depending on head, q_values(obs) returns (B, A).
      - head.q_target: nn.Module
          Target Q-network Q'(s, a) used to form stable TD targets.
      - head.device: torch.device

    Methods:
      - head.logp(obs, action) -> Tensor (B, 1)
          Log-prob under current policy π of the sampled action.
      - head.probs(obs) -> Tensor (B, A)
          Action probabilities π(a|s).
      - head.q_values(obs) -> Tensor (B, A)
          Q(s, ·) under the current critic. If double-Q, head may already apply
          min(Q1, Q2) or a similar aggregation.
      - head.q_values_target(obs) -> Tensor (B, A)
          Q'(s, ·) under the target critic (again, head may aggregate if double-Q).

    Off-policy info expected in batch
    ---------------------------------
      - batch.behavior_logp or batch.logp : log μ(a|s), shape (B,) or (B,1)
        (μ is the behavior policy that generated the data)

    Optional for bias correction
    ----------------------------
      - batch.behavior_probs : μ(a|s) probabilities, shape (B, A)
        If present, ACER can include the bias-correction term that accounts for
        truncation in importance weights across all actions.

    PER support (optional)
    ----------------------
      - batch may contain prioritized replay fields (weights/indices/etc.).
      - get_per_weights(...) returns per-sample weights w (B,1) or None.
    """

    def __init__(
        self,
        *,
        head: Any,
        # -----------------------------
        # Core ACER hyperparameters
        # -----------------------------
        gamma: float = 0.99,
        c_bar: float = 10.0,
        entropy_coef: float = 0.0,
        critic_is: bool = False,
        # -----------------------------
        # Target updates
        # -----------------------------
        target_update_interval: int = 1,
        tau: float = 0.005,
        # -----------------------------
        # Optimizer / scheduler config (ActorCriticCore)
        # -----------------------------
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
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
        # Grad / AMP
        # -----------------------------
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # -----------------------------
        # PER epsilon (core-side proxy only)
        # -----------------------------
        per_eps: float = 1e-6,
    ) -> None:
        # ActorCriticCore sets up:
        # - actor_opt / critic_opt
        # - actor_sched / critic_sched
        # - AMP scaler if enabled
        # - self.device from head
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

        # Store ACER hyperparameters
        self.gamma = float(gamma)
        self.c_bar = float(c_bar)
        self.entropy_coef = float(entropy_coef)
        self.critic_is = bool(critic_is)

        # Target update configuration
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)

        # Gradient clipping / PER
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # Defensive validation (fail fast on obvious misconfiguration)
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.c_bar <= 0.0:
            raise ValueError(f"c_bar must be > 0, got {self.c_bar}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        # Ensure the target critic is frozen once (core owns target-freeze responsibility).
        q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            self._freeze_target(q_target)

    # =============================================================================
    # Batch helpers
    # =============================================================================
    def _get_behavior_logp(self, batch: Any) -> th.Tensor:
        """
        Extract behavior log-prob log μ(a|s) from batch and normalize to shape (B, 1).

        Accepted fields
        ---------------
        - batch.behavior_logp
        - batch.logp (fallback)

        Returns
        -------
        th.Tensor
            Behavior log-prob tensor of shape (B, 1) on self.device.

        Raises
        ------
        ValueError
            If behavior log-prob is missing or the final shape is not (B,1).
        """
        if hasattr(batch, "behavior_logp"):
            log_mu = batch.behavior_logp
        elif hasattr(batch, "logp"):
            log_mu = batch.logp
        else:
            raise ValueError("ACER requires behavior logp in batch (behavior_logp or logp).")

        log_mu = log_mu.to(self.device)

        # Normalize (B,) -> (B,1) for consistent arithmetic with log_pi.
        if log_mu.dim() == 1:
            log_mu = log_mu.unsqueeze(-1)

        if log_mu.dim() != 2 or log_mu.shape[1] != 1:
            raise ValueError(f"behavior_logp must be shape (B,1) or (B,), got {tuple(log_mu.shape)}")

        return log_mu

    def _get_logits(self, obs_t: th.Tensor) -> th.Tensor:
        """
        Best-effort logits extraction for stable log-softmax / entropy computations.

        Preference order
        ---------------
        1) head.dist(obs).logits
        2) head.logits(obs)              (if head provides a direct logits method)
        3) head.actor(obs)               (assume actor forward returns logits)

        Parameters
        ----------
        obs_t : th.Tensor
            Batched observations tensor on self.device.

        Returns
        -------
        th.Tensor
            Logits tensor of shape (B, A).
        """
        d = getattr(self.head, "dist", None)
        if callable(d):
            dist = self.head.dist(obs_t)
            logits = getattr(dist, "logits", None)
            if logits is not None:
                return logits

        fn = getattr(self.head, "logits", None)
        if callable(fn):
            return fn(obs_t)

        return self.head.actor(obs_t)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one ACER update step from a replay batch.

        Returns
        -------
        Dict[str, float]
            Scalar metrics for logging. (Note: this function currently also returns
            a NumPy array under "per/td_errors"; strictly speaking that is not float.)
        """
        # Internal update counter (typically used for schedulers/target update cadence).
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch tensors to device and normalize shapes
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)                  # (B, obs_dim)
        act = batch.actions.to(self.device).long()                # (B,) or (B,1) -> used as indices
        rew = to_column(batch.rewards.to(self.device))               # (B,1)
        next_obs = batch.next_observations.to(self.device)        # (B, obs_dim)
        done = to_column(batch.dones.to(self.device))                # (B,1), 1 if terminal else 0

        B = int(obs.shape[0])

        # PER weights (if batch carries PER fields). Returns (B,1) or None.
        w = get_per_weights(batch, B, device=self.device)

        # ---------------------------------------------------------------------
        # Importance sampling ratios between target policy π and behavior μ
        # ---------------------------------------------------------------------
        log_mu = self._get_behavior_logp(batch)                   # log μ(a|s), (B,1)

        log_pi = self.head.logp(obs, act)                         # log π(a|s), expected (B,1)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(-1)
        if log_pi.dim() != 2 or log_pi.shape[1] != 1:
            raise ValueError(f"head.logp must return (B,1) or (B,), got {tuple(log_pi.shape)}")

        # ρ = π(a|s) / μ(a|s) = exp(log_pi - log_mu)
        # Clamp for numerical safety.
        rho = th.exp(log_pi - log_mu).clamp(max=1e6)              # (B,1)

        # Truncated importance weight c = min(ρ, c_bar).
        c = th.clamp(rho, max=self.c_bar)                         # (B,1)

        # ============================================================
        # 1) Critic target: y = r + γ(1-d) * Vπ(s')
        #
        # Vπ(s') = Σ_a π(a|s') * Q'(s',a)
        # Uses target critic Q' for stability.
        # ============================================================
        with th.no_grad():
            pi_next = self.head.probs(next_obs)                             # (B,A)
            q_next_t = self.head.q_values_target(next_obs, reduce="min")    # (B,A)
            v_next_t = (pi_next * q_next_t).sum(dim=1, keepdim=True)        # (B,1)
            target_q = rew + self.gamma * (1.0 - done) * v_next_t           # (B,1)

        # ============================================================
        # 2) Critic update: minimize TD regression error
        #
        # loss = E[ 0.5 (y - Q(s,a))^2 ]
        # Optionally: truncated IS on critic loss (critic_is)
        # Optionally: multiply PER weights w
        # ============================================================
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            q_all = self.head.q_values(obs, reduce="min")         # (B,A) (grad-enabled)
            q_sa = q_all.gather(1, act.view(-1, 1))               # (B,1)
            td = (target_q - q_sa)                                # (B,1)

            loss_ps = 0.5 * td.pow(2)                             # (B,1)

            # Optional: apply truncated IS to the critic (not always used in ACER variants).
            if self.critic_is:
                loss_ps = loss_ps * th.clamp(rho, max=self.c_bar)

            # Optional: PER weights.
            if w is not None:
                loss_ps = loss_ps * w

            return loss_ps.mean(), td

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            # AMP branch: compute loss under autocast, scale gradients, then step optimizer.
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ============================================================
        # 3) Advantage for actor: A(s,a) = Q(s,a) - V(s)
        #
        # Stop-grad Q to avoid backprop through critic during actor update.
        # ============================================================
        with th.no_grad():
            pi = self.head.probs(obs)                             # (B,A)
            q_all_ng = self.head.q_values(obs, reduce="min")      # (B,A) (no_grad)
            v_s = (pi * q_all_ng).sum(dim=1, keepdim=True)        # (B,1)

            q_sa_ng = q_all_ng.gather(1, act.view(-1, 1))         # (B,1)
            adv_sa = q_sa_ng - v_s                                # (B,1)

            # Advantage for all actions (used by bias correction if enabled).
            adv_all = q_all_ng - v_s                              # (B,A)

        # ============================================================
        # 4) Actor loss: truncated IS + optional bias correction + entropy
        #
        # Main term (sampled action):
        #   L_main = - E[ c * A(s,a) * log π(a|s) ]
        #
        # Bias correction (requires μ(a|s) probs for all actions):
        #   Uses weights w_bc = max(ρ(a) - c_bar, 0) and a correction term over actions.
        #
        # Entropy regularization:
        #   L_ent = -entropy_coef * E[ H(π(·|s)) ]
        # ============================================================
        main_term = -(c * adv_sa * log_pi).mean()

        correction = th.zeros((), device=self.device)
        correction_on = False

        if hasattr(batch, "behavior_probs"):
            # Behavior policy probabilities μ(a|s) for all actions.
            mu_probs = batch.behavior_probs.to(self.device)       # (B,A)
            pi_probs = pi                                         # (B,A)

            # ρ(a) for all actions; clamp for safety.
            rho_all = (pi_probs / (mu_probs + 1e-8)).clamp(max=1e6)  # (B,A)

            # Bias correction weights only apply where ρ(a) exceeds truncation threshold.
            w_bc = th.clamp(rho_all - self.c_bar, min=0.0)        # (B,A)

            # Need log π(a|s) for all actions; obtain stable logits path.
            logits = self._get_logits(obs)                        # (B,A)
            log_pi_all = F.log_softmax(logits, dim=-1)            # (B,A)

            # Correction term over actions (ACER-style):
            # Negative sign for gradient ascent objective translated into loss minimization.
            correction = -((w_bc * pi_probs * log_pi_all * adv_all).sum(dim=1, keepdim=True)).mean()
            correction_on = True

        entropy_term = th.zeros((), device=self.device)
        if self.entropy_coef != 0.0:
            logits = self._get_logits(obs)
            pi_probs2 = F.softmax(logits, dim=-1)
            log_pi_all2 = F.log_softmax(logits, dim=-1)

            # Entropy H(π) = -Σ_a π(a) log π(a)
            entropy = -(pi_probs2 * log_pi_all2).sum(dim=-1, keepdim=True)  # (B,1)
            entropy_term = -self.entropy_coef * entropy.mean()

        actor_loss = main_term + correction + entropy_term

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss_amp = actor_loss
            self.scaler.scale(actor_loss_amp).backward()
            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ============================================================
        # 5) Target update: Q' <- Q (hard/soft), then freeze remains in effect
        #
        # ActorCriticCore typically handles:
        # - cadence via interval (target_update_interval)
        # - hard vs soft via tau
        # ============================================================
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ============================================================
        # 6) PER priorities (optional): use |TD| as priority signal
        # ============================================================
        with th.no_grad():
            td_abs = td.abs().view(-1)  # (B,)

        # NOTE:
        # - This return type annotation says Dict[str, float], but "per/td_errors"
        #   is a NumPy array. If you want strict typing, either:
        #     a) drop it from return dict and set it on batch for PER,
        #     b) change return type to Dict[str, Any],
        #     c) log only scalar summary stats (mean/max) here.
        return {
            "loss/critic": float(to_scalar(critic_loss)),
            "loss/actor": float(to_scalar(actor_loss)),
            "is/rho_mean": float(to_scalar(rho.mean())),
            "is/c_mean": float(to_scalar(c.mean())),
            "adv/mean": float(to_scalar(adv_sa.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "acer/correction_on": float(1.0 if correction_on else 0.0),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        Notes
        -----
        ActorCriticCore state_dict typically already contains optimizer/scheduler state,
        AMP scaler state, and internal update counters. This method appends ACER-specific
        hyperparameters for inspection/debugging.

        Important
        ---------
        Hyperparameters are constructor-owned; storing them here is informational.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "c_bar": float(self.c_bar),
                "entropy_coef": float(self.entropy_coef),
                "critic_is": bool(self.critic_is),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Load core state.

        Policy
        ------
        - Delegates to ActorCriticCore for optimizer/scheduler/counter state restore.
        - Does NOT silently override ctor-owned hyperparameters (gamma, c_bar, etc.).
          If you want hyperparameter restore, do it explicitly at construction time.
        """
        super().load_state_dict(state)
        return
