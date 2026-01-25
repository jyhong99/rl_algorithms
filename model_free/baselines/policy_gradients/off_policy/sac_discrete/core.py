from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import copy

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.utils.common_utils import to_scalar, to_column


class SACDiscreteCore(ActorCriticCore):
    """
    Discrete SAC Update Engine built on ActorCriticCore infrastructure.

    What this core updates
    ----------------------
    - Actor: categorical policy π(a|s) over discrete actions
    - Critic: twin Q networks Q1(s,·), Q2(s,·) each output shape (B, A)
    - Target critic: Polyak/EMA copy of critic used for bootstrap targets
    - Entropy temperature α (optional, learned via log_alpha)

    Ownership / responsibilities
    ----------------------------
    - ActorCriticCore owns:
        * actor_opt / actor_sched
        * critic_opt / critic_sched
        * AMP scaler (if enabled)
        * update_calls counter + persistence

    - This SACDiscreteCore additionally owns:
        * critic_target (if head doesn't provide one)
        * log_alpha + alpha optimizer/scheduler (optional)

    Expected head interface (duck-typed)
    ------------------------------------
    Required:
      - head.actor: nn.Module with get_dist(obs)->Categorical-like dist or logits
      - head.critic: nn.Module, critic(obs)->(q1,q2) each (B, A)  OR critic(obs)->tuple
      - head.device

    Optional:
      - head.critic_target: nn.Module (same signature as critic)
      - BaseHead utilities (hard_update/soft_update/freeze_target) are used indirectly
        via ActorCriticCore helper methods.
    """

    def __init__(
        self,
        *,
        head: Any,
        # core hparams
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        # alpha / entropy
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # optimizers (actor/critic handled by ActorCriticCore)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # alpha optim
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        # schedulers
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        alpha_sched_name: str = "none",
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
        # PER
        per_eps: float = 1e-6,
    ) -> None:
        # ActorCriticCore validates:
        # - head.actor is nn.Module
        # - head.critic is nn.Module
        # and creates actor_opt/critic_opt (+ schedulers).
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

        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # ----- target entropy default (DISCRETE) -----
        # Discrete SAC often targets entropy around log(|A|) (max entropy).
        # A common choice: target_entropy = -log(|A|) or -|A| depending on conventions.
        # Here we keep your sign convention: alpha_loss uses (ent - target_entropy).
        if target_entropy is None:
            # Prefer head.n_actions; fallback to head.action_dim if some heads reuse name.
            n_actions = getattr(self.head, "n_actions", None)
            if n_actions is None:
                n_actions = getattr(self.head, "action_dim", None)
            if n_actions is None:
                raise ValueError("SACDiscreteCore needs head.n_actions (or head.action_dim) to infer target_entropy.")
            self.target_entropy = float(math.log(float(int(n_actions))))  # positive entropy scale
        else:
            self.target_entropy = float(target_entropy)

        # ----- critic_target ownership -----
        # If head already provides critic_target, use it. Else, keep a private deepcopy.
        ct = getattr(self.head, "critic_target", None)
        if ct is not None:
            self.critic_target = ct
        else:
            self.critic_target = copy.deepcopy(self.head.critic).to(self.device)

        # Core owns freeze responsibility (target should not receive grads)
        self.head.freeze_target(self.head.critic_target)

        # ----- alpha parameter (log-space) -----
        log_alpha_init = float(math.log(float(alpha_init)))
        self.log_alpha = th.tensor(
            log_alpha_init,
            device=self.device,
            requires_grad=bool(self.auto_alpha),
        )

        # ----- alpha optimizer/scheduler -----
        self.alpha_opt = None
        self.alpha_sched = None
        if self.auto_alpha:
            self.alpha_opt = build_optimizer(
                [self.log_alpha],
                name=str(alpha_optim_name),
                lr=float(alpha_lr),
                weight_decay=float(alpha_weight_decay),
            )
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

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """Entropy temperature α = exp(log_alpha). Always positive."""
        return self.log_alpha.exp()

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        One gradient update using a replay batch.

        Batch contract (typical OffPolicyAlgorithm)
        -------------------------------------------
        - observations:      (B, obs_dim) tensor
        - actions:           (B,) or (B,1) integer tensor (discrete action indices)
        - rewards:           (B,) or (B,1)
        - next_observations: (B, obs_dim)
        - dones:             (B,) or (B,1)
        - optional: weights  (B,1) for PER
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()  
        rew = to_column(batch.rewards.to(self.device))  # -> (B,1)
        nxt = batch.next_observations.to(self.device)
        done = to_column(batch.dones.to(self.device))   # -> (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ---------------------------------------------------------------------
        # Target:
        #   V(s') = Σ_a π(a|s') [ min(Q1_t, Q2_t)(s',a) - α log π(a|s') ]
        #   y = r + γ (1-done) V(s')
        # ---------------------------------------------------------------------
        with th.no_grad():
            # Actor distribution for next state
            dist_next = self.head.dist(nxt)

            # Prefer dist.logits if present; else fall back to network forward
            logits_next = getattr(dist_next, "logits", None)
            if logits_next is None:
                logits_next = self.head.actor(nxt)

            logp_next_all = F.log_softmax(logits_next, dim=-1)  # (B,A)
            prob_next_all = logp_next_all.exp()                 # (B,A)

            # Target critics output Q(s',·) for all actions
            q1_t, q2_t = self.head.q_values_target_pair(nxt)    # each (B,A)
            min_q_t = th.min(q1_t, q2_t)                        # (B,A)

            # Expected soft value under π
            v_next = th.sum(
                prob_next_all * (min_q_t - self.alpha * logp_next_all),
                dim=-1,
                keepdim=True,
            )  # (B,1)

            target_q = rew + self.gamma * (1.0 - done) * v_next  # (B,1)

        # ---------------------------------------------------------------------
        # Critic update (PER-weighted)
        # Critic outputs Q(s,·); we regress Q(s,a_taken) toward target_q.
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            q1, q2 = self.head.q_values_pair(obs)                      # each (B,A)

            act_idx = act.view(-1).long()
            q1_sa = q1.gather(1, act_idx.view(-1, 1))
            q2_sa = q2.gather(1, act_idx.view(-1, 1))

            td1 = target_q - q1_sa
            td2 = target_q - q2_sa

            # Per-sample critic loss (keep (B,1) shape for PER multiply)
            per_sample = 0.5 * (td1.pow(2) + td2.pow(2))       # (B,1)
            loss = per_sample.mean() if w is None else (w * per_sample).mean()

            # TD-error magnitude for PER priority update (B,)
            td_abs = 0.5 * (td1.abs() + td2.abs()).view(-1)
            return loss, td_abs

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td()

            self.scaler.scale(critic_loss).backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.critic_opt)

            # AMP path: step must go through scaler
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update
        # Objective (minimization form):
        #   L_pi = E_s [ Σ_a π(a|s) ( α log π(a|s) - minQ(s,a) ) ]
        # (Equivalent to maximizing E[minQ + α H] depending on sign convention.)
        # ---------------------------------------------------------------------
        def _actor_loss_and_entropy() -> Tuple[th.Tensor, th.Tensor]:
            dist = self.head.dist(obs)
            logits = getattr(dist, "logits", None)
            if logits is None:
                logits = self.head.actor(obs)

            logp_all = F.log_softmax(logits, dim=-1)           # (B,A)
            prob_all = logp_all.exp()                          # (B,A)

            # Use critics as baseline; no grad through Q for policy update
            with th.no_grad():
                q1_pi, q2_pi = self.head.q_values_pair(obs)    # each (B,A)
                min_q_pi = th.min(q1_pi, q2_pi)                # (B,A)

            # L_pi = E[ Σ_a π(a|s) ( α logπ(a|s) - minQ(s,a) ) ]
            per_state_obj = th.sum(prob_all * (self.alpha * logp_all - min_q_pi), dim=-1, keepdim=True)  # (B,1)
            loss = per_state_obj.mean()

            # Entropy H(π) = -Σ π logπ
            ent = -(prob_all * logp_all).sum(dim=-1, keepdim=True)  # (B,1)
            return loss, ent

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, ent = _actor_loss_and_entropy()

            self.scaler.scale(actor_loss).backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.actor_opt)

            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss, ent = _actor_loss_and_entropy()
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Alpha update (optional)
        #
        # We want entropy to track target_entropy.
        # Using your convention:
        #   alpha_loss = - log_alpha * (H(pi) - H_target)
        # If H > H_target -> decrease alpha; if H < H_target -> increase alpha.
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (ent.detach() - self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(to_scalar(alpha_loss))

        # ---------------------------------------------------------------------
        # Target update: critic_target <- critic
        # (Polyak if tau < 1 else hard)
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=self.critic_target,
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )
        # _maybe_update_target() is expected to re-freeze target params.

        td_abs_np = td_abs.clamp(min=self.per_eps).detach().cpu().numpy()

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        out: Dict[str, Any] = {
            "loss/critic": float(to_scalar(critic_loss)),
            "loss/actor": float(to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "stats/alpha": float(to_scalar(self.alpha)),
            "stats/entropy": float(to_scalar(ent.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "sac/target_updated": float(
                1.0
                if (self.target_update_interval > 0 and (self.update_calls % self.target_update_interval == 0))
                else 0.0
            ),
            "per/td_errors": td_abs_np,
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize:
        - base core state (update_calls + actor/critic opt/sched)
        - critic_target weights (if core-owned or even if head-owned; safe to store)
        - log_alpha + alpha opt/sched state
        - key hyperparameters (for reproducibility)
        """
        s = super().state_dict()
        s.update(
            {
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": float(to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "target_entropy": float(self.target_entropy),
                "max_grad_norm": float(self.max_grad_norm),
                "auto_alpha": bool(self.auto_alpha),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core + optimizers, then restore critic_target + alpha states.
        """
        super().load_state_dict(state)

        if "critic_target" in state and state["critic_target"] is not None:
            self.critic_target.load_state_dict(state["critic_target"])
            self._freeze_target(self.critic_target)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
