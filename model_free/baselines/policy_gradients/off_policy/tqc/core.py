from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import get_per_weights, quantile_huber_loss
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.utils.common_utils import to_scalar, to_column


class TQCCore(ActorCriticCore):
    """
    TQC Update Engine (TQCCore) built on ActorCriticCore infrastructure.

    What this core does
    -------------------
    Implements Truncated Quantile Critics (TQC):
      - Critic learns a distribution over returns via quantiles Z(s,a)
      - Target distribution is built from target critic quantiles at next state/action
      - Truncation: drop the largest `top_quantiles_to_drop` quantiles after sorting
        (reduces overestimation; core idea of TQC)

    Head contract (duck-typed)
    --------------------------
    Required:
      - head.actor: nn.Module
      - head.critic: nn.Module              returns quantiles (B,C,N)
      - head.critic_target: nn.Module       returns quantiles (B,C,N)
      - head.device
      - head.sample_action_and_logp(obs) -> (a, logp)  where logp is (B,1) preferred

    Notes
    -----
    - Actor/Critic optimizers+schedulers + update_calls persistence are handled by ActorCriticCore.
    - This core additionally owns:
        * log_alpha (+ alpha optimizer/scheduler if auto_alpha=True)
        * truncation setting, gamma/tau, target_entropy
    """

    def __init__(
        self,
        *,
        head: Any,
        # core hparams
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        top_quantiles_to_drop: int = 2,
        # entropy / alpha
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
        # alpha optim/sched (core-owned)
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
    ) -> None:
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        tau : float
            Polyak averaging coefficient for target updates.
        target_update_interval : int
            How often to update the target critic (in update calls).
        top_quantiles_to_drop : int
            Number of largest quantiles to drop after sorting flattened quantiles
            from the target critic. Typical: 2~5 depending on C*N.
        auto_alpha : bool
            Whether to learn entropy temperature alpha (log-space parameterization).
        alpha_init : float
            Initial alpha value (temperature), used as exp(log_alpha_init).
        target_entropy : Optional[float]
            SAC-style entropy target. If None, defaults to -action_dim.
        """
        super().__init__(
            head=head,
            use_amp=use_amp,
            # optim (ActorCriticCore builds actor_opt/critic_opt)
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            # sched (ActorCriticCore builds actor_sched/critic_sched)
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

        # ----- core hyperparameters -----
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.top_quantiles_to_drop = int(top_quantiles_to_drop)

        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)

        # ----- target entropy default -----
        # SAC heuristic: target_entropy = -|A|
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # ----- alpha parameter (log-space) -----
        # Use math.log for clarity; th.log(th.tensor(...)) also works but is indirect.
        log_alpha_init = float(th.log(th.tensor(float(alpha_init))).item())
        self.log_alpha = th.tensor(
            log_alpha_init,
            device=self.device,
            requires_grad=bool(self.auto_alpha),
        )

        # ----- alpha optimizer/scheduler (core-owned) -----
        # ActorCriticCore does NOT manage alpha; we do it here.
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

        # Core owns freeze responsibility:
        # target critic should be eval + requires_grad=False to avoid accidental grads.
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """Entropy temperature alpha = exp(log_alpha)."""
        return self.log_alpha.exp()

    # =============================================================================
    # Truncation helper
    # =============================================================================
    @staticmethod
    def _truncate_quantiles(z: th.Tensor, top_drop: int) -> th.Tensor:
        """
        Truncate the target quantile set by dropping the top `top_drop` quantiles.

        Parameters
        ----------
        z : torch.Tensor
            Quantiles with shape (B, C, N).
              B=batch, C=ensemble size, N=quantiles per net
        top_drop : int
            Number of highest quantiles to drop after sorting.

        Returns
        -------
        z_trunc : torch.Tensor
            Truncated and sorted quantiles with shape (B, K),
            where K = C*N - top_drop.

        Implementation details
        ----------------------
        1) Flatten ensemble+quantiles: (B, C*N)
        2) Sort ascending
        3) Keep only the lowest K elements (drop the largest top_drop)
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z to be (B,C,N), got {tuple(z.shape)}")

        b, c, n = z.shape
        flat = z.reshape(b, c * n)              # (B, C*N)
        flat_sorted, _ = th.sort(flat, dim=1)   # ascending sort by quantile value

        drop = int(top_drop)
        if drop < 0 or drop >= c * n:
            raise ValueError(f"top_drop must be in [0, {c*n-1}], got {drop}")

        if drop == 0:
            return flat_sorted
        return flat_sorted[:, : (c * n - drop)]  # keep lower quantiles only

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one gradient update from a replay batch.

        Batch contract
        --------------
        batch must provide (tensors):
          - observations:      (B, obs_dim)
          - actions:           (B, action_dim)
          - rewards:           (B,) or (B,1)
          - next_observations: (B, obs_dim)
          - dones:             (B,) or (B,1)
          - (optional) PER weights via get_per_weights(batch, ...)

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar logs + PER td_errors vector.
        """
        self._bump()  # increments update_calls, handles AMP bookkeeping, etc.

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = to_column(batch.rewards.to(self.device))   # enforce (B,1)
        nxt = batch.next_observations.to(self.device)
        done = to_column(batch.dones.to(self.device))    # enforce (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ---------------------------------------------------------------------
        # Target truncated distribution:
        #   1) sample next action from current actor
        #   2) get target critic quantiles Z_t(s', a')
        #   3) truncate: drop top quantiles across ensemble
        #   4) Bellman backup per kept quantile:
        #        y = r + gamma(1-d) * ( z_trunc - alpha * logpi(a'|s') )
        # ---------------------------------------------------------------------
        with th.no_grad():
            next_a, next_logp = self.head.sample_action_and_logp(nxt)
            if next_logp.dim() == 1:
                next_logp = next_logp.unsqueeze(1)  # enforce (B,1)

            z_next = self.head.quantiles_target(nxt, next_a)  # (B,C,N)
            z_trunc = self._truncate_quantiles(z_next, self.top_quantiles_to_drop)  # (B,K)

            # Bellman backup on each retained quantile (distributional target)
            target = rew + self.gamma * (1.0 - done) * (z_trunc - self.alpha * next_logp)  # (B,K)

            # quantile_huber_loss commonly expects target shaped like current:
            # current: (B,C,N), target: (B,C,K) (K may differ due to truncation)
            target_quantiles = target.unsqueeze(1).expand(-1, z_next.shape[1], -1)  # (B,C,K)

        # ---------------------------------------------------------------------
        # Critic update: distributional regression via quantile Huber loss
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            current = self.head.quantiles(obs, act)  # (B,C,N)

            # quantile_huber_loss should:
            # - compute loss across quantile pairs between current and target
            # - optionally apply PER weights
            loss, td_err = quantile_huber_loss(
                current_quantiles=current,
                target_quantiles=target_quantiles,
                cum_prob=None,               # your util may infer taus internally
                weights=w,                   # (B,1) or None
            )

            # Build td_abs as PER feedback vector shape (B,).
            # td_err format depends on your helper; handle typical cases robustly.
            if isinstance(td_err, th.Tensor):
                if td_err.dim() == 1:
                    td_abs = td_err.detach()
                elif td_err.dim() == 2 and td_err.shape[1] == 1:
                    td_abs = td_err.detach().squeeze(1)
                else:
                    td_abs = td_err.detach().mean(dim=tuple(range(1, td_err.dim())))
            else:
                # Fallback: proxy TD using conservative scalar Q estimates.
                q_cur = current.mean(dim=-1).min(dim=1).values       # (B,)
                q_tgt = target_quantiles.mean(dim=-1).min(dim=1).values  # (B,)
                td_abs = (q_cur - q_tgt).abs().detach()

            return loss, td_abs

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
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
        # Actor update (SAC-style):
        #   J_pi = E[ alpha * logpi(a|s) - Q(s,a) ]
        # where Q(s,a) is formed conservatively from quantiles:
        #   - mean over N quantiles per critic -> (B,C)
        #   - min over C critics -> (B,1)
        # ---------------------------------------------------------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            new_a, logp = self.head.sample_action_and_logp(obs)
            if logp.dim() == 1:
                logp = logp.unsqueeze(1)  # (B,1)

            z = self.head.quantiles(obs, new_a)               # (B,C,N)
            q_c = z.mean(dim=-1)                           # (B,C)  mean over quantiles
            q_min = th.min(q_c, dim=1).values.unsqueeze(1) # (B,1)  conservative over ensemble

            loss = (self.alpha * logp - q_min).mean()
            return loss, logp, q_min

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, logp, q_pi = _actor_loss()
            self.scaler.scale(actor_loss).backward()
            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss, logp, q_pi = _actor_loss()
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Alpha update (optional)
        #
        # You are following the SAC continuous convention:
        #   alpha_loss = -E[ log_alpha * (logp + target_entropy) ]
        # so that when entropy is above target, alpha decreases, and vice versa.
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            if self.alpha_sched is not None:
                self.alpha_sched.step()
            alpha_loss_val = to_scalar(alpha_loss)

        # ---------------------------------------------------------------------
        # Target critic update: critic_target <- critic
        # (core helper handles hard/soft schedule + freezing)
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ---------------------------------------------------------------------
        # Logging stats
        # ---------------------------------------------------------------------
        with th.no_grad():
            z_cur = self.head.critic(obs, act)  # (B,C,N)
            q_mean = z_cur.mean()               # scalar (distribution mean proxy)

        out: Dict[str, Any] = {
            "loss/critic": to_scalar(critic_loss),
            "loss/actor": to_scalar(actor_loss),
            "loss/alpha": float(alpha_loss_val),
            "alpha": to_scalar(self.alpha),
            "q/quantile_mean": to_scalar(q_mean),
            "q/pi_min_mean": to_scalar(q_pi.mean()),
            "logp_mean": to_scalar(logp.mean()),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "tqc/target_updated": float(
                1.0
                if (self.target_update_interval > 0 and self.update_calls % self.target_update_interval == 0)
                else 0.0
            ),
            # PER feedback vector: expected shape (B,)
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
        ActorCriticCore already saves:
          - update_calls
          - actor optimizer/scheduler state
          - critic optimizer/scheduler state

        We add:
          - log_alpha
          - alpha optimizer/scheduler (if enabled)
          - core hyperparameters (useful for reproducibility/debug)
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "top_quantiles_to_drop": int(self.top_quantiles_to_drop),
                "target_entropy": float(self.target_entropy),
                "max_grad_norm": float(self.max_grad_norm),
                "auto_alpha": bool(self.auto_alpha),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state:
          - inherited actor/critic opt/sched + update_calls
          - log_alpha
          - alpha opt/sched (if enabled)
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
