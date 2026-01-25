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


class REDQCore(ActorCriticCore):
    """
    REDQ update engine (core) built on top of ActorCriticCore.

    Role separation (important)
    ---------------------------
    - Head (policy module) owns networks and inference:
        actor, critics ensemble, target critics ensemble, sampling actions/logp, etc.
    - Core owns optimization/update logic:
        losses, optimizers/schedulers, gradient steps, target updates, logging, PER TD-errors.

    Expected head interface (duck-typed)
    ------------------------------------
    Required:
      - head.actor: nn.Module
      - head.critics: nn.ModuleList of Q networks, each Q(obs, act) -> (B,1)
      - head.critics_target: nn.ModuleList of target Q networks, same shape
      - head.sample_action_and_logp(obs) -> (action, logp) where:
            action: (B, action_dim)
            logp:   (B,1) or (B,)
      - head.device: torch.device (or str resolved by base class)
      - head.action_dim: int (used for default entropy target)

    Optional:
      - head.q_values_target_subset_min(obs, act, subset_size=...) -> (B,1)
        (If missing, core provides a fallback subset-min implementation.)
      - head.soft_update_target(tau=...) for convenience.

    Batch contract
    --------------
      - batch.observations:       (B, obs_dim) tensor
      - batch.actions:            (B, action_dim) tensor
      - batch.rewards:            (B,) or (B,1) tensor
      - batch.next_observations:  (B, obs_dim) tensor
      - batch.dones:              (B,) or (B,1) tensor
      - Optional: batch.weights for PER (handled by get_per_weights)

    Notes on REDQ target
    --------------------
    REDQ typically uses:
      target = r + gamma*(1-done)*( min_{i in subset} Q_i^t(s',a') - alpha*logpi(a'|s') )
    where subset is sampled uniformly from the target critic ensemble.
    """

    def __init__(
        self,
        *,
        head: Any,
        # RL hparams
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        # REDQ subset override
        num_target_subset: Optional[int] = None,
        # entropy / temperature
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # optim
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
        # sched
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
        # ------------------------------------------------------------------
        # 1) Build base ActorCriticCore first
        # ------------------------------------------------------------------
        # ActorCriticCore typically:
        # - stores self.head, self.device, self.use_amp, self.scaler, etc.
        # - builds actor_opt/sched and critic_opt/sched on some default critic params
        #
        # REDQ uses an ensemble of critics => we will override critic optimizer/scheduler below.
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

        # ------------------------------------------------------------------
        # 2) REDQ specific: critic optimizer must cover ALL ensemble critics
        # ------------------------------------------------------------------
        # Base class likely assumed a single critic; rebuild optimizer to include all params.
        critic_params = [p for q in self.head.critics for p in q.parameters()]
        self.critic_opt = build_optimizer(
            critic_params,
            name=str(critic_optim_name),
            lr=float(critic_lr),
            weight_decay=float(critic_weight_decay),
        )
        self.critic_sched = build_scheduler(
            self.critic_opt,
            name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        # ------------------------------------------------------------------
        # 3) Store hparams
        # ------------------------------------------------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)

        self.num_target_subset_override: Optional[int] = None
        if num_target_subset is not None:
            self.num_target_subset_override = int(num_target_subset)

        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # ------------------------------------------------------------------
        # 4) Entropy target
        # ------------------------------------------------------------------
        # SAC common default: target_entropy = -|A| (for continuous control).
        # You intentionally used -log(|A|) previously; we preserve that behavior.
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -math.log(float(action_dim))
        else:
            self.target_entropy = float(target_entropy)

        # ------------------------------------------------------------------
        # 5) Temperature (alpha): log_alpha is the optimized parameter
        # ------------------------------------------------------------------
        # We optimize log(alpha) for numerical stability and to keep alpha positive.
        init_log_alpha = float(th.log(th.tensor(float(alpha_init))).item())
        self.log_alpha = th.tensor(
            init_log_alpha,
            device=self.device,
            requires_grad=bool(self.auto_alpha),
        )

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

        # ------------------------------------------------------------------
        # 6) Ensure target critics are frozen (core enforces "no grads into target")
        # ------------------------------------------------------------------
        for q_t in self.head.critics_target:
            self.head.freeze_target(q_t)


    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> th.Tensor:
        """Current temperature alpha = exp(log_alpha)."""
        return self.log_alpha.exp()

    def _subset_size(self) -> int:
        """
        Determine REDQ subset size (m) for subset-min computation.

        Priority:
        1) explicit override in core (num_target_subset_override)
        2) head.num_target_subset
        3) head.cfg.num_target_subset
        4) fallback to 2
        """
        if self.num_target_subset_override is not None:
            return int(self.num_target_subset_override)
        if hasattr(self.head, "num_target_subset"):
            return int(getattr(self.head, "num_target_subset"))
        cfg = getattr(self.head, "cfg", None)
        if cfg is not None and hasattr(cfg, "num_target_subset"):
            return int(getattr(cfg, "num_target_subset"))
        return 2

    @th.no_grad()
    def _subset_min(
        self,
        critics: Sequence[Any],
        obs: th.Tensor,
        act: th.Tensor,
        subset_size: int,
    ) -> th.Tensor:
        """
        Compute min Q over a random subset of critics.

        Parameters
        ----------
        critics : Sequence[Any]
            A list/ModuleList of Q networks, each Q(obs, act)->(B,1).
        obs : torch.Tensor
            Observation tensor of shape (B, obs_dim).
        act : torch.Tensor
            Action tensor of shape (B, action_dim).
        subset_size : int
            Number of critics to sample uniformly without replacement.

        Returns
        -------
        q_min : torch.Tensor
            Shape (B,1), min over sampled subset.

        Notes
        -----
        - Sampling uses torch.randperm on obs.device (usually self.device).
        - Determinism can be achieved by controlling torch RNG seeds.
        """
        n = len(critics)
        m = int(subset_size)
        if m <= 0 or m > n:
            raise ValueError(f"subset_size must be in [1, {n}], got {m}")

        idx = th.randperm(n, device=obs.device)[:m].tolist()
        qs = [critics[i](obs, act) for i in idx]     # list[(B,1)]
        q_stack = th.stack(qs, dim=0)                # (m, B, 1)
        return th.min(q_stack, dim=0).values         # (B,1)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one REDQ update step from a replay batch.

        Steps
        -----
        1) Build REDQ target using target critics subset-min and entropy term.
        2) Update critic ensemble with MSE to target.
        3) Update actor using entropy-regularized objective.
        4) Update alpha (temperature) if auto_alpha enabled.
        5) Soft-update target critics periodically.

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar metrics for logging + PER TD-errors array.
        """
        self._bump()  # increments self.update_calls and other counters in base core

        # -------------------
        # Move batch to device + normalize shapes
        # -------------------
        obs = batch.observations.to(self.device)            # (B, obs_dim)
        act = batch.actions.to(self.device)                 # (B, action_dim)
        rew = to_column(batch.rewards.to(self.device))      # (B,1)
        nxt = batch.next_observations.to(self.device)       # (B, obs_dim)
        done = to_column(batch.dones.to(self.device))       # (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)   # (B,1) or None
        m = self._subset_size()                             # subset size for REDQ

        # -------------------
        # Target computation (no grad)
        # -------------------
        with th.no_grad():
            # a' ~ pi(s') and log pi(a'|s')
            next_a, next_logp = self.head.sample_action_and_logp(nxt)
            if next_logp.dim() == 1:
                next_logp = next_logp.unsqueeze(1)          # ensure (B,1)

            # Prefer head-provided subset-min (keeps logic centralized in head),
            # else fallback to core helper.
            fn = getattr(self.head, "q_values_target_subset_min", None)
            if callable(fn):
                q_min_t = fn(nxt, next_a, subset_size=m)    # (B,1)
            else:
                q_min_t = self._subset_min(self.head.critics_target, nxt, next_a, subset_size=m)

            # REDQ/SAC target: r + gamma*(1-done)*(Q_min_target - alpha*logp)
            target_q = rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * next_logp)  # (B,1)

        # -------------------
        # Critic update (ensemble)
        # -------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute:
            - critic loss: sum_i MSE(Q_i(s,a), target) (optionally PER-weighted)
            - td_abs: |mean_i Q_i(s,a) - target| for PER priority updates
            """
            qs = [q(obs, act) for q in self.head.critics]    # list[(B,1)]

            # Accumulate per-sample loss across ensemble
            per_sample = th.zeros((B, 1), device=self.device)
            for qi in qs:
                per_sample = per_sample + F.mse_loss(qi, target_q, reduction="none")  # (B,1)

            # TD-error based on mean Q (common choice for PER priority)
            q_mean = th.stack(qs, dim=0).mean(dim=0)         # (B,1)
            td = (q_mean - target_q).detach().squeeze(1).abs()  # (B,)

            # PER weighting if available
            if w is None:
                loss = per_sample.mean()
            else:
                loss = (w * per_sample).mean()
            return loss, td

        self.critic_opt.zero_grad(set_to_none=True)
        if self.use_amp:
            # AMP path: forward in autocast, backward via scaler
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()

            # Gradient clipping must see unscaled grads; base helper should handle scaler/optimizer if provided.
            self._clip_params(
                (p for q in self.head.critics for p in q.parameters()),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            # We call scaler.update() once per iteration (after actor step below) to keep your existing pattern.
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(
                (p for q in self.head.critics for p in q.parameters()),
                max_grad_norm=self.max_grad_norm,
            )
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # -------------------
        # Actor update
        # -------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Actor objective (SAC-style):
              maximize E[ Q(s, a~pi) - alpha*logpi(a|s) ]
            => minimize E[ alpha*logpi(a|s) - Q(s,a) ]

            Here Q(s,a) is computed as subset-min over ONLINE critics.
            (Some implementations use mean or a single critic; subset-min is conservative.)
            """
            new_a, logp = self.head.sample_action_and_logp(obs)
            if logp.dim() == 1:
                logp = logp.unsqueeze(1)                    # (B,1)

            q_min = self._subset_min(self.head.critics, obs, new_a, subset_size=m)  # (B,1)
            loss = (self.alpha * logp - q_min).mean()
            return loss, logp, q_min

        self.actor_opt.zero_grad(set_to_none=True)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, logp, q_pi = _actor_loss()
            self.scaler.scale(actor_loss).backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.actor_opt)
            self.scaler.step(self.actor_opt)
            self.scaler.update()  # one update per iter is sufficient
        else:
            actor_loss, logp, q_pi = _actor_loss()
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # -------------------
        # Alpha update (temperature)
        # -------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            # Standard SAC temperature loss:
            #   L(alpha) = - E[ log_alpha * (logpi(a|s) + target_entropy) ]
            # Gradient pushes alpha up if entropy is below target, and down if above.
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            if self.alpha_sched is not None:
                self.alpha_sched.step()
            alpha_loss_val = to_scalar(alpha_loss)

        # -------------------
        # Target update (Polyak) for ALL target critics
        # -------------------
        do_target = (self.target_update_interval > 0) and (self.update_calls % self.target_update_interval == 0)
        if do_target:
            # Prefer head-level helper if available (keeps target update logic centralized)
            fn2 = getattr(self.head, "soft_update_target", None)
            if callable(fn2):
                fn2(tau=self.tau)
            else:
                # Manual Polyak for each pair
                for q_t, q in zip(self.head.critics_target, self.head.critics):
                    self._maybe_update_target(
                        target=q_t,
                        source=q,
                        interval=1,   # already gated by do_target
                        tau=self.tau,
                    )

            for q_t in self.head.critics_target:
                self.head.freeze_target(q_t)

        # -------------------
        # Logging (no grad)
        # -------------------
        with th.no_grad():
            q_all = [q(obs, act) for q in self.head.critics]     # list[(B,1)]
            q_mean_scalar = th.stack(q_all, dim=0).mean()        # scalar tensor

        out: Dict[str, Any] = {
            # losses
            "loss/critic": to_scalar(critic_loss),
            "loss/actor": to_scalar(actor_loss),
            "loss/alpha": float(alpha_loss_val),

            # alpha / entropy
            "alpha": to_scalar(self.alpha),
            "logp_mean": to_scalar(logp.mean()),

            # Q stats
            "q/ensemble_mean": to_scalar(q_mean_scalar),
            "q/pi_min_mean": to_scalar(q_pi.mean()),

            # LR stats
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "redq/target_updated": float(1.0 if do_target else 0.0),

            # PER: return per-sample TD errors for priority update upstream
            "per/td_errors": td_abs.clamp(min=self.per_eps).detach().cpu().numpy(),
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend base core state_dict with alpha-related states.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "auto_alpha": bool(self.auto_alpha),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state including alpha optimizer/scheduler.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
