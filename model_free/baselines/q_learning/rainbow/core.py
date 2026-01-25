from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th

from model_free.common.utils.common_utils import to_column, to_scalar
from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.policy_utils import get_per_weights, distribution_projection


class RainbowCore(QLearningCore):
    """
    Rainbow (C51) update engine (discrete) refactored to reuse QLearningCore.

    What this core does
    -------------------
    Implements the distributional Bellman update used in Rainbow's C51:
      1) predict the online categorical distribution p_theta(z | s, a) for taken action a
      2) build a projected target categorical distribution T_z p_bar(z | s', a*)
      3) minimize cross-entropy between target distribution and current distribution

    Reused from QLearningCore
    -------------------------
    - Optimizer + scheduler initialization: self.opt / self.sched
    - AMP scaler + update counter: self._bump(), self.update_calls
    - Gradient clipping helper: self._clip_params(...)
    - Target update helper (hard/soft): self._maybe_update_target(...)

    Expected head contract (duck-typed)
    -----------------------------------
    Required attributes / methods:
      - head.q: nn.Module
          * head.q(obs) -> expected Q-values, shape (B, A)
          * head.q.dist(obs) -> categorical distribution, shape (B, A, K)
          * head.q.reset_noise() -> None (optional; for NoisyNet)
      - head.q_target: nn.Module
          * head.q_target(obs) -> expected Q-values, shape (B, A)
          * head.q_target.dist(obs) -> categorical distribution, shape (B, A, K)
          * head.q_target.reset_noise() -> None (optional)
      - head.support: torch.Tensor of shape (K,)
      - head.v_min: float
      - head.v_max: float

    Notes
    -----
    - PER weighting: get_per_weights(batch, B) returns (B,1) or None.
      If present, we compute a weighted mean loss.
    - PER priorities proxy: we return per-sample cross-entropy magnitude as "per/td_errors".
      (Strictly speaking, this is not TD error, but serves as a stable priority signal.)
    """

    def __init__(
        self,
        *,
        head: Any,
        # -----------------------------
        # TD / target update
        # -----------------------------
        gamma: float = 0.99,
        n_step: int = 1,
        target_update_interval: int = 1000,
        tau: float = 0.0,          # tau>0 => soft update at interval; else hard copy
        double_dqn: bool = True,   # select a* using online Q, evaluate on target dist
        # -----------------------------
        # Grad / AMP
        # -----------------------------
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # -----------------------------
        # PER proxy epsilon
        # -----------------------------
        per_eps: float = 1e-6,
        # numeric stability for log(p)
        log_eps: float = 1e-6,
        # -----------------------------
        # Optimizer / scheduler (QLearningCore)
        # -----------------------------
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
        # sched shared knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        super().__init__(
            head=head,
            use_amp=use_amp,
            optim_name=optim_name,
            lr=lr,
            weight_decay=weight_decay,
            sched_name=sched_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            poly_power=poly_power,
            step_size=step_size,
            sched_gamma=sched_gamma,
            milestones=milestones,
        )

        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)

        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)
        self.log_eps = float(log_eps)

        # ------------------------------------------------------------------
        # Hyperparameter validation
        # ------------------------------------------------------------------
        if self.n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got: {self.n_step}")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got: {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")
        if self.log_eps <= 0.0:
            raise ValueError(f"log_eps must be > 0, got {self.log_eps}")

        # Freeze target net once at init (core "owns" this responsibility).
        q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            self._freeze_target(q_target)

    # =============================================================================
    # Helpers
    # =============================================================================
    def _get_nstep_fields(self, batch: Any) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Prefer n-step fields when they are present AND not None.
        Fallback to 1-step fields otherwise.

        Returns
        -------
        rewards : torch.Tensor
        dones   : torch.Tensor
        next_obs: torch.Tensor
        """
        # If n_step == 1, don't even try n-step fields.
        if int(self.n_step) <= 1:
            return (
                batch.rewards.to(self.device),
                batch.dones.to(self.device),
                batch.next_observations.to(self.device),
            )

        r = getattr(batch, "n_step_returns", None)
        d = getattr(batch, "n_step_dones", None)
        ns = getattr(batch, "n_step_next_observations", None)

        # Use n-step only if all three are not None.
        if (r is not None) and (d is not None) and (ns is not None):
            # Defensive: ensure they are tensors or tensor-like
            return (
                r.to(self.device),
                d.to(self.device),
                ns.to(self.device),
            )

        # Fallback to 1-step fields
        return (
            batch.rewards.to(self.device),
            batch.dones.to(self.device),
            batch.next_observations.to(self.device),
        )

    def _gamma_n(self) -> float:
        """
        Compute gamma^n for the Bellman backup.

        Important:
        - If the replay buffer already computed n-step returns for `n_step`,
          the discount exponent must match that same n.
        """
        return float(self.gamma) ** max(1, int(self.n_step))

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one Rainbow(C51) update from a replay batch.

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar logs plus PER feedback vector "per/td_errors" of shape (B,).
        """
        # Bump update counter and AMP bookkeeping (QLearningCore utility).
        self._bump()

        # ------------------------------------------------------------------
        # Move batch to device.
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)       # (B, obs_dim)
        act = batch.actions.to(self.device).long()     # (B,) discrete action indices

        # Potentially use n-step targets if replay provides them.
        rew, done, nxt = self._get_nstep_fields(batch)
        rew = to_column(rew)                           # (B,1)
        done = to_column(done)                         # (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ------------------------------------------------------------------
        # Current distribution for taken action.
        #
        # head.q.dist(obs): (B, A, K)
        # gather along action dim -> (B, 1, K)
        # squeeze -> (B, K)
        # ------------------------------------------------------------------
        dist_all = self.head.q.dist(obs)               # (B, A, K)
        if dist_all.dim() != 3:
            raise ValueError(f"head.q.dist(obs) must be (B,A,K), got {tuple(dist_all.shape)}")

        K = int(dist_all.shape[-1])
        dist_a = dist_all.gather(
            1, act.view(-1, 1, 1).expand(-1, 1, K)
        ).squeeze(1)                                   # (B, K)

        # ------------------------------------------------------------------
        # Target distribution (no grad): select a* then apply C51 projection.
        #
        # a* selection:
        #   - Double DQN: argmax_a Q_online(s', a)
        #   - Otherwise:  argmax_a Q_target(s', a)
        #
        # Then obtain next_dist = p_target(z | s', a*) and project:
        #   Tz p(z|s',a*) onto fixed support [v_min, v_max] with K atoms.
        # ------------------------------------------------------------------
        with th.no_grad():
            if self.double_dqn:
                # Common Rainbow behavior: resample online noise before evaluating greedy action.
                if hasattr(self.head.q, "reset_noise"):
                    self.head.q.reset_noise()
                q_next_online = self.head.q_values(nxt)        # (B, A) expected Q
                a_star = th.argmax(q_next_online, dim=-1)  # (B,)
            else:
                q_next_t = self.head.q_values_target(nxt)      # (B, A)
                a_star = th.argmax(q_next_t, dim=-1)    # (B,)

            next_dist_all = self.head.q_target.dist(nxt)  # (B, A, K)
            if next_dist_all.dim() != 3:
                raise ValueError(
                    f"head.q_target.dist(nxt) must be (B,A,K), got {tuple(next_dist_all.shape)}"
                )
            if int(next_dist_all.shape[-1]) != K:
                raise ValueError(
                    f"Atom count mismatch: online K={K} vs target K={int(next_dist_all.shape[-1])}"
                )

            next_dist = next_dist_all.gather(
                1, a_star.view(-1, 1, 1).expand(-1, 1, K)
            ).squeeze(1)                                 # (B, K)

            target_dist = distribution_projection(
                next_dist=next_dist,                     # (B, K)
                rewards=rew,                             # (B, 1)
                dones=done,                              # (B, 1)
                gamma=self._gamma_n(),                 # scalar gamma^n
                support=self.head.support,               # (K,)
                v_min=float(self.head.v_min),
                v_max=float(self.head.v_max),
            )                                             # (B, K)

        # ------------------------------------------------------------------
        # Cross-entropy loss:
        #   L_i = - sum_k target_dist[i,k] * log(dist_a[i,k])
        #
        # PER weighting:
        #   If weights w are provided, compute weighted mean over batch.
        # ------------------------------------------------------------------
        logp = th.log(dist_a.clamp(min=self.log_eps))      # (B, K)
        per_sample = -(target_dist * logp).sum(dim=-1)     # (B,)

        loss = per_sample.mean() if w is None else (per_sample.view(-1, 1) * w).mean()

        # ------------------------------------------------------------------
        # Optimization step (optionally AMP).
        # ------------------------------------------------------------------
        self.opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                loss_amp = loss
            self.scaler.scale(loss_amp).backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm)
            self.opt.step()

        if self.sched is not None:
            self.sched.step()

        # ------------------------------------------------------------------
        # Target update (hard/soft). The helper should re-freeze after update.
        # ------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ------------------------------------------------------------------
        # Metrics + PER proxy vector.
        #
        # q_taken is expected Q(s,a) for taken action derived from dist_a:
        #   q = sum_k p_k * support_k
        #
        # per/td_errors:
        #   we use |cross_entropy| as a stable per-sample priority proxy.
        # ------------------------------------------------------------------
        with th.no_grad():
            support = self.head.support.to(self.device)
            if support.dim() != 1 or int(support.shape[0]) != K:
                raise ValueError(f"head.support must be (K,), got {tuple(support.shape)} vs K={K}")

            q_taken = (dist_a * support.view(1, -1)).sum(dim=-1)         # (B,)
            td_abs = per_sample.detach().abs().clamp(min=self.per_eps)   # (B,)

        return {
            "loss/q": float(to_scalar(loss)),
            "q/mean": float(to_scalar(q_taken.mean())),
            "target/mean": float(to_scalar(target_dist.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        QLearningCore already includes:
          - update_calls
          - optimizer / scheduler state (under a nested key)
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "n_step": int(self.n_step),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "double_dqn": bool(self.double_dqn),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
                "log_eps": float(self.log_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state (optimizer/scheduler/update counter).

        Note:
        - Hyperparameters are considered constructor-owned and are NOT overwritten.
        """
        super().load_state_dict(state)
        return
