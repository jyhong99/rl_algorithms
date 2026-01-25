from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import to_scalar, to_column
from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.policy_utils import get_per_weights


class DQNCore(QLearningCore):
    """
    DQN Update Engine (DQNCore)

    Summary
    -------
    Implements the standard DQN / Double DQN update on top of the shared
    `QLearningCore` infrastructure (optimizer/scheduler/AMP/target updates).

    What this core owns
    -------------------
    - TD hyperparameters: gamma, target update schedule (interval/tau)
    - Variant toggles: Double DQN, Huber vs MSE
    - Gradient clipping and PER-related logging

    What `QLearningCore` provides (inherited)
    ----------------------------------------
    - `self.opt` / `self.sched` (optimizer + optional scheduler for Q network)
    - `self.device` (resolved from head)
    - AMP scaler + update counter via `_bump()`
    - `_clip_params()` helper for gradient norm clipping
    - `_maybe_update_target()` helper for periodic hard/soft target updates

    Expected head (duck-typed)
    --------------------------
    - head.q: nn.Module
        Online Q-network. Forward: q(obs) -> (B, A).
    - head.q_target: nn.Module
        Target Q-network. Forward: q_target(obs) -> (B, A).
    - head.device: torch.device
        Device used by the head (and thus by the core).
    - optional: head.hard_update / head.soft_update / head.freeze_target
        Common utilities exposed by a BaseHead implementation.
    """

    def __init__(
        self,
        *,
        head: Any,
        # TD / target update
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        # variants / loss
        double_dqn: bool = True,
        huber: bool = True,
        # grad / amp
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # PER epsilon for priorities output (core-side proxy only)
        per_eps: float = 1e-6,
        # optimizer/scheduler (inherited)
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
        # scheduler shared knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        # Delegate optimizer/scheduler/AMP wiring to QLearningCore.
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

        # Store TD / target-update hyperparameters.
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)

        # Algorithm variants.
        self.double_dqn = bool(double_dqn)
        self.huber = bool(huber)

        # Training stability knobs.
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # Basic validation (fail fast for invalid configs).
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        self.head.freeze_target(self.head.q_target)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one gradient update using a replay batch.

        Batch contract
        --------------
        batch must provide (tensors):
          - observations:      (B, obs_dim)
          - actions:           (B,) or (B,1) (discrete action indices)
          - rewards:           (B,) or (B,1)
          - next_observations: (B, obs_dim)
          - dones:             (B,) or (B,1)
          - (optional) PER weights via `get_per_weights(batch, ...)`

        Returns
        -------
        metrics : Dict[str, float] (plus a numpy vector for PER td_errors)
            Logging scalars and PER priority feedback.
        """
        # Bump internal update counter; manages AMP bookkeeping in base core.
        self._bump()

        # Move batch to device and normalize shapes.
        obs = batch.observations.to(self.device)                 # (B, obs_dim)
        act = batch.actions.to(self.device).long()               # (B,) or (B,1)
        rew = to_column(batch.rewards.to(self.device))           # -> (B,1)
        next_obs = batch.next_observations.to(self.device)       # (B, obs_dim)
        done = to_column(batch.dones.to(self.device))            # -> (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)

        # ------------------------------------------------------------------
        # Online estimate: Q(s,a)
        # ------------------------------------------------------------------
        # Compute all action values Q(s, ·) then gather Q(s,a) for chosen actions.
        q_all = self.head.q_values(obs)                           # (B, A)
        q_sa = q_all.gather(1, act.view(-1, 1))                   # (B, 1)

        # ------------------------------------------------------------------
        # Bootstrapped target:
        #   y = r + gamma * (1-done) * Q_target(s', a*)
        #
        # Double DQN:
        #   a* = argmax_a Q_online(s',a)
        #   Q_target evaluated only at that action
        # ------------------------------------------------------------------
        with th.no_grad():
            q_next_target_all = self.head.q_values_target(next_obs)      # (B, A)

            if self.double_dqn:
                # Action selection from online net (reduces overestimation bias).
                a_star = th.argmax(self.head.q_values(next_obs), dim=-1, keepdim=True)  # (B,1)
                q_next = q_next_target_all.gather(1, a_star)                     # (B,1)
            else:
                # Vanilla DQN target: max over target net directly.
                q_next = q_next_target_all.max(dim=1, keepdim=True).values       # (B,1)

            target = rew + self.gamma * (1.0 - done) * q_next     # (B, 1)

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        # Compute elementwise TD loss to support PER weighting.
        if self.huber:
            # Smooth L1 is typically more robust to outliers than MSE.
            per_sample = F.smooth_l1_loss(q_sa, target, reduction="none")        # (B,1)
        else:
            per_sample = F.mse_loss(q_sa, target, reduction="none")              # (B,1)

        # If PER is enabled, weight each sample by its importance weight.
        loss = per_sample.mean() if w is None else (per_sample * w).mean()

        # ------------------------------------------------------------------
        # Optim step (online Q network only)
        # ------------------------------------------------------------------
        self.opt.zero_grad(set_to_none=True)

        if self.use_amp:
            # AMP branch: scale gradients and step via GradScaler.
            with th.cuda.amp.autocast(enabled=True):
                loss_amp = loss
            self.scaler.scale(loss_amp).backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            # FP32 branch.
            loss.backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm)
            self.opt.step()

        # Step scheduler after optimizer step (if configured).
        if self.sched is not None:
            self.sched.step()

        # ------------------------------------------------------------------
        # PER priorities (TD error proxy)
        # ------------------------------------------------------------------
        # Provide per-sample TD error magnitude as priority signal.
        with th.no_grad():
            td_error = (target - q_sa).abs().view(-1)             # (B,)

        # ------------------------------------------------------------------
        # Target update
        # ------------------------------------------------------------------
        # Uses base helper which can implement:
        # - hard updates every `interval` steps (tau=0 or explicit hard copy)
        # - soft/Polyak updates when tau>0
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        return {
            "loss/q": float(to_scalar(loss)),
            "q/mean": float(to_scalar(q_sa.mean())),
            "target/mean": float(to_scalar(target.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            # Numpy vector used by PER buffer to update priorities.
            "per/td_errors": td_error.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend base core state with DQN-specific hyperparameters.

        Base `QLearningCore.state_dict()` typically includes:
          - update_calls
          - optimizer state (+ scheduler state if present)

        This core adds hyperparameters for reproducibility/debugging.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "double_dqn": bool(self.double_dqn),
                "huber": bool(self.huber),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base core state (optimizer/scheduler/update counters).

        Note
        ----
        Hyperparameters are constructor-owned. We intentionally do not override
        them from the checkpoint to avoid silently changing runtime behavior.
        """
        super().load_state_dict(state)
        return
