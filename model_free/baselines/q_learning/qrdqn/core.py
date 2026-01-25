# qrdqn_core.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th

from model_free.common.utils.common_utils import to_column, to_scalar
from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.policy_utils import get_per_weights, quantile_huber_loss


class QRDQNCore(QLearningCore):
    """
    QR-DQN Update Engine (Discrete), refactored to reuse QLearningCore.

    Overview
    --------
    QR-DQN is a distributional RL method that learns a return distribution per action.
    The critic outputs N quantiles per action:
        Z(s, a) = [z_1, ..., z_N]  where each z_i is a quantile estimate.

    This core implements the QR-DQN temporal-difference update:
      1) Build the distributional Bellman target using the target network.
      2) Regress online quantiles for the taken action toward the target quantiles
         using a quantile Huber loss (a.k.a. quantile regression with Huber smoothing).
      3) Optionally use Double DQN action selection to reduce overestimation.
      4) Periodically update the target network using hard/soft updates.

    Reuses from QLearningCore
    ------------------------
    - self.head, self.device
    - update bookkeeping: _bump() / update_calls
    - optimizer + scheduler for Q-net: self.opt / self.sched
    - AMP scaler (if enabled): self.scaler
    - gradient clipping helper: _clip_params(...)
    - target update helper: _maybe_update_target(...)

    Expected head (duck-typed)
    --------------------------
    Required:
      - head.q: nn.Module
          forward(q(obs)) -> quantiles of shape (B, N, A)
      - head.q_target: nn.Module
          forward(q_target(obs)) -> quantiles of shape (B, N, A)
      - head.device: torch.device

    Batch contract (duck-typed)
    ---------------------------
    batch must provide tensors:
      - observations:      (B, obs_dim)
      - actions:           (B,) or (B,1)  (discrete action indices)
      - rewards:           (B,) or (B,1)
      - next_observations: (B, obs_dim)
      - dones:             (B,) or (B,1)
      - optional PER weights (via get_per_weights)

    Notes on PER
    ------------
    - PER weights are read as w = get_per_weights(batch, B) -> (B,1) or None.
    - We export "per/td_errors" as a proxy priority signal. For distributional loss,
      this is inherently heuristic; we use the td_error returned by quantile_huber_loss
      as best-effort.
    """

    def __init__(
        self,
        *,
        head: Any,
        # TD / target update
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,  # tau>0 => soft update at interval, else hard update
        double_dqn: bool = True,
        # grad / amp
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # PER epsilon for priorities output (core-side proxy only)
        per_eps: float = 1e-6,
        # optimizer/scheduler (QLearningCore)
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
        """
        Parameters
        ----------
        gamma : float
            Discount factor in [0, 1).
        target_update_interval : int
            How often to update q_target from q, measured in *update calls*.
            - 0: never update target
            - 1: update every update call
        tau : float
            Polyak coefficient for soft updates:
              target <- (1 - tau)*target + tau*source
            If tau == 0, the helper typically performs a hard copy at interval.
        double_dqn : bool
            If True, use Double DQN action selection:
              a* = argmax_a E[Z_online(s',a)]
              target uses Z_target(s', a*)
        max_grad_norm : float
            Gradient clipping norm. 0 disables clipping.
        use_amp : bool
            Enable torch.cuda.amp for mixed-precision (CUDA only).
        per_eps : float
            Small epsilon used when exporting PER td_errors to avoid zeros.
        """
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

        # --- core hyperparameters ---
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)

        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # --- sanity checks ---
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
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one QR-DQN update from a replay batch.

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar logs plus PER feedback vector under "per/td_errors" (B,).
        """
        self._bump()

        # ------------------------------------------------------------------
        # Move batch to device and normalize shapes.
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)                 # (B, obs_dim)
        act = batch.actions.to(self.device).long()               # (B,) or (B,1) -> action indices
        rew = to_column(batch.rewards.to(self.device))           # (B,1)
        nxt = batch.next_observations.to(self.device)            # (B, obs_dim)
        done = to_column(batch.dones.to(self.device))            # (B,1)

        B = int(obs.shape[0])
        w = get_per_weights(batch, B, device=self.device)        # (B,1) or None

        # ------------------------------------------------------------------
        # Current quantiles for the *taken* action.
        #
        # head.quantiles(obs): (B, N, A)
        # gather on action dim -> (B, N, 1) -> squeeze -> (B, N)
        # ------------------------------------------------------------------
        cur_all = self.head.quantiles(obs)                       # (B, N, A)
        if cur_all.dim() != 3:
            raise ValueError(f"head.quantiles(obs) must be (B,N,A), got {tuple(cur_all.shape)}")

        N = int(cur_all.shape[1])
        cur = cur_all.gather(
            2, act.view(-1, 1, 1).expand(-1, N, 1)
        ).squeeze(-1)                                            # (B, N)

        # ------------------------------------------------------------------
        # Build target quantiles (no grad).
        #
        # Target distribution:
        #   Z_target(s', a*) where a* is chosen by:
        #     - Double DQN: argmax_a Q_online_mean(s',a)
        #     - Vanilla:    argmax_a Q_target_mean(s',a)
        #
        # Bellman backup per quantile:
        #   y = r + gamma * (1 - done) * Z_target(s', a*)
        # ------------------------------------------------------------------
        with th.no_grad():
            nxt_t_all = self.head.quantiles_target(nxt)          # (B, N, A)
            if nxt_t_all.dim() != 3:
                raise ValueError(
                    f"head.quantiles_target(nxt) must be (B,N,A), got {tuple(nxt_t_all.shape)}"
                )
            if int(nxt_t_all.shape[1]) != N:
                raise ValueError(
                    f"Quantile count mismatch: online N={N} vs target N={int(nxt_t_all.shape[1])}"
                )

            if self.double_dqn:
                # a* = argmax_a E[Z_online(s',a)]
                # Let head handle "mean over quantiles" via q_values(...)
                nxt_q_mean_online = self.head.q_values(nxt)      # (B, A)
                a_star = th.argmax(nxt_q_mean_online, dim=-1)    # (B,)
            else:
                # a* = argmax_a E[Z_target(s',a)]
                nxt_q_mean_target = self.head.q_values_target(nxt)  # (B, A)
                a_star = th.argmax(nxt_q_mean_target, dim=-1)       # (B,)

            # Select target quantiles for greedy action a*:
            nxt_t = nxt_t_all.gather(
                2, a_star.view(-1, 1, 1).expand(-1, N, 1)
            ).squeeze(-1)                                        # (B, N)

            # Bellman target per quantile (broadcast rew/done to (B,N))
            target = rew.expand(-1, N) + self.gamma * (1.0 - done.expand(-1, N)) * nxt_t  # (B, N)

        # ------------------------------------------------------------------
        # Quantile regression loss.
        # ------------------------------------------------------------------
        loss, td_error = quantile_huber_loss(
            current_quantiles=cur,          # (B, N)
            target_quantiles=target,        # (B, N)
            cum_prob=None,                  # let helper create tau midpoints
            weights=w,                      # (B,1) or None
        )

        # ------------------------------------------------------------------
        # Optimizer step (optionally AMP).
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
        # Target update (generic helper).
        #
        # NOTE:
        # - We still pass `source=self.head.q` because the actual module being trained
        #   is the online quantile network stored in head.q.
        # - If your framework prefers updating via head methods (hard_update/soft_update),
        #   keep using _maybe_update_target as you already do.
        # ------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ------------------------------------------------------------------
        # Metrics + PER td_errors proxy
        # ------------------------------------------------------------------
        with th.no_grad():
            q_mean_taken = cur.mean(dim=1)  # (B,)

            td = td_error.detach()
            if td.dim() > 1:
                td = td.view(-1)
            td_abs = td.abs().clamp(min=self.per_eps)

        return {
            "loss/q": float(to_scalar(loss)),
            "q/mean": float(to_scalar(q_mean_taken.mean())),
            "target/mean": float(to_scalar(target.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend QLearningCore.state_dict with QR-DQN-specific hyperparameters.

        Note: optimizer/scheduler/update_calls are already handled by QLearningCore.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "double_dqn": bool(self.double_dqn),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore inherited state (optimizer/scheduler/update_calls).

        Hyperparameters are constructor-owned by design, so we do not override them
        from the checkpoint silently.
        """
        super().load_state_dict(state)
        return
