from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th

from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.utils.policy_utils import _get_per_weights, _quantile_huber_loss


class QRDQNCore(QLearningCore):
    """
    Quantile Regression DQN (QR-DQN) update engine for discrete action spaces.

    QR-DQN is a distributional RL algorithm that learns an approximation of the
    return distribution ``Z(s,a)`` rather than only the expectation ``Q(s,a)``.
    The distribution is represented by a fixed set of ``N`` quantile locations
    per action.

    Core responsibilities
    ---------------------
    For each gradient update call, this core:

    1. Computes **online** quantiles for the taken action:
       ``Z_θ(s,a)`` with shape ``(B, N)``.
    2. Computes **target** quantiles for the greedy next action:
       ``Z̄(s', a*)`` with shape ``(B, N)``.
    3. Builds the distributional Bellman target (per-quantile backup):
       ``y = r + γ (1-done) Z̄(s', a*)``.
    4. Regresses online quantiles toward targets using the quantile Huber loss.
    5. Applies an optimizer step (optionally AMP) and steps an LR scheduler.
    6. Updates the target network periodically (hard/soft updates).

    What is reused from :class:`~model_free.common.policies.base_core.QLearningCore`
    -------------------------------------------------------------------------------
    - ``self.head`` and ``self.device`` resolution
    - update bookkeeping via ``_bump()`` and ``update_calls``
    - optimizer / scheduler wiring: ``self.opt`` and ``self.sched``
    - AMP scaler: ``self.scaler`` (when ``use_amp=True``)
    - gradient clipping helper: ``_clip_params(...)``
    - target update helper: ``_maybe_update_target(...)``

    Expected head (duck-typed)
    --------------------------
    Required attributes/methods on ``head``:

    - ``head.q`` : torch.nn.Module
        Online quantile network. Forward output shape: ``(B, N, A)``.
    - ``head.q_target`` : torch.nn.Module
        Target quantile network. Forward output shape: ``(B, N, A)``.
    - ``head.quantiles(obs)`` : callable
        Returns online quantiles ``(B, N, A)``.
    - ``head.quantiles_target(obs)`` : callable
        Returns target quantiles ``(B, N, A)`` (no-grad recommended).
    - ``head.q_values(obs)`` : callable
        Returns expected Q-values ``(B, A)`` from online quantiles (mean over N).
    - ``head.q_values_target(obs)`` : callable
        Returns expected Q-values ``(B, A)`` from target quantiles.
    - optional ``head.freeze_target(module)`` : callable
        Freezes ``q_target`` (no gradients) and enforces eval mode.

    Batch contract (duck-typed)
    ---------------------------
    ``batch`` must provide torch tensors:
    - ``batch.observations``      : ``(B, obs_dim)``
    - ``batch.actions``           : ``(B,)`` or ``(B,1)``  (discrete action indices)
    - ``batch.rewards``           : ``(B,)`` or ``(B,1)``
    - ``batch.next_observations`` : ``(B, obs_dim)``
    - ``batch.dones``             : ``(B,)`` or ``(B,1)``
    - optional PER fields consumed by ``_get_per_weights``

    Notes on PER integration
    ------------------------
    - PER weights are fetched via ``_get_per_weights(batch, B, device=...)`` and are
      expected to be shaped ``(B,1)`` or ``None``.
    - We export ``"per/td_errors"`` as a priority proxy. For distributional methods
      there is no single canonical TD error; we use the auxiliary error returned by
      ``_quantile_huber_loss`` and take its absolute value as best-effort.

    Parameters
    ----------
    head : Any
        Policy head providing online/target quantile networks and helper methods.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    target_update_interval : int, default=1000
        Target update period measured in *update calls*.
        - ``0`` disables target updates.
        - ``1`` updates every update.
    tau : float, default=0.0
        Soft-update coefficient in ``[0, 1]`` used by the target update helper.
        Typical semantics:
        - ``tau <= 0`` => hard update at interval
        - ``tau > 0``  => Polyak/soft update at interval
    double_dqn : bool, default=True
        If True, uses Double DQN action selection:
        - select ``a*`` using online expected Q-values
        - evaluate target quantiles using target network at ``a*``
    max_grad_norm : float, default=0.0
        Global norm clipping threshold (``0`` disables clipping).
    use_amp : bool, default=False
        Enable torch.cuda.amp mixed precision (CUDA only).
    per_eps : float, default=1e-6
        Minimum clamp for exported PER priority signal to avoid zeros.
    optim_name : str, default="adamw"
        Optimizer name passed to QLearningCore's optimizer builder.
    lr : float, default=3e-4
        Learning rate.
    weight_decay : float, default=0.0
        Weight decay.
    sched_name : str, default="none"
        Scheduler name passed to QLearningCore's scheduler builder.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration knobs passed through to QLearningCore.

    Raises
    ------
    ValueError
        If hyperparameters are out of valid ranges (e.g., ``gamma`` not in ``[0,1)``).
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        double_dqn: bool = True,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        per_eps: float = 1e-6,
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
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
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)

        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

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

        # Convention: freeze target network for deterministic behavior and to prevent grads.
        self.head.freeze_target(self.head.q_target)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one QR-DQN update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch satisfying the "Batch contract" described in the class
            docstring.

        Returns
        -------
        metrics : Dict[str, Any]
            Dictionary of training statistics. Includes:
            - ``"loss/q"`` : float
                Scalar quantile regression loss.
            - ``"q/mean"`` : float
                Mean expected Q-value for taken actions (mean over quantiles).
            - ``"target/mean"`` : float
                Mean Bellman target (mean over batch and quantiles).
            - ``"lr"`` : float
                Current learning rate from optimizer param group 0.
            - ``"per/td_errors"`` : np.ndarray, shape (B,)
                Priority proxy vector suitable for PER buffers.
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch to device and normalize shapes.
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)                  # (B, obs_dim)
        act = batch.actions.to(self.device).long()                # (B,) or (B,1)
        rew = _to_column(batch.rewards.to(self.device))           # (B,1)
        nxt = batch.next_observations.to(self.device)             # (B, obs_dim)
        done = _to_column(batch.dones.to(self.device))            # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)        # (B,1) or None

        # ---------------------------------------------------------------------
        # Current quantiles for the taken action.
        #
        # cur_all: (B, N, A)
        # cur:     (B, N) quantiles corresponding to chosen action a_t
        # ---------------------------------------------------------------------
        cur_all = self.head.quantiles(obs)                        # (B, N, A)
        if cur_all.dim() != 3:
            raise ValueError(f"head.quantiles(obs) must be (B,N,A), got {tuple(cur_all.shape)}")

        N = int(cur_all.shape[1])
        cur = (
            cur_all.gather(2, act.view(-1, 1, 1).expand(-1, N, 1))
            .squeeze(-1)
        )                                                         # (B, N)

        # ---------------------------------------------------------------------
        # Build target quantiles (no-grad).
        #
        # - Compute target quantiles for next state: nxt_t_all = (B, N, A)
        # - Choose greedy next action a* (Double DQN optional)
        # - Select target quantiles nxt_t = Z_target(s', a*) -> (B, N)
        # - Bellman backup per quantile:
        #     y = r + gamma * (1 - done) * nxt_t
        # ---------------------------------------------------------------------
        with th.no_grad():
            nxt_t_all = self.head.quantiles_target(nxt)           # (B, N, A)
            if nxt_t_all.dim() != 3:
                raise ValueError(
                    f"head.quantiles_target(nxt) must be (B,N,A), got {tuple(nxt_t_all.shape)}"
                )
            if int(nxt_t_all.shape[1]) != N:
                raise ValueError(
                    f"Quantile count mismatch: online N={N} vs target N={int(nxt_t_all.shape[1])}"
                )

            if self.double_dqn:
                # Double DQN: action selection by online expected Q-values.
                nxt_q_mean_online = self.head.q_values(nxt)       # (B, A)
                a_star = th.argmax(nxt_q_mean_online, dim=-1)     # (B,)
            else:
                # Vanilla: action selection by target expected Q-values.
                nxt_q_mean_target = self.head.q_values_target(nxt)  # (B, A)
                a_star = th.argmax(nxt_q_mean_target, dim=-1)       # (B,)

            nxt_t = (
                nxt_t_all.gather(2, a_star.view(-1, 1, 1).expand(-1, N, 1))
                .squeeze(-1)
            )                                                      # (B, N)

            target = (
                rew.expand(-1, N)
                + self.gamma * (1.0 - done.expand(-1, N)) * nxt_t
            )                                                      # (B, N)

        # ---------------------------------------------------------------------
        # Quantile regression loss (optionally PER-weighted).
        # ---------------------------------------------------------------------
        loss, td_error = _quantile_huber_loss(
            current_quantiles=cur,          # (B, N)
            target_quantiles=target,        # (B, N)
            cum_prob=None,                  # helper creates tau midpoints
            weights=w,                      # (B,1) or None
        )

        # ---------------------------------------------------------------------
        # Optimizer step (online net only) with optional AMP.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Target update (hard or soft) using the shared helper.
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ---------------------------------------------------------------------
        # Metrics + PER priority proxy
        # ---------------------------------------------------------------------
        with th.no_grad():
            q_mean_taken = cur.mean(dim=1)  # (B,)

            td = td_error.detach()
            if td.dim() > 1:
                td = td.view(-1)
            td_abs = td.abs().clamp(min=self.per_eps)

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_mean_taken.mean())),
            "target/mean": float(_to_scalar(target.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state, extending :meth:`QLearningCore.state_dict`.

        The base :class:`QLearningCore` state typically includes:
        - update counters (e.g., ``update_calls``)
        - optimizer state
        - scheduler state (if present)

        This method adds QR-DQN-specific hyperparameters for reproducibility and
        debugging. Hyperparameters are still constructor-owned on restore.

        Returns
        -------
        state : Dict[str, Any]
            Serializable state dictionary.
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
        Restore inherited state (optimizer/scheduler/update counters).

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.

        Notes
        -----
        Hyperparameters are intentionally **not** overridden from the checkpoint.
        They are treated as constructor-owned to avoid silently changing runtime
        behavior when resuming.
        """
        super().load_state_dict(state)
        return
