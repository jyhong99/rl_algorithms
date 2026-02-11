from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.policy_utils import _get_per_weights


class DQNCore(QLearningCore):
    """
    DQN / Double DQN update engine built on top of :class:`QLearningCore`.

    This core implements the standard temporal-difference (TD) update used by
    DQN-style algorithms for **discrete action** spaces. It relies on the shared
    :class:`QLearningCore` base class for optimizer/scheduler management, AMP
    support, update counters, and target-network update utilities.

    Algorithm
    ---------
    Let :math:`Q_\\theta` be the online network and :math:`Q_{\\bar\\theta}` the
    target network.

    Online estimate:

    .. math::
        Q_\\theta(s, a)

    Target:

    - Vanilla DQN:
      .. math::
          y = r + \\gamma (1-d) \\max_{a'} Q_{\\bar\\theta}(s', a')

    - Double DQN:
      .. math::
          a^* = \\arg\\max_{a'} Q_{\\theta}(s', a'), \\quad
          y = r + \\gamma (1-d) Q_{\\bar\\theta}(s', a^*)

    Loss (per-sample):

    - Huber (Smooth L1) if ``huber=True`` else MSE.
    - If PER weights ``w_i`` are provided, the loss is importance-weighted.

    Target updates
    --------------
    Target updates are delegated to ``_maybe_update_target`` from the base class:

    - Hard update every ``target_update_interval`` steps (typical DQN) when ``tau=0``.
    - Soft/Polyak update when ``tau>0`` (if supported by base implementation).

    Expected head interface (duck-typed)
    ------------------------------------
    The head must provide:

    - ``head.q`` : torch.nn.Module
        Online Q-network.
    - ``head.q_target`` : torch.nn.Module
        Target Q-network.
    - ``head.q_values(obs) -> Tensor`` of shape (B, A)
        Online Q-values.
    - ``head.q_values_target(obs) -> Tensor`` of shape (B, A)
        Target Q-values.
    - ``head.freeze_target(module) -> None``
        Utility to freeze/eval the target network (recommended).

    Notes
    -----
    - This core returns a NumPy vector ``per/td_errors`` as a priority proxy for PER.
      The core itself does not update priorities; the replay buffer is expected to
      consume these values.
    - ``per_eps`` is stored as metadata for reproducibility / buffer-side use. It is
      not directly applied in the loss here.
    """

    def __init__(
        self,
        *,
        head: Any,
        # ---------------------------------------------------------------------
        # TD / target update
        # ---------------------------------------------------------------------
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        # ---------------------------------------------------------------------
        # Variants / loss
        # ---------------------------------------------------------------------
        double_dqn: bool = True,
        huber: bool = True,
        # ---------------------------------------------------------------------
        # Stability / AMP
        # ---------------------------------------------------------------------
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # ---------------------------------------------------------------------
        # PER metadata (core-side only)
        # ---------------------------------------------------------------------
        per_eps: float = 1e-6,
        # ---------------------------------------------------------------------
        # Optimizer / scheduler (handled by QLearningCore)
        # ---------------------------------------------------------------------
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
        # Scheduler knobs (shared)
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
        head : Any
            Q-learning head that provides online/target Q networks and helper methods.
            See class docstring for the expected interface.
        gamma : float, default=0.99
            Discount factor. Must satisfy ``0 <= gamma < 1``.
        target_update_interval : int, default=1000
            Interval (in gradient updates) for target-network updates.
            - If 0: base implementation may interpret as "always update" or "never";
              follow your ``_maybe_update_target`` semantics.
        tau : float, default=0.0
            Soft-update coefficient for Polyak averaging.
            - ``tau = 0`` typically corresponds to hard updates only.
            - ``0 < tau <= 1`` enables soft updates (if base supports it).

        double_dqn : bool, default=True
            If True, use Double DQN target computation (action selection from online net).
        huber : bool, default=True
            If True, use Huber loss (Smooth L1). Else use MSE.

        max_grad_norm : float, default=0.0
            Global gradient clipping threshold.
            - 0.0 typically means "no clipping" (depending on BaseCore implementation).
        use_amp : bool, default=False
            Enable AMP autocast + GradScaler (handled by base class).

        per_eps : float, default=1e-6
            PER epsilon metadata for priority computation on the replay-buffer side.
            This core returns TD errors; the buffer may compute priorities as
            ``(|td_error| + per_eps)``.
        optim_name : str, default="adamw"
            Optimizer name for online Q parameters.
        lr : float, default=3e-4
            Learning rate.
        weight_decay : float, default=0.0
            Weight decay.
        sched_name : str, default="none"
            Scheduler name (optional).
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler knobs passed to the base class.

        Raises
        ------
        ValueError
            If any hyperparameter is out of range.
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

        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)

        self.double_dqn = bool(double_dqn)
        self.huber = bool(huber)

        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(
                f"target_update_interval must be >= 0, got {self.target_update_interval}"
            )
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        # Enforce common invariant: target net is frozen/eval.
        self.head.freeze_target(self.head.q_target)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one DQN-style TD update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch container providing (tensors):
            observations : torch.Tensor
                Shape (B, obs_dim).
            actions : torch.Tensor
                Discrete action indices. Shape (B,) or (B,1).
            rewards : torch.Tensor
                Shape (B,) or (B,1).
            next_observations : torch.Tensor
                Shape (B, obs_dim).
            dones : torch.Tensor
                Terminal flags (0/1). Shape (B,) or (B,1).

            PER integration (optional)
            --------------------------
            If your replay buffer supports prioritized replay, ``_get_per_weights``
            should be able to extract importance-sampling weights from the batch.
            The returned ``w`` must be broadcastable to (B,1).

        Returns
        -------
        metrics : Dict[str, float]
            Logging scalars plus a NumPy TD-error vector used for PER updates:

            - ``loss/q`` : float
                Scalar TD loss.
            - ``q/mean`` : float
                Mean Q(s,a) on the batch.
            - ``target/mean`` : float
                Mean TD target y on the batch.
            - ``lr`` : float
                Current learning rate.
            - ``per/td_errors`` : np.ndarray
                Absolute TD error per sample, shape (B,).

        Notes
        -----
        - Target computation is wrapped in ``torch.no_grad()`` to prevent gradients
          through the target path.
        - For Double DQN, the argmax action is selected using the online network.
        - Shape normalization is done via ``_to_column`` to unify to (B,1).
        """
        self._bump()

        # ------------------------------------------------------------------
        # Move batch to device and normalize shapes
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)                      # (B, obs_dim)
        act = batch.actions.to(self.device).long()                    # (B,) or (B,1)
        rew = _to_column(batch.rewards.to(self.device))               # (B,1)
        next_obs = batch.next_observations.to(self.device)            # (B, obs_dim)
        done = _to_column(batch.dones.to(self.device))                # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)            # None or (B,1)/(B,)

        # ------------------------------------------------------------------
        # Online estimate: Q(s,a)
        # ------------------------------------------------------------------
        q_all = self.head.q_values(obs)                               # (B, A)
        q_sa = q_all.gather(1, act.view(-1, 1))                       # (B, 1)

        # ------------------------------------------------------------------
        # Bootstrapped target
        # ------------------------------------------------------------------
        with th.no_grad():
            q_next_target_all = self.head.q_values_target(next_obs)   # (B, A)

            if self.double_dqn:
                # Action selection from online net; evaluation from target net.
                a_star = th.argmax(
                    self.head.q_values(next_obs), dim=-1, keepdim=True
                )                                                    # (B,1)
                q_next = q_next_target_all.gather(1, a_star)          # (B,1)
            else:
                q_next = q_next_target_all.max(dim=1, keepdim=True).values  # (B,1)

            target = rew + self.gamma * (1.0 - done) * q_next         # (B,1)

        # ------------------------------------------------------------------
        # TD loss (elementwise -> supports PER weighting)
        # ------------------------------------------------------------------
        if self.huber:
            per_sample = F.smooth_l1_loss(q_sa, target, reduction="none")    # (B,1)
        else:
            per_sample = F.mse_loss(q_sa, target, reduction="none")          # (B,1)

        if w is None:
            loss = per_sample.mean()
        else:
            # Ensure (B,1) broadcast
            w_col = _to_column(w) if isinstance(w, th.Tensor) else w
            loss = (per_sample * w_col).mean()  # type: ignore[operator]

        # ------------------------------------------------------------------
        # Optimizer step (online network only)
        # ------------------------------------------------------------------
        self.opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                loss_amp = loss
            self.scaler.scale(loss_amp).backward()
            self._clip_params(
                self.head.q.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.opt,
            )
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm)
            self.opt.step()

        if self.sched is not None:
            self.sched.step()

        # ------------------------------------------------------------------
        # PER priority proxy: |TD error|
        # ------------------------------------------------------------------
        with th.no_grad():
            td_error = (target - q_sa).abs().view(-1)                 # (B,)

        # ------------------------------------------------------------------
        # Target update (hard/soft depending on base helper)
        # ------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_sa.mean())),
            "target/mean": float(_to_scalar(target.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "per/td_errors": td_error.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state including DQN-specific hyperparameters.

        Returns
        -------
        state : Dict[str, Any]
            Serializable state dict.

        Notes
        -----
        Base :class:`QLearningCore.state_dict` is expected to include:
        - update counters / bookkeeping
        - optimizer state
        - scheduler state (if configured)

        This method extends the base state with DQN hyperparameters to support
        reproducibility and debugging.
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

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.

        Notes
        -----
        - Hyperparameters are constructor-owned in this implementation and are
          intentionally not overridden from the checkpoint to avoid silently
          changing runtime behavior.
        - If you want hyperparameter restoration, implement it explicitly and
          validate compatibility (e.g., gamma, double_dqn, etc.).
        """
        super().load_state_dict(state)
