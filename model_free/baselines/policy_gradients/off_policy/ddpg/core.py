from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.utils.policy_utils import _get_per_weights


class DDPGCore(ActorCriticCore):
    """
    DDPG update engine built on :class:`~model_free.common.policies.base_core.ActorCriticCore`.

    This core implements the classic DDPG optimization step using the project's
    head/core separation:

    - The **head** owns the networks (actor/critic + their targets) and exposes
      explicit Q-access methods.
    - The **core** owns the optimization logic: loss computation, optimizer steps,
      scheduler steps, gradient clipping, AMP, and target-update cadence.

    Key design choice
    -----------------
    This core intentionally avoids calling internal modules like
    ``head.critic(...)`` / ``head.critic_target(...)`` directly to compute Q.
    Instead, it uses the head's explicit interfaces:

    - ``head.q_values(obs, action) -> Q(s,a)``
    - ``head.q_values_target(obs, action) -> Q_targ(s,a)``

    This makes the core robust to head implementation details (e.g., different
    critic architectures or naming), while still allowing :class:`ActorCriticCore`
    to discover parameters to optimize via ``head.actor`` and ``head.critic``.

    Parameters
    ----------
    head : Any
        Deterministic actor-critic head (duck-typed).

        Required attributes (discovered/used by ActorCriticCore)
        --------------------------------------------------------
        - ``head.actor`` : ``nn.Module``
        - ``head.critic`` : ``nn.Module``
        - ``head.actor_target`` : ``nn.Module``
        - ``head.critic_target`` : ``nn.Module``
        - ``head.device`` : ``torch.device`` or ``str``

        Required methods (used by DDPGCore)
        ----------------------------------
        - ``head.q_values(obs, action) -> torch.Tensor``
          Returns Q(s,a). Shape (B,1) or (B,).
        - ``head.q_values_target(obs, action) -> torch.Tensor``
          Returns Q_targ(s,a). Shape (B,1) or (B,).

    gamma : float, default=0.99
        Discount factor :math:`\\gamma`. Must satisfy ``0 <= gamma < 1``.
    tau : float, default=0.005
        Soft update coefficient for target networks.
        ``tau=1`` corresponds to a hard update.
    target_update_interval : int, default=1
        Target update cadence in optimizer steps. If 0, disables target updates.

    actor_optim_name, critic_optim_name : str
        Optimizer identifiers resolved by :class:`ActorCriticCore`.
    actor_lr, critic_lr : float
        Learning rates.
    actor_weight_decay, critic_weight_decay : float
        Weight decay.

    actor_sched_name, critic_sched_name : str
        Scheduler identifiers resolved by :class:`ActorCriticCore`.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration forwarded to the base core.

    max_grad_norm : float, default=0.0
        Gradient global-norm clipping threshold. If 0, clipping is disabled.
    use_amp : bool, default=False
        Enable automatic mixed precision (AMP) via torch.cuda.amp.

    Batch Contract
    --------------
    ``update_from_batch(batch)`` expects:

    - ``batch.observations`` : Tensor (B, obs_dim)
    - ``batch.actions`` : Tensor (B, action_dim)
    - ``batch.rewards`` : Tensor (B,) or (B,1)
    - ``batch.next_observations`` : Tensor (B, obs_dim)
    - ``batch.dones`` : Tensor (B,) or (B,1)

    PER Contract (optional)
    -----------------------
    If PER sampling weights are present (as supported by ``_get_per_weights``),
    critic loss is importance-weighted:

    - weights returned by ``_get_per_weights`` : Tensor (B,1)

    Priority Feedback (optional)
    ----------------------------
    This core returns ``"per/td_errors"`` as a NumPy array of shape (B,), which
    off-policy wrappers can use to update priorities.

    Notes
    -----
    - Actor loss is the negative Q under the current actor:
      :math:`L_\\pi = -\\mathbb{E}[Q(s,\\pi(s))]`.
    - Critic loss is mean-squared TD error toward:
      :math:`y = r + \\gamma(1-d)Q_{\\text{targ}}(s',\\pi_{\\text{targ}}(s'))`.
    """

    def __init__(
        self,
        *,
        head: Any,
        # DDPG core hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        # Optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # Schedulers (optional)
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # Grad / AMP
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__(
            head=head,
            use_amp=use_amp,
            # Optimizers
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            # Schedulers
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

        # ---------------------------------------------------------------------
        # Store hyperparameters
        # ---------------------------------------------------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        # ---------------------------------------------------------------------
        # Defensive validation
        # ---------------------------------------------------------------------
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # ---------------------------------------------------------------------
        # Freeze target modules once (core owns freeze responsibility)
        # ---------------------------------------------------------------------
        at = getattr(self.head, "actor_target", None)
        ct = getattr(self.head, "critic_target", None)
        if at is not None:
            self._freeze_target(at)
        if ct is not None:
            self._freeze_target(ct)

        # ---------------------------------------------------------------------
        # Validate required Q interfaces
        # ---------------------------------------------------------------------
        if not callable(getattr(self.head, "q_values", None)):
            raise ValueError("DDPGCore requires head.q_values(obs, action).")
        if not callable(getattr(self.head, "q_values_target", None)):
            raise ValueError("DDPGCore requires head.q_values_target(obs, action).")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Run one DDPG update using a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch with fields described in the class docstring.

        Returns
        -------
        Dict[str, float]
            Scalar metrics for logging. Additionally returns PER feedback:

            - ``out["per/td_errors"]`` : np.ndarray of shape (B,)

        Notes
        -----
        The update performs, in order:

        1) Critic step:
           minimize MSE between Q(s,a) and TD target y.

        2) Actor step:
           maximize Q(s, π(s)) (implemented as minimizing -Q).

        3) Scheduler steps (if configured).

        4) Target updates (actor_target and critic_target) at the configured cadence.
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch to device and normalize shapes
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)  # (B, obs_dim)
        act = batch.actions.to(self.device)  # (B, action_dim)
        rew = _to_column(batch.rewards.to(self.device))  # (B,1)
        nxt = batch.next_observations.to(self.device)  # (B, obs_dim)
        done = _to_column(batch.dones.to(self.device))  # (B,1)

        B = int(obs.shape[0])

        # PER weights if present: (B,1) or None.
        w = _get_per_weights(batch, B, device=self.device)

        # ---------------------------------------------------------------------
        # Target Q:
        #   y = r + γ(1-done) * Q_targ(s', π_targ(s'))
        # ---------------------------------------------------------------------
        with th.no_grad():
            act_fn = getattr(self.head.actor_target, "act", None)
            if callable(act_fn):
                next_a, _ = act_fn(nxt, deterministic=True)
            else:
                next_a = self.head.actor_target(nxt)

            q_t = self.head.q_values_target(nxt, next_a)
            q_t = _to_column(q_t)  # (B,1)

            target_q = rew + self.gamma * (1.0 - done) * q_t  # (B,1)

        # ---------------------------------------------------------------------
        # Critic update (optionally PER-weighted)
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            q = self.head.q_values(obs, act)
            q = _to_column(q)  # (B,1)

            # Per-sample MSE for PER weighting.
            per_sample = F.mse_loss(q, target_q, reduction="none")  # (B,1)

            # TD error magnitude for PER priorities.
            td_abs = (q - target_q).detach().squeeze(1).abs()  # (B,)

            if w is None:
                loss = per_sample.mean()
            else:
                loss = (w * per_sample).mean()

            return loss, td_abs, q

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs, q_now = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()

            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )

            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs, q_now = _critic_loss_and_td()
            critic_loss.backward()

            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)

            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update:
        #   maximize Q(s, π(s))  <=>  minimize -Q(s, π(s))
        # ---------------------------------------------------------------------
        def _actor_loss() -> th.Tensor:
            act_fn = getattr(self.head.actor, "act", None)
            if callable(act_fn):
                pi, _ = act_fn(obs, deterministic=True)
            else:
                pi = self.head.actor(obs)

            q_pi = self.head.q_values(obs, pi)
            q_pi = _to_column(q_pi)
            return (-q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss = _actor_loss()
            self.scaler.scale(actor_loss).backward()

            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )

            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss = _actor_loss()
            actor_loss.backward()

            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Target update gate (cadence controlled here; update method handles tau)
        # ---------------------------------------------------------------------
        did_target = (
            self.target_update_interval > 0
            and (self.update_calls % self.target_update_interval == 0)
        )
        if did_target:
            self._maybe_update_target(
                target=getattr(self.head, "actor_target", None),
                source=self.head.actor,
                interval=self.target_update_interval,
                tau=self.tau,
            )
            self._maybe_update_target(
                target=getattr(self.head, "critic_target", None),
                source=self.head.critic,
                interval=self.target_update_interval,
                tau=self.tau,
            )

        out: Dict[str, float] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "q/q_mean": float(_to_scalar(q_now.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "ddpg/did_target_update": float(1.0 if did_target else 0.0),
        }

        # Non-scalar PER feedback (consumed by the algorithm wrapper for priority updates).
        out["per/td_errors"] = td_abs.detach().cpu().numpy()  # type: ignore[assignment]
        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state and store DDPG metadata.

        Returns
        -------
        Dict[str, Any]
            State mapping produced by the base core (optimizers, schedulers, AMP,
            counters) plus DDPG hyperparameters for inspection/debugging.

        Notes
        -----
        Hyperparameters are constructor-owned. They are stored here as metadata and
        are not automatically restored by :meth:`load_state_dict`.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state (optimizers/schedulers/AMP/counters).

        Parameters
        ----------
        state : Mapping[str, Any]
            State mapping produced by :meth:`state_dict`.

        Notes
        -----
        - Delegates to :class:`ActorCriticCore` for restoring optimizer/scheduler/counter state.
        - Does **not** override constructor-owned hyperparameters (gamma, tau, etc.).
        """
        super().load_state_dict(state)
        return
