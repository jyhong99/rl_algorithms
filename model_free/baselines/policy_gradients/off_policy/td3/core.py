from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.utils.policy_utils import _get_per_weights


class TD3Core(ActorCriticCore):
    """
    Twin Delayed DDPG (TD3) update engine built on :class:`ActorCriticCore`.

    This core implements TD3's update rules while reusing shared infrastructure
    from :class:`ActorCriticCore` for:
      - optimizer / scheduler construction (actor + critic)
      - AMP GradScaler support (optional)
      - update counters and generic persistence helpers
      - Polyak target updates via ``_maybe_update_target``

    Separation of concerns
    ----------------------
    - **Head** owns networks and inference utilities:
        * ``actor``, ``critic``, ``actor_target``, ``critic_target``
        * TD3 target policy smoothing via ``head.target_action(...)``
        * Q helpers: ``head.q_values(...)``, ``head.q_values_target(...)``

    - **Core** owns optimization and update logic:
        * TD targets
        * critic regression
        * delayed actor update
        * target network updates (when actor updates, gated by interval)
        * PER TD-errors (if replay provides weights)

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes
        - ``head.actor`` : torch.nn.Module
            Deterministic actor network :math:`\\pi(s)`.
        - ``head.critic`` : torch.nn.Module
            Twin critic network; expected to support ``q_values(obs, act) -> (q1, q2)``.
        - ``head.actor_target`` : torch.nn.Module
            Target actor :math:`\\pi'(s)`.
        - ``head.critic_target`` : torch.nn.Module
            Target twin critics :math:`Q_1'(s,a), Q_2'(s,a)`.
        - ``head.device`` : torch.device or str
            Device used for forward passes.

    Required methods
        - ``head.target_action(next_obs, *, noise_std, noise_clip) -> torch.Tensor``
            TD3 target policy smoothing action :math:`a'`.

        - ``head.q_values(obs, act) -> Tuple[Tensor, Tensor]``
            Returns ``(q1, q2)`` each shaped ``(B, 1)``.

        - ``head.q_values_target(obs, act) -> Tuple[Tensor, Tensor]``
            Returns target critics output ``(q1_t, q2_t)`` each shaped ``(B, 1)``.

    Optional (used indirectly)
        - ``head.freeze_target(module)``
            Disables gradients for target networks (safety).

    Batch contract
    --------------
    ``update_from_batch(batch)`` expects an object with:
        - ``batch.observations``      : Tensor, shape ``(B, obs_dim)``
        - ``batch.actions``           : Tensor, shape ``(B, action_dim)``
        - ``batch.rewards``           : Tensor, shape ``(B,)`` or ``(B, 1)``
        - ``batch.next_observations`` : Tensor, shape ``(B, obs_dim)``
        - ``batch.dones``             : Tensor, shape ``(B,)`` or ``(B, 1)``
        - Optional ``batch.weights``  : Tensor, shape ``(B,)`` or ``(B, 1)`` (PER)

    Notes
    -----
    TD3 key mechanics:
      1) **Target policy smoothing** for critic targets
      2) **Twin critics** and min-reduction for TD target
      3) **Delayed policy updates** (actor updated every ``policy_delay`` steps)
      4) **Target updates** usually coincide with actor updates (and may be gated)
    """

    def __init__(
        self,
        *,
        head: Any,
        # TD3 hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        target_update_interval: int = 1,
        # optimizers (built by ActorCriticCore)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # schedulers (optional; built by ActorCriticCore)
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
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
        head : Any
            Policy head providing actor/critic networks and TD3-specific helpers.
            See class docstring for the required interface.
        gamma : float, default=0.99
            Discount factor :math:`\\gamma`.
        tau : float, default=0.005
            Polyak coefficient for target updates. ``tau=1`` becomes a hard update.
        policy_noise : float, default=0.2
            Stddev of Gaussian noise used for TD3 target policy smoothing.
        noise_clip : float, default=0.5
            Clip range for target policy smoothing noise.
        policy_delay : int, default=2
            Actor update period (in critic update calls). Actor updates when
            ``update_calls % policy_delay == 0``.
        target_update_interval : int, default=1
            Additional gate for target updates when actor updates. Targets update when
            ``update_calls % target_update_interval == 0`` (inside actor-update branch).
        actor_optim_name, critic_optim_name : str
            Names forwarded to your optimizer builder in :class:`ActorCriticCore`.
        actor_lr, critic_lr : float
            Learning rates for actor and critic optimizers.
        actor_weight_decay, critic_weight_decay : float
            Weight decay values for actor and critic optimizers.
        actor_sched_name, critic_sched_name : str
            Scheduler names forwarded to :class:`ActorCriticCore`.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler hyperparameters forwarded to :class:`ActorCriticCore`.
        max_grad_norm : float, default=0.0
            Global norm clip threshold. ``0.0`` disables clipping.
        use_amp : bool, default=False
            If True, enables AMP autocast + GradScaler update paths.
        """
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
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        # Fail-fast validation
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.policy_delay < 0:
            raise ValueError(f"policy_delay must be >= 0, got {self.policy_delay}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # Ensure targets never receive grads from the core.
        self.head.freeze_target(self.head.actor_target)
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one TD3 update step from a replay batch.

        Algorithm
        ---------
        Critic update (every call)
            1) Compute smoothed target action:
               ``a' = head.target_action(s', noise_std=policy_noise, noise_clip=noise_clip)``
            2) Compute target Q:
               ``y = r + gamma * (1-done) * min(Q1_t(s',a'), Q2_t(s',a'))``
            3) Regress both critics to ``y`` using MSE (PER-weighted if provided)

        Actor update (delayed)
            Every ``policy_delay`` calls:
            4) Update actor by maximizing ``Q1(s, pi(s))`` (minimizing ``-Q1``)

        Target updates (when actor updates)
            5) Polyak-update targets, additionally gated by ``target_update_interval``.

        Parameters
        ----------
        batch : Any
            Replay batch satisfying the class-level "Batch contract".

        Returns
        -------
        metrics : Dict[str, float]
            Scalar metrics suitable for logging.

            Notes:
            - This method also returns PER TD-errors under key ``"per/td_errors"`` as a
              numpy array. The return type annotation is kept as ``Dict[str, float]``
              to match your existing patterns, but the value is non-scalar.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))         # (B,1)
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))          # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)      # (B,1) or None

        # ---------------------------------------------------------------------
        # TD target (no grad)
        # ---------------------------------------------------------------------
        with th.no_grad():
            next_a = self.head.target_action(
                nxt,
                noise_std=float(self.policy_noise),
                noise_clip=float(self.noise_clip),
            )
            q1_t, q2_t = self.head.q_values_target(nxt, next_a)  # each (B,1)
            q_t = th.min(q1_t, q2_t)
            target_q = rew + self.gamma * (1.0 - done) * q_t     # (B,1)

        # ---------------------------------------------------------------------
        # Critic update (PER-weighted)
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute critic loss and TD-error proxy for PER.

            Returns
            -------
            loss : torch.Tensor
                Scalar critic loss (sum of twin losses, mean-reduced).
            td_abs : torch.Tensor
                Shape ``(B,)`` absolute TD error proxy for PER priorities.
            q1 : torch.Tensor
                Current Q1(s,a), shape ``(B,1)`` (for logging).
            q2 : torch.Tensor
                Current Q2(s,a), shape ``(B,1)`` (for logging).
            """
            q1, q2 = self.head.q_values(obs, act)

            l1 = F.mse_loss(q1, target_q, reduction="none")      # (B,1)
            l2 = F.mse_loss(q2, target_q, reduction="none")      # (B,1)
            per_sample = l1 + l2                                  # (B,1)

            td = th.min(q1, q2) - target_q                        # (B,1)
            td_abs = td.abs().detach().squeeze(1)                 # (B,)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
            return loss, td_abs, q1, q2

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update (delayed)
        # ---------------------------------------------------------------------
        did_actor = (self.policy_delay > 0) and (self.update_calls % self.policy_delay == 0)
        actor_loss_scalar = 0.0

        if did_actor:

            def _actor_loss() -> th.Tensor:
                """
                TD3 actor objective.

                Maximization form:
                    maximize E[ Q1(s, pi(s)) ]

                Minimization form (implemented):
                    minimize E[ -Q1(s, pi(s)) ]
                """
                act_fn = getattr(self.head.actor, "act", None)
                if callable(act_fn):
                    pi, _ = act_fn(obs, deterministic=True)
                else:
                    pi = self.head.actor(obs)

                q1_pi, _q2_pi = self.head.q_values(obs, pi)
                return (-q1_pi).mean()

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

            actor_loss_scalar = float(_to_scalar(actor_loss))

            # -----------------------------------------------------------------
            # Target updates (only when actor updates)
            # -----------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        out: Dict[str, float] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(actor_loss_scalar),
            "q/q1_mean": float(_to_scalar(q1_now.mean())),
            "q/q2_mean": float(_to_scalar(q2_now.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "td3/did_actor_update": float(1.0 if did_actor else 0.0),
        }

        # PER feedback (non-scalar). Kept as numpy array for replay priority updates.
        out["per/td_errors"] = td_abs.detach().cpu().numpy()  # type: ignore[assignment]
        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend base :meth:`ActorCriticCore.state_dict` with TD3 hyperparameters.

        Returns
        -------
        state : Dict[str, Any]
            Includes base core state (optimizers/schedulers/update counters) plus
            TD3-specific hyperparameters for reproducibility/debugging.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "policy_noise": float(self.policy_noise),
                "noise_clip": float(self.noise_clip),
                "policy_delay": int(self.policy_delay),
                "target_update_interval": int(self.target_update_interval),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore optimizer/scheduler state and counters, then re-freeze targets.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.

        Notes
        -----
        This method intentionally does **not** overwrite hyperparameters from the
        checkpoint to avoid surprising behavior when loading into a differently
        configured run. If you want hparam restoration, do it explicitly at the
        experiment/config level.

        After restoring base state, we re-freeze targets to guarantee that no
        gradients flow into target networks, even if the checkpoint came from a
        different training context.
        """
        super().load_state_dict(state)

        self.head.freeze_target(self.head.actor_target)
        self.head.freeze_target(self.head.critic_target)
