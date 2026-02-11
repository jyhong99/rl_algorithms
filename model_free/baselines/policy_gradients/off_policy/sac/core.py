from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import _get_per_weights
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class SACCore(ActorCriticCore):
    """
    Soft Actor-Critic (SAC) update engine.

    This core implements the SAC learning rule for continuous-control settings while
    reusing :class:`~model_free.common.policies.base_core.ActorCriticCore` for shared
    training infrastructure (optimizers, schedulers, AMP, counters, and persistence).

    Overview
    --------
    SAC optimizes a maximum-entropy objective. Each update iteration typically performs:

    1. **Target construction (no grad)**

       .. math::

          y = r + \\gamma (1-d) \\left( \\min(Q_1^t(s', a'), Q_2^t(s', a')) - \\alpha \\log \\pi(a'|s') \\right)

       where :math:`a' \\sim \\pi(\\cdot|s')`.

    2. **Critic update (twin Q regression)**

       .. math::

          J_Q = \\mathbb{E}\\left[ (Q_1(s,a) - y)^2 + (Q_2(s,a) - y)^2 \\right]

       If PER is enabled, this regression may be importance-weighted.

    3. **Actor update (entropy-regularized)**

       .. math::

          J_\\pi = \\mathbb{E}\\left[ \\alpha \\log \\pi(a|s) - \\min(Q_1(s,a), Q_2(s,a)) \\right]

       with :math:`a \\sim \\pi(\\cdot|s)`.

    4. **Temperature update (optional; auto-alpha)**

       .. math::

          J_\\alpha = -\\mathbb{E}\\left[ \\log \\alpha \\, (\\log \\pi(a|s) + \\mathcal{H}_{\\text{target}}) \\right]

       This pushes entropy toward a target value.

    5. **Target critic update (Polyak / hard)** driven by the base core helper.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes (discovered by :class:`ActorCriticCore`)
    - ``head.actor`` : torch.nn.Module
    - ``head.critic`` : torch.nn.Module
        Must support ``critic(obs, act) -> (q1, q2)`` with each output shaped ``(B, 1)``.
    - ``head.critic_target`` : torch.nn.Module
        Same signature as ``head.critic``.
    - ``head.device`` : torch.device or str (base core normalizes to ``self.device``).

    Required methods (used directly by this core)
    - ``head.sample_action_and_logp(obs) -> (action, logp)``
        - ``action``: tensor of shape ``(B, action_dim)``
        - ``logp``: tensor of shape ``(B, 1)`` preferred (``(B,)`` accepted and normalized)

    Shape conventions
    -----------------
    - Rewards and done flags are normalized to ``(B, 1)`` via :func:`_to_column`.
    - Log-probabilities are normalized to ``(B, 1)`` via :func:`_to_column`.

    Notes
    -----
    - This core returns ``"per/td_errors"`` as a NumPy array (non-scalar) to support
      PER priority updates in an outer algorithm wrapper. If you want strict typing,
      use ``Dict[str, Any]`` as the return type or omit the array and log only scalar
      aggregates.
    """

    def __init__(
        self,
        *,
        head: Any,
        # SAC hyperparameters
        gamma: float = 0.99,
        # target update
        tau: float = 0.005,
        target_update_interval: int = 1,
        # entropy temperature
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # alpha optimizer/scheduler
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        alpha_sched_name: str = "none",
        # sched shared knobs
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
        # actor/critic opt/sched (built by ActorCriticCore)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
    ) -> None:
        # ---------------------------------------------------------------------
        # Build base training infrastructure (actor/critic optimizers/schedulers)
        # ---------------------------------------------------------------------
        super().__init__(
            head=head,
            use_amp=bool(use_amp),
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
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
        # Hyperparameters + validation
        # ---------------------------------------------------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # ---------------------------------------------------------------------
        # Target entropy
        # ---------------------------------------------------------------------
        # Common SAC heuristic: target_entropy = -|A| (continuous action dimension).
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # ---------------------------------------------------------------------
        # Temperature parameter alpha (optimize log_alpha)
        # ---------------------------------------------------------------------
        self.auto_alpha = bool(auto_alpha)
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")

        self.log_alpha = th.tensor(
            float(math.log(float(alpha_init))),
            device=self.device,
            requires_grad=self.auto_alpha,
        )

        # alpha optimizer/scheduler (separate from ActorCriticCore)
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

        # Enforce target critic to be frozen (no gradients into target params).
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """
        Current entropy temperature :math:`\\alpha = \\exp(\\log \\alpha)`.

        Returns
        -------
        torch.Tensor
            Scalar tensor on ``self.device``.
        """
        return self.log_alpha.exp()

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one SAC update from a sampled replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch object (duck-typed). Expected fields:

            - ``observations`` : torch.Tensor, shape (B, obs_dim)
            - ``actions`` : torch.Tensor, shape (B, action_dim)
            - ``rewards`` : torch.Tensor, shape (B,) or (B, 1)
            - ``next_observations`` : torch.Tensor, shape (B, obs_dim)
            - ``dones`` : torch.Tensor, shape (B,) or (B, 1)

            Optional PER fields may be present and will be interpreted by
            :func:`~model_free.common.utils.policy_utils._get_per_weights`.

        Returns
        -------
        metrics : Dict[str, Any]
            Logging metrics. All values are scalars except:

            - ``"per/td_errors"`` : np.ndarray, shape (B,)
              Absolute TD-error proxy for PER priority updates upstream.

        Notes
        -----
        - This method increments internal update counters via ``self._bump()``.
        - AMP behavior is controlled by ``self.use_amp`` from the base core.
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch to device + normalize shapes
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))            # (B,1)
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))             # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)         # (B,1) or None

        # ---------------------------------------------------------------------
        # Target Q computation (no grad)
        # ---------------------------------------------------------------------
        with th.no_grad():
            nxt_a, nxt_logp = self.head.sample_action_and_logp(nxt)
            nxt_logp = _to_column(nxt_logp)                        # (B,1)

            q1_t, q2_t = self.head.q_values_target(nxt, nxt_a)     # each (B,1)
            q_min_t = th.min(q1_t, q2_t)                           # (B,1)

            target_q = rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * nxt_logp)

        # ---------------------------------------------------------------------
        # Critic update
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute critic regression loss and a TD-error magnitude proxy.

            Returns
            -------
            loss : torch.Tensor
                Scalar loss tensor.
            td_abs : torch.Tensor
                Shape (B,) absolute TD error proxy suitable for PER priorities.
            """
            q1, q2 = self.head.q_values(obs, act)                  # each (B,1)

            per_sample = (
                F.mse_loss(q1, target_q, reduction="none")
                + F.mse_loss(q2, target_q, reduction="none")
            )                                                      # (B,1)

            # TD magnitude proxy (min(Q1,Q2) matches the target structure)
            td = th.min(q1, q2) - target_q
            td_abs = td.abs().detach().squeeze(1)                  # (B,)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
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
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update
        # ---------------------------------------------------------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute SAC actor loss on the current batch.

            Returns
            -------
            loss : torch.Tensor
                Scalar actor loss.
            logp : torch.Tensor
                Shape (B,1) log π(a|s) for sampled actions.
            q_pi : torch.Tensor
                Shape (B,1) min(Q1,Q2)(s,a) for sampled actions.
            """
            new_a, logp = self.head.sample_action_and_logp(obs)
            logp = _to_column(logp)

            q1_pi, q2_pi = self.head.q_values(obs, new_a)
            q_pi = th.min(q1_pi, q2_pi)

            loss = (self.alpha * logp - q_pi).mean()
            return loss, logp, q_pi

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
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            # Detach logp to prevent alpha loss from backpropagating into the actor.
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(_to_scalar(alpha_loss))

        # ---------------------------------------------------------------------
        # Target critic update (Polyak / hard depending on core helper)
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        with th.no_grad():
            q1_b, q2_b = self.head.critic(obs, act)

        out: Dict[str, Any] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "alpha": float(_to_scalar(self.alpha)),
            "q/q1_mean": float(_to_scalar(q1_b.mean())),
            "q/q2_mean": float(_to_scalar(q2_b.mean())),
            "q/pi_min_mean": float(_to_scalar(q_pi.mean())),
            "logp_mean": float(_to_scalar(logp.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
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
        Serialize core state, including alpha state if auto-alpha is enabled.

        Returns
        -------
        Dict[str, Any]
            State dictionary including:
            - base core state (optimizers/schedulers/counters)
            - ``log_alpha`` and alpha optimizer/scheduler state (if present)
            - ``auto_alpha`` and ``target_entropy`` (for inspection/debugging)

        Notes
        -----
        Hyperparameters are typically constructor-owned; these fields are stored
        primarily for reproducibility and debugging.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(_to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "auto_alpha": bool(self.auto_alpha),
                "target_entropy": float(self.target_entropy),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state, including alpha optimizer/scheduler if present.

        Parameters
        ----------
        state : Mapping[str, Any]
            State payload produced by :meth:`state_dict`.

        Notes
        -----
        - This method assumes constructor configuration is compatible with the
          serialized state (e.g., auto_alpha enabled/disabled consistently).
        - Optimizer reconstruction is not performed here; it is expected to be
          done in ``__init__`` and then populated with loaded state.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        # Best-effort restore of configuration metadata (does not rebuild optimizers)
        if "auto_alpha" in state:
            self.auto_alpha = bool(state["auto_alpha"])
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
