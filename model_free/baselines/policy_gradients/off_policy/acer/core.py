from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.policy_utils import _get_per_weights


class ACERCore(ActorCriticCore):
    """
    ACER update engine for discrete action spaces (single-step / TD(0) variant).

    This core implements a practical, 1-step ACER-style update on top of
    :class:`~model_free.common.policies.base_core.ActorCriticCore`.

    The update decomposes into:

    1) **Critic (TD regression)**

       Uses a TD(0)-style target:

       .. math::
           y = r + \\gamma (1-d) V_\\pi(s')

       where:

       .. math::
           V_\\pi(s') = \\sum_a \\pi(a\\mid s') Q_{\\text{targ}}(s', a)

       The target critic is used for stability.

    2) **Actor (off-policy policy gradient)**

       Uses truncated importance sampling (IS):

       .. math::
           \\rho = \\frac{\\pi(a\\mid s)}{\\mu(a\\mid s)}, \\quad
           c = \\min(\\rho, \\bar c)

       and a sampled-action loss:

       .. math::
           L_{\\text{main}} = -\\mathbb{E}[c\\, A(s,a)\\, \\log \\pi(a\\mid s)]

       with:

       .. math::
           A(s,a) = Q(s,a) - V_\\pi(s)

    3) **Optional bias correction (requires behavior probs)**

       If :math:`\\mu(a\\mid s)` is available for *all* actions, a correction term
       can be added to account for truncation (implementation-specific details).

    4) **Optional entropy regularization**

       Adds:

       .. math::
           L_{\\text{ent}} = -\\beta\\, \\mathbb{E}[H(\\pi(\\cdot\\mid s))]

    Parameters
    ----------
    head : Any
        Policy head object (duck-typed). Expected to provide at least:

        - ``head.actor`` : ``nn.Module`` producing logits (or compatible API)
        - ``head.critic`` : critic module (online)
        - ``head.critic_target`` : critic module (target), or ``head.q_target`` in older naming
        - ``head.logp(obs, act) -> Tensor`` : log π(a|s), shape (B,1) or (B,)
        - ``head.probs(obs) -> Tensor`` : π(·|s), shape (B,A)
        - ``head.q_values(obs, reduce=...) -> Tensor`` : Q(s,·), shape (B,A)
        - ``head.q_values_target(obs, reduce=...) -> Tensor`` : Q_targ(s,·), shape (B,A)

        Notes
        -----
        Your codebase appears to use ``critic`` / ``critic_target`` naming in the head,
        but this core also checks ``q_target`` once for freezing compatibility.

    gamma : float, default=0.99
        Discount factor :math:`\\gamma`. Must satisfy ``0 <= gamma < 1``.
    c_bar : float, default=10.0
        Truncation threshold :math:`\\bar c` for importance weights. Must be > 0.
    entropy_coef : float, default=0.0
        Entropy regularization coefficient. Set to 0 to disable.
    critic_is : bool, default=False
        If ``True``, applies truncated IS weights to critic loss as well.
        (Not always used in ACER variants; keep ``False`` unless you intend it.)

    target_update_interval : int, default=1
        Update cadence in optimizer steps. If 0, disables target updates.
    tau : float, default=0.005
        Soft update coefficient. ``tau=1`` corresponds to a hard copy.

    actor_optim_name, critic_optim_name : str
        Optimizer identifiers handled by :class:`ActorCriticCore`.
    actor_lr, critic_lr : float
        Learning rates.
    actor_weight_decay, critic_weight_decay : float
        Weight decay values.
    actor_sched_name, critic_sched_name : str
        Scheduler identifiers handled by :class:`ActorCriticCore`.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration forwarded to the base core.

    max_grad_norm : float, default=0.0
        Global norm clipping threshold. If 0, clipping is disabled.
    use_amp : bool, default=False
        Enables torch AMP for mixed precision.
    per_eps : float, default=1e-6
        Small epsilon used by PER integrations (core-side proxy only).

    Batch Contract
    --------------
    The input ``batch`` must provide:

    - ``batch.observations`` : Tensor (B, obs_dim)
    - ``batch.actions`` : Tensor (B,) or (B,1)
    - ``batch.rewards`` : Tensor (B,) or (B,1)
    - ``batch.next_observations`` : Tensor (B, obs_dim)
    - ``batch.dones`` : Tensor (B,) or (B,1)

    For off-policy ratios, behavior log-prob is required via:

    - ``batch.behavior_logp`` or fallback ``batch.logp`` : log μ(a|s), (B,) or (B,1)

    Optional bias correction requires:

    - ``batch.behavior_probs`` : μ(·|s), Tensor (B, A)

    Optional PER support:
    - PER fields as expected by ``_get_per_weights(...)``

    Returns
    -------
    dict
        Scalar metrics suitable for logging. This implementation also returns a
        NumPy array under ``"per/td_errors"`` (useful for PER), which is not a
        float. If you want strict typing, change the return type to ``Dict[str, Any]``
        or log only summary statistics here.
    """

    def __init__(
        self,
        *,
        head: Any,
        # -----------------------------
        # Core ACER hyperparameters
        # -----------------------------
        gamma: float = 0.99,
        c_bar: float = 10.0,
        entropy_coef: float = 0.0,
        critic_is: bool = False,
        # -----------------------------
        # Target updates
        # -----------------------------
        target_update_interval: int = 1,
        tau: float = 0.005,
        # -----------------------------
        # Optimizer / scheduler config (ActorCriticCore)
        # -----------------------------
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # -----------------------------
        # Grad / AMP
        # -----------------------------
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # -----------------------------
        # PER epsilon (core-side proxy only)
        # -----------------------------
        per_eps: float = 1e-6,
    ) -> None:
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

        # -----------------------------
        # ACER hyperparameters
        # -----------------------------
        self.gamma = float(gamma)
        self.c_bar = float(c_bar)
        self.entropy_coef = float(entropy_coef)
        self.critic_is = bool(critic_is)

        # -----------------------------
        # Target update config
        # -----------------------------
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)

        # -----------------------------
        # Grad / PER
        # -----------------------------
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # -----------------------------
        # Defensive validation
        # -----------------------------
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.c_bar <= 0.0:
            raise ValueError(f"c_bar must be > 0, got {self.c_bar}")
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

        # Freeze target critic once (core owns "target is non-trainable" policy).
        #
        # Naming note:
        # - newer heads: critic_target
        # - legacy compatibility: q_target
        q_target = getattr(self.head, "critic_target", None)
        if q_target is None:
            q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            self._freeze_target(q_target)

    # =============================================================================
    # Batch helpers
    # =============================================================================
    def _get_behavior_logp(self, batch: Any) -> th.Tensor:
        """
        Extract behavior log-probabilities log μ(a|s) and normalize to (B, 1).

        Parameters
        ----------
        batch : Any
            Batch object providing either ``behavior_logp`` or ``logp``.

            Accepted fields
            ---------------
            - ``batch.behavior_logp`` : Tensor (B,) or (B,1)
            - ``batch.logp`` : Tensor (B,) or (B,1) (fallback)

        Returns
        -------
        torch.Tensor
            Behavior log-probabilities on ``self.device`` with shape (B, 1).

        Raises
        ------
        ValueError
            If neither behavior log-prob field exists, or if the resulting tensor
            does not have shape (B, 1).

        Notes
        -----
        ACER's importance sampling ratio uses:

        .. math::
            \\rho = \\exp(\\log \\pi(a\\mid s) - \\log \\mu(a\\mid s))

        Therefore log μ(a|s) is required for the sampled action.
        """
        if hasattr(batch, "behavior_logp"):
            log_mu = batch.behavior_logp
        elif hasattr(batch, "logp"):
            log_mu = batch.logp
        else:
            raise ValueError("ACER requires behavior logp in batch (behavior_logp or logp).")

        log_mu = log_mu.to(self.device)

        # Normalize (B,) -> (B,1)
        if log_mu.dim() == 1:
            log_mu = log_mu.unsqueeze(-1)

        if log_mu.dim() != 2 or log_mu.shape[1] != 1:
            raise ValueError(f"behavior_logp must be shape (B,1) or (B,), got {tuple(log_mu.shape)}")

        return log_mu

    def _get_logits(self, obs_t: th.Tensor) -> th.Tensor:
        """
        Retrieve policy logits for stable entropy / log-softmax computations.

        This method is intentionally "best-effort" to support multiple head/actor APIs.

        Preference order
        ---------------
        1) ``head.dist(obs).logits`` (if ``head.dist`` exists)
        2) ``head.logits(obs)``      (if head exposes logits directly)
        3) ``head.actor(obs)``       (assume actor forward returns logits)

        Parameters
        ----------
        obs_t : torch.Tensor
            Batched observations tensor on ``self.device`` with shape (B, obs_dim).

        Returns
        -------
        torch.Tensor
            Logits tensor with shape (B, A), where A is the number of actions.

        Notes
        -----
        - This function does not detach; call-sites should control autograd context.
        - If your actor returns a distribution object instead of logits, implement
          ``head.dist`` or ``head.logits`` for clarity.
        """
        dist_fn = getattr(self.head, "dist", None)
        if callable(dist_fn):
            dist = dist_fn(obs_t)
            logits = getattr(dist, "logits", None)
            if logits is not None:
                return logits

        logits_fn = getattr(self.head, "logits", None)
        if callable(logits_fn):
            return logits_fn(obs_t)

        return self.head.actor(obs_t)

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one ACER update using a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch providing tensors described in the class docstring
            (observations, actions, rewards, next_observations, dones, behavior_logp).

        Returns
        -------
        Dict[str, float]
            Logging metrics. This function also includes ``"per/td_errors"`` as a
            NumPy array (not a float), despite the annotation. If you want strict
            typing, change the return type to ``Dict[str, Any]`` or log only scalar
            summaries for TD errors.

        Raises
        ------
        ValueError
            If required batch fields are missing or have incompatible shapes.

        Notes
        -----
        The update order is:
        1) critic step
        2) actor step
        3) scheduler steps (if configured)
        4) optional target update
        5) optional TD-error export for PER
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch tensors to device and normalize shapes
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)  # (B, obs_dim)

        # Actions are indices for gather; normalize to (B,) long then view(-1,1).
        act = batch.actions.to(self.device)
        if act.dim() == 2 and act.shape[1] == 1:
            act = act.squeeze(1)
        act = act.long()  # (B,)

        rew = _to_column(batch.rewards.to(self.device))  # (B,1)
        next_obs = batch.next_observations.to(self.device)  # (B, obs_dim)
        done = _to_column(batch.dones.to(self.device))  # (B,1)

        B = int(obs.shape[0])

        # PER weights (if present). Shape (B,1) or None.
        w = _get_per_weights(batch, B, device=self.device)

        # ---------------------------------------------------------------------
        # Importance sampling ratios: ρ = π(a|s)/μ(a|s)
        # ---------------------------------------------------------------------
        log_mu = self._get_behavior_logp(batch)  # (B,1)

        log_pi = self.head.logp(obs, act)  # expected (B,1) or (B,)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(-1)
        if log_pi.dim() != 2 or log_pi.shape[1] != 1:
            raise ValueError(f"head.logp must return (B,1) or (B,), got {tuple(log_pi.shape)}")

        rho = th.exp(log_pi - log_mu).clamp(max=1e6)  # (B,1)
        c = th.clamp(rho, max=self.c_bar)  # (B,1)

        # ============================================================
        # 1) Target for critic: y = r + γ(1-d) * Vπ(s')
        #
        # Vπ(s') = Σ_a π(a|s') * Q_target(s',a)
        # ============================================================
        with th.no_grad():
            pi_next = self.head.probs(next_obs)  # (B,A)
            q_next_t = self.head.q_values_target(next_obs, reduce="min")  # (B,A)
            v_next_t = (pi_next * q_next_t).sum(dim=1, keepdim=True)  # (B,1)
            target_q = rew + self.gamma * (1.0 - done) * v_next_t  # (B,1)

        # ============================================================
        # 2) Critic update: TD regression (optionally IS-weighted, PER-weighted)
        # ============================================================
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            q_all = self.head.q_values(obs, reduce="min")  # (B,A) (grad-enabled)
            q_sa = q_all.gather(1, act.view(-1, 1))  # (B,1)
            td = target_q - q_sa  # (B,1)

            loss_ps = 0.5 * td.pow(2)  # (B,1)

            if self.critic_is:
                loss_ps = loss_ps * th.clamp(rho, max=self.c_bar)

            if w is not None:
                loss_ps = loss_ps * w

            return loss_ps.mean(), td

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ============================================================
        # 3) Advantage for actor: A(s,a) = Q(s,a) - Vπ(s)
        #
        # Compute under no_grad to avoid actor loss backprop into critic.
        # ============================================================
        with th.no_grad():
            pi = self.head.probs(obs)  # (B,A)
            q_all_ng = self.head.q_values(obs, reduce="min")  # (B,A)
            v_s = (pi * q_all_ng).sum(dim=1, keepdim=True)  # (B,1)

            q_sa_ng = q_all_ng.gather(1, act.view(-1, 1))  # (B,1)
            adv_sa = q_sa_ng - v_s  # (B,1)

            # Advantage over all actions (for bias correction when enabled).
            adv_all = q_all_ng - v_s  # (B,A)

        # ============================================================
        # 4) Actor loss: truncated IS + optional bias correction + entropy
        # ============================================================
        main_term = -(c * adv_sa * log_pi).mean()

        correction = th.zeros((), device=self.device)
        correction_on = False

        if hasattr(batch, "behavior_probs"):
            mu_probs = batch.behavior_probs.to(self.device)  # (B,A)
            pi_probs = pi  # (B,A)

            # ρ(a) for all actions.
            rho_all = (pi_probs / (mu_probs + 1e-8)).clamp(max=1e6)  # (B,A)

            # Weight only where truncation applies.
            w_bc = th.clamp(rho_all - self.c_bar, min=0.0)  # (B,A)

            # log π(·|s) for all actions.
            logits = self._get_logits(obs)  # (B,A)
            log_pi_all = F.log_softmax(logits, dim=-1)  # (B,A)

            # Bias correction term (implementation follows your current convention).
            correction = -(
                (w_bc * pi_probs * log_pi_all * adv_all).sum(dim=1, keepdim=True)
            ).mean()
            correction_on = True

        entropy_term = th.zeros((), device=self.device)
        if self.entropy_coef != 0.0:
            logits = self._get_logits(obs)
            pi_probs2 = F.softmax(logits, dim=-1)
            log_pi_all2 = F.log_softmax(logits, dim=-1)

            entropy = -(pi_probs2 * log_pi_all2).sum(dim=-1, keepdim=True)  # (B,1)
            entropy_term = -self.entropy_coef * entropy.mean()

        actor_loss = main_term + correction + entropy_term

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss_amp = actor_loss
            self.scaler.scale(actor_loss_amp).backward()
            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ============================================================
        # 5) Target update: critic_target <- critic (hard/soft)
        # ============================================================
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ============================================================
        # 6) TD errors for PER (optional): |TD|
        # ============================================================
        with th.no_grad():
            td_abs = td.abs().view(-1)  # (B,)

        return {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "is/rho_mean": float(_to_scalar(rho.mean())),
            "is/c_mean": float(_to_scalar(c.mean())),
            "adv/mean": float(_to_scalar(adv_sa.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "acer/correction_on": float(1.0 if correction_on else 0.0),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state (optimizers, schedulers, counters) plus ACER metadata.

        Returns
        -------
        Dict[str, Any]
            Serializable state mapping.

            The base :class:`ActorCriticCore` typically includes:
            - actor/critic optimizer state
            - actor/critic scheduler state (if enabled)
            - AMP scaler state (if enabled)
            - internal update counters / step trackers

            This override appends ACER-specific hyperparameters as informational
            metadata for debugging and inspection.

        Notes
        -----
        Hyperparameters are constructor-owned; storing them here does not imply that
        :meth:`load_state_dict` will override constructor values.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "c_bar": float(self.c_bar),
                "entropy_coef": float(self.entropy_coef),
                "critic_is": bool(self.critic_is),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state (optimizers/schedulers/counters) from a serialized mapping.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dict produced by :meth:`state_dict`.

        Notes
        -----
        - Delegates to :class:`ActorCriticCore` for restoring optimizer/scheduler/counter
          state.
        - Does **not** silently override constructor-owned hyperparameters (gamma, c_bar,
          etc.). If you want hyperparameter restore, reconstruct the core with the desired
          hyperparameters explicitly.
        """
        super().load_state_dict(state)
        return
