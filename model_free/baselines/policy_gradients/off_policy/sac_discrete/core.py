from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import copy
import math

import torch as th
import torch.nn.functional as F

from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.utils.policy_utils import _get_per_weights


class SACDiscreteCore(ActorCriticCore):
    """
    Discrete Soft Actor-Critic (SAC) update engine.

    This core implements the update rules for a discrete-action SAC variant while
    reusing :class:`~model_free.common.policies.base_core.ActorCriticCore` for
    shared infrastructure (optimizers/schedulers, AMP scaler, counters, and
    persistence helpers).

    Overview
    --------
    The discrete SAC objective differs from the continuous version in that the
    actor is a categorical policy over actions, and the critic produces Q-values
    for *all* actions at once:

    - Actor: :math:`\\pi(a\\mid s)` (Categorical over ``A`` actions)
    - Critic: twin Q functions :math:`Q_1(s, \\cdot), Q_2(s, \\cdot)` each shaped
      ``(B, A)``.
    - Target critic: lagged copy of the critic used for stable bootstrapping.
    - Temperature: :math:`\\alpha` (optionally learned by optimizing ``log_alpha``)

    The update uses the “soft value” target:
    .. math::
        V(s') = \\sum_a \\pi(a\\mid s')\\left[\\min(Q_1^t, Q_2^t)(s',a) - \\alpha\\log\\pi(a\\mid s')\\right]

        y = r + \\gamma(1-d)\\,V(s')

    Parameters
    ----------
    head : Any
        Policy head that owns the actor/critic modules and exposes the required
        interfaces listed in **Expected head interface** below.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    tau : float, default=0.005
        Polyak coefficient for target updates in ``(0, 1]``.
    target_update_interval : int, default=1
        Perform a target update every ``target_update_interval`` core update calls.
        Set ``0`` to disable target updates.
    auto_alpha : bool, default=True
        If True, learn temperature by optimizing ``log_alpha``.
    alpha_init : float, default=0.2
        Initial alpha value (temperature). Stored as ``log_alpha = log(alpha_init)``.
    target_entropy : Optional[float], default=None
        Target entropy for temperature learning. If None, a default is inferred
        as ``log(|A|)`` (positive entropy scale). The alpha loss uses the core’s
        sign convention (see Notes).
    actor_optim_name, critic_optim_name : str
        Optimizer names passed to your builder utilities for actor/critic.
    actor_lr, critic_lr : float
        Learning rates for actor/critic optimizers.
    actor_weight_decay, critic_weight_decay : float
        Weight decay for actor/critic optimizers.
    alpha_optim_name : str, default="adamw"
        Optimizer name for alpha (temperature) optimizer (if ``auto_alpha=True``).
    alpha_lr : float, default=3e-4
        Learning rate for alpha optimizer (if enabled).
    alpha_weight_decay : float, default=0.0
        Weight decay for alpha optimizer (if enabled).
    actor_sched_name, critic_sched_name, alpha_sched_name : str
        Scheduler names passed to your scheduler builder.
    total_steps : int, default=0
        Total training steps used by some schedulers.
    warmup_steps : int, default=0
        Warmup steps used by some schedulers.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for polynomial/cosine schedules (builder-dependent).
    poly_power : float, default=1.0
        Polynomial decay power (builder-dependent).
    step_size : int, default=1000
        Step size for step-based schedulers (builder-dependent).
    sched_gamma : float, default=0.99
        Decay rate for step/exponential schedulers (builder-dependent).
    milestones : Sequence[int], default=()
        Milestones for multi-step schedulers (builder-dependent).
    max_grad_norm : float, default=0.0
        If > 0, apply gradient clipping with this L2 norm bound.
    use_amp : bool, default=False
        If True, use torch AMP autocast + GradScaler in updates.
    per_eps : float, default=1e-6
        Epsilon used to clamp TD errors before returning them to PER.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes:
    - ``head.actor`` : nn.Module
    - ``head.critic`` : nn.Module
    - ``head.device`` : torch.device (or string handled by base class)

    Required methods:
    - ``head.dist(obs)`` -> distribution with either ``.logits`` or compatible output
    - ``head.q_values_pair(obs)`` -> (q1, q2), each ``(B, A)``
    - ``head.q_values_target_pair(obs)`` -> (q1_t, q2_t), each ``(B, A)``
    - ``head.freeze_target(module)`` (or base head helper reachable via ``head``)

    Optional attributes:
    - ``head.critic_target`` : nn.Module. If missing, this core deep-copies
      ``head.critic`` into a private target.

    Batch contract
    --------------
    ``update_from_batch(batch)`` expects:
    - ``batch.observations`` : ``(B, obs_dim)``
    - ``batch.actions`` : ``(B,)`` or ``(B,1)`` integer action indices
    - ``batch.rewards`` : ``(B,)`` or ``(B,1)``
    - ``batch.next_observations`` : ``(B, obs_dim)``
    - ``batch.dones`` : ``(B,)`` or ``(B,1)``
    - Optional: ``batch.weights`` for PER importance weights

    Notes
    -----
    **Entropy/alpha sign convention**
        This implementation uses:

        - Actor loss:
          :math:`\\mathbb{E}[\\sum_a \\pi(a\\mid s)(\\alpha\\log\\pi(a\\mid s) - \\min Q(s,a))]`

        - Alpha loss (when auto_alpha enabled):
          :math:`-\\log\\alpha \\cdot (H(\\pi) - H_{target})`

        With this convention:
        - If entropy ``H`` is *below* the target, alpha tends to increase.
        - If entropy ``H`` is *above* the target, alpha tends to decrease.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
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
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        per_eps: float = 1e-6,
    ) -> None:
        # Build actor/critic optimizers + schedulers via ActorCriticCore.
        super().__init__(
            head=head,
            use_amp=use_amp,
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

        # -----------------------------
        # Hyperparameters / validation
        # -----------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)

        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        # -----------------------------
        # Target entropy default (discrete)
        # -----------------------------
        if target_entropy is None:
            n_actions = getattr(self.head, "n_actions", None)
            if n_actions is None:
                n_actions = getattr(self.head, "action_dim", None)
            if n_actions is None:
                raise ValueError(
                    "SACDiscreteCore needs head.n_actions (or head.action_dim) to infer target_entropy."
                )
            self.target_entropy = float(math.log(float(int(n_actions))))
        else:
            self.target_entropy = float(target_entropy)

        ct = getattr(self.head, "critic_target", None)
        if ct is None:
            ct = copy.deepcopy(self.head.critic).to(self.device)
            setattr(self.head, "critic_target", ct)

        self.critic_target = ct

        self.head.freeze_target(self.critic_target)

        # -----------------------------
        # Temperature parameter alpha (log-space)
        # -----------------------------
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")

        self.log_alpha = th.tensor(
            float(math.log(float(alpha_init))),
            device=self.device,
            requires_grad=self.auto_alpha,
        )

        # -----------------------------
        # Alpha optimizer/scheduler (separate from ActorCriticCore)
        # -----------------------------
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

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """
        Current entropy temperature.

        Returns
        -------
        torch.Tensor
            Scalar tensor ``alpha = exp(log_alpha)`` on ``self.device``.
        """
        return self.log_alpha.exp()

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Run one Discrete SAC update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch with fields described in the class docstring.

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar metrics suitable for logging plus PER feedback:
            - ``metrics["per/td_errors"]`` : ``np.ndarray`` of shape ``(B,)``

        Notes
        -----
        - Rewards/dones are normalized to ``(B,1)`` via :func:`_to_column`.
        - If PER weights are present, critic loss is importance-weighted.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ---------------------------------------------------------------------
        # Target computation (no grad)
        #   V(s') = Σ_a π(a|s') [ min(Q_t)(s',a) - α log π(a|s') ]
        #   y = r + γ (1-d) V(s')
        # ---------------------------------------------------------------------
        with th.no_grad():
            dist_next = self.head.dist(nxt)

            logits_next = getattr(dist_next, "logits", None)
            if logits_next is None:
                logits_next = self.head.actor(nxt)

            logp_next_all = F.log_softmax(logits_next, dim=-1)  # (B,A)
            prob_next_all = logp_next_all.exp()                 # (B,A)

            # IMPORTANT: use this core's target critic module, not head.* implicitly.
            q1_t, q2_t = self.critic_target(nxt)                # each (B,A)
            min_q_t = th.min(q1_t, q2_t)                        # (B,A)

            v_next = th.sum(
                prob_next_all * (min_q_t - self.alpha * logp_next_all),
                dim=-1,
                keepdim=True,
            )  # (B,1)

            target_q = rew + self.gamma * (1.0 - done) * v_next  # (B,1)

        # ---------------------------------------------------------------------
        # Critic update (PER-weighted)
        # Regress Q(s,a_taken) toward target_q.
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            q1, q2 = self.head.q_values_pair(obs)  # each (B,A)

            act_idx = act.view(-1).long()
            q1_sa = q1.gather(1, act_idx.view(-1, 1))  # (B,1)
            q2_sa = q2.gather(1, act_idx.view(-1, 1))  # (B,1)

            td1 = target_q - q1_sa
            td2 = target_q - q2_sa

            per_sample = 0.5 * (td1.pow(2) + td2.pow(2))  # (B,1)
            loss = per_sample.mean() if w is None else (w * per_sample).mean()

            td_abs = 0.5 * (td1.abs() + td2.abs()).view(-1)  # (B,)
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
            self.scaler.update()
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update
        #   L_pi = E_s[ Σ_a π(a|s) ( α logπ(a|s) - minQ(s,a) ) ]
        # ---------------------------------------------------------------------
        def _actor_loss_and_entropy() -> Tuple[th.Tensor, th.Tensor]:
            dist = self.head.dist(obs)

            logits = getattr(dist, "logits", None)
            if logits is None:
                logits = self.head.actor(obs)

            logp_all = F.log_softmax(logits, dim=-1)  # (B,A)
            prob_all = logp_all.exp()                 # (B,A)

            # Do not backprop through critic into actor for this objective path.
            with th.no_grad():
                q1_pi, q2_pi = self.head.q_values_pair(obs)  # each (B,A)
                min_q_pi = th.min(q1_pi, q2_pi)              # (B,A)

            per_state = th.sum(prob_all * (self.alpha * logp_all - min_q_pi), dim=-1, keepdim=True)  # (B,1)
            loss = per_state.mean()

            ent = -(prob_all * logp_all).sum(dim=-1, keepdim=True)  # (B,1)
            return loss, ent

        self.actor_opt.zero_grad(set_to_none=True)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, ent = _actor_loss_and_entropy()

            self.scaler.scale(actor_loss).backward()
            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss, ent = _actor_loss_and_entropy()
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Alpha update (optional)
        #   L_alpha = - log_alpha * (H(pi) - H_target)
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (ent.detach() - self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(_to_scalar(alpha_loss))

        # ---------------------------------------------------------------------
        # Target critic update (Polyak / hard)
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=self.critic_target,
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # PER feedback
        td_abs_np = td_abs.clamp(min=self.per_eps).detach().cpu().numpy()

        out: Dict[str, Any] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "stats/alpha": float(_to_scalar(self.alpha)),
            "stats/entropy": float(_to_scalar(ent.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "sac/target_updated": float(
                1.0
                if (self.target_update_interval > 0 and (self.update_calls % self.target_update_interval == 0))
                else 0.0
            ),
            "per/td_errors": td_abs_np,
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        Returns
        -------
        Dict[str, Any]
            A state dictionary that includes:
            - base core state (via ``super().state_dict()``)
            - target critic parameters
            - alpha parameter and optional optimizer/scheduler state
            - key hyperparameters (for reproducibility)
        """
        s = super().state_dict()
        s.update(
            {
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": float(_to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "target_entropy": float(self.target_entropy),
                "max_grad_norm": float(self.max_grad_norm),
                "auto_alpha": bool(self.auto_alpha),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.

        Notes
        -----
        - Restores base core state first (actor/critic opt/sched and counters).
        - Restores target critic weights if present.
        - Restores ``log_alpha`` and alpha optimizer/scheduler state if enabled.
        """
        super().load_state_dict(state)

        if state.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(state["critic_target"])
            self._freeze_target(self.critic_target)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
