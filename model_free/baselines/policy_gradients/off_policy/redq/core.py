from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.utils.policy_utils import _get_per_weights


class REDQCore(ActorCriticCore):
    r"""
    REDQ update engine built on :class:`~model_free.common.policies.base_core.ActorCriticCore`.

    This core implements a SAC-like stochastic actor update combined with an
    ensemble of Q critics and a REDQ-style *random subset minimum* for target
    value estimation.

    Role separation
    --------------
    The project follows a strict separation of concerns:

    - **Head (policy module)** owns network modules and inference utilities:
      actor, critic ensemble, target critic ensemble, and sampling
      (e.g., ``sample_action_and_logp``).

    - **Core (update engine)** owns optimization logic:
      loss construction, optimizer/scheduler steps, AMP, gradient clipping,
      target updates, and logging.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes
        head.actor : torch.nn.Module
            Stochastic policy network.
        head.critics : torch.nn.ModuleList
            Online critic ensemble. Each ``Q_i(obs, act) -> (B,1)``.
        head.critics_target : torch.nn.ModuleList
            Target critic ensemble. Same structure/shapes as ``head.critics``.
        head.device : torch.device or str
            Device for computation (resolved by base core).
        head.action_dim : int
            Action dimension. Used for default entropy target if
            ``target_entropy`` is not provided.

    Required methods
        head.sample_action_and_logp(obs) -> Tuple[action, logp]
            Samples actions from the policy and returns their log-probabilities.

            - action : torch.Tensor
              Shape ``(B, action_dim)``.
            - logp : torch.Tensor
              Shape ``(B,1)`` or ``(B,)``.

    Optional methods
        head.q_values_target_subset_min(obs, act, subset_size=...) -> torch.Tensor
            If present, the core delegates REDQ target min-reduction to the head.
        head.soft_update_target(tau=...)
            If present, the core can delegate target Polyak updates to the head.
        head.freeze_target(module)
            Used to enforce target no-grad (often provided by base head).

    Batch contract
    --------------
    ``update_from_batch(batch)`` expects:

    - batch.observations       : torch.Tensor, shape ``(B, obs_dim)``
    - batch.actions            : torch.Tensor, shape ``(B, action_dim)``
    - batch.rewards            : torch.Tensor, shape ``(B,)`` or ``(B,1)``
    - batch.next_observations  : torch.Tensor, shape ``(B, obs_dim)``
    - batch.dones              : torch.Tensor, shape ``(B,)`` or ``(B,1)``

    PER contract (optional)
    -----------------------
    If PER weights are present (as supported by ``_get_per_weights``), the critic
    loss is importance-weighted.

    REDQ target definition
    ----------------------
    A typical REDQ target mirrors SAC:

    .. math::
        y = r + \gamma (1-d)\left(\min_{i\in\mathcal{I}} Q_i^t(s',a') - \alpha \log \pi(a'|s')\right)

    where :math:`\mathcal{I}` is a uniformly sampled subset of the target critic
    ensemble indices.

    Notes
    -----
    - Critic loss uses the **sum** of per-critic MSEs to the target.
    - PER priorities typically use an absolute TD error derived from the
      ensemble mean: ``|mean_i Q_i(s,a) - y|``.
    - The temperature :math:`\alpha` is optimized in log-space (``log_alpha``)
      for numerical stability when ``auto_alpha=True``.
    """

    def __init__(
        self,
        *,
        head: Any,
        # RL hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        # REDQ subset override
        num_target_subset: Optional[int] = None,
        # Entropy / temperature
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # Optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # Alpha optimizer
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        # Schedulers
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
        # Grad / AMP
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # PER
        per_eps: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Policy head providing actor, critic ensembles, target critic ensembles,
            and action sampling/logp.
        gamma : float, default=0.99
            Discount factor. Must satisfy ``0 <= gamma < 1``.
        tau : float, default=0.005
            Polyak update coefficient for target critics. Must satisfy ``0 <= tau <= 1``.
        target_update_interval : int, default=1
            Target update cadence in update calls. If 0, disables target updates.
        num_target_subset : int, optional
            If provided, overrides the subset size used for REDQ min-reduction.
        auto_alpha : bool, default=True
            If True, optimize temperature (entropy coefficient) automatically.
        alpha_init : float, default=0.2
            Initial alpha value. Stored/optimized in log-space (log_alpha).
        target_entropy : float, optional
            Target entropy for auto-alpha. If None, uses a continuous-control
            heuristic: ``-log(action_dim)`` (preserves your prior behavior).
        actor_optim_name, critic_optim_name : str
            Optimizer identifiers for actor and critic parameters.
        actor_lr, critic_lr : float
            Learning rates.
        actor_weight_decay, critic_weight_decay : float
            Weight decay values.
        alpha_optim_name, alpha_lr, alpha_weight_decay : Any
            Optimizer config for log_alpha when auto_alpha is enabled.
        actor_sched_name, critic_sched_name, alpha_sched_name : str
            Scheduler identifiers for actor/critic/alpha.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler configuration forwarded to scheduler builders.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold (global norm). If 0, clipping is disabled.
        use_amp : bool, default=False
            Enable AMP (mixed precision) via torch.cuda.amp.
        per_eps : float, default=1e-6
            Small epsilon used when reporting TD errors for PER priorities.

        Raises
        ------
        ValueError
            If obvious hyperparameter constraints are violated.
        """
        # ------------------------------------------------------------------
        # 1) Build base ActorCriticCore first
        # ------------------------------------------------------------------
        # Base core typically:
        # - stores self.head, self.device, AMP scaler, counters, etc.
        # - builds actor optimizer/scheduler
        # - builds a critic optimizer/scheduler assuming a single critic module
        #
        # REDQ uses an ensemble of critics, so we rebuild critic optimizer/scheduler
        # below to cover *all* critic parameters.
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

        # ------------------------------------------------------------------
        # 2) Rebuild critic optimizer/scheduler to include ALL ensemble critics
        # ------------------------------------------------------------------
        critic_params = [p for q in self.head.critics for p in q.parameters()]
        self.critic_opt = build_optimizer(
            critic_params,
            name=str(critic_optim_name),
            lr=float(critic_lr),
            weight_decay=float(critic_weight_decay),
        )
        self.critic_sched = build_scheduler(
            self.critic_opt,
            name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        # ------------------------------------------------------------------
        # 3) Store hyperparameters
        # ------------------------------------------------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)

        self.num_target_subset_override: Optional[int] = None
        if num_target_subset is not None:
            self.num_target_subset_override = int(num_target_subset)

        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        # Minimal sanity checks
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        # ------------------------------------------------------------------
        # 4) Entropy target
        # ------------------------------------------------------------------
        # SAC common heuristic is target_entropy = -|A|.
        # You previously used -log(|A|); preserve that behavior unless overridden.
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -math.log(float(action_dim))
        else:
            self.target_entropy = float(target_entropy)

        # ------------------------------------------------------------------
        # 5) Temperature (alpha): optimize log(alpha) for stability
        # ------------------------------------------------------------------
        init_log_alpha = float(th.log(th.tensor(float(alpha_init))).item())
        self.log_alpha = th.tensor(
            init_log_alpha,
            device=self.device,
            requires_grad=bool(self.auto_alpha),
        )

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

        # ------------------------------------------------------------------
        # 6) Freeze all target critics (enforce "no grads into targets")
        # ------------------------------------------------------------------
        for q_t in self.head.critics_target:
            # Prefer base-head freeze helper if available; else fall back to base core.
            fn_freeze = getattr(self.head, "freeze_target", None)
            if callable(fn_freeze):
                fn_freeze(q_t)
            else:
                self._freeze_target(q_t)

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
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

    def _subset_size(self) -> int:
        """
        Determine REDQ subset size used for subset-min computations.

        Resolution priority
        -------------------
        1) ``self.num_target_subset_override`` (core-level explicit override)
        2) ``head.num_target_subset`` (head attribute)
        3) ``head.cfg.num_target_subset`` (if head holds a cfg object)
        4) default to 2

        Returns
        -------
        int
            Subset size ``m`` (positive integer).
        """
        if self.num_target_subset_override is not None:
            return int(self.num_target_subset_override)
        if hasattr(self.head, "num_target_subset"):
            return int(getattr(self.head, "num_target_subset"))
        cfg = getattr(self.head, "cfg", None)
        if cfg is not None and hasattr(cfg, "num_target_subset"):
            return int(getattr(cfg, "num_target_subset"))
        return 2

    @th.no_grad()
    def _subset_min(
        self,
        critics: Sequence[Any],
        obs: th.Tensor,
        act: th.Tensor,
        subset_size: int,
    ) -> th.Tensor:
        """
        Compute :math:`\\min_{i \\in \\mathcal{I}} Q_i(s,a)` over a random subset.

        Parameters
        ----------
        critics : Sequence[Any]
            Sequence of Q networks. Each element must be callable:
            ``Q(obs, act) -> (B,1)``.
        obs : torch.Tensor
            Batched observations of shape ``(B, obs_dim)``.
        act : torch.Tensor
            Batched actions of shape ``(B, action_dim)``.
        subset_size : int
            Subset size ``m`` sampled uniformly without replacement.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B,1)`` containing the minimum Q across the subset.

        Raises
        ------
        ValueError
            If ``subset_size`` is not in ``[1, len(critics)]``.

        Notes
        -----
        - Sampling uses ``torch.randperm`` on ``obs.device`` for CPU/GPU consistency.
        - Deterministic subsets require controlling the PyTorch RNG seed.
        """
        n = len(critics)
        m = int(subset_size)
        if m <= 0 or m > n:
            raise ValueError(f"subset_size must be in [1, {n}], got {m}")

        idx = th.randperm(n, device=obs.device)[:m].tolist()
        qs = [critics[i](obs, act) for i in idx]  # list[(B,1)]
        q_stack = th.stack(qs, dim=0)  # (m, B, 1)
        return th.min(q_stack, dim=0).values  # (B,1)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one REDQ update step from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch satisfying the contract described in the class docstring.

        Returns
        -------
        Dict[str, Any]
            Metrics dictionary containing scalar values for logging and a
            non-scalar TD-error vector for PER priority updates:

            - ``"per/td_errors"`` : np.ndarray, shape ``(B,)``

        Notes
        -----
        The update consists of:

        1) Target construction using target critics subset-min and entropy term.
        2) Critic ensemble update by minimizing sum of per-critic MSE losses.
        3) Actor update using entropy-regularized objective (SAC-like).
        4) Alpha (temperature) update if ``auto_alpha`` is enabled.
        5) Target critics Polyak update on the configured cadence.
        """
        self._bump()

        # -------------------
        # Move batch to device and normalize shapes
        # -------------------
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)  # (B,1) or None
        m = self._subset_size()

        # -------------------
        # Target computation (no grad)
        # -------------------
        with th.no_grad():
            next_a, next_logp = self.head.sample_action_and_logp(nxt)
            if next_logp.dim() == 1:
                next_logp = next_logp.unsqueeze(1)

            fn_subset = getattr(self.head, "q_values_target_subset_min", None)
            if callable(fn_subset):
                q_min_t = fn_subset(nxt, next_a, subset_size=m)  # (B,1)
            else:
                q_min_t = self._subset_min(self.head.critics_target, nxt, next_a, subset_size=m)

            target_q = rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * next_logp)

        # -------------------
        # Critic update (ensemble)
        # -------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute critic loss and TD-error signal.

            Returns
            -------
            loss : torch.Tensor
                Scalar loss: sum of per-critic MSE losses averaged over batch.
            td_abs : torch.Tensor
                Absolute TD error used for PER priority update, shape (B,).
                Defined as: ``|mean_i Q_i(s,a) - target|``.
            """
            qs = [q(obs, act) for q in self.head.critics]  # list[(B,1)]

            per_sample = th.zeros((B, 1), device=self.device)
            for qi in qs:
                per_sample = per_sample + F.mse_loss(qi, target_q, reduction="none")

            q_mean = th.stack(qs, dim=0).mean(dim=0)  # (B,1)
            td_abs = (q_mean - target_q).detach().squeeze(1).abs()  # (B,)

            if w is None:
                loss = per_sample.mean()
            else:
                loss = (w * per_sample).mean()
            return loss, td_abs

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()

            self._clip_params(
                (p for q in self.head.critics for p in q.parameters()),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            # scaler.update() is called after the actor step (single update per iter).
        else:
            critic_loss, td_abs = _critic_loss_and_td()
            critic_loss.backward()

            self._clip_params(
                (p for q in self.head.critics for p in q.parameters()),
                max_grad_norm=self.max_grad_norm,
            )
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # -------------------
        # Actor update (SAC-like)
        # -------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute actor loss and supporting stats.

            Actor objective (entropy-regularized)
            ------------------------------------
            Maximize:  E[ Q(s, a~pi) - alpha * logpi(a|s) ]
            Minimize:  E[ alpha * logpi(a|s) - Q(s, a~pi) ]

            Returns
            -------
            loss : torch.Tensor
                Scalar actor loss.
            logp : torch.Tensor
                Log-probabilities of sampled actions, shape (B,1).
            q_min : torch.Tensor
                Subset-min Q estimates for sampled actions, shape (B,1).
            """
            new_a, logp = self.head.sample_action_and_logp(obs)
            if logp.dim() == 1:
                logp = logp.unsqueeze(1)

            q_min = self._subset_min(self.head.critics, obs, new_a, subset_size=m)
            loss = (self.alpha * logp - q_min).mean()
            return loss, logp, q_min

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

        # -------------------
        # Alpha update (temperature)
        # -------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            # SAC temperature loss:
            #   L = - E[ log_alpha * (logpi + target_entropy) ]
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(_to_scalar(alpha_loss))

        # -------------------
        # Target update (Polyak) for ALL target critics
        # -------------------
        do_target = (self.target_update_interval > 0) and (self.update_calls % self.target_update_interval == 0)
        if do_target:
            fn_soft = getattr(self.head, "soft_update_target", None)
            if callable(fn_soft):
                fn_soft(tau=self.tau)
            else:
                for q_t, q in zip(self.head.critics_target, self.head.critics):
                    self._maybe_update_target(
                        target=q_t,
                        source=q,
                        interval=1,  # already gated by do_target
                        tau=self.tau,
                    )

            # Re-freeze targets defensively (guards against accidental unfreezing).
            for q_t in self.head.critics_target:
                fn_freeze = getattr(self.head, "freeze_target", None)
                if callable(fn_freeze):
                    fn_freeze(q_t)
                else:
                    self._freeze_target(q_t)

        # -------------------
        # Logging stats (no grad)
        # -------------------
        with th.no_grad():
            q_all = [q(obs, act) for q in self.head.critics]  # list[(B,1)]
            q_mean_scalar = th.stack(q_all, dim=0).mean()  # scalar

        out: Dict[str, Any] = {
            # Losses
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            # Alpha / entropy
            "alpha": float(_to_scalar(self.alpha)),
            "logp_mean": float(_to_scalar(logp.mean())),
            # Q statistics
            "q/ensemble_mean": float(_to_scalar(q_mean_scalar)),
            "q/pi_min_mean": float(_to_scalar(q_pi.mean())),
            # LR statistics
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "redq/target_updated": float(1.0 if do_target else 0.0),
            # PER: per-sample TD errors for priority update upstream
            "per/td_errors": td_abs.clamp(min=self.per_eps).detach().cpu().numpy(),
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Extend base core serialization with alpha state.

        Returns
        -------
        Dict[str, Any]
            Base core state (optimizers/schedulers/AMP/counters) plus alpha fields:
            - ``log_alpha`` : float
            - ``auto_alpha`` : bool
            - ``alpha`` : optimizer/scheduler state for temperature (if enabled)

        Notes
        -----
        Hyperparameters (gamma, tau, etc.) are constructor-owned and are not
        restored from this state dict unless you do so explicitly.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "auto_alpha": bool(self.auto_alpha),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base core state and alpha optimizer/scheduler state (if enabled).

        Parameters
        ----------
        state : Mapping[str, Any]
            State mapping produced by :meth:`state_dict`.

        Notes
        -----
        - Delegates to :class:`ActorCriticCore` for restoring base optimizer/scheduler/AMP/counters.
        - Restores ``log_alpha`` value if present.
        - Does not silently toggle ``auto_alpha``; constructor decides whether alpha is optimized.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
