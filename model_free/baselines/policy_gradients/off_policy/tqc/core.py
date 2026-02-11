from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import torch as th

from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.utils.policy_utils import _get_per_weights, _quantile_huber_loss


class TQCCore(ActorCriticCore):
    """
    Truncated Quantile Critics (TQC) update engine.

    This core implements TQC (Kuznetsov et al.) on top of the shared
    :class:`~model_free.common.policies.base_core.ActorCriticCore` infrastructure.

    High-level algorithm
    --------------------
    TQC extends SAC by learning a *distribution* over returns via a quantile critic
    ensemble and computing targets from a **truncated** target distribution:

    1) Sample next action from current policy:
       :math:`a' \\sim \\pi(\\cdot\\mid s')`
    2) Compute target quantiles:
       :math:`Z_t(s',a') \\in \\mathbb{R}^{C\\times N}`
    3) Flatten + sort all quantiles, then drop the largest ``top_quantiles_to_drop``:
       :math:`\\tilde{Z}_t(s',a') \\in \\mathbb{R}^{K}` where :math:`K=C\\cdot N-\\text{drop}`
    4) Build per-quantile Bellman backup with entropy term:
       :math:`y = r + \\gamma(1-d)\\,(\\tilde{Z}_t(s',a') - \\alpha\\log\\pi(a'\\mid s'))`
    5) Update critic by minimizing quantile regression loss (Huber quantile loss).
    6) Update actor using a conservative scalar value derived from quantiles.
    7) Optionally update temperature :math:`\\alpha` to match a target entropy.
    8) Polyak update target critic at ``target_update_interval``.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes
    - ``head.actor`` : torch.nn.Module
        Stochastic squashed Gaussian policy (SAC-style).
    - ``head.critic`` : torch.nn.Module
        Quantile critic ensemble; forward returns quantiles shaped ``(B, C, N)``.
    - ``head.critic_target`` : torch.nn.Module
        Target critic with same signature as ``head.critic``.
    - ``head.action_dim`` : int
        Used to infer default ``target_entropy`` when not provided.

    Required methods
    - ``head.sample_action_and_logp(obs) -> (action, logp)``
        ``action`` shaped ``(B, action_dim)`` and ``logp`` preferably ``(B, 1)``.
    - ``head.quantiles(obs, action) -> torch.Tensor``
        Returns ``(B, C, N)``.
    - ``head.quantiles_target(obs, action) -> torch.Tensor``
        Returns ``(B, C, N)`` without gradients.
    - ``head.freeze_target(module)``
        Sets target parameters ``requires_grad=False`` (and typically puts into eval mode).

    Notes
    -----
    - Actor/Critic optimizers, schedulers, AMP scaler, and update counters are managed
      by :class:`ActorCriticCore`. This class additionally owns ``log_alpha`` and its
      optimizer/scheduler when ``auto_alpha=True``.
    - PER integration: per-sample weights are pulled via ``_get_per_weights`` and a
      TD-error proxy is returned under ``"per/td_errors"`` for priority updates.
    """

    def __init__(
        self,
        *,
        head: Any,
        # Core hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        top_quantiles_to_drop: int = 2,
        # Entropy temperature (SAC-style)
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # Actor/Critic optimizers (handled by ActorCriticCore)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # Alpha optimizer (core-owned)
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
        # Gradient / AMP
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Head instance satisfying the "Expected head interface" described in the
            class docstring (TQCHead-like).
        gamma : float, default=0.99
            Discount factor :math:`\\gamma`.
        tau : float, default=0.005
            Polyak averaging factor for target critic updates.
        target_update_interval : int, default=1
            Target update frequency in **update calls**. If ``0``, disables target updates.
        top_quantiles_to_drop : int, default=2
            Number of **largest** quantiles to drop after sorting the flattened target
            quantiles (TQC truncation). Must satisfy ``0 <= drop < C*N``.
        auto_alpha : bool, default=True
            If True, learn temperature :math:`\\alpha` by optimizing ``log_alpha``.
        alpha_init : float, default=0.2
            Initial temperature value. Internally stores ``log_alpha = log(alpha_init)``.
        target_entropy : float or None, default=None
            Target entropy for SAC-style temperature tuning. If None, defaults to
            ``-action_dim`` (continuous SAC heuristic).
        actor_optim_name, critic_optim_name : str
            Optimizer identifiers passed to your optimizer builder for actor/critic.
        actor_lr, critic_lr : float
            Actor/critic learning rates.
        actor_weight_decay, critic_weight_decay : float
            Actor/critic weight decay.
        alpha_optim_name : str
            Optimizer identifier for ``log_alpha`` (only used when ``auto_alpha=True``).
        alpha_lr : float
            Learning rate for ``log_alpha``.
        alpha_weight_decay : float
            Weight decay for ``log_alpha`` (usually 0).
        actor_sched_name, critic_sched_name, alpha_sched_name : str
            Scheduler identifiers passed to your scheduler builder.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler configuration forwarded to your scheduler builder.
        max_grad_norm : float, default=0.0
            Global norm clipping threshold. ``0.0`` disables clipping.
        use_amp : bool, default=False
            If True, uses CUDA AMP autocast + GradScaler for mixed-precision training.

        Raises
        ------
        ValueError
            If any hyperparameter is outside an expected range.
        """
        super().__init__(
            head=head,
            use_amp=bool(use_amp),
            # Optimizers (ActorCriticCore builds actor_opt/critic_opt)
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            # Schedulers (ActorCriticCore builds actor_sched/critic_sched)
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
        self.top_quantiles_to_drop = int(top_quantiles_to_drop)
        self.auto_alpha = bool(auto_alpha)
        self.max_grad_norm = float(max_grad_norm)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # -----------------------------
        # Target entropy default (SAC heuristic)
        # -----------------------------
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # -----------------------------
        # Temperature parameter (log-space)
        # -----------------------------
        # Keep log(alpha) for numerical stability; alpha = exp(log_alpha).
        log_alpha_init = float(math.log(float(alpha_init)))
        self.log_alpha = th.tensor(
            log_alpha_init,
            device=self.device,
            requires_grad=self.auto_alpha,
        )

        # -----------------------------
        # Alpha optimizer/scheduler (core-owned)
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

        # Ensure target critic does not receive gradients.
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """
        Entropy temperature :math:`\\alpha`.

        Returns
        -------
        torch.Tensor
            Scalar tensor computed as ``exp(log_alpha)``.
        """
        return self.log_alpha.exp()

    # =============================================================================
    # Truncation helper
    # =============================================================================
    @staticmethod
    def _truncate_quantiles(z: th.Tensor, top_drop: int) -> th.Tensor:
        """
        Truncate a set of quantiles by dropping the largest values.

        Parameters
        ----------
        z : torch.Tensor
            Quantiles with shape ``(B, C, N)``:
            ``B`` = batch size, ``C`` = ensemble size, ``N`` = quantiles per critic.
        top_drop : int
            Number of highest quantiles to drop after sorting. Must satisfy
            ``0 <= top_drop < C*N``.

        Returns
        -------
        torch.Tensor
            Sorted, truncated quantiles with shape ``(B, K)``, where
            ``K = C*N - top_drop``.

        Raises
        ------
        ValueError
            If ``z`` is not 3D or if ``top_drop`` is outside ``[0, C*N-1]``.
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z to be (B,C,N), got {tuple(z.shape)}")

        b, c, n = z.shape
        flat = z.reshape(b, c * n)  # (B, C*N)
        flat_sorted, _ = th.sort(flat, dim=1)

        drop = int(top_drop)
        if drop < 0 or drop >= c * n:
            raise ValueError(f"top_drop must be in [0, {c*n-1}], got {drop}")

        return flat_sorted if drop == 0 else flat_sorted[:, : (c * n - drop)]

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Run one TQC gradient update from a replay batch.

        Parameters
        ----------
        batch : Any
            A batch object (typically from your replay buffer) providing:
            ``observations``, ``actions``, ``rewards``, ``next_observations``, ``dones``,
            and optionally PER importance weights (retrieved via ``_get_per_weights``).

            Expected tensor shapes:
            - observations:      ``(B, obs_dim)``
            - actions:           ``(B, action_dim)``
            - rewards:           ``(B,)`` or ``(B, 1)``
            - next_observations: ``(B, obs_dim)``
            - dones:             ``(B,)`` or ``(B, 1)``

        Returns
        -------
        Dict[str, Any]
            Scalar metrics for logging plus PER feedback:

            - ``loss/critic`` : float
            - ``loss/actor`` : float
            - ``loss/alpha`` : float
            - ``alpha`` : float
            - ``q/quantile_mean`` : float
            - ``q/pi_min_mean`` : float
            - ``logp_mean`` : float
            - ``lr/actor`` : float
            - ``lr/critic`` : float
            - ``lr/alpha`` : float (if auto-alpha enabled)
            - ``tqc/target_updated`` : float {0.0, 1.0}
            - ``per/td_errors`` : np.ndarray of shape ``(B,)``
        """
        self._bump()

        # -----------------------------
        # Move batch to device and normalize shapes
        # -----------------------------
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))  # (B,1)
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))  # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ---------------------------------------------------------------------
        # Target truncated distribution (no grad)
        # ---------------------------------------------------------------------
        with th.no_grad():
            next_a, next_logp = self.head.sample_action_and_logp(nxt)
            next_logp = _to_column(next_logp)  # robust (B,) -> (B,1)

            z_next = self.head.quantiles_target(nxt, next_a)  # (B,C,N)
            z_trunc = self._truncate_quantiles(z_next, self.top_quantiles_to_drop)  # (B,K)

            # Per-quantile Bellman backup with entropy regularization.
            target = rew + self.gamma * (1.0 - done) * (z_trunc - self.alpha * next_logp)  # (B,K)

            # Expand to match loss helper expectation: (B,C,K)
            target_quantiles = target.unsqueeze(1).expand(-1, z_next.shape[1], -1)

        # ---------------------------------------------------------------------
        # Critic update: quantile regression (Huber)
        # ---------------------------------------------------------------------
        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute critic loss and a TD-error proxy for PER.

            Returns
            -------
            loss : torch.Tensor
                Scalar critic loss.
            td_abs : torch.Tensor
                Per-sample error magnitude proxy with shape ``(B,)``.
            """
            current = self.head.quantiles(obs, act)  # (B,C,N)

            loss, td_err = _quantile_huber_loss(
                current_quantiles=current,
                target_quantiles=target_quantiles,
                cum_prob=None,  # helper may infer taus/quantile fractions internally
                weights=w,
            )

            # Robust TD proxy extraction for PER.
            if isinstance(td_err, th.Tensor):
                if td_err.ndim == 1:
                    td_abs = td_err.detach()
                elif td_err.ndim == 2 and td_err.shape[1] == 1:
                    td_abs = td_err.detach().squeeze(1)
                else:
                    td_abs = td_err.detach().mean(dim=tuple(range(1, td_err.ndim)))
            else:
                # Fallback: compare conservative scalar summaries.
                q_cur = current.mean(dim=-1).min(dim=1).values  # (B,)
                q_tgt = target_quantiles.mean(dim=-1).min(dim=1).values  # (B,)
                td_abs = (q_cur - q_tgt).abs().detach()

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
        # Actor update (SAC-style) using conservative scalar Q proxy
        # ---------------------------------------------------------------------
        def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute actor loss and diagnostics.

            Returns
            -------
            loss : torch.Tensor
                Scalar actor loss.
            logp : torch.Tensor
                Log-probabilities of sampled actions, shape ``(B,1)``.
            q_min : torch.Tensor
                Conservative scalar Q proxy, shape ``(B,1)``.
            """
            new_a, logp = self.head.sample_action_and_logp(obs)
            logp = _to_column(logp)

            z = self.head.quantiles(obs, new_a)  # (B,C,N)
            q_c = z.mean(dim=-1)  # (B,C)
            q_min = th.min(q_c, dim=1).values.unsqueeze(1)  # (B,1)

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

        # ---------------------------------------------------------------------
        # Alpha update (optional, SAC-style)
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(_to_scalar(alpha_loss))

        # ---------------------------------------------------------------------
        # Target critic update
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
            z_cur = self.head.critic(obs, act)  # (B,C,N)
            q_mean = z_cur.mean()

        out: Dict[str, Any] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "alpha": float(_to_scalar(self.alpha)),
            "q/quantile_mean": float(_to_scalar(q_mean)),
            "q/pi_min_mean": float(_to_scalar(q_pi.mean())),
            "logp_mean": float(_to_scalar(logp.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "tqc/target_updated": float(
                1.0
                if (self.target_update_interval > 0 and (self.update_calls % self.target_update_interval == 0))
                else 0.0
            ),
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
        Serialize core state (base + TQC additions).

        ActorCriticCore already serializes:
        - update counter (update_calls)
        - actor optimizer/scheduler state
        - critic optimizer/scheduler state

        This method additionally stores:
        - temperature state (log_alpha + alpha opt/sched state)
        - key hyperparameters for reproducibility/debugging

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(_to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "top_quantiles_to_drop": int(self.top_quantiles_to_drop),
                "target_entropy": float(self.target_entropy),
                "max_grad_norm": float(self.max_grad_norm),
                "auto_alpha": bool(self.auto_alpha),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state (base + TQC additions).

        Parameters
        ----------
        state : Mapping[str, Any]
            State dict produced by :meth:`state_dict`.

        Notes
        -----
        - Optimizer/scheduler reconstruction is assumed to match constructor config.
          This method restores optimizer/scheduler *state* but does not rebuild them.
        - Target critic freezing is enforced in ``__init__`` and target update helper.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
