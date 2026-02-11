from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple, List

import torch as th
import torch.nn.functional as F

from model_free.common.utils.common_utils import _to_scalar, _to_column
from model_free.common.policies.base_core import BaseCore
from model_free.common.optimizers.optimizer_builder import build_optimizer
from model_free.common.optimizers.scheduler_builder import build_scheduler


class VPGCore(BaseCore):
    """
    Vanilla Policy Gradient (VPG) update engine for continuous control.

    This core performs a single on-policy gradient update per call using:

    - Policy loss:
        .. math::
            \\mathcal{L}_{\\pi} = -\\mathbb{E}[\\log \\pi_\\theta(a\\mid s)\\, A(s,a)]

    - Entropy regularization (optional):
        .. math::
            \\mathcal{L}_{H} = -\\mathbb{E}[H(\\pi_\\theta(\\cdot\\mid s))]

      The total objective adds ``ent_coef * L_H`` (equivalently: adds
      ``+ent_coef * E[entropy]`` to the maximization objective).

    - Value loss (optional baseline):
        .. math::
            \\mathcal{L}_V = \\tfrac{1}{2}\\,\\mathrm{MSE}(V_\\phi(s), R)

      Included only if a value baseline is enabled and scaled by ``vf_coef``.

    Baseline policy (critic usage)
    ------------------------------
    Baseline usage is dictated by the head:

    - If the head has attribute ``use_baseline``, follow it strictly.
    - Else, infer baseline usage as ``(head.critic is not None)``.

    Notes
    -----
    - If the baseline is disabled, critic optimizer/scheduler are not created and
      ``value_loss`` is always a scalar zero.
    - Advantage normalization is intentionally NOT handled here; do it in the rollout
      buffer / algorithm.
    - This core does not compute advantages except for a REINFORCE fallback when
      ``batch.advantages`` is missing (uses returns).
    """

    def __init__(
        self,
        *,
        head: Any,
        # coefficients
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        # optimizers
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # schedulers
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # optional scheduler knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # misc
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Policy head providing at least:
            - ``head.actor.get_dist(obs)`` -> distribution with ``log_prob`` and ``entropy``
            - ``head.critic(obs)`` -> value prediction ``V(s)`` (only if baseline enabled)
            - ``head.device`` -> torch device placement (string or torch.device)

            Baseline usage is determined by:
            - ``head.use_baseline`` if present, otherwise
            - ``(head.critic is not None)``.
        vf_coef : float, default=0.5
            Coefficient for value loss when baseline is enabled.
        ent_coef : float, default=0.0
            Coefficient for entropy regularization. Typical values are small (e.g., 0 to 1e-2).
        actor_optim_name : str, default="adamw"
            Optimizer name for actor parameters (resolved by ``build_optimizer``).
        actor_lr : float, default=3e-4
            Actor learning rate.
        actor_weight_decay : float, default=0.0
            Actor weight decay.
        critic_optim_name : str, default="adamw"
            Optimizer name for critic parameters (only if baseline enabled).
        critic_lr : float, default=3e-4
            Critic learning rate (only if baseline enabled).
        critic_weight_decay : float, default=0.0
            Critic weight decay (only if baseline enabled).
        actor_sched_name : str, default="none"
            Scheduler name for actor LR schedule (resolved by ``build_scheduler``).
        critic_sched_name : str, default="none"
            Scheduler name for critic LR schedule (only if baseline enabled).
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler knobs passed through to ``build_scheduler``. Exact meaning depends on scheduler.
        max_grad_norm : float, default=0.5
            Global norm clipping threshold. Set to 0 to disable clipping.
        use_amp : bool, default=False
            Enable AMP autocast + gradient scaling. Uses BaseCore scaler utilities.

        Raises
        ------
        ValueError
            If ``max_grad_norm`` is negative or baseline configuration is inconsistent
            (e.g., head indicates baseline enabled but ``head.critic`` is missing).
        """
        super().__init__(head=head, use_amp=use_amp)

        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # ------------------------------------------------------------------
        # Baseline usage is dictated by the head
        # ------------------------------------------------------------------
        head_has_critic = getattr(self.head, "critic", None) is not None
        if hasattr(self.head, "use_baseline"):
            self.use_baseline = bool(getattr(self.head, "use_baseline"))
        else:
            self.use_baseline = bool(head_has_critic)

        self._has_critic = bool(head_has_critic)
        if self.use_baseline and not self._has_critic:
            raise ValueError(
                "Head indicates baseline enabled (use_baseline=True) but head.critic is None."
            )

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------
        self.actor_opt = build_optimizer(
            self.head.actor.parameters(),
            name=str(actor_optim_name),
            lr=float(actor_lr),
            weight_decay=float(actor_weight_decay),
        )

        self.critic_opt = None
        if self.use_baseline:
            self.critic_opt = build_optimizer(
                self.head.critic.parameters(),  # type: ignore[attr-defined]
                name=str(critic_optim_name),
                lr=float(critic_lr),
                weight_decay=float(critic_weight_decay),
            )

        # ------------------------------------------------------------------
        # Schedulers (best-effort; may return None)
        # ------------------------------------------------------------------
        self.actor_sched = build_scheduler(
            self.actor_opt,
            name=str(actor_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        self.critic_sched = None
        if self.critic_opt is not None:
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

    # ============================================================
    # Schedulers
    # ============================================================
    def _step_scheds(self) -> None:
        """
        Step actor/critic schedulers if they exist.

        Notes
        -----
        Some algorithm drivers call this hook after each update. This method is
        safe to call regardless of whether schedulers are configured.
        """
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

    # ============================================================
    # Update
    # ============================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one VPG update using an on-policy batch.

        Parameters
        ----------
        batch : Any
            Batch container providing (at minimum):
            - ``observations`` : torch.Tensor, shape (B, obs_dim)
            - ``actions``      : torch.Tensor, shape (B, act_dim) or (act_dim,) if B=1
            - ``returns``      : torch.Tensor, shape (B,) or (B,1)
            - ``advantages``   : torch.Tensor, shape (B,) or (B,1), optional

            If ``advantages`` is missing, the update falls back to REINFORCE
            behavior by using returns as advantages.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar metrics for logging, including:
            - ``loss/policy``, ``loss/entropy``, ``loss/total``
            - ``stats/entropy``
            - learning rates
            - optional value metrics if baseline is enabled

        Notes
        -----
        - This core does NOT compute advantages (except a REINFORCE fallback).
          If advantages are provided, they are treated as fixed targets for policy
          gradients (detached).
        - Entropy and log-prob shapes depend on your distribution implementation.
          They are normalized to column vectors via ``_to_column`` for consistent
          reductions.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        ret = _to_column(batch.returns.to(self.device))

        adv = getattr(batch, "advantages", None)
        if adv is None:
            adv = ret
        else:
            adv = _to_column(adv.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute the total loss and its components.

            Returns
            -------
            total_loss : torch.Tensor
                Scalar tensor for backprop. Includes policy loss, entropy regularization,
                and (optionally) value loss.
            policy_loss : torch.Tensor
                Scalar tensor: negative surrogate objective ``-(logp * adv).mean()``.
            value_loss : torch.Tensor
                Scalar tensor: 0 if baseline disabled, else ``0.5 * MSE(V, returns)``.
            ent_loss : torch.Tensor
                Scalar tensor: negative mean entropy (so adding ent_coef encourages exploration).
            ent_mean : torch.Tensor
                Scalar tensor: mean entropy (for logging).
            v_mean : torch.Tensor
                Scalar tensor: mean value prediction (0 if baseline disabled).
            """
            dist = self.head.actor.get_dist(obs)

            logp = _to_column(dist.log_prob(act))
            entropy = _to_column(dist.entropy())

            policy_loss = -(logp * adv.detach()).mean()
            ent_loss = -entropy.mean()

            value_loss = th.zeros((), device=self.device)
            v_mean = th.zeros((), device=self.device)
            if self.use_baseline:
                v = _to_column(self.head.critic(obs))  # type: ignore[attr-defined]
                value_loss = 0.5 * F.mse_loss(v, ret)
                v_mean = v.mean()

            total_loss = policy_loss + self.ent_coef * ent_loss
            if self.use_baseline:
                total_loss = total_loss + self.vf_coef * value_loss

            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v_mean

        # ------------------------------------------------------------------
        # Zero gradients
        # ------------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        if self.critic_opt is not None:
            self.critic_opt.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------
        # Backward + optimizer steps (optional AMP)
        # ------------------------------------------------------------------
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            params: List[th.nn.Parameter] = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            if self.critic_opt is not None:
                self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            params = list(self.head.actor.parameters())
            if self.use_baseline:
                params += list(self.head.critic.parameters())  # type: ignore[attr-defined]
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.actor_opt.step()
            if self.critic_opt is not None:
                self.critic_opt.step()

        # ------------------------------------------------------------------
        # Scheduler steps
        # ------------------------------------------------------------------
        self._step_scheds()

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        out: Dict[str, float] = {
            "loss/policy": float(_to_scalar(policy_loss)),
            "loss/entropy": float(_to_scalar(ent_loss)),
            "loss/total": float(_to_scalar(total_loss)),
            "stats/entropy": float(_to_scalar(ent_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
        }

        if self.use_baseline:
            out["loss/value"] = float(_to_scalar(value_loss))
            out["stats/value_mean"] = float(_to_scalar(v_mean))
            out["lr/critic"] = float(self.critic_opt.param_groups[0]["lr"]) if self.critic_opt is not None else 0.0

        return out

    # ============================================================
    # Persistence
    # ============================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state.

        Returns
        -------
        state : Dict[str, Any]
            Serializable state dictionary.

        Notes
        -----
        The state includes:
        - actor optimizer/scheduler state
        - critic optimizer/scheduler state (or None if baseline disabled)
        - hyperparameters required to resume training consistently
        """
        s = super().state_dict()
        s.update(
            {
                "actor": self._save_opt_sched(self.actor_opt, self.actor_sched),
                "critic": None
                if self.critic_opt is None
                else self._save_opt_sched(self.critic_opt, self.critic_sched),
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "max_grad_norm": float(self.max_grad_norm),
                "has_baseline": bool(self.use_baseline),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state from a serialized dictionary.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.

        Raises
        ------
        ValueError
            If checkpoint baseline configuration is incompatible with the current head.

        Notes
        -----
        Compatibility rules:
        - If current head baseline is enabled, checkpoint must contain critic optimizer state.
        - If current head baseline is disabled, checkpoint must NOT contain critic optimizer state.
        """
        super().load_state_dict(state)

        if "actor" in state:
            self._load_opt_sched(self.actor_opt, self.actor_sched, state["actor"])

        ckpt_critic = state.get("critic", None)

        if self.use_baseline:
            if ckpt_critic is None:
                raise ValueError(
                    "Checkpoint has no critic optimizer state but head baseline is enabled."
                )
            if self.critic_opt is None:
                raise ValueError(
                    "Head baseline enabled but critic optimizer is None (internal inconsistency)."
                )
            self._load_opt_sched(self.critic_opt, self.critic_sched, ckpt_critic)
        else:
            if ckpt_critic is not None:
                raise ValueError(
                    "Checkpoint contains critic optimizer state but head baseline is disabled."
                )
