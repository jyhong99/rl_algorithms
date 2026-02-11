from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar


class ACKTRCore(ActorCriticCore):
    """
    ACKTR update core for **continuous** action spaces (Gaussian actor).

    This class implements the per-update training step for ACKTR-style methods.

    Conceptual background
    ---------------------
    ACKTR (Actor-Critic using Kronecker-Factored Trust Region) typically differs from
    vanilla A2C/PPO in the **optimizer** rather than the objective:

    - Objective: usually the same A2C-style composite loss
      (policy loss + value loss + entropy bonus)
    - Optimizer: K-FAC (Kronecker-Factored Approximate Curvature) to approximate a
      natural-gradient step.

    This core assumes:
    - Continuous actions only (diagonal Gaussian policy).
    - A state-value critic :math:`V(s)` (not Q(s,a)).
    - Optimizer builder supports K-FAC and may require model hooks/flags for Fisher
      statistics (e.g., ``fisher_backprop``).

    Loss
    ----
    For a rollout minibatch (B samples):

    - Policy gradient term
      .. math::
          L_\\pi = -\\mathbb{E}[\\log \\pi(a\\mid s)\\, A(s,a)]

    - Value regression term
      .. math::
          L_V = \\tfrac{1}{2}\\,\\mathrm{MSE}(V(s), R)

    - Entropy bonus term (implemented as a loss)
      .. math::
          L_H = -\\mathbb{E}[H(\\pi(\\cdot\\mid s))]

    Total objective (minimized)
      .. math::
          L = L_\\pi + \\text{vf\\_coef}\\,L_V + \\text{ent\\_coef}\\,L_H

    Batch contract
    --------------
    The incoming `batch` must expose the following tensor fields:

    - ``batch.observations`` : torch.Tensor, shape ``(B, obs_dim)``
    - ``batch.actions``      : torch.Tensor, shape ``(B, action_dim)`` (float)
      (may also be ``(action_dim,)`` for a single sample depending on upstream code)
    - ``batch.returns``      : torch.Tensor, shape ``(B,)`` or ``(B, 1)``
    - ``batch.advantages``   : torch.Tensor, shape ``(B,)`` or ``(B, 1)``

    Distribution contract (continuous)
    ----------------------------------
    The head must satisfy:

    - ``head.actor.get_dist(obs)`` -> distribution `dist`
    - ``dist.log_prob(action)`` returns either:
        - ``(B,)`` if already reduced over action dims, or
        - ``(B, action_dim)`` if per-dimension values are returned
    - ``dist.entropy()`` returns either ``(B,)`` or ``(B, action_dim)``

    This implementation reduces per-dimension outputs to one scalar per sample by
    summing over the last dimension, and then enforces column shape ``(B, 1)`` via
    :func:`_to_column`.

    AMP and K-FAC
    -------------
    AMP is often discouraged with K-FAC because curvature statistics can be sensitive
    to reduced precision. AMP support is kept as a best-effort option for parity with
    other cores.

    Notes
    -----
    - K-FAC implementations frequently require toggling Fisher-statistics accumulation
      during backward. This core attempts to set ``optimizer.fisher_backprop`` if present.
    - Gradient clipping is applied globally across actor + critic parameters.
      Some K-FAC setups prefer trust-region control instead of clipping; set
      ``max_grad_norm=0`` to disable clipping.
    """

    def __init__(
        self,
        *,
        head: Any,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        actor_optim_name: str = "kfac",
        actor_lr: float = 0.25,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "kfac",
        critic_lr: float = 0.25,
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
        # ---------------------------------------------------------------------
        # K-FAC-specific knobs (passed through build_optimizer via *_optim_kwargs)
        # ---------------------------------------------------------------------
        actor_damping: float = 1e-2,
        actor_momentum: float = 0.9,
        actor_eps: float = 0.95,
        actor_Ts: int = 1,
        actor_Tf: int = 10,
        actor_max_lr: float = 1.0,
        actor_trust_region: float = 2e-3,
        critic_damping: float = 1e-2,
        critic_momentum: float = 0.9,
        critic_eps: float = 0.95,
        critic_Ts: int = 1,
        critic_Tf: int = 10,
        critic_max_lr: float = 1.0,
        critic_trust_region: float = 2e-3,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Actor-critic head providing:
            - ``head.actor.get_dist(obs)`` -> Gaussian-like distribution
            - ``head.critic(obs)`` -> value prediction :math:`V(s)`
            - device handling compatible with :class:`ActorCriticCore`
        vf_coef : float, default=0.5
            Coefficient applied to the value-loss term.
        ent_coef : float, default=0.0
            Coefficient applied to the entropy-loss term.

            Notes
            -----
            Entropy is implemented as ``ent_loss = -entropy.mean()``. Therefore a
            positive ``ent_coef`` encourages exploration (higher entropy).
        max_grad_norm : float, default=0.0
            Global gradient norm clipping threshold.

            - Set to ``0`` to disable clipping.
            - Must be non-negative.
        use_amp : bool, default=False
            Enable CUDA Automatic Mixed Precision (AMP) for forward/backward.

            Notes
            -----
            AMP is best-effort and only meaningful on CUDA. With K-FAC it may
            affect curvature statistics; use with care.

        actor_optim_name, critic_optim_name : str, default="kfac"
            Optimizer identifiers for actor and critic. Typically ``"kfac"``.
        actor_lr, critic_lr : float, default=0.25
            Learning rates for K-FAC updates (semantics are optimizer-dependent).
        actor_weight_decay, critic_weight_decay : float, default=0.0
            Weight decay for actor and critic optimizers.
        actor_sched_name, critic_sched_name : str, default="none"
            Scheduler identifiers (if supported by the base core).
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Shared scheduler parameters passed through to the base class.

        actor_damping, critic_damping : float, default=1e-2
            Damping term for K-FAC (stabilizes curvature inverse).
        actor_momentum, critic_momentum : float, default=0.9
            Momentum term used by the K-FAC optimizer (implementation-dependent).
        actor_eps, critic_eps : float, default=0.95
            Exponential moving-average factor for K-FAC statistics (implementation-dependent).
        actor_Ts, critic_Ts : int, default=1
            Frequency (in steps) to update Kronecker factors (implementation-dependent).
        actor_Tf, critic_Tf : int, default=10
            Frequency (in steps) to update inverse factors (implementation-dependent).
        actor_max_lr, critic_max_lr : float, default=1.0
            Maximum learning rate used by trust-region logic (implementation-dependent).
        actor_trust_region, critic_trust_region : float, default=2e-3
            Trust-region / KL constraint used by the optimizer (implementation-dependent).

        Raises
        ------
        ValueError
            If ``max_grad_norm < 0``.
        """
        super().__init__(
            head=head,
            use_amp=use_amp,
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            # Pass K-FAC hyperparameters through to optimizer builder.
            actor_optim_kwargs={
                "damping": float(actor_damping),
                "momentum": float(actor_momentum),
                "kfac_eps": float(actor_eps),
                "Ts": int(actor_Ts),
                "Tf": int(actor_Tf),
                "max_lr": float(actor_max_lr),
                "trust_region": float(actor_trust_region),
                # "model": self.head.actor  # optional; base may auto-inject
            },
            critic_optim_kwargs={
                "damping": float(critic_damping),
                "momentum": float(critic_momentum),
                "kfac_eps": float(critic_eps),
                "Ts": int(critic_Ts),
                "Tf": int(critic_Tf),
                "max_lr": float(critic_max_lr),
                "trust_region": float(critic_trust_region),
                # "model": self.head.critic  # optional; base may auto-inject
            },
            actor_sched_name=str(actor_sched_name),
            critic_sched_name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            sched_gamma=float(sched_gamma),
            milestones=milestones,
        )

        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

    # =============================================================================
    # Internal helpers: K-FAC fisher_backprop toggle (best-effort)
    # =============================================================================
    @staticmethod
    def _maybe_set_fisher_backprop(opt: Any, enabled: bool) -> None:
        """
        Enable/disable Fisher-statistics backprop if the optimizer supports it.

        Some K-FAC implementations expose a boolean flag (e.g., ``fisher_backprop``)
        controlling whether the backward pass accumulates curvature statistics.

        Parameters
        ----------
        opt : Any
            Optimizer instance (may be a wrapper around a torch optimizer).
        enabled : bool
            If True, enable Fisher-statistics accumulation; otherwise disable it.

        Notes
        -----
        This is best-effort:
        - If the attribute does not exist, this function does nothing.
        """
        if hasattr(opt, "fisher_backprop"):
            setattr(opt, "fisher_backprop", bool(enabled))

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """
        Return a readable learning rate for logging.

        Parameters
        ----------
        opt : Any
            Optimizer instance. Some K-FAC implementations wrap a torch optimizer
            as ``opt.optim``.

        Returns
        -------
        float
            The first param-group learning rate if available, otherwise NaN.

        Notes
        -----
        Supports both layouts:
        - ``opt.optim.param_groups[0]["lr"]``
        - ``opt.param_groups[0]["lr"]``
        """
        if hasattr(opt, "optim") and hasattr(opt.optim, "param_groups"):
            return float(opt.optim.param_groups[0]["lr"])
        if hasattr(opt, "param_groups"):
            return float(opt.param_groups[0]["lr"])
        return float("nan")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one ACKTR update step.

        Parameters
        ----------
        batch : Any
            Rollout minibatch exposing attributes:
            ``observations``, ``actions``, ``returns``, ``advantages``.

        Returns
        -------
        Dict[str, float]
            Scalar training metrics, typically including:

            - ``loss/policy``     : policy gradient loss
            - ``loss/value``      : value regression loss
            - ``loss/entropy``    : entropy loss (negative entropy)
            - ``loss/total``      : combined loss
            - ``stats/entropy``   : mean entropy (positive quantity)
            - ``stats/value_mean``: mean predicted value
            - ``lr/actor``        : current actor learning rate (best-effort)
            - ``lr/critic``       : current critic learning rate (best-effort)

        Notes
        -----
        - Advantages are detached for the policy term.
        - Fisher-statistics accumulation is enabled during backward (if supported)
          and disabled after stepping.
        - Global gradient clipping is applied across actor + critic parameters.
        """
        self._bump()

        # ---- Move tensors to device and standardize shapes
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        ret = _to_column(batch.returns.to(self.device))
        adv = _to_column(batch.advantages.to(self.device))

        def _forward_losses() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute losses and key statistics.

            Returns
            -------
            total_loss : torch.Tensor
                Scalar tensor used for backward.
            policy_loss : torch.Tensor
                Scalar tensor.
            value_loss : torch.Tensor
                Scalar tensor.
            ent_loss : torch.Tensor
                Scalar tensor (negative entropy).
            ent_mean : torch.Tensor
                Mean entropy (positive), used for logging.
            v_mean : torch.Tensor
                Mean predicted value, used for logging.
            """
            # Actor distribution π(.|s)
            dist = self.head.actor.get_dist(obs)

            # log_prob/entropy may be (B,) or (B, action_dim); reduce if needed.
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            logp = _to_column(logp)          # (B, 1)
            entropy = _to_column(entropy)    # (B, 1)

            # Critic value V(s)
            v = _to_column(self.head.critic(obs))

            policy_loss = -(logp * adv.detach()).mean()
            value_loss = 0.5 * F.mse_loss(v, ret)
            ent_loss = -entropy.mean()

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v.mean()

        # ---- Clear gradients
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        # ---- Enable Fisher-statistics accumulation during backward (if supported)
        self._maybe_set_fisher_backprop(self.actor_opt, True)
        self._maybe_set_fisher_backprop(self.critic_opt, True)

        params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())

        # AMP path (CUDA only, best-effort; not generally recommended with K-FAC)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Best-effort unscale before clipping. This may fail for non-torch
            # optimizer wrappers; ignore failures.
            try:
                self.scaler.unscale_(self.actor_opt)
            except Exception:
                pass
            try:
                self.scaler.unscale_(self.critic_opt)
            except Exception:
                pass

            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()

        # Standard FP32 path
        else:
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # ---- Disable Fisher-statistics accumulation after the step (if supported)
        self._maybe_set_fisher_backprop(self.actor_opt, False)
        self._maybe_set_fisher_backprop(self.critic_opt, False)

        # ---- Step LR schedulers (if present)
        self._step_scheds()

        return {
            "loss/policy": float(_to_scalar(policy_loss)),
            "loss/value": float(_to_scalar(value_loss)),
            "loss/entropy": float(_to_scalar(ent_loss)),
            "loss/total": float(_to_scalar(total_loss)),
            "stats/entropy": float(_to_scalar(ent_mean)),
            "stats/value_mean": float(_to_scalar(v_mean)),
            "lr/actor": float(self._get_optimizer_lr(self.actor_opt)),
            "lr/critic": float(self._get_optimizer_lr(self.critic_opt)),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Return a serialized core state for checkpointing.

        This extends :meth:`ActorCriticCore.state_dict` with ACKTR-specific
        hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary including:

            - base core state (optimizers, schedulers, counters, AMP scaler, etc.)
            - ``vf_coef`` : float
            - ``ent_coef`` : float
            - ``max_grad_norm`` : float
        """
        s = super().state_dict()
        s.update(
            {
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "max_grad_norm": float(self.max_grad_norm),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state from a checkpoint dictionary.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary previously produced by :meth:`state_dict`.

        Notes
        -----
        - Restores base-class state first (optimizers/schedulers/counters/AMP scaler).
        - Then restores ACKTR-specific hyperparameters when present.
        """
        super().load_state_dict(state)

        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
