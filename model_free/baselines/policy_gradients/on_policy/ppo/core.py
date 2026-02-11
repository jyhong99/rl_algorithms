from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _reduce_joint, _to_column, _to_scalar


class PPOCore(ActorCriticCore):
    """
    PPO update core for **continuous** action spaces (single minibatch step).

    This class implements the PPO-Clip objective for one minibatch and performs
    one optimizer step for both actor and critic.

    Responsibilities
    ----------------
    - Compute PPO losses from a rollout minibatch:
      - clipped policy surrogate
      - value regression (optionally clipped)
      - entropy bonus
    - Perform backward pass and optimizer steps (actor + critic).
    - Apply global gradient clipping across actor + critic parameters.
    - Optionally use AMP for forward/backward (best-effort on CUDA).
    - Optionally emit a minibatch-level early-stop signal based on target KL.

    Base-class integration
    ----------------------
    :class:`ActorCriticCore` provides common infrastructure:
    - device management and update counter
    - construction and ownership of actor/critic optimizers
    - optional schedulers and scheduler stepping
    - AMP GradScaler and clip helper (``_clip_params``)

    Batch contract
    --------------
    The minibatch object is duck-typed and must provide:

    observations : torch.Tensor
        Shape ``(B, obs_dim)``.
    actions : torch.Tensor
        Shape ``(B, action_dim)`` for continuous actions.
    log_probs : torch.Tensor
        Old log-probabilities ``log π_old(a|s)`` recorded at rollout time.
        Shape ``(B,)`` or ``(B, 1)``.
    values : torch.Tensor
        Old value predictions ``V_old(s)`` recorded at rollout time.
        Shape ``(B,)`` or ``(B, 1)``.
    returns : torch.Tensor
        Return targets. Shape ``(B,)`` or ``(B, 1)``.
    advantages : torch.Tensor
        Advantage estimates. Shape ``(B,)`` or ``(B, 1)``.

    Distribution contract
    ---------------------
    The head must satisfy:

    - ``head.actor.get_dist(obs)`` -> distribution `dist`
    - ``dist.log_prob(actions)`` returns either:
        - ``(B,)`` if already reduced over action dims, or
        - ``(B, action_dim)`` if per-dimension values are returned
    - ``dist.entropy()`` returns either ``(B,)`` or ``(B, action_dim)``

    PPO requires *joint* log-prob and entropy per sample. This implementation uses
    :func:`_reduce_joint` to collapse per-dimension outputs to ``(B,)``, then
    standardizes to column shape ``(B, 1)`` via :func:`_to_column`.

    PPO objective (clip)
    --------------------
    Let

    - :math:`r(\\theta) = \\exp(\\log \\pi_\\theta(a|s) - \\log \\pi_{\\text{old}}(a|s))`
    - :math:`A` be the (possibly normalized) advantage

    Policy loss:
    .. math::
        L_\\pi = -\\mathbb{E}[\\min(r A, \\mathrm{clip}(r, 1-\\epsilon, 1+\\epsilon) A)]

    Value loss:
    - unclipped:
      .. math::
          L_V = \\tfrac{1}{2}\\,\\mathrm{MSE}(V(s), R)
    - clipped variant (if ``clip_vloss=True``):
      .. math::
          V_{\\text{clip}} = V_{\\text{old}} + \\mathrm{clip}(V - V_{\\text{old}}, -\\epsilon, \\epsilon)

      .. math::
          L_V = \\tfrac{1}{2}\\,\\mathbb{E}[\\max((V-R)^2, (V_{\\text{clip}}-R)^2)]

    Entropy bonus (as a loss term):
    .. math::
        L_H = -\\mathbb{E}[H(\\pi(\\cdot|s))]

    Total:
    .. math::
        L = L_\\pi + \\text{vf\\_coef}\\,L_V + \\text{ent\\_coef}\\,L_H

    KL early stopping (minibatch-level)
    -----------------------------------
    If ``target_kl`` is provided, an approximate KL is computed per minibatch and a
    scalar flag ``train/early_stop`` is returned:

    - 1.0 if ``approx_kl > kl_stop_multiplier * target_kl`` else 0.0

    The outer training loop can use this flag to break out of remaining epochs.

    Notes
    -----
    - Advantage normalization, if desired, should happen upstream (buffer/algorithm).
    - This core assumes continuous actions only; discrete PPO would use a different head/core.
    """

    def __init__(
        self,
        *,
        head: Any,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        clip_vloss: bool = True,
        target_kl: Optional[float] = None,
        kl_stop_multiplier: float = 1.0,
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
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
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Actor-critic head providing:
            - ``head.actor.get_dist(obs)`` -> distribution
            - ``head.critic(obs)`` -> value prediction :math:`V(s)`
            - ``head.device`` compatible with :class:`ActorCriticCore`
        clip_range : float, default=0.2
            PPO clipping parameter :math:`\\epsilon`.
        vf_coef : float, default=0.5
            Coefficient applied to the value-loss term.
        ent_coef : float, default=0.0
            Coefficient applied to the entropy-loss term.

            Notes
            -----
            Entropy is implemented as ``ent_loss = -entropy.mean()``, so a positive
            ``ent_coef`` encourages higher entropy.
        clip_vloss : bool, default=True
            If True, apply PPO-style value clipping around the old values.

        target_kl : float | None, default=None
            Target KL threshold for early stopping. If provided, must be > 0.
        kl_stop_multiplier : float, default=1.0
            Stop when ``approx_kl > kl_stop_multiplier * target_kl``. Must be > 0.

        max_grad_norm : float, default=0.5
            Global gradient norm clip threshold. Set to 0 to disable clipping.
        use_amp : bool, default=False
            Enable CUDA AMP for forward/backward (best-effort).

        actor_optim_name, critic_optim_name : str, default="adamw"
            Optimizer identifiers understood by your optimizer builder.
        actor_lr, critic_lr : float, default=3e-4
            Learning rates.
        actor_weight_decay, critic_weight_decay : float, default=0.0
            Weight decay values.

        actor_sched_name, critic_sched_name : str, default="none"
            Scheduler identifiers.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler parameters forwarded to :class:`ActorCriticCore`.

        Raises
        ------
        ValueError
            If ``max_grad_norm < 0`` or invalid KL configuration is provided.
        """
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

        # ---- PPO hyperparameters
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.clip_vloss = bool(clip_vloss)

        # ---- KL early stop configuration
        self.target_kl = None if target_kl is None else float(target_kl)
        self.kl_stop_multiplier = float(kl_stop_multiplier)

        # ---- Gradient clipping
        self.max_grad_norm = float(max_grad_norm)

        # ---- Validate configuration
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError(f"target_kl must be > 0 when provided, got {self.target_kl}")
        if self.kl_stop_multiplier <= 0.0:
            raise ValueError(f"kl_stop_multiplier must be > 0, got {self.kl_stop_multiplier}")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one PPO minibatch update (actor + critic).

        Parameters
        ----------
        batch : Any
            Rollout minibatch providing the fields described in the class docstring.

        Returns
        -------
        Dict[str, float]
            Scalar training metrics. Typical keys:

            Losses
            - ``loss/policy``  : clipped policy surrogate loss
            - ``loss/value``   : value loss (clipped or unclipped)
            - ``loss/entropy`` : entropy loss (negative entropy)
            - ``loss/total``   : total loss

            Stats
            - ``stats/approx_kl`` : approximate KL divergence (scalar)
            - ``stats/clip_frac`` : fraction of samples where ratio was clipped
            - ``stats/entropy``   : mean entropy (positive)
            - ``stats/value_mean``: mean predicted value

            Training signals
            - ``train/early_stop`` : 1.0 if KL early-stop condition met else 0.0
            - ``train/target_kl``  : configured target_kl or 0.0 if disabled

            Learning rates
            - ``lr/actor``  : actor LR (param group 0)
            - ``lr/critic`` : critic LR (param group 0)
        """
        self._bump()

        # ---- Move data to device
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        # Old (behavior) log-prob/value stored during rollout
        old_logp = _to_column(batch.log_probs.to(self.device))
        old_v = _to_column(batch.values.to(self.device))

        # Targets
        ret = _to_column(batch.returns.to(self.device))
        adv = _to_column(batch.advantages.to(self.device))

        clip_eps = self.clip_range

        def _forward_losses() -> Tuple[
            th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor
        ]:
            """
            Compute PPO losses and diagnostics for the current minibatch.

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
            approx_kl : torch.Tensor
                Scalar tensor (approximate KL divergence).
            clip_frac : torch.Tensor
                Scalar tensor (fraction clipped).
            ent_mean : torch.Tensor
                Scalar tensor (mean entropy, positive).
            v_mean : torch.Tensor
                Scalar tensor (mean predicted value).
            """
            # Current policy distribution π(.|s)
            dist = self.head.actor.get_dist(obs)

            # New log-prob and entropy (possibly per-dimension; reduce to joint)
            new_logp = _to_column(_reduce_joint(dist.log_prob(act)))
            entropy = _to_column(_reduce_joint(dist.entropy()))

            # Current critic value V(s)
            v = _to_column(self.head.critic(obs))

            # Policy ratio: exp(log π - log π_old)
            log_ratio = new_logp - old_logp
            ratio = th.exp(log_ratio)

            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = th.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -th.min(surr1, surr2).mean()

            # Diagnostics
            approx_kl = (ratio - 1.0 - log_ratio).mean()
            clip_frac = (th.abs(ratio - 1.0) > clip_eps).float().mean()

            # Value loss (optionally clipped)
            if self.clip_vloss:
                v_clipped = old_v + th.clamp(v - old_v, -clip_eps, clip_eps)
                v_loss_unclipped = (v - ret).pow(2)
                v_loss_clipped = (v_clipped - ret).pow(2)
                value_loss = 0.5 * th.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(v, ret)

            # Entropy bonus (as loss term: negative sign)
            ent_loss = -entropy.mean()

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss

            return (
                total_loss,
                policy_loss,
                value_loss,
                ent_loss,
                approx_kl,
                clip_frac,
                entropy.mean(),
                v.mean(),
            )

        # ---- Backward + optimizer steps
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_frac, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Global clip across actor+critic (common PPO practice).
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_frac, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # ---- Step LR schedulers (if enabled)
        self._step_scheds()

        # ---- target_kl early-stop (minibatch-level signal)
        early_stop = 0.0
        if self.target_kl is not None:
            if float(_to_scalar(approx_kl)) > self.kl_stop_multiplier * self.target_kl:
                early_stop = 1.0

        return {
            "loss/policy": float(_to_scalar(policy_loss)),
            "loss/value": float(_to_scalar(value_loss)),
            "loss/entropy": float(_to_scalar(ent_loss)),
            "loss/total": float(_to_scalar(total_loss)),
            "stats/approx_kl": float(_to_scalar(approx_kl)),
            "stats/clip_frac": float(_to_scalar(clip_frac)),
            "stats/entropy": float(_to_scalar(ent_mean)),
            "stats/value_mean": float(_to_scalar(v_mean)),
            "train/early_stop": float(early_stop),
            "train/target_kl": float(self.target_kl) if self.target_kl is not None else 0.0,
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Return a serialized core state for checkpointing.

        This extends :meth:`ActorCriticCore.state_dict` with PPO-specific
        hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary including:

            - base core state (optimizers, schedulers, counters, AMP scaler, etc.)
            - PPO hyperparameters:
              ``clip_range``, ``vf_coef``, ``ent_coef``, ``clip_vloss``,
              ``target_kl``, ``kl_stop_multiplier``, ``max_grad_norm``
        """
        s = super().state_dict()
        s.update(
            {
                "clip_range": float(self.clip_range),
                "vf_coef": float(self.vf_coef),
                "ent_coef": float(self.ent_coef),
                "clip_vloss": bool(self.clip_vloss),
                "target_kl": None if self.target_kl is None else float(self.target_kl),
                "kl_stop_multiplier": float(self.kl_stop_multiplier),
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
        - Then restores PPO hyperparameters when present. Some codebases treat PPO
          hyperparameters as static config and may choose not to overwrite them
          from checkpoints; this implementation restores them for round-trip symmetry.
        """
        super().load_state_dict(state)

        if "clip_range" in state:
            self.clip_range = float(state["clip_range"])
        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "clip_vloss" in state:
            self.clip_vloss = bool(state["clip_vloss"])
        if "target_kl" in state:
            self.target_kl = None if state["target_kl"] is None else float(state["target_kl"])
        if "kl_stop_multiplier" in state:
            self.kl_stop_multiplier = float(state["kl_stop_multiplier"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
