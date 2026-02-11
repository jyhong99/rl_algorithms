from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar


class A2CCore(ActorCriticCore):
    """
    A2C update core for **continuous** action spaces.

    This class implements the *update step* (i.e., computing losses, backprop,
    optimizer steps, optional AMP, and scheduler stepping) for an A2C-style
    actor-critic model.

    Overview
    --------
    Given a rollout minibatch containing observations, actions, returns, and
    advantages, this core performs **one** gradient update:

    - Policy gradient term
      .. math::
          L_\\pi = -\\mathbb{E}[\\log \\pi(a\\mid s)\\, A(s,a)]

    - Value regression term
      .. math::
          L_V = \\tfrac{1}{2}\\, \\mathrm{MSE}(V(s), R)

    - Entropy bonus term (implemented as a loss)
      .. math::
          L_H = -\\mathbb{E}[H(\\pi(\\cdot\\mid s))]

    Total objective (minimized)
      .. math::
          L = L_\\pi + \\text{vf\\_coef}\\, L_V + \\text{ent\\_coef}\\, L_H

    Batch contract
    --------------
    The incoming `batch` must expose these tensor fields:

    - ``batch.observations`` : torch.Tensor, shape ``(B, obs_dim)``
    - ``batch.actions``      : torch.Tensor, shape ``(B, act_dim)``
      (may also be ``(act_dim,)`` or scalar for ``act_dim == 1`` depending on env)
    - ``batch.returns``      : torch.Tensor, shape ``(B,)`` or ``(B, 1)``
    - ``batch.advantages``   : torch.Tensor, shape ``(B,)`` or ``(B, 1)``

    Distribution contract (continuous)
    ----------------------------------
    The head must satisfy:

    - ``head.actor.get_dist(obs)`` returns a distribution object `dist`
    - `dist` supports:
        - ``dist.log_prob(action)``
        - ``dist.entropy()``

    Some distributions return per-action-dimension values:

    - ``log_prob`` may be ``(B,)`` or ``(B, act_dim)``
    - ``entropy``  may be ``(B,)`` or ``(B, act_dim)``

    This implementation reduces to one scalar per sample by summing the last
    dimension when necessary, then enforces column shape ``(B, 1)`` via
    :func:`_to_column`.

    Implementation notes
    --------------------
    - Advantages are treated as constants in the policy term via ``adv.detach()``.
    - Gradient clipping is applied *globally* across actor+critic parameters.
    - This core assumes both actor and critic exist (no baseline toggle).
    - Advantage normalization (if desired) should be handled upstream.
    """

    def __init__(
        self,
        *,
        head: Any,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        actor_optim_name: str = "adamw",
        actor_lr: float = 7e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 7e-4,
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
            Actor-critic head with the following attributes:

            - ``head.actor.get_dist(obs)`` -> distribution
            - ``head.critic(obs)`` -> state value ``V(s)``
            - ``head.device`` or equivalent device management via the base class

            The head is typically a module like ``A2CHead`` that bundles actor/critic.
        vf_coef : float, default=0.5
            Coefficient applied to the value loss term.
        ent_coef : float, default=0.0
            Coefficient applied to the entropy loss term.

            Notes
            -----
            Entropy is implemented as a **loss** `ent_loss = -entropy.mean()`.
            Therefore a *positive* ``ent_coef`` encourages exploration.
        max_grad_norm : float, default=0.5
            Global gradient norm clipping threshold.

            - Set to ``0`` to disable clipping.
            - Must be non-negative.
        use_amp : bool, default=False
            Enable CUDA Automatic Mixed Precision (AMP) for forward/backward.

            Notes
            -----
            AMP is best-effort: it is only meaningful on CUDA devices.
        actor_optim_name, critic_optim_name : str, default="adamw"
            Optimizer identifiers understood by the base class / optimizer builder.
        actor_lr, critic_lr : float, default=7e-4
            Learning rates for actor and critic optimizers.
        actor_weight_decay, critic_weight_decay : float, default=0.0
            Weight decay (L2 regularization) for actor and critic optimizers.
        actor_sched_name, critic_sched_name : str, default="none"
            Scheduler identifiers understood by the base class / scheduler builder.
        total_steps : int, default=0
            Total number of environment steps (or update steps) for scheduler planning.
        warmup_steps : int, default=0
            Warmup steps for schedulers that support warmup.
        min_lr_ratio : float, default=0.0
            Minimum LR ratio for decay schedules (base-class dependent).
        poly_power : float, default=1.0
            Power parameter for polynomial decay schedules (if used).
        step_size : int, default=1000
            Step size for step-based schedulers (if used).
        sched_gamma : float, default=0.99
            Gamma for exponential/step schedulers (if used).
        milestones : Sequence[int], default=()
            Milestones for multi-step schedulers (if used).

        Raises
        ------
        ValueError
            If ``max_grad_norm < 0``.
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

        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Perform one A2C update step using a rollout minibatch.

        Parameters
        ----------
        batch : Any
            Rollout minibatch object exposing attributes described in the class docstring:
            ``observations``, ``actions``, ``returns``, ``advantages``.

        Returns
        -------
        Dict[str, float]
            Scalar logging metrics. Typical keys:

            - ``loss/policy``   : policy gradient loss
            - ``loss/value``    : value regression loss
            - ``loss/entropy``  : entropy loss (negative entropy)
            - ``loss/total``    : combined loss
            - ``stats/entropy`` : mean entropy (positive quantity)
            - ``stats/value_mean`` : mean predicted value
            - ``lr/actor``      : current actor learning rate
            - ``lr/critic``     : current critic learning rate

        Notes
        -----
        - Uses :func:`_to_column` to enforce shapes ``(B, 1)`` for returns/advantages
          and for reduced log-prob/entropy/value outputs.
        - Advantages are detached for the policy loss.
        - Clips gradients across *both* actor and critic parameters.
        """
        # Base-class bookkeeping (e.g., update counters, AMP scaler checks, etc.)
        self._bump()

        # ---- Move tensors to device and normalize shapes
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
                Scalar tensor (0-dim) used for backward.
            policy_loss : torch.Tensor
                Scalar tensor.
            value_loss : torch.Tensor
                Scalar tensor.
            ent_loss : torch.Tensor
                Scalar tensor (negative entropy).
            ent_mean : torch.Tensor
                Mean entropy (positive quantity), used for logging.
            v_mean : torch.Tensor
                Mean predicted value, used for logging.
            """
            # ---- Policy distribution π(.|s)
            dist = self.head.actor.get_dist(obs)

            # ---- log π(a|s) and H(π(.|s))
            logp = dist.log_prob(act)
            entropy = dist.entropy()

            # Reduce to one scalar per sample if per-dimension outputs are returned.
            # - e.g., Normal without Independent -> (B, act_dim)
            # - e.g., Independent(Normal(...), 1) -> (B,)
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            logp = _to_column(logp)          # (B, 1)
            entropy = _to_column(entropy)    # (B, 1)

            # ---- Critic value V(s)
            v = _to_column(self.head.critic(obs))  # (B, 1)

            # ---- Loss terms
            policy_loss = -(logp * adv.detach()).mean()
            value_loss = 0.5 * F.mse_loss(v, ret)
            ent_loss = -entropy.mean()

            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            return total_loss, policy_loss, value_loss, ent_loss, entropy.mean(), v.mean()

        # Reset grads (set_to_none=True is usually faster and avoids accumulating zeros)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        # Collect params for global clipping across both modules
        params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())

        # ---- AMP path (CUDA only, best-effort)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # Clip *unscaled* grads (base helper should handle AMP-safe behavior)
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()

        # ---- Standard FP32 path
        else:
            total_loss, policy_loss, value_loss, ent_loss, ent_mean, v_mean = _forward_losses()
            total_loss.backward()

            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # Step schedulers (if enabled/configured by the base class)
        self._step_scheds()

        return {
            "loss/policy": float(_to_scalar(policy_loss)),
            "loss/value": float(_to_scalar(value_loss)),
            "loss/entropy": float(_to_scalar(ent_loss)),
            "loss/total": float(_to_scalar(total_loss)),
            "stats/entropy": float(_to_scalar(ent_mean)),
            "stats/value_mean": float(_to_scalar(v_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Return a serialized state for checkpointing.

        This extends the base :meth:`ActorCriticCore.state_dict` with A2C-specific
        hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary including:

            - base core state (optimizers, schedulers, counters, AMP scaler, etc.)
            - ``vf_coef`` : float
            - ``ent_coef`` : float
            - ``max_grad_norm`` : float

        Notes
        -----
        - Optimizer and scheduler states are managed by the base class.
        - The head's parameters are usually saved by the head/module itself; this
          method focuses on the *core* (update engine) state.
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
        - Then restores A2C-specific hyperparameters when present.
        """
        super().load_state_dict(state)

        if "vf_coef" in state:
            self.vf_coef = float(state["vf_coef"])
        if "ent_coef" in state:
            self.ent_coef = float(state["ent_coef"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
