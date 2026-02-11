from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from model_free.common.policies.base_core import ActorCriticCore
from model_free.common.utils.common_utils import _to_column, _to_scalar


class PPODiscreteCore(ActorCriticCore):
    """
    PPO update engine for **discrete** action spaces (categorical policies).

    This core implements PPO-Clip for categorical actors, mirroring the continuous
    PPO core but with discrete-action conventions:

    - Actions are discrete indices (typically ``LongTensor`` of shape ``(B,)``).
    - The actor produces a ``Categorical``-like distribution via
      ``head.actor.get_dist(obs)``.

    One-call semantics
    ------------------
    :meth:`update_from_batch` performs **exactly one** minibatch update. Looping over
    epochs/minibatches is the responsibility of the higher-level trainer
    (e.g., ``OnPolicyAlgorithm``).

    PPO objective (discrete)
    ------------------------
    Given rollout-time saved values (old policy and old critic):

    - Policy ratio:
      ``ratio = exp(new_logp - old_logp)``
    - Clipped surrogate:
      ``L_pi = -E[min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)]``

    Value loss
    ----------
    - If ``clip_vloss=True``:
      ``v_clipped = old_v + clip(v - old_v, -eps, eps)``
      ``L_v = 0.5 * E[max((v-ret)^2, (v_clipped-ret)^2)]``
    - Else:
      ``L_v = 0.5 * MSE(v, ret)``

    Entropy bonus
    -------------
    Entropy is added as a bonus by minimizing the *negative* entropy:

    ``L_ent = -E[H(pi(.|s))]``

    Total loss
    ----------
    ``L = L_pi + vf_coef * L_v + ent_coef * L_ent``

    Batch contract
    --------------
    The minibatch object is duck-typed and must provide:

    - ``observations`` : Tensor, shape ``(B, obs_dim)``
    - ``actions``      : Tensor, shape ``(B,)`` or ``(B,1)`` or scalar
    - ``log_probs``    : old log-prob, shape ``(B,)`` or ``(B,1)``
    - ``values``       : old values, shape ``(B,)`` or ``(B,1)``
    - ``returns``      : returns, shape ``(B,)`` or ``(B,1)``
    - ``advantages``   : advantages, shape ``(B,)`` or ``(B,1)``

    Notes
    -----
    - This core standardizes key vectors to column shape ``(B, 1)`` via
      :func:`_to_column`.
    - If ``target_kl`` is configured, an approximate KL diagnostic is computed per
      minibatch and an ``early_stop`` flag is returned (the outer loop decides
      whether to stop).
    """

    def __init__(
        self,
        *,
        head: Any,
        # PPO hyperparameters
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        clip_vloss: bool = True,
        # target_kl (optional)
        target_kl: Optional[float] = None,
        kl_stop_multiplier: float = 1.0,
        # grad / amp
        max_grad_norm: float = 0.5,
        use_amp: bool = False,
        # actor/critic opt/sched (inherited)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # scheduler shared knobs
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
            Discrete actor-critic head providing:
            - ``head.actor.get_dist(obs)`` -> categorical-like distribution
            - ``head.critic(obs)`` -> value tensor
            - ``head._normalize_discrete_action(action)`` -> LongTensor (B,)
        clip_range : float, default=0.2
            PPO clip epsilon :math:`\\epsilon`.
        vf_coef : float, default=0.5
            Coefficient for value loss.
        ent_coef : float, default=0.0
            Coefficient for entropy bonus (implemented via negative entropy loss).
        clip_vloss : bool, default=True
            Whether to apply PPO value clipping around rollout-time values.
        target_kl : float | None, default=None
            If provided, enable minibatch-level KL early-stop signal.
        kl_stop_multiplier : float, default=1.0
            Early-stop threshold multiplier. Stop if
            ``approx_kl > kl_stop_multiplier * target_kl``.
        max_grad_norm : float, default=0.5
            Global gradient norm clipping threshold. Must be ``>= 0``.
        use_amp : bool, default=False
            Enable CUDA AMP (best-effort).

        actor_optim_name, critic_optim_name : str, default="adamw"
            Optimizer names for the base core optimizer builder.
        actor_lr, critic_lr : float, default=3e-4
            Learning rates.
        actor_weight_decay, critic_weight_decay : float, default=0.0
            Weight decay values.

        actor_sched_name, critic_sched_name : str, default="none"
            Scheduler names for the base core scheduler builder.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Scheduler parameters forwarded to the base core.

        Raises
        ------
        ValueError
            If invalid hyperparameters are provided (negative grad norm, non-positive
            target_kl, non-positive kl_stop_multiplier).
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

        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.clip_vloss = bool(clip_vloss)

        self.target_kl = None if target_kl is None else float(target_kl)
        self.kl_stop_multiplier = float(kl_stop_multiplier)

        self.max_grad_norm = float(max_grad_norm)
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
        Perform **one** PPO minibatch update for discrete actions.

        Parameters
        ----------
        batch : Any
            Rollout minibatch with required fields documented in the class docstring.

        Returns
        -------
        Dict[str, float]
            Scalar metrics suitable for logging. Keys include:
            - ``loss/policy``, ``loss/value``, ``loss/entropy``, ``loss/total``
            - ``stats/approx_kl``, ``stats/clip_frac``, ``stats/entropy``, ``stats/value_mean``
            - ``train/early_stop``, ``train/target_kl``
            - ``lr/actor``, ``lr/critic``
        """
        self._bump()

        # ------------------------------------------------------------------
        # Move tensors to device + normalize shapes/dtypes
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)

        # Discrete actions must be LongTensor of shape (B,)
        act_raw = batch.actions.to(self.device)
        act = self.head._normalize_discrete_action(act_raw)

        # Rollout-time (behavior policy) stats
        old_logp = _to_column(batch.log_probs.to(self.device))  # (B,1)
        old_v = _to_column(batch.values.to(self.device))        # (B,1)

        # Targets
        ret = _to_column(batch.returns.to(self.device))         # (B,1)
        adv = _to_column(batch.advantages.to(self.device))      # (B,1)

        clip_eps = self.clip_range

        def _forward_losses() -> Tuple[
            th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor
        ]:
            """
            Compute PPO losses/diagnostics for the current minibatch.

            Returns
            -------
            total_loss : torch.Tensor
            policy_loss : torch.Tensor
            value_loss : torch.Tensor
            ent_loss : torch.Tensor
            approx_kl : torch.Tensor
            clip_frac : torch.Tensor
            ent_mean : torch.Tensor
            v_mean : torch.Tensor
            """
            # Current policy π(.|s)
            dist = self.head.actor.get_dist(obs)

            # Categorical log_prob/entropy are typically (B,)
            new_logp = _to_column(dist.log_prob(act))
            entropy = _to_column(dist.entropy())

            # Current value V(s)
            v = _to_column(self.head.critic(obs))

            # PPO ratio
            log_ratio = new_logp - old_logp
            ratio = th.exp(log_ratio)

            # Clipped policy surrogate
            surr1 = ratio * adv
            surr2 = th.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -th.min(surr1, surr2).mean()

            # Diagnostics
            approx_kl = (ratio - 1.0 - log_ratio).mean()
            clip_frac = (th.abs(ratio - 1.0) > clip_eps).float().mean()

            # Value loss (optional clipping)
            if self.clip_vloss:
                v_clipped = old_v + th.clamp(v - old_v, -clip_eps, clip_eps)
                v_loss_unclipped = (v - ret).pow(2)
                v_loss_clipped = (v_clipped - ret).pow(2)
                value_loss = 0.5 * th.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(v, ret)

            # Entropy bonus as a loss term (negative sign)
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

        # ------------------------------------------------------------------
        # Backward + optimizer step
        # ------------------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                (
                    total_loss,
                    policy_loss,
                    value_loss,
                    ent_loss,
                    approx_kl,
                    clip_frac,
                    ent_mean,
                    v_mean,
                ) = _forward_losses()

            self.scaler.scale(total_loss).backward()

            # One global clip across actor + critic parameters
            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm, optimizer=None)

            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            (
                total_loss,
                policy_loss,
                value_loss,
                ent_loss,
                approx_kl,
                clip_frac,
                ent_mean,
                v_mean,
            ) = _forward_losses()

            total_loss.backward()

            params = list(self.head.actor.parameters()) + list(self.head.critic.parameters())
            self._clip_params(params, max_grad_norm=self.max_grad_norm)

            self.actor_opt.step()
            self.critic_opt.step()

        # Step schedulers (if configured in base core)
        self._step_scheds()

        # ------------------------------------------------------------------
        # target_kl early-stop (per minibatch)
        # ------------------------------------------------------------------
        early_stop = 0.0
        if self.target_kl is not None:
            if float(_to_scalar(approx_kl)) > self.kl_stop_multiplier * self.target_kl:
                early_stop = 1.0

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
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
        Extend :meth:`ActorCriticCore.state_dict` with PPO-discrete hyperparameters.

        Returns
        -------
        Dict[str, Any]
            State dictionary including:
            - base core state (optimizers, schedulers, AMP scaler, update counter)
            - PPO-discrete hyperparameters under ``state["ppo_discrete"]``
        """
        s = super().state_dict()
        s.update(
            {
                "ppo_discrete": {
                    "clip_range": float(self.clip_range),
                    "vf_coef": float(self.vf_coef),
                    "ent_coef": float(self.ent_coef),
                    "clip_vloss": bool(self.clip_vloss),
                    "target_kl": None if self.target_kl is None else float(self.target_kl),
                    "kl_stop_multiplier": float(self.kl_stop_multiplier),
                    "max_grad_norm": float(self.max_grad_norm),
                }
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore base state and PPO-discrete hyperparameters.

        Parameters
        ----------
        state : Mapping[str, Any]
            State mapping produced by :meth:`state_dict`.

        Notes
        -----
        - The base class restores optimizer/scheduler/scaler state.
        - PPO-discrete hyperparameters are restored from ``state["ppo_discrete"]``
          if present, otherwise this method falls back to top-level keys for
          backward compatibility with older checkpoints.
        """
        super().load_state_dict(state)

        ppo_state: Mapping[str, Any] = state.get("ppo_discrete", state)  # backward-compatible

        if "clip_range" in ppo_state:
            self.clip_range = float(ppo_state["clip_range"])
        if "vf_coef" in ppo_state:
            self.vf_coef = float(ppo_state["vf_coef"])
        if "ent_coef" in ppo_state:
            self.ent_coef = float(ppo_state["ent_coef"])
        if "clip_vloss" in ppo_state:
            self.clip_vloss = bool(ppo_state["clip_vloss"])
        if "target_kl" in ppo_state:
            self.target_kl = None if ppo_state["target_kl"] is None else float(ppo_state["target_kl"])
        if "kl_stop_multiplier" in ppo_state:
            self.kl_stop_multiplier = float(ppo_state["kl_stop_multiplier"])
        if "max_grad_norm" in ppo_state:
            self.max_grad_norm = float(ppo_state["max_grad_norm"])
