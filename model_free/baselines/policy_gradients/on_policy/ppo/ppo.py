from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from .head import PPOHead
from .core import PPOCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


# =============================================================================
# Builder (config-free): same style as your acktr() / a2c() builders
# =============================================================================
def ppo(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters (continuous policy)
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Gaussian std parameterization (continuous-only)
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # -------------------------------------------------------------------------
    # PPO update (core) hyperparameters
    # -------------------------------------------------------------------------
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    clip_vloss: bool = True,
    # PPO early stopping (optional)
    target_kl: Optional[float] = None,
    kl_stop_multiplier: float = 1.0,
    # Gradient clipping + AMP
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    # -------------------------------------------------------------------------
    # (Optional) learning-rate schedulers
    # -------------------------------------------------------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # -------------------------------------------------------------------------
    # OnPolicyAlgorithm rollout / training schedule
    # -------------------------------------------------------------------------
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a complete PPO OnPolicyAlgorithm (config-free).

    This factory wires together the three core components:
      1) PPOHead  : actor/critic networks (Gaussian actor + V(s) critic)
      2) PPOCore  : PPO update rule (clipped objective, value loss, entropy bonus)
      3) OnPolicyAlgorithm : rollout collection + batching + update scheduling

    Returns
    -------
    algo : OnPolicyAlgorithm
        Typical usage:
            algo.setup(env)
            a = algo.act(obs)
            algo.on_env_step(transition)
            if algo.ready_to_update():
                metrics = algo.update()

    Notes
    -----
    - This builder targets continuous action spaces only (Gaussian policy).
    - Advantage normalization is typically handled by OnPolicyAlgorithm/buffer,
      not inside PPOCore.
    """

    # -------------------------------------------------------------------------
    # 1) Head: build actor+critic networks
    # -------------------------------------------------------------------------
    # PPOHead is responsible for:
    # - constructing actor (Gaussian distribution policy)
    # - constructing critic (state-value V(s))
    # - providing act(...) and evaluate_actions(...) APIs expected by OnPolicyAlgorithm
    head = PPOHead(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
    )

    # -------------------------------------------------------------------------
    # 2) Core: build PPO update engine (one minibatch step per call)
    # -------------------------------------------------------------------------
    # PPOCore handles:
    # - clipped policy loss
    # - value loss (+ optional value clipping)
    # - entropy bonus
    # - optimizer steps for actor/critic
    # - optional target-KL early-stop signal
    core = PPOCore(
        head=head,
        clip_range=float(clip_range),
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        clip_vloss=bool(clip_vloss),
        target_kl=(None if target_kl is None else float(target_kl)),
        kl_stop_multiplier=float(kl_stop_multiplier),
        # Optimizers
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # Schedulers
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # Gradient clipping + AMP
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # -------------------------------------------------------------------------
    # 3) Algorithm: rollout collection + update scheduling
    # -------------------------------------------------------------------------
    # OnPolicyAlgorithm is responsible for:
    # - storing rollouts (obs/action/reward/done)
    # - computing returns/advantages (e.g., via GAE)
    # - slicing into minibatches over multiple epochs
    # - calling core.update_from_batch(...) repeatedly
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )

    return algo