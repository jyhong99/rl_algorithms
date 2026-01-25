from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .head import A2CHead
from .core import A2CCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def a2c(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters (continuous only)
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Gaussian policy parameters
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # -------------------------------------------------------------------------
    # A2C update (core) hyperparameters
    # -------------------------------------------------------------------------
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
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
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,  # continuous actions stored as float
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build an A2C OnPolicyAlgorithm for CONTINUOUS actions (Gaussian policy).

    Pipeline
    --------
    1) Head:
       - Builds the actor+critic networks (Gaussian policy + V(s) baseline).
    2) Core:
       - Implements one update step (policy/value/entropy losses, optimizer steps,
         gradient clipping, optional AMP, optional schedulers).
    3) OnPolicyAlgorithm:
       - Orchestrates rollout collection, batching, and the update schedule.

    Notes
    -----
    - This builder is continuous-only (action_dim required).
    - Advantage normalization (if enabled) is typically handled at the algorithm/buffer level,
      not inside the core.
    """

    # -------------------------------------------------------------------------
    # 1) Head: build actor+critic networks (continuous)
    # -------------------------------------------------------------------------
    head = A2CHead(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
        device=device,
    )

    # -------------------------------------------------------------------------
    # 2) Core: build the update engine (continuous)
    # -------------------------------------------------------------------------
    core = A2CCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # optimizers
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # schedulers
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

    # -------------------------------------------------------------------------
    # 3) Algorithm: rollout collection + update scheduling
    # -------------------------------------------------------------------------
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=None if minibatch_size is None else int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
