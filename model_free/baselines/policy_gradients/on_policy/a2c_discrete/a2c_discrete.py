from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import torch as th

from .head import A2CDiscreteHead
from .core import A2CDiscreteCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def a2c_discrete(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters (discrete only)
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
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
    minibatch_size: int = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.int64,  # discrete actions are stored as integer indices
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build an A2C OnPolicyAlgorithm for DISCRETE actions (categorical policy).

    High-level structure
    --------------------
    1) Head:
       - Discrete actor  : categorical π(a|s) over `n_actions`
       - Critic          : V(s) baseline
       Implemented by `A2CDiscreteHead`.

    2) Core:
       - Performs gradient updates for actor/critic using the rollout minibatch.
       - Enforces discrete action conventions (actions -> LongTensor (B,) before log_prob).
       Implemented by `A2CDiscreteCore`.

    3) Algorithm:
       - Handles rollout collection, advantage/return computation, and update scheduling.
       Implemented by `OnPolicyAlgorithm`.

    Notes on dtypes
    --------------
    - dtype_obs: typically float32
    - dtype_act: int64 for discrete actions (categorical indices)
    """

    # -------------------------------------------------------------------------
    # 1) Head: build actor + critic networks (discrete)
    # -------------------------------------------------------------------------
    head = A2CDiscreteHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # -------------------------------------------------------------------------
    # 2) Core: build the update engine (discrete)
    # -------------------------------------------------------------------------
    core = A2CDiscreteCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
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
        # Stability / AMP
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # -------------------------------------------------------------------------
    # 3) Algorithm: rollout collection + update scheduling
    # -------------------------------------------------------------------------
    # Important:
    # - minibatch_size can be None to mean "full batch" depending on your implementation.
    # - update_epochs=1 is typical for classic A2C-style (single pass per rollout).
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=minibatch_size if minibatch_size is None else int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
