from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import torch as th

from .head import ACKTRHead
from .core import ACKTRCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def acktr(
    *,
    # =============================================================================
    # Environment I/O sizes
    # =============================================================================
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",

    # =============================================================================
    # Network (head) hyperparameters (continuous only)
    # =============================================================================
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Gaussian policy std parameterization (continuous actions)
    log_std_mode: str = "param",
    log_std_init: float = -0.5,

    # =============================================================================
    # ACKTR update (core) hyperparameters
    # =============================================================================
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.0,   # 0.0 often means "no clipping" depending on your core
    use_amp: bool = False,

    # =============================================================================
    # K-FAC / ACKTR-specific optimizer knobs (passed to ACKTRCore -> build_optimizer)
    #
    # Note:
    # - ACKTR typically uses K-FAC to approximate natural gradients.
    # - The head is standard actor/critic; the "ACKTR-ness" lives in the optimizer/core.
    # =============================================================================
    actor_optim_name: str = "kfac",
    actor_lr: float = 0.25,
    actor_weight_decay: float = 0.0,
    actor_damping: float = 1e-2,
    actor_momentum: float = 0.9,
    actor_eps: float = 0.95,
    actor_Ts: int = 1,
    actor_Tf: int = 10,
    actor_max_lr: float = 1.0,
    actor_trust_region: float = 2e-3,

    critic_optim_name: str = "kfac",
    critic_lr: float = 0.25,
    critic_weight_decay: float = 0.0,
    critic_damping: float = 1e-2,
    critic_momentum: float = 0.9,
    critic_eps: float = 0.95,
    critic_Ts: int = 1,
    critic_Tf: int = 10,
    critic_max_lr: float = 1.0,
    critic_trust_region: float = 2e-3,

    # =============================================================================
    # (Optional) LR schedulers
    #
    # In many ACKTR/K-FAC setups, schedulers are not used (name="none").
    # Still exposed for parity with other algorithms.
    # =============================================================================
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),

    # =============================================================================
    # OnPolicyAlgorithm rollout / training schedule
    # =============================================================================
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 1,       # ACKTR is commonly 1 epoch per rollout
    minibatch_size: int = None,   # None => full-batch (depending on your OnPolicyAlgorithm)
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,  # continuous actions stored as float
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a complete ACKTR OnPolicyAlgorithm (continuous actions only).

    Pipeline (what this function constructs)
    ---------------------------------------
    1) Head (ACKTRHead):
       - Actor: Gaussian policy network (continuous actions)
       - Critic: V(s) value network
    2) Core (ACKTRCore):
       - A2C-style losses (policy + value + entropy)
       - K-FAC optimizer wiring (damping, trust region, hooks, etc.)
    3) Algorithm (OnPolicyAlgorithm):
       - Rollout collection, return/advantage computation (if supported),
         minibatch iteration, and calling core.update_from_batch().

    Returns
    -------
    algo : OnPolicyAlgorithm
        Typical usage:
          algo.setup(env)
          action = algo.act(obs)
          algo.on_env_step(transition_dict)
          if algo.ready_to_update():
              metrics = algo.update()
    """

    # =============================================================================
    # 1) Head: build actor + critic networks
    # =============================================================================
    head = ACKTRHead(
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

    # =============================================================================
    # 2) Core: build the update engine (K-FAC / ACKTR)
    # =============================================================================
    core = ACKTRCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),

        # ----- actor K-FAC knobs -----
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        actor_damping=float(actor_damping),
        actor_momentum=float(actor_momentum),
        actor_eps=float(actor_eps),
        actor_Ts=int(actor_Ts),
        actor_Tf=int(actor_Tf),
        actor_max_lr=float(actor_max_lr),
        actor_trust_region=float(actor_trust_region),

        # ----- critic K-FAC knobs -----
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        critic_damping=float(critic_damping),
        critic_momentum=float(critic_momentum),
        critic_eps=float(critic_eps),
        critic_Ts=int(critic_Ts),
        critic_Tf=int(critic_Tf),
        critic_max_lr=float(critic_max_lr),
        critic_trust_region=float(critic_trust_region),

        # ----- schedulers (optional) -----
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),

        # ----- grad clipping / AMP -----
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # =============================================================================
    # 3) Algorithm: rollout + update scheduling
    # =============================================================================
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
