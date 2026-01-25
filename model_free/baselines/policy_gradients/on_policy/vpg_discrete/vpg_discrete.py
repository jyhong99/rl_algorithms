from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .head import VPGDiscreteHead
from .core import VPGDiscreteCore

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def vpg_discrete(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    use_baseline: bool = True,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # VPG update (core) hyperparams
    # -----------------------------
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    # -----------------------------
    # Optimizers
    # -----------------------------
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    # -----------------------------
    # (Optional) schedulers
    # -----------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # -----------------------------
    # OnPolicyAlgorithm rollout / schedule
    # -----------------------------
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.int64,  # Discrete actions are typically stored as int64
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a DISCRETE-action VPG OnPolicyAlgorithm (config-free style).

    What this builder constructs
    ----------------------------
    - Head: VPGDiscreteHead
        * Actor: categorical policy π(a|s)
        * Critic (optional): value baseline V(s)

    - Core: VPGDiscreteCore
        * Policy gradient update
        * Optional critic regression update (if baseline enabled)

    - Algorithm: OnPolicyAlgorithm
        * Rollout collection
        * GAE / return computation
        * Update scheduling (when ready_to_update triggers)

    Baseline contract
    -----------------
    - `use_baseline` controls whether the head creates a critic:
        * use_baseline=False => REINFORCE-style (actor-only)
        * use_baseline=True  => actor + value baseline (variance reduction)
    - The core does NOT decide baseline usage independently.
      It follows the head settings and builds critic optimizer/scheduler only
      if the critic exists.

    Notes
    -----
    - For VPG, the common setting is:
        update_epochs = 1
        minibatch_size = None (full-batch update)
    - `gae_lambda` is included for consistency with the shared OnPolicyAlgorithm.
      Whether "true VPG" uses GAE is your design choice; in practice GAE is often
      used even for simple policy gradient implementations.
    """

    # ------------------------------------------------------------------
    # Head: networks (actor + optional value baseline critic)
    # ------------------------------------------------------------------
    head = VPGDiscreteHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        use_baseline=bool(use_baseline),
    )

    # ------------------------------------------------------------------
    # Core: update engine (baseline behavior follows head)
    # ------------------------------------------------------------------
    core = VPGDiscreteCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
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
        # misc
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # ------------------------------------------------------------------
    # Algorithm: rollout + update scheduling
    # ------------------------------------------------------------------
    # OnPolicyAlgorithm owns the rollout buffer and triggers core updates once
    # enough transitions are collected.
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        # None => "full batch" if your OnPolicyAlgorithm supports it
        minibatch_size=None if minibatch_size is None else int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
