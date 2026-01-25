from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .head import VPGHead
from .core import VPGCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def vpg(
    *,
    obs_dim: int,
    action_dim: int,
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
    # Gaussian params (continuous actions)
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
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
    dtype_act: Any = np.float32,
    # advantage normalization (algorithm-side)
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a complete (continuous) VPG OnPolicyAlgorithm (config-free builder style).

    What this builder wires together
    --------------------------------
    - Head (VPGHead):
        * Actor: unsquashed diagonal Gaussian policy π(a|s)
        * Critic: optional baseline V(s) (enabled by use_baseline)
    - Core (VPGCore):
        * Updates actor (always)
        * Updates critic only if baseline is enabled by the head
        * Supports optional entropy bonus and value loss scaling
    - Algorithm (OnPolicyAlgorithm):
        * Collects rollouts, computes returns/advantages (typically via GAE),
          and calls core.update_from_batch() when ready.

    Baseline policy (important)
    ---------------------------
    - `use_baseline` controls whether the head constructs a critic.
    - The core follows the head configuration:
        * baseline OFF -> actor-only REINFORCE-style updates (no value loss)
        * baseline ON  -> actor + critic updates

    Notes / typical VPG defaults
    ----------------------------
    - VPG is commonly trained with one update per rollout:
        update_epochs=1 and minibatch_size=None (full-batch)
    - If you enable minibatching in your OnPolicyAlgorithm, it should be used with care:
      vanilla VPG is typically derived/used as a full-batch method.
    """

    # ------------------------------------------------------------------
    # Head: networks (actor + optional baseline critic)
    # ------------------------------------------------------------------
    head = VPGHead(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        use_baseline=bool(use_baseline),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
        device=device,
    )

    # ------------------------------------------------------------------
    # Core: update engine (baseline behavior follows head)
    # ------------------------------------------------------------------
    # Even if vf_coef/critic_* are provided, the core will only construct and step
    # a critic optimizer if the head has baseline enabled (i.e., critic exists).
    core = VPGCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        # actor optimizer
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        # critic optimizer (used only when baseline is enabled)
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # schedulers (optional)
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
    # Algorithm: rollout collection + update scheduling
    # ------------------------------------------------------------------
    # minibatch_size:
    # - None => assume full-batch update inside OnPolicyAlgorithm
    # - else => cast to int (implementation-dependent)
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
