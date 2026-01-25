from __future__ import annotations

from typing import Any, Tuple, Union

import torch as th

from .head import ACERHead
from .core import ACERCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def acer(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    dueling_mode: bool = False,
    double_q: bool = True,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # ACER update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    c_bar: float = 10.0,
    entropy_coef: float = 0.0,
    critic_is: bool = False,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    per_eps: float = 1e-6,
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
    # OffPolicyAlgorithm schedule / replay
    # -----------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    warmup_env_steps: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # PER (optional: ACER can still benefit for critic regression)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Construct a full ACER OffPolicyAlgorithm (discrete) by composing:
      - Head: neural networks (actor + critic + target critic)
      - Core: update logic (ACER losses and target updates)
      - Algorithm wrapper: replay buffer + update scheduling + PER plumbing

    Why a factory function
    ----------------------
    This provides a "config-free" ergonomic builder: you pass hyperparameters once
    and receive a ready-to-setup OffPolicyAlgorithm instance.

    Critical data requirements (off-policy)
    ---------------------------------------
    ACER uses importance sampling ratios ρ = π(a|s) / μ(a|s), so replay must store:
      - behavior log-prob: batch.behavior_logp (preferred) or batch.logp

    Bias correction term (optional)
    -------------------------------
    If you want the ACER bias-correction term (for truncation), replay must also store:
      - behavior_probs: batch.behavior_probs (μ(a|s) over all actions)

    Returns
    -------
    OffPolicyAlgorithm
        Typical usage pattern (high level):
          - algo.setup(env)
          - action = algo.act(obs)
          - algo.on_env_step(transition)
          - if algo.ready_to_update(): algo.update()

    Notes
    -----
    - PER is optional here; it primarily affects critic regression sampling/weighting.
      The core may also expose TD errors for priority updates (depending on your wrapper).
    - Scheduler-related arguments (total_steps, warmup_steps, etc.) are forwarded into
      the ActorCriticCore scheduler builder.
    """

    # ---------------------------------------------------------------------
    # 1) Head: policy and value networks
    #
    # - actor: categorical policy over discrete actions
    # - q:     Q(s,a) critic (optionally double Q and/or dueling)
    # - q_target: target critic for stable TD targets
    # ---------------------------------------------------------------------
    head = ACERHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        double_q=bool(double_q),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # ---------------------------------------------------------------------
    # 2) Core: ACER update engine
    #
    # - defines critic target + loss, actor loss (truncated IS + bias correction),
    #   entropy regularization, and target network update cadence.
    #
    # - also owns optimizers/schedulers for actor and critic via ActorCriticCore.
    # ---------------------------------------------------------------------
    core = ACERCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        c_bar=float(c_bar),
        entropy_coef=float(entropy_coef),
        critic_is=bool(critic_is),
        # Optimizers (actor / critic)
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # Schedulers (optional)
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # Grad/AMP
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER priority epsilon (core-side proxy)
        per_eps=float(per_eps),
    )

    # ---------------------------------------------------------------------
    # 3) Algorithm: replay buffer + scheduling + PER integration
    #
    # Scheduling knobs (typical semantics)
    # -----------------------------------
    # - warmup_steps / warmup_env_steps: how many env steps to collect before updating
    # - update_after: first env step at which updates are allowed
    # - update_every: perform updates every N env steps (after update_after)
    # - utd / gradient_steps: how many gradient steps to run per update call
    # - max_updates_per_call: safety cap to avoid long stalls in a single call
    #
    # ACER-specific replay requirements
    # --------------------------------
    # store_behavior_logp=True  -> store log μ(a|s) into replay
    # store_behavior_probs=True -> store μ(a|s) probs (enables bias correction)
    # ---------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        warmup_steps=int(warmup_env_steps),
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER (optional)
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        # ACER requires behavior policy info in replay batches
        store_behavior_logp=True,
        store_behavior_probs=True,
    )

    return algo
