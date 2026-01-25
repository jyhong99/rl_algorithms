from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th

from .head import TQCHead
from .core import TQCCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def tqc(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Actor distribution params (SAC-like squashed Gaussian)
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    # Quantile critic params (TQC-specific)
    n_quantiles: int = 25,  # N: quantiles per critic
    n_nets: int = 2,        # C: number of critic nets (ensemble size)
    # -----------------------------
    # TQC update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    top_quantiles_to_drop: int = 2,   # drop the highest quantiles (overestimation control)
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
    max_grad_norm: float = 0.0,
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
    alpha_optim_name: str = "adamw",
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 0.0,
    # -----------------------------
    # (Optional) schedulers
    # -----------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    alpha_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # -----------------------------
    # OffPolicyAlgorithm schedule / replay
    # -----------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    warmup_env_steps: int = 10_000,   # steps to populate replay before updates
    update_after: int = 1_000,        # begin updates after this many env steps
    update_every: int = 1,            # update cadence (in env steps)
    utd: float = 1.0,                 # updates-to-data ratio
    gradient_steps: int = 1,          # gradient steps per update call
    max_updates_per_call: int = 1_000,
    # PER (Prioritized Experience Replay) config
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete TQC OffPolicyAlgorithm (config-free).

    High-level composition
    ----------------------
    - TQCHead:
        * Actor: squashed Gaussian (SAC-style)
        * Critic: quantile critic ensemble producing Z(s,a) with shape (B, C, N)
        * Critic_target: target ensemble (frozen) for Bellman backup
    - TQCCore:
        * Implements TQC update:
            - Truncate target quantiles by dropping the largest `top_quantiles_to_drop`
            - Quantile regression (Huber) for critic
            - SAC-style actor update using conservative Q from quantiles
            - Optional auto-alpha temperature tuning
            - Periodic soft target update
    - OffPolicyAlgorithm:
        * Replay buffer + scheduling + update loop (UTD, warmup, PER, etc.)

    Returns
    -------
    algo : OffPolicyAlgorithm
        Typical usage:
          algo.setup(env)
          a = algo.act(obs)
          algo.on_env_step(transition)
          if algo.ready_to_update():
              metrics = algo.update()
    """

    # -----------------------------
    # Head: networks (actor + quantile critic ensemble + target)
    # -----------------------------
    head = TQCHead(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        # Actor distribution configuration
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
        # Quantile critic ensemble configuration
        n_quantiles=int(n_quantiles),
        n_nets=int(n_nets),
    )

    # -----------------------------
    # Core: update engine (owns optimizers/schedulers + alpha, executes gradient steps)
    # -----------------------------
    core = TQCCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        top_quantiles_to_drop=int(top_quantiles_to_drop),
        auto_alpha=bool(auto_alpha),
        alpha_init=float(alpha_init),
        target_entropy=(None if target_entropy is None else float(target_entropy)),
        # Optimizers
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        alpha_optim_name=str(alpha_optim_name),
        alpha_lr=float(alpha_lr),
        alpha_weight_decay=float(alpha_weight_decay),
        # Schedulers
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        alpha_sched_name=str(alpha_sched_name),
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
    )

    # -----------------------------
    # Algorithm: replay + scheduling + (optional) PER
    # -----------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # Replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # Schedule
        warmup_steps=int(warmup_env_steps),
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER configuration (OffPolicyAlgorithm + buffer handle the details)
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
    )
    return algo
