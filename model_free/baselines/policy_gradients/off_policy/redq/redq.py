from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th

from .head import REDQHead
from .core import REDQCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def redq(
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
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    num_critics: int = 10,
    num_target_subset: int = 2,
    # -----------------------------
    # REDQ update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    # core-side override (None -> use head default num_target_subset)
    num_target_subset_override: Optional[int] = None,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
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
    warmup_env_steps: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # PER (replay config)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete REDQ OffPolicyAlgorithm (factory function).

    This function wires together:
      1) Head  (REDQHead):   neural networks + action sampling/inference
      2) Core  (REDQCore):   loss computation + optimization/update rules
      3) Algo  (OffPolicyAlgorithm): replay buffer + update scheduling + PER plumbing

    Parameters (high-level)
    -----------------------
    obs_dim, action_dim:
        Environment observation/action dimensions (continuous actions assumed).
    device:
        Device for the online networks and training updates (e.g., "cpu", "cuda").
        Note: Ray rollout workers may override to CPU via head factory spec.

    Head hyperparams:
        MLP sizes, activation, initialization, policy log-std settings, and
        REDQ ensemble size (num_critics) + target subset size (num_target_subset).

    Core hyperparams:
        gamma/tau/target update interval, entropy temperature configuration,
        gradient clipping and AMP, plus PER epsilon for TD-error stability.

    Optimizers / schedulers:
        Names and parameters forwarded to your optimizer/scheduler builders.

    Replay / schedule:
        OffPolicyAlgorithm settings that control replay buffer size, batch size,
        warmup and update cadence, and PER parameters.

    Returns
    -------
    algo : OffPolicyAlgorithm
        A ready-to-setup algorithm object.

    Typical usage
    -------------
      algo = redq(obs_dim=..., action_dim=..., device="cuda")
      algo.setup(env)
      a = algo.act(obs)
      algo.on_env_step(transition)
      if algo.ready_to_update():
          metrics = algo.update()
    """

    # ------------------------------------------------------------------
    # 1) Head: policy + critic ensembles (network construction only)
    # ------------------------------------------------------------------
    # The head owns:
    # - actor (SAC-style squashed Gaussian)
    # - critic ensemble and target critic ensemble
    # - action sampling utilities (for entropy-regularized objectives)
    head = REDQHead(
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
        num_critics=int(num_critics),
        num_target_subset=int(num_target_subset),
    )

    # ------------------------------------------------------------------
    # 2) Core: update engine (losses + optimizers + target updates)
    # ------------------------------------------------------------------
    # The core owns:
    # - critic/actor/alpha losses
    # - optimizers and optional schedulers
    # - Polyak target updates (or delegates to head if head provides helper)
    # - PER TD-error computation (for priority updates in replay)
    core = REDQCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),

        # If provided, core uses this subset size for REDQ target min-reduction.
        # If None, core will use head.num_target_subset (or head.cfg.num_target_subset).
        num_target_subset=num_target_subset_override,

        # Entropy temperature handling (SAC-style)
        auto_alpha=bool(auto_alpha),
        alpha_init=float(alpha_init),
        target_entropy=(None if target_entropy is None else float(target_entropy)),

        # Optimizer hyperparams
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        alpha_optim_name=str(alpha_optim_name),
        alpha_lr=float(alpha_lr),
        alpha_weight_decay=float(alpha_weight_decay),

        # Optional scheduler hyperparams
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

        # Training stability / performance knobs
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),

        # PER: numeric stability for TD-error clamping
        per_eps=float(per_eps),
    )

    # ------------------------------------------------------------------
    # 3) OffPolicyAlgorithm: replay buffer + update scheduling + PER glue
    # ------------------------------------------------------------------
    # OffPolicyAlgorithm is the "driver":
    # - stores replay buffer (uniform or PER)
    # - decides when to start updating (warmup_steps / update_after)
    # - decides update frequency (update_every, utd, gradient_steps)
    # - calls core.update_from_batch(...) and propagates TD-errors back to PER
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,

        # Replay buffer sizing and batch sampling
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),

        # Warmup: how many env steps to collect before learning begins
        warmup_steps=int(warmup_env_steps),

        # Update schedule controls
        update_after=int(update_after),            # minimum env steps before any update
        update_every=int(update_every),            # update frequency in env steps
        utd=float(utd),                            # update-to-data ratio (how many updates per env step)
        gradient_steps=int(gradient_steps),        # number of gradient steps per update call
        max_updates_per_call=int(max_updates_per_call),

        # PER configuration (if enabled)
        use_per=bool(use_per),
        per_alpha=float(per_alpha),                # priority exponent
        per_beta=float(per_beta),                  # importance sampling exponent (initial)
        per_eps=float(per_eps),                    # small constant to avoid zero priority
        per_beta_final=float(per_beta_final),      # final beta for annealing
        per_beta_anneal_steps=int(per_beta_anneal_steps),
    )
    return algo