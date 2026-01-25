from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from .head import TD3Head
from .core import TD3Core
from model_free.common.noises.noise_builder import build_noise
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def td3(
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
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    # -----------------------------
    # Exploration noise (Head-side)
    # -----------------------------
    exploration_noise: Optional[str] = None,
    noise_mu: float = 0.0,
    noise_sigma: float = 0.1,
    ou_theta: float = 0.15,
    ou_dt: float = 1e-2,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    action_noise_eps: float = 1e-6,
    action_noise_low: Optional[Union[float, Sequence[float]]] = None,
    action_noise_high: Optional[Union[float, Sequence[float]]] = None,
    # -----------------------------
    # TD3 update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    target_update_interval: int = 1,
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
    # -----------------------------
    # (Optional) schedulers
    # -----------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
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
    # PER (Prioritized Experience Replay)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete TD3 OffPolicyAlgorithm (config-free).

    Overview
    --------
    This factory wires together three layers:
      1) TD3Head:  actor/critic networks + target networks + exploration noise
      2) TD3Core:  update rule (critic every step, actor delayed, target updates)
      3) OffPolicyAlgorithm: replay buffer + scheduling (when to collect/update)

    Key design choice (important)
    -----------------------------
    Exploration noise is owned and applied by the *head* (TD3Head), not by the
    algorithm driver. This keeps OffPolicyAlgorithm generic across off-policy
    methods (SAC/TD3/REDQ/etc.).

    Contract
    --------
    - Exploration noise is stored inside `TD3Head` (head-side).
    - OffPolicyAlgorithm should NOT receive action_noise nor sample it.
    - Episode-boundary reset is requested by OffPolicyAlgorithm via
      `head.reset_exploration_noise()` when available (enabled by reset_noise_on_done=True).
    """

    # ------------------------------------------------------------------
    # 1) Build exploration noise object (head-side ownership)
    # ------------------------------------------------------------------
    # build_noise() is your centralized noise factory. It supports:
    # - Gaussian / OU / Uniform (depending on `kind`)
    # - optional action-dependent clipping/range via action_noise_low/high
    # - dtype/device handling
    noise = build_noise(
        kind=exploration_noise,       # e.g. "gaussian", "ou", "uniform", or None
        action_dim=int(action_dim),   # noise dimension must match action dimension
        device=device,                # keep noise tensors on same device as policy
        noise_mu=float(noise_mu),
        noise_sigma=float(noise_sigma),
        ou_theta=float(ou_theta),
        ou_dt=float(ou_dt),
        uniform_low=float(uniform_low),
        uniform_high=float(uniform_high),
        action_noise_eps=float(action_noise_eps),
        action_noise_low=action_noise_low,
        action_noise_high=action_noise_high,
    )

    # ------------------------------------------------------------------
    # 2) Head: actor/critic networks + target networks + exploration noise
    # ------------------------------------------------------------------
    # TD3Head is a deterministic actor-critic head:
    # - actor(obs) -> action (optionally clamped to bounds)
    # - critic(obs, act) -> (q1, q2)
    # - actor_target / critic_target maintained for TD targets
    # - noise is applied during `act(..., deterministic=False)` (head decides)
    head = TD3Head(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_sizes=tuple(int(x) for x in hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        # Action bounds:
        # - If provided: TD3Head will clamp both exploration actions and target actions.
        # - If None: actions are unbounded (not typical for Box envs, but allowed).
        action_low=action_low,
        action_high=action_high,
        # Exploration noise lives in the head (NOT in OffPolicyAlgorithm)
        noise=noise,
    )

    # ------------------------------------------------------------------
    # 3) Core: update rule (TD3)
    # ------------------------------------------------------------------
    # TD3Core performs:
    # - critic update every call
    # - actor update every `policy_delay` calls
    # - target updates (Polyak) via tau, optionally gated by target_update_interval
    # - target policy smoothing via (policy_noise, noise_clip)
    core = TD3Core(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        policy_noise=float(policy_noise),     # std for target action smoothing noise
        noise_clip=float(noise_clip),         # clip for target smoothing noise
        policy_delay=int(policy_delay),       # delayed policy update
        target_update_interval=int(target_update_interval),
        # actor/critic optimizer & scheduler settings (reused infra in ActorCriticCore)
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        # stability knobs
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # ------------------------------------------------------------------
    # 4) Algorithm: replay buffer + scheduling
    # ------------------------------------------------------------------
    # OffPolicyAlgorithm responsibilities:
    # - store transitions into replay
    # - decide when to start updating (warmup/update_after)
    # - decide how many updates to run (utd, gradient_steps, max_updates_per_call)
    # - provide PER sampling + report per/beta metrics (if use_per=True)
    # - call head.reset_exploration_noise() on episode end (if enabled)
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        warmup_steps=int(warmup_env_steps),
        # scheduling
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        # IMPORTANT:
        # Ensure the exploration noise state (e.g., OU process) is reset when an episode ends.
        # OffPolicyAlgorithm will call head.reset_exploration_noise() if the method exists.
        reset_noise_on_done=True,
    )
    return algo
