from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from .head import DDPGHead
from .core import DDPGCore
from model_free.common.noises.noise_builder import build_noise
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def ddpg(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # =========================================================================
    # Network (head) hyperparams
    # =========================================================================
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    # =========================================================================
    # Exploration noise (HEAD-side)
    # =========================================================================
    exploration_noise: Optional[str] = None,   # e.g. "gaussian", "ou", "uniform", None
    noise_mu: float = 0.0,
    noise_sigma: float = 0.1,
    ou_theta: float = 0.15,
    ou_dt: float = 1e-2,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    action_noise_eps: float = 1e-6,
    action_noise_low: Optional[Union[float, Sequence[float]]] = None,
    action_noise_high: Optional[Union[float, Sequence[float]]] = None,
    noise_clip: Optional[float] = None,        # optional clamp on sampled eps (head-side)
    # =========================================================================
    # DDPG update (core) hyperparams
    # =========================================================================
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # =========================================================================
    # Optimizers
    # =========================================================================
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    # =========================================================================
    # (Optional) schedulers
    # =========================================================================
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # =========================================================================
    # OffPolicyAlgorithm schedule / replay
    # =========================================================================
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    warmup_env_steps: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # =========================================================================
    # PER (Prioritized Experience Replay)
    # =========================================================================
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete DDPG OffPolicyAlgorithm (config-free).

    What this factory returns
    -------------------------
    It returns an `OffPolicyAlgorithm` that is composed of:

      - DDPGHead: actor/critic + target networks + optional exploration noise
      - DDPGCore: the gradient update engine (critic regression + actor policy update)
      - Replay buffer driver: provided by OffPolicyAlgorithm (uniform or PER)

    Key design notes
    ----------------
    1) Exploration noise is handled by the head (DDPGHead.noise).
       - Noise is applied in head.act(...) when deterministic=False.
       - OffPolicyAlgorithm never owns or samples noise.

    2) Scheduling and replay are controlled by OffPolicyAlgorithm:
       - warmup_steps, update_after, update_every, utd, gradient_steps, PER params, etc.

    3) This factory is "config-free":
       - No dependency on external config objects
       - All hyperparameters are explicit function arguments
    """

    # =========================================================================
    # 1) Head-side exploration noise object
    # =========================================================================
    # The noise object is constructed once here and then stored inside DDPGHead.
    #
    # Common choices:
    #   - exploration_noise=None:     deterministic behavior (no added noise)
    #   - exploration_noise="gaussian": Gaussian noise
    #   - exploration_noise="ou":       Ornstein-Uhlenbeck noise (temporally correlated)
    #   - exploration_noise="uniform":  Uniform noise
    #
    # Note:
    # - This noise is typically sampled in "policy action space" (depends on head.act()).
    # - noise_clip (below) can optionally clamp the sampled noise epsilon to limit spikes.
    noise = build_noise(
        kind=exploration_noise,
        action_dim=int(action_dim),
        device=device,
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

    # =========================================================================
    # 2) Head: networks (+ exploration policy)
    # =========================================================================
    # The head is responsible for:
    # - maintaining actor/critic networks
    # - maintaining target networks
    # - action selection (act)
    # - Q evaluation helpers (q_values / q_values_target), depending on your Head base class
    #
    # action_low/high (optional):
    # - If provided, the actor can clamp/rescale its output into env bounds.
    head = DDPGHead(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_sizes=tuple(int(x) for x in hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        action_low=action_low,
        action_high=action_high,
        noise=noise,
        noise_clip=None if noise_clip is None else float(noise_clip),
    )

    # =========================================================================
    # 3) Core: update engine (learning)
    # =========================================================================
    # The core is responsible for:
    # - critic regression towards TD targets
    # - actor optimization via deterministic policy gradient
    # - optional target soft/hard updates through ActorCriticCore utilities
    # - AMP / grad clipping
    core = DDPGCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        # optim
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # sched
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # grad/amp
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # =========================================================================
    # 4) Algorithm: replay buffer + update scheduling driver
    # =========================================================================
    # OffPolicyAlgorithm is the outer loop driver:
    # - stores transitions into replay buffer
    # - decides when to update (based on update_after/update_every/utd/budget)
    # - samples batches from buffer (uniform or PER)
    # - calls core.update_from_batch(batch)
    #
    # reset_noise_on_done=True:
    # - If an episode ends, OffPolicyAlgorithm will call head.reset_exploration_noise()
    #   if the head implements it.
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
        # PER
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        # head-side noise hygiene
        reset_noise_on_done=True,
    )

    return algo