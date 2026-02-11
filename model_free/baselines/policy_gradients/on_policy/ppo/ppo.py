from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm

from .core import PPOCore
from .head import PPOHead


def ppo(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters (continuous policy)
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Gaussian std parameterization (continuous-only)
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # -------------------------------------------------------------------------
    # PPO update (core) hyperparameters
    # -------------------------------------------------------------------------
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    clip_vloss: bool = True,
    # PPO early stopping (optional)
    target_kl: Optional[float] = None,
    kl_stop_multiplier: float = 1.0,
    # Gradient clipping + AMP
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
    update_epochs: int = 10,
    minibatch_size: int = 64,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a PPO :class:`~model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`
    for **continuous** action spaces (Gaussian policy).

    This is a config-free builder in the same style as your ``a2c()`` / ``acktr()``
    factories. It wires together the three layers of the on-policy stack:

    1) **Head** (:class:`PPOHead`)
       - Actor: diagonal Gaussian policy :math:`\\pi(a\\mid s)` (unsquashed)
       - Critic: state-value baseline :math:`V(s)`

    2) **Core** (:class:`PPOCore`)
       - PPO-Clip objective on each minibatch
       - optional value clipping (``clip_vloss``)
       - entropy bonus
       - optimizer steps for actor and critic
       - optional minibatch-level early-stop signal based on target KL

    3) **Algorithm** (:class:`OnPolicyAlgorithm`)
       - rollout collection of length ``rollout_steps``
       - return / advantage computation (e.g., GAE-λ)
       - minibatch iteration over multiple epochs

    Parameters
    ----------
    obs_dim : int
        Dimension of flattened observations.
    action_dim : int
        Dimension of continuous actions.
    device : str | torch.device, default="cpu"
        Device used by the learner (head + core). Rollout workers (if any) may
        still run on CPU depending on your infrastructure.

    hidden_sizes : tuple[int, ...], default=(64, 64)
        MLP hidden sizes used for both actor and critic networks.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class for MLP layers (e.g., ``nn.ReLU``, ``nn.Tanh``).
    init_type : str, default="orthogonal"
        Weight initialization scheme identifier understood by your network builders.
    gain : float, default=1.0
        Optional initialization gain passed through to network builders.
    bias : float, default=0.0
        Optional initialization bias passed through to network builders.
    log_std_mode : str, default="param"
        Gaussian log-standard-deviation parameterization mode used by the actor.
    log_std_init : float, default=-0.5
        Initial log-std value.

    clip_range : float, default=0.2
        PPO clipping parameter :math:`\\epsilon`.
    vf_coef : float, default=0.5
        Value-loss coefficient.
    ent_coef : float, default=0.0
        Entropy coefficient (implemented as ``ent_loss = -entropy.mean()``, so a
        positive ``ent_coef`` encourages exploration).
    clip_vloss : bool, default=True
        If True, apply PPO-style value clipping around old value predictions.

    target_kl : float | None, default=None
        Target KL threshold for early stopping. If set, the core reports
        ``train/early_stop=1.0`` when the minibatch KL exceeds
        ``kl_stop_multiplier * target_kl``.
    kl_stop_multiplier : float, default=1.0
        Multiplier used with ``target_kl`` for early stopping.

    max_grad_norm : float, default=0.5
        Global gradient norm clipping threshold (0 disables clipping, depending on core).
    use_amp : bool, default=False
        Enable CUDA AMP for forward/backward (best-effort).

    actor_optim_name, critic_optim_name : str, default="adamw"
        Optimizer identifiers for the optimizer builder used by :class:`ActorCriticCore`.
    actor_lr, critic_lr : float, default=3e-4
        Learning rates.
    actor_weight_decay, critic_weight_decay : float, default=0.0
        Weight decay values.

    actor_sched_name, critic_sched_name : str, default="none"
        Scheduler identifiers (if supported).
    total_steps : int, default=0
        Total training steps used by certain schedules.
    warmup_steps : int, default=0
        Warmup steps for schedules that support warmup.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for decay schedules.
    poly_power : float, default=1.0
        Power for polynomial LR decay.
    step_size : int, default=1000
        Step size for step-based schedules.
    sched_gamma : float, default=0.99
        Gamma/decay for exponential or step schedules.
    milestones : tuple[int, ...], default=()
        Milestones for multi-step schedules.

    rollout_steps : int, default=2048
        Number of environment steps collected per rollout iteration.
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE-λ parameter for advantage estimation.
    update_epochs : int, default=10
        Number of epochs over the rollout buffer per iteration.
    minibatch_size : int, default=64
        Minibatch size used when iterating the rollout buffer.
    dtype_obs : Any, default=numpy.float32
        Numpy dtype used to store observations in the rollout buffer.
    dtype_act : Any, default=numpy.float32
        Numpy dtype used to store actions in the rollout buffer.
    normalize_advantages : bool, default=False
        Whether to normalize advantages before updates (typically in algorithm/buffer).
    adv_eps : float, default=1e-8
        Epsilon used when normalizing advantages.

    Returns
    -------
    OnPolicyAlgorithm
        Fully constructed on-policy algorithm instance configured for PPO training.

    Notes
    -----
    - This builder is **continuous-only** (Gaussian actor).
    - If you want advantage normalization, prefer doing it in the buffer/algorithm
      for consistency across cores.
    """
    # -------------------------------------------------------------------------
    # 1) Head: actor + critic networks
    # -------------------------------------------------------------------------
    head = PPOHead(
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
    )

    # -------------------------------------------------------------------------
    # 2) Core: PPO minibatch update engine
    # -------------------------------------------------------------------------
    core = PPOCore(
        head=head,
        clip_range=float(clip_range),
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        clip_vloss=bool(clip_vloss),
        target_kl=None if target_kl is None else float(target_kl),
        kl_stop_multiplier=float(kl_stop_multiplier),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
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
    )

    # -------------------------------------------------------------------------
    # 3) Algorithm: rollout collection + batching + update schedule
    # -------------------------------------------------------------------------
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
