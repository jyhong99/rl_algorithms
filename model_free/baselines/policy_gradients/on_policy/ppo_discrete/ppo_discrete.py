from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm

from .core import PPODiscreteCore
from .head import PPODiscreteHead


def ppo_discrete(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -------------------------------------------------------------------------
    # PPO update (core) hyperparameters
    # -------------------------------------------------------------------------
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    clip_vloss: bool = True,
    target_kl: Optional[float] = None,
    kl_stop_multiplier: float = 1.0,
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
    dtype_act: Any = np.int64,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a PPO on-policy algorithm for **discrete** action spaces.

    This is a config-free factory that wires together:
    1) :class:`PPODiscreteHead`  (networks + action/value interfaces)
    2) :class:`PPODiscreteCore`  (PPO-Clip update rule and optimizer steps)
    3) :class:`OnPolicyAlgorithm` (rollout collection, GAE/returns, batching, epochs)

    The discrete PPO pipeline
    -------------------------
    Head
        - Actor: categorical policy :math:`\\pi(a\\mid s)` over ``n_actions``.
        - Critic: state-value baseline :math:`V(s)`.

    Core
        - PPO clipped surrogate objective using rollout-time stored ``old_logp``.
        - Value loss with optional PPO value clipping around rollout-time ``old_v``.
        - Entropy bonus (implemented as negative entropy loss).
        - Gradient clipping and optional AMP.

    Algorithm
        - Collect ``rollout_steps`` transitions.
        - Compute returns and advantages (e.g., GAE).
        - Iterate for ``update_epochs`` and sample minibatches of ``minibatch_size``.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    n_actions : int
        Number of discrete actions (size of categorical distribution).
    device : str | torch.device, default="cpu"
        Torch device for the learner-side modules (e.g., "cpu", "cuda").

    hidden_sizes : Tuple[int, ...], default=(64, 64)
        Hidden layer sizes for actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function (class or callable) used in MLPs.
    init_type : str, default="orthogonal"
        Initialization scheme name passed to your network builders.
    gain : float, default=1.0
        Initialization gain (library-specific).
    bias : float, default=0.0
        Bias initialization constant (library-specific).

    clip_range : float, default=0.2
        PPO clipping parameter :math:`\\epsilon`.
    vf_coef : float, default=0.5
        Coefficient for value loss.
    ent_coef : float, default=0.0
        Coefficient for entropy bonus.
    clip_vloss : bool, default=True
        If True, apply PPO value clipping using rollout-time values.
    target_kl : float | None, default=None
        If set, compute an approximate KL per minibatch and report an early-stop flag.
    kl_stop_multiplier : float, default=1.0
        Early-stop threshold multiplier:
        stop signal if ``approx_kl > kl_stop_multiplier * target_kl``.
    max_grad_norm : float, default=0.5
        Global gradient clipping max norm (>= 0). Use 0 to disable in your core
        if that is your convention.
    use_amp : bool, default=False
        Enable CUDA automatic mixed precision (best-effort).

    actor_optim_name, critic_optim_name : str, default="adamw"
        Optimizer names passed to the base optimizer builder.
    actor_lr, critic_lr : float, default=3e-4
        Learning rates.
    actor_weight_decay, critic_weight_decay : float, default=0.0
        Weight decay values.

    actor_sched_name, critic_sched_name : str, default="none"
        Scheduler names passed to the base scheduler builder.
    total_steps : int, default=0
        Total steps used by schedulers (if enabled).
    warmup_steps : int, default=0
        Warmup steps used by schedulers (if enabled).
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for some scheduler types (if enabled).
    poly_power : float, default=1.0
        Polynomial decay power (if enabled).
    step_size : int, default=1000
        Step size for step schedulers (if enabled).
    sched_gamma : float, default=0.99
        Gamma for exponential/step schedulers (if enabled).
    milestones : Tuple[int, ...], default=()
        Milestones for multi-step schedulers (if enabled).

    rollout_steps : int, default=2048
        Number of environment steps collected per rollout before an update.
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE lambda for advantage estimation.
    update_epochs : int, default=10
        Number of passes over the rollout buffer per update.
    minibatch_size : int, default=64
        Minibatch size used for PPO updates.
    dtype_obs : Any, default=numpy.float32
        Numpy dtype used to store observations in the rollout buffer.
    dtype_act : Any, default=numpy.int64
        Numpy dtype used to store actions in the rollout buffer.
        For discrete actions, this should typically be an integer dtype.
    normalize_advantages : bool, default=False
        Whether the algorithm/buffer normalizes advantages before updates.
    adv_eps : float, default=1e-8
        Epsilon used in advantage normalization (if enabled).

    Returns
    -------
    OnPolicyAlgorithm
        Fully constructed PPO algorithm instance for discrete actions.

    Examples
    --------
    >>> algo = ppo_discrete(obs_dim=4, n_actions=2, device="cpu")
    >>> algo.setup(env)
    >>> a = algo.act(obs, deterministic=False)
    >>> algo.on_env_step(transition)
    >>> if algo.ready_to_update():
    ...     metrics = algo.update()
    """
    # -------------------------------------------------------------------------
    # 1) Head: actor + critic networks
    # -------------------------------------------------------------------------
    head = PPODiscreteHead(
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
    # 2) Core: PPO update engine (one minibatch update per call)
    # -------------------------------------------------------------------------
    core = PPODiscreteCore(
        head=head,
        clip_range=float(clip_range),
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        clip_vloss=bool(clip_vloss),
        target_kl=None if target_kl is None else float(target_kl),
        kl_stop_multiplier=float(kl_stop_multiplier),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
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
    )

    # -------------------------------------------------------------------------
    # 3) Algorithm: rollout collection + update scheduling
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
