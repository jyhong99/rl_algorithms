from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm

from .core import A2CCore
from .head import A2CHead


def a2c(
    *,
    # -------------------------------------------------------------------------
    # Environment I/O sizes
    # -------------------------------------------------------------------------
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters (continuous only)
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Gaussian policy parameters
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # -------------------------------------------------------------------------
    # A2C update (core) hyperparameters
    # -------------------------------------------------------------------------
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
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
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build an A2C :class:`~model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`
    for **continuous** action spaces.

    This is a *builder* function that wires together three layers:

    1) **Head** (:class:`A2CHead`)
       Owns the neural networks:
       - Actor: diagonal Gaussian policy :math:`\\pi(a\\mid s)`
       - Critic: state-value baseline :math:`V(s)`

    2) **Core** (:class:`A2CCore`)
       Owns the update rule and optimization:
       - policy/value/entropy losses
       - optimizer steps (actor + critic)
       - optional AMP
       - global grad clipping
       - optional LR schedulers (via base core)

    3) **Algorithm** (:class:`OnPolicyAlgorithm`)
       Owns rollout collection and training schedule:
       - collects `rollout_steps` transitions
       - computes returns/advantages (GAE-λ)
       - runs `update_epochs` epochs of minibatch updates

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened), i.e., number of features in ``obs``.
    action_dim : int
        Continuous action dimension (flattened).
    device : str | torch.device, default="cpu"
        Torch device used by the learner/head (e.g., ``"cpu"``, ``"cuda:0"``).

        Notes
        -----
        Ray rollout workers (if used) may still force CPU for worker-side policies
        via :meth:`A2CHead.get_ray_policy_factory_spec`.

    hidden_sizes : tuple[int, ...], default=(64, 64)
        MLP hidden layer sizes shared by actor and critic networks.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class for MLPs (e.g., ``nn.ReLU``, ``nn.Tanh``).
    init_type : str, default="orthogonal"
        Weight initialization scheme name understood by your network builders.
    gain : float, default=1.0
        Optional initialization gain passed through to network builders.
    bias : float, default=0.0
        Optional initialization bias passed through to network builders.
    log_std_mode : str, default="param"
        Log-standard-deviation parameterization for the Gaussian actor.

        Common options (implementation-dependent)
        - ``"param"``: trainable, state-independent log-std vector.
        - other modes may exist in your codebase.
    log_std_init : float, default=-0.5
        Initial log-std value (used when ``log_std_mode="param"``).

    vf_coef : float, default=0.5
        Value-loss coefficient used by :class:`A2CCore`.
    ent_coef : float, default=0.0
        Entropy coefficient used by :class:`A2CCore`.

        Notes
        -----
        Entropy is implemented as a loss term ``ent_loss = -entropy.mean()``, so
        positive ``ent_coef`` encourages exploration.
    max_grad_norm : float, default=0.5
        Global gradient norm clipping threshold (0 disables clipping).
    use_amp : bool, default=False
        Enable CUDA AMP in the core (best-effort; meaningful on CUDA).

    actor_optim_name, critic_optim_name : str, default="adamw"
        Optimizer identifiers for actor and critic optimizer builders.
    actor_lr, critic_lr : float, default=3e-4
        Learning rates for actor and critic.
    actor_weight_decay, critic_weight_decay : float, default=0.0
        Weight decay (L2) for actor and critic.

    actor_sched_name, critic_sched_name : str, default="none"
        Scheduler identifiers (if supported by your base core).
    total_steps : int, default=0
        Total steps used by schedules requiring a horizon (polynomial, cosine, etc.).
    warmup_steps : int, default=0
        Warmup steps for schedules that support warmup.
    min_lr_ratio : float, default=0.0
        Minimum LR as a ratio of the base LR for decay schedules.
    poly_power : float, default=1.0
        Power for polynomial decay schedules.
    step_size : int, default=1000
        Step size for step-based schedulers.
    sched_gamma : float, default=0.99
        Decay factor for exponential/step schedulers.
    milestones : tuple[int, ...], default=()
        Milestones for multi-step schedulers.

    rollout_steps : int, default=2048
        Number of environment steps collected per rollout before updates.
    gamma : float, default=0.99
        Discount factor for returns/GAE.
    gae_lambda : float, default=0.95
        GAE-λ parameter.
    update_epochs : int, default=1
        Number of epochs over the rollout buffer per iteration.

        Notes
        -----
        Classic A2C uses ``update_epochs=1`` (single pass), while PPO often uses
        multiple epochs.
    minibatch_size : int | None, default=None
        Minibatch size for updates. If ``None``, the algorithm may use the full
        rollout batch (implementation-dependent in :class:`OnPolicyAlgorithm`).
    dtype_obs : Any, default=numpy.float32
        Numpy dtype used to store observations in the rollout buffer.
    dtype_act : Any, default=numpy.float32
        Numpy dtype used to store actions in the rollout buffer.

        Notes
        -----
        Continuous actions are stored as floats.
    normalize_advantages : bool, default=False
        Whether the algorithm/buffer normalizes advantages before updates.
    adv_eps : float, default=1e-8
        Small epsilon used when normalizing advantages (to avoid division by zero).

    Returns
    -------
    OnPolicyAlgorithm
        Fully constructed on-policy algorithm object with A2C head/core.

    Notes
    -----
    - This builder is **continuous-only**; discrete/categorical actions require a
      different head/core pair.
    - Advantage computation (GAE) and any advantage normalization typically happens
      in :class:`OnPolicyAlgorithm` and/or its rollout buffer, not in the core.
    """
    # -------------------------------------------------------------------------
    # 1) Head: actor + critic networks (continuous Gaussian policy + V(s))
    # -------------------------------------------------------------------------
    head = A2CHead(
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

    # -------------------------------------------------------------------------
    # 2) Core: update rule + optimizers/schedulers
    # -------------------------------------------------------------------------
    core = A2CCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
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
    # 3) Algorithm: rollout collection + advantage/return computation + updates
    # -------------------------------------------------------------------------
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
