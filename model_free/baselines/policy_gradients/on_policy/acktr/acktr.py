from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm

from .core import ACKTRCore
from .head import ACKTRHead


def acktr(
    *,
    # =============================================================================
    # Environment I/O sizes
    # =============================================================================
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # =============================================================================
    # Network (head) hyperparameters (continuous only)
    # =============================================================================
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # =============================================================================
    # ACKTR update (core) hyperparameters
    # =============================================================================
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # =============================================================================
    # K-FAC / ACKTR-specific optimizer knobs (passed to ACKTRCore -> build_optimizer)
    #
    # Note
    # ----
    # ACKTR typically uses K-FAC to approximate natural gradients. The head remains
    # a standard actor-critic container; the "ACKTR-ness" lives in the optimizer/core.
    # =============================================================================
    actor_optim_name: str = "kfac",
    actor_lr: float = 0.25,
    actor_weight_decay: float = 0.0,
    actor_damping: float = 1e-2,
    actor_momentum: float = 0.9,
    actor_eps: float = 0.95,
    actor_Ts: int = 1,
    actor_Tf: int = 10,
    actor_max_lr: float = 1.0,
    actor_trust_region: float = 2e-3,
    critic_optim_name: str = "kfac",
    critic_lr: float = 0.25,
    critic_weight_decay: float = 0.0,
    critic_damping: float = 1e-2,
    critic_momentum: float = 0.9,
    critic_eps: float = 0.95,
    critic_Ts: int = 1,
    critic_Tf: int = 10,
    critic_max_lr: float = 1.0,
    critic_trust_region: float = 2e-3,
    # =============================================================================
    # (Optional) LR schedulers
    # =============================================================================
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # =============================================================================
    # OnPolicyAlgorithm rollout / training schedule
    # =============================================================================
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
    Build an ACKTR :class:`~model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`
    for **continuous** action spaces.

    This is a *builder* function that wires together three layers:

    1) **Head** (:class:`ACKTRHead`)
       Owns the neural networks:
       - Actor: diagonal Gaussian policy :math:`\\pi(a\\mid s)` (unsquashed)
       - Critic: state-value baseline :math:`V(s)`

    2) **Core** (:class:`ACKTRCore`)
       Owns the update rule and optimization:
       - A2C-style losses (policy/value/entropy)
       - K-FAC optimizer configuration (damping, trust region, factor update cadence, etc.)
       - optional AMP (best-effort; often not recommended with K-FAC)
       - optional global gradient clipping

    3) **Algorithm** (:class:`OnPolicyAlgorithm`)
       Owns rollout collection and training schedule:
       - collects ``rollout_steps`` transitions
       - computes returns/advantages (GAE-λ)
       - runs ``update_epochs`` epochs of minibatch updates

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened), i.e., number of features in ``obs``.
    action_dim : int
        Continuous action dimension (flattened).
    device : str | torch.device, default="cpu"
        Torch device used by the learner/head (e.g., ``"cpu"``, ``"cuda:0"``).

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
        Log-standard-deviation parameterization mode for the Gaussian actor.
    log_std_init : float, default=-0.5
        Initial log-std value.

    vf_coef : float, default=0.5
        Value-loss coefficient used by :class:`ACKTRCore`.
    ent_coef : float, default=0.0
        Entropy coefficient used by :class:`ACKTRCore`.

        Notes
        -----
        Entropy is typically used as an exploration bonus. In the core, it is often
        implemented as a loss term ``ent_loss = -entropy.mean()``; thus a positive
        ``ent_coef`` encourages higher entropy.
    max_grad_norm : float, default=0.0
        Global gradient norm clipping threshold.

        Notes
        -----
        - Set to ``0`` to disable clipping (depending on your core's convention).
        - Some ACKTR/K-FAC setups rely on trust-region logic rather than clipping.
    use_amp : bool, default=False
        Enable CUDA AMP in the core (best-effort; meaningful on CUDA).

        Notes
        -----
        With K-FAC, AMP may affect curvature statistics. Use with care.

    actor_optim_name, critic_optim_name : str, default="kfac"
        Optimizer identifiers (typically ``"kfac"``).
    actor_lr, critic_lr : float, default=0.25
        Learning rates for actor/critic K-FAC optimizers (semantics are optimizer-dependent).
    actor_weight_decay, critic_weight_decay : float, default=0.0
        Weight decay for actor/critic optimizers.
    actor_damping, critic_damping : float, default=1e-2
        Damping term for K-FAC.
    actor_momentum, critic_momentum : float, default=0.9
        Momentum term for K-FAC (implementation-dependent).
    actor_eps, critic_eps : float, default=0.95
        Exponential moving-average factor for K-FAC statistics (implementation-dependent).
    actor_Ts, critic_Ts : int, default=1
        Factor update period (implementation-dependent).
    actor_Tf, critic_Tf : int, default=10
        Inverse-factor update period (implementation-dependent).
    actor_max_lr, critic_max_lr : float, default=1.0
        Maximum learning rate used by trust-region logic (implementation-dependent).
    actor_trust_region, critic_trust_region : float, default=2e-3
        Trust-region / KL constraint (implementation-dependent).

    actor_sched_name, critic_sched_name : str, default="none"
        Scheduler identifiers (if supported by your base core).
    total_steps : int, default=0
        Total steps used by schedules requiring a horizon.
    warmup_steps : int, default=0
        Warmup steps for schedules that support warmup.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for decay schedules.
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
        Classic ACKTR setups often use a single epoch/passage per rollout.
    minibatch_size : int | None, default=None
        Minibatch size for updates. If ``None``, the algorithm may treat the entire
        rollout as a single batch (implementation-dependent in :class:`OnPolicyAlgorithm`).
    dtype_obs : Any, default=numpy.float32
        Numpy dtype used to store observations in the rollout buffer.
    dtype_act : Any, default=numpy.float32
        Numpy dtype used to store actions in the rollout buffer.
    normalize_advantages : bool, default=False
        Whether the algorithm/buffer normalizes advantages before updates.
    adv_eps : float, default=1e-8
        Small epsilon used when normalizing advantages (to avoid division by zero).

    Returns
    -------
    OnPolicyAlgorithm
        Fully constructed on-policy algorithm object with ACKTR head/core.

    Notes
    -----
    - This builder is **continuous-only**.
    - The "ACKTR-ness" is primarily determined by using K-FAC optimizers inside
      :class:`ACKTRCore` (and your optimizer implementation), not by the head.
    """
    # =============================================================================
    # 1) Head: actor + critic networks
    # =============================================================================
    head = ACKTRHead(
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

    # =============================================================================
    # 2) Core: update engine (A2C losses + K-FAC optimizer wiring)
    # =============================================================================
    core = ACKTRCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # ----- actor K-FAC knobs -----
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        actor_damping=float(actor_damping),
        actor_momentum=float(actor_momentum),
        actor_eps=float(actor_eps),
        actor_Ts=int(actor_Ts),
        actor_Tf=int(actor_Tf),
        actor_max_lr=float(actor_max_lr),
        actor_trust_region=float(actor_trust_region),
        # ----- critic K-FAC knobs -----
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        critic_damping=float(critic_damping),
        critic_momentum=float(critic_momentum),
        critic_eps=float(critic_eps),
        critic_Ts=int(critic_Ts),
        critic_Tf=int(critic_Tf),
        critic_max_lr=float(critic_max_lr),
        critic_trust_region=float(critic_trust_region),
        # ----- schedulers (optional) -----
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

    # =============================================================================
    # 3) Algorithm: rollout collection + update scheduling
    # =============================================================================
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
