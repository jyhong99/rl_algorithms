from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from model_free.common.noises.noise_builder import build_noise
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm

from .core import TD3Core
from .head import TD3Head


def td3(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    # -------------------------------------------------------------------------
    # Exploration noise (head-owned)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # TD3 update (core) hyperparameters
    # -------------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    target_update_interval: int = 1,
    max_grad_norm: float = 0.0,
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
    # (Optional) schedulers
    # -------------------------------------------------------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    # -------------------------------------------------------------------------
    # OffPolicyAlgorithm schedule / replay
    # -------------------------------------------------------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # -------------------------------------------------------------------------
    # Prioritized Experience Replay (PER)
    # -------------------------------------------------------------------------
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete TD3 :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`.

    This factory function assembles a full TD3 stack by composing:

    1) :class:`~.head.TD3Head`
        Owns neural networks and inference utilities:
          - deterministic actor and twin critics
          - target actor and target critics
          - optional action bounds clamping
          - exploration noise (head-owned), applied during action selection

    2) :class:`~.core.TD3Core`
        Owns TD3 update logic:
          - critic update every step
          - delayed actor update
          - TD3 target policy smoothing (Gaussian noise + clipping)
          - Polyak target updates

    3) :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
        Owns replay and scheduling:
          - replay buffer management (uniform or PER)
          - update schedule (update_after, update_every, UTD, gradient_steps)
          - PER plumbing (sampling weights and priority feedback)
          - optional episode-boundary noise reset via ``head.reset_exploration_noise()``

    Parameters
    ----------
    obs_dim : int
        Observation dimension of the environment. The factory assumes observations are
        vectorized to shape ``(B, obs_dim)`` at training time.
    action_dim : int
        Action dimension (continuous). TD3 assumes actions are continuous vectors of
        shape ``(B, action_dim)``.
    device : Union[str, torch.device], default="cpu"
        Device used for training (online networks, optimizers, forward/backward passes).
        Note: Ray worker policies may override to CPU in the head-side factory spec.

    Network (head) hyperparameters
    ------------------------------
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        MLP hidden layer sizes for both actor and critics.
    activation_fn : Any, default=torch.nn.ReLU
        Activation class (e.g., ``torch.nn.ReLU``).
    init_type : str, default="orthogonal"
        Initialization strategy name forwarded to your network builders.
    gain : float, default=1.0
        Gain multiplier for initialization (if supported by your builders).
    bias : float, default=0.0
        Bias initialization constant (if supported by your builders).
    action_low : Optional[np.ndarray], default=None
        Lower action bounds. If provided, must be shape ``(action_dim,)``.
        Used by the head to clamp both exploration actions and target-smoothed actions.
    action_high : Optional[np.ndarray], default=None
        Upper action bounds. If provided, must be shape ``(action_dim,)`` and paired with
        ``action_low``.

    Exploration noise (head-owned)
    ------------------------------
    exploration_noise : Optional[str], default=None
        Exploration noise kind. Expected values depend on ``build_noise`` (e.g.,
        ``"gaussian"``, ``"ou"``, ``"uniform"``). ``None`` disables exploration noise.
    noise_mu : float, default=0.0
        Mean for Gaussian/OU noise (if applicable).
    noise_sigma : float, default=0.1
        Stddev for Gaussian/OU noise (if applicable).
    ou_theta : float, default=0.15
        OU drift parameter (if using OU noise).
    ou_dt : float, default=1e-2
        OU time step (if using OU noise).
    uniform_low : float, default=-1.0
        Low bound for uniform noise (if using uniform noise).
    uniform_high : float, default=1.0
        High bound for uniform noise (if using uniform noise).
    action_noise_eps : float, default=1e-6
        Small epsilon used by some action-dependent noise wrappers to avoid degenerate
        ranges or division by zero.
    action_noise_low : Optional[Union[float, Sequence[float]]], default=None
        Optional per-dimension lower clip/range for action-dependent noise modules.
    action_noise_high : Optional[Union[float, Sequence[float]]], default=None
        Optional per-dimension upper clip/range for action-dependent noise modules.

    TD3 update (core) hyperparameters
    --------------------------------
    gamma : float, default=0.99
        Discount factor :math:`\\gamma`.
    tau : float, default=0.005
        Polyak coefficient for target updates.
    policy_noise : float, default=0.2
        Stddev of Gaussian noise used in TD3 target policy smoothing
        (applied to the *target actor* output inside the core/head).
    noise_clip : float, default=0.5
        Clip range for the TD3 target policy smoothing noise.
    policy_delay : int, default=2
        Actor update period (in critic update calls).
    target_update_interval : int, default=1
        Additional gate for target updates when actor updates.
    max_grad_norm : float, default=0.0
        Global gradient norm clip threshold. ``0.0`` disables clipping.
    use_amp : bool, default=False
        Enables mixed precision training (AMP) if supported by the device.

    Optimizers / schedulers
    -----------------------
    actor_optim_name, critic_optim_name : str
        Optimizer names forwarded to the optimizer builder used in :class:`TD3Core`.
    actor_lr, critic_lr : float
        Learning rates for actor and critic.
    actor_weight_decay, critic_weight_decay : float
        Weight decay values for actor and critic optimizers.
    actor_sched_name, critic_sched_name : str
        Scheduler names forwarded to the scheduler builder used in :class:`TD3Core`.
    total_steps, warmup_steps, min_lr_ratio, poly_power : int/float
        Scheduler hyperparameters forwarded to the scheduler builder.

    Replay / schedule (OffPolicyAlgorithm)
    --------------------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity (number of transitions).
    batch_size : int, default=256
        Minibatch size for updates.
    update_after : int, default=1_000
        Minimum number of environment steps before any update is allowed.
    update_every : int, default=1
        Update cadence in environment steps.
    utd : float, default=1.0
        Update-to-data ratio: roughly how many updates per environment step.
    gradient_steps : int, default=1
        Number of gradient steps per update call.
    max_updates_per_call : int, default=1_000
        Upper bound to prevent extremely long update bursts when UTD is large.

    PER configuration
    -----------------
    use_per : bool, default=True
        Enables prioritized replay if True.
    per_alpha : float, default=0.6
        Priority exponent (how strongly priorities affect sampling).
    per_beta : float, default=0.4
        Importance sampling exponent (initial value).
    per_eps : float, default=1e-6
        Small constant added to TD errors to avoid zero priorities.
    per_beta_final : float, default=1.0
        Final beta value after annealing.
    per_beta_anneal_steps : int, default=200_000
        Number of steps over which beta is annealed from ``per_beta`` to ``per_beta_final``.

    Returns
    -------
    algo : OffPolicyAlgorithm
        A ready-to-use off-policy algorithm instance. Typical workflow::

            algo = td3(obs_dim=..., action_dim=..., device="cuda")
            algo.setup(env)
            action = algo.act(obs)
            algo.on_env_step(transition)
            if algo.ready_to_update():
                metrics = algo.update()

    Notes
    -----
    Exploration noise ownership is intentionally placed in the head:

    - The algorithm driver remains generic across off-policy methods (SAC/TD3/REDQ/etc.).
    - The head can manage stateful noise processes (e.g., OU) and reset them on episode
      boundaries when ``reset_noise_on_done=True`` in :class:`OffPolicyAlgorithm`.
    """
    # -------------------------------------------------------------------------
    # 1) Build exploration noise (head-owned)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2) Head: networks + targets + exploration noise
    # -------------------------------------------------------------------------
    head = TD3Head(
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
    )

    # -------------------------------------------------------------------------
    # 3) Core: TD3 update logic
    # -------------------------------------------------------------------------
    core = TD3Core(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        policy_noise=float(policy_noise),
        noise_clip=float(noise_clip),
        policy_delay=int(policy_delay),
        target_update_interval=int(target_update_interval),
        # optim/sched
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
        # stability
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # -------------------------------------------------------------------------
    # 4) Algorithm driver: replay + scheduling (+ PER)
    # -------------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # schedule
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
        # Episode-boundary noise reset (if head implements reset_exploration_noise()).
        reset_noise_on_done=True,
    )
    return algo
