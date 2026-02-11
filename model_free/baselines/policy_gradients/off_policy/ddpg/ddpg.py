from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from model_free.common.noises.noise_builder import build_noise
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm

from .core import DDPGCore
from .head import DDPGHead


def ddpg(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # =========================================================================
    # Network (head) hyperparameters
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
    noise_clip: Optional[float] = None,
    # =========================================================================
    # DDPG update (core) hyperparameters
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
    Build a complete DDPG off-policy algorithm instance (config-free factory).

    This factory constructs and wires together the three standard layers in this
    codebase:

    1) **Head** (:class:`~.head.DDPGHead`)
       Owns networks and action-selection behavior:
       - actor : deterministic policy :math:`\\pi(s)`
       - critic : state-action value :math:`Q(s,a)`
       - target networks : :math:`\\pi_{\\text{targ}}` and :math:`Q_{\\text{targ}}`
       - (optional) exploration noise stored on the head and applied in ``act``

    2) **Core** (:class:`~.core.DDPGCore`)
       Owns learning logic:
       - critic TD regression
       - deterministic policy gradient actor update
       - target network update cadence (hard/soft, via ``tau``)
       - optimizers/schedulers, AMP, gradient clipping

    3) **Algorithm wrapper** (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
       Owns environment-facing mechanics:
       - replay buffer (uniform or PER)
       - update scheduling (update_after / update_every / utd / gradient_steps)
       - batch sampling and PER bookkeeping
       - calls ``core.update_from_batch(batch)``

    Parameters
    ----------
    obs_dim : int
        Observation (state) vector dimension.
    action_dim : int
        Action vector dimension.
    device : str or torch.device, default="cpu"
        Device used for head/core computation.

    Network hyperparameters (head)
    ------------------------------
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        Hidden layer widths for actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation constructor used in MLP blocks.
    init_type : str, default="orthogonal"
        Initialization scheme forwarded to network constructors.
    gain : float, default=1.0
        Initialization gain forwarded to network constructors.
    bias : float, default=0.0
        Bias initialization forwarded to network constructors.
    action_low : np.ndarray, optional
        Lower bound for actions, shape ``(action_dim,)``. If provided alongside
        ``action_high``, the actor may rescale/clamp outputs accordingly.
    action_high : np.ndarray, optional
        Upper bound for actions, shape ``(action_dim,)``.

    Exploration noise (head-side)
    -----------------------------
    exploration_noise : str or None, default=None
        Noise family identifier passed to :func:`build_noise`.

        Common values (depending on your noise builder)
        - ``None``: no noise (pure deterministic)
        - ``"gaussian"``: Gaussian noise
        - ``"ou"``: Ornstein-Uhlenbeck noise (temporally correlated)
        - ``"uniform"``: Uniform noise

    noise_mu : float, default=0.0
        Mean of Gaussian/OU noise (if applicable).
    noise_sigma : float, default=0.1
        Standard deviation of Gaussian noise and OU diffusion scale (if applicable).
    ou_theta : float, default=0.15
        Mean reversion rate for OU noise.
    ou_dt : float, default=1e-2
        Discretization timestep for OU noise.
    uniform_low : float, default=-1.0
        Lower bound for Uniform noise.
    uniform_high : float, default=1.0
        Upper bound for Uniform noise.
    action_noise_eps : float, default=1e-6
        Small epsilon used by action-dependent noise variants (if applicable).
    action_noise_low : float or Sequence[float], optional
        Optional per-dimension lower bounds for action-dependent noise scaling.
    action_noise_high : float or Sequence[float], optional
        Optional per-dimension upper bounds for action-dependent noise scaling.
    noise_clip : float, optional
        Optional clamp applied to sampled noise epsilon inside the head (if your
        ``DDPGHead.act`` uses it). Useful to limit rare noise spikes.

    DDPG update hyperparameters (core)
    ----------------------------------
    gamma : float, default=0.99
        Discount factor :math:`\\gamma`.
    tau : float, default=0.005
        Target soft-update coefficient. ``tau=1`` approximates a hard update.
    target_update_interval : int, default=1
        Update cadence for target networks in optimizer steps. If 0, disables
        target updates.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold (global norm). If 0, clipping is disabled.
    use_amp : bool, default=False
        Enable AMP (mixed precision) in the core.

    Optimizers
    ----------
    actor_optim_name, critic_optim_name : str
        Optimizer identifiers resolved by :class:`ActorCriticCore`.
    actor_lr, critic_lr : float
        Learning rates.
    actor_weight_decay, critic_weight_decay : float
        Weight decay values.

    Schedulers (optional)
    ---------------------
    actor_sched_name, critic_sched_name : str
        Scheduler identifiers resolved by :class:`ActorCriticCore`.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration forwarded to the base core.

    Off-policy replay / update scheduling
    -------------------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Batch size sampled from replay.
    update_after : int, default=1_000
        Minimum number of environment steps before updates are allowed.
    update_every : int, default=1
        Perform updates every N environment steps (after ``update_after``).
    utd : float, default=1.0
        Update-to-data ratio (wrapper-defined semantics).
    gradient_steps : int, default=1
        Gradient steps per update call (wrapper-defined semantics).
    max_updates_per_call : int, default=1_000
        Safety cap on total update steps per call to avoid long stalls.

    PER (optional)
    --------------
    use_per : bool, default=True
        Enable prioritized sampling and importance weights.
    per_alpha : float, default=0.6
        Priority exponent.
    per_beta : float, default=0.4
        Initial importance-weight exponent.
    per_eps : float, default=1e-6
        Priority epsilon.
    per_beta_final : float, default=1.0
        Final beta after annealing.
    per_beta_anneal_steps : int, default=200_000
        Annealing horizon for beta.

    Returns
    -------
    OffPolicyAlgorithm
        Configured algorithm instance. Typical usage pattern:

        - ``algo.setup(env)``
        - ``action = algo.act(obs)``
        - ``algo.on_env_step(transition)``
        - ``if algo.ready_to_update(): algo.update()``

    Notes
    -----
    - Exploration noise is created *once* here and stored inside the head
      (``DDPGHead.noise``). The outer algorithm wrapper does not sample noise.
    - ``reset_noise_on_done=True`` instructs the wrapper to call
      ``head.reset_exploration_noise()`` (if implemented) at episode boundaries,
      which is useful for stateful noise processes (e.g., OU noise).
    - PER is optional for DDPG; it often improves critic learning by focusing on
      high-TD-error transitions.
    """

    # =========================================================================
    # 1) Build the head-side exploration noise object
    # =========================================================================
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
    # 2) Head: networks + (optional) exploration noise
    # =========================================================================
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
    core = DDPGCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
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
        # Grad / AMP
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # =========================================================================
    # 4) Algorithm wrapper: replay buffer + update scheduling + PER plumbing
    # =========================================================================
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
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
        # Noise hygiene (stateful noises like OU benefit from resets)
        reset_noise_on_done=True,
    )

    return algo
