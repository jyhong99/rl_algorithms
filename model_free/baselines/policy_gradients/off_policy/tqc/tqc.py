from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th

from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm

from .core import TQCCore
from .head import TQCHead


def tqc(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # Actor distribution (SAC-like squashed Gaussian)
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    # Quantile critic (TQC-specific)
    n_quantiles: int = 25,
    n_nets: int = 2,
    # ---------------------------------------------------------------------
    # TQC update (core) hyperparameters
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    top_quantiles_to_drop: int = 2,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # ---------------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------------
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    alpha_optim_name: str = "adamw",
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) schedulers
    # ---------------------------------------------------------------------
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
    # ---------------------------------------------------------------------
    # OffPolicyAlgorithm schedule / replay
    # ---------------------------------------------------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # Prioritized Experience Replay (PER)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete Truncated Quantile Critics (TQC) off-policy algorithm.

    This factory function composes the three-layer structure used throughout your
    off-policy codebase:

    - **Head** (:class:`~.head.TQCHead`)
        Owns neural networks and action sampling primitives.

        * Actor: squashed Gaussian policy (SAC-style)
        * Critic: quantile critic ensemble producing a return distribution
          :math:`Z(s,a)` with shape ``(B, C, N)``
        * Target critic: frozen copy of critic updated via Polyak averaging

    - **Core** (:class:`~.core.TQCCore`)
        Implements the gradient update rules:

        * Truncated target distribution: after sorting the flattened target quantiles,
          drop the largest ``top_quantiles_to_drop`` quantiles (overestimation control).
        * Critic update via quantile regression (Huber quantile loss).
        * Actor update in SAC form using a conservative scalar Q proxy derived from
          the critic quantiles.
        * Optional temperature learning (``auto_alpha``) to match ``target_entropy``.
        * Periodic target critic updates controlled by ``target_update_interval`` and ``tau``.

    - **Algorithm driver** (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
        Owns the replay buffer and update scheduling:

        * replay sizing: ``buffer_size``, sampling ``batch_size``
        * warmup/start update gate: ``update_after``
        * update cadence: ``update_every``, UTD ratio ``utd``, ``gradient_steps``
        * per-call cap: ``max_updates_per_call``
        * PER plumbing (if enabled) using TD-error feedback returned by the core

    Parameters
    ----------
    obs_dim : int
        Observation dimension of the environment.
    action_dim : int
        Action dimension of the environment (continuous actions assumed).
    device : str or torch.device, default="cpu"
        Device used for learner-side networks and updates (e.g., "cpu", "cuda").

    hidden_sizes : tuple of int, default=(256, 256)
        MLP hidden layer sizes used by both actor and critic networks in the head.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class for network construction.
    init_type : str, default="orthogonal"
        Weight initialization strategy identifier used by your network builders.
    gain : float, default=1.0
        Gain forwarded to initialization.
    bias : float, default=0.0
        Bias initialization value forwarded to initialization.

    log_std_mode : str, default="layer"
        Actor log-standard-deviation parameterization mode. Interpretation depends
        on your :class:`~model_free.common.networks.policy_networks.ContinuousPolicyNetwork`.
    log_std_init : float, default=-0.5
        Initial log standard deviation value.

    n_quantiles : int, default=25
        Number of quantiles per critic head (``N``).
    n_nets : int, default=2
        Number of critic ensemble members (``C``).

    gamma : float, default=0.99
        Discount factor.
    tau : float, default=0.005
        Polyak averaging factor used for target critic updates.
    target_update_interval : int, default=1
        Target update period measured in *core update calls*. If 0, disables target updates.
    top_quantiles_to_drop : int, default=2
        Number of highest quantiles to drop after sorting flattened target quantiles.
        Must satisfy ``0 <= drop < C*N`` (validated inside the core).
    auto_alpha : bool, default=True
        If True, learn temperature :math:`\\alpha` by optimizing ``log_alpha``.
    alpha_init : float, default=0.2
        Initial temperature value (alpha = exp(log_alpha)).
    target_entropy : float or None, default=None
        Target entropy for SAC-style temperature tuning. If None, the core typically
        uses the heuristic ``-action_dim``.
    max_grad_norm : float, default=0.0
        Gradient norm clipping threshold. ``0.0`` disables clipping.
    use_amp : bool, default=False
        If True, enable CUDA AMP mixed-precision updates in the core.

    actor_optim_name, critic_optim_name, alpha_optim_name : str
        Optimizer identifiers forwarded to your optimizer builder.
    actor_lr, critic_lr, alpha_lr : float
        Learning rates for actor/critic/alpha.
    actor_weight_decay, critic_weight_decay, alpha_weight_decay : float
        Weight decay for actor/critic/alpha optimizers.

    actor_sched_name, critic_sched_name, alpha_sched_name : str
        Scheduler identifiers forwarded to your scheduler builder.
    total_steps : int, default=0
        Total training steps (used by some schedulers; 0 may mean "disabled/unknown").
    warmup_steps : int, default=0
        LR warmup steps for schedulers that support warmup.
    min_lr_ratio : float, default=0.0
        Minimum learning rate as a fraction of base LR (scheduler-specific).
    poly_power : float, default=1.0
        Polynomial decay exponent (scheduler-specific).
    step_size : int, default=1000
        Step interval for step-based schedulers.
    sched_gamma : float, default=0.99
        Multiplicative decay for step/exponential schedulers.
    milestones : Sequence[int], default=()
        Milestones for multi-step schedulers.

    buffer_size : int, default=1_000_000
        Replay buffer capacity (number of transitions).
    batch_size : int, default=256
        Minibatch size sampled from replay.
    update_after : int, default=1_000
        Minimum environment steps before any gradient updates start.
    update_every : int, default=1
        Update cadence in environment steps.
    utd : float, default=1.0
        Updates-to-data ratio (how many updates per env step, aggregated by the driver).
    gradient_steps : int, default=1
        Number of gradient steps per update call (driver-level).
    max_updates_per_call : int, default=1_000
        Hard cap on updates performed in one driver update call (safety limit).

    use_per : bool, default=True
        Whether to use prioritized experience replay.
    per_alpha : float, default=0.6
        Priority exponent for PER.
    per_beta : float, default=0.4
        Initial importance-sampling exponent for PER.
    per_eps : float, default=1e-6
        Small constant added/clamp used for numerical stability in PER priorities.
    per_beta_final : float, default=1.0
        Final beta value after annealing.
    per_beta_anneal_steps : int, default=200_000
        Number of environment steps (or update steps, depending on driver) over which
        PER beta anneals from ``per_beta`` to ``per_beta_final``.

    Returns
    -------
    OffPolicyAlgorithm
        A ready-to-setup algorithm instance composed as:

        - ``algo.head`` is a :class:`~.head.TQCHead`
        - ``algo.core`` is a :class:`~.core.TQCCore`

        Typical usage::

            algo = tqc(obs_dim=..., action_dim=..., device="cuda")
            algo.setup(env)
            action = algo.act(obs)
            algo.on_env_step(transition)
            if algo.ready_to_update():
                metrics = algo.update()

    See Also
    --------
    TQCHead
        Network container providing actor + quantile critics.
    TQCCore
        Update engine implementing truncation + quantile regression + SAC-style updates.
    OffPolicyAlgorithm
        Replay buffer and scheduling driver shared across off-policy methods.
    """
    # ---------------------------------------------------------------------
    # 1) Head: networks (actor + quantile critic ensemble + target critic)
    # ---------------------------------------------------------------------
    head = TQCHead(
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
        n_quantiles=int(n_quantiles),
        n_nets=int(n_nets),
    )

    # ---------------------------------------------------------------------
    # 2) Core: update engine (losses, optimizers/schedulers, alpha, targets)
    # ---------------------------------------------------------------------
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
        # Stability / performance knobs
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # ---------------------------------------------------------------------
    # 3) Algorithm driver: replay + scheduling + (optional) PER
    # ---------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # Replay buffer
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # Update schedule
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
    )
    return algo
