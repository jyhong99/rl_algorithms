from __future__ import annotations

from typing import Any, Tuple, Union

import torch as th

from .core import ACERCore
from .head import ACERHead
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def acer(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    dueling_mode: bool = False,
    double_q: bool = True,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # ACER update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    c_bar: float = 10.0,
    entropy_coef: float = 0.0,
    critic_is: bool = False,
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
    # -----------------------------
    # (Optional) schedulers
    # -----------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # -----------------------------
    # OffPolicyAlgorithm schedule / replay
    # -----------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # PER (optional)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a discrete ACER algorithm by composing head + core + off-policy wrapper.

    This factory returns a fully-wired :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
    instance configured for ACER-style training in **discrete** action spaces.

    Architecture
    ------------
    The returned algorithm is a composition of three layers:

    1) **Head** (:class:`~.head.ACERHead`)
       Holds neural networks and model-side utilities:
       - actor : categorical policy :math:`\\pi(a\\mid s)`
       - critic : :math:`Q(s,a)` (optionally Double Q and/or dueling)
       - target critic : :math:`Q_{\\text{targ}}(s,a)` for stable TD targets
       - policy helpers (e.g., ``logp``, ``probs``, ``q_values``, ``q_values_target``)

    2) **Core** (:class:`~.core.ACERCore`)
       Implements the optimization step given a replay batch:
       - critic TD regression toward :math:`r + \\gamma(1-d)V_\\pi(s')`
       - actor update with truncated importance sampling (IS)
       - optional bias correction if behavior policy probabilities are available
       - optional entropy regularization
       - target network update cadence (hard/soft) and optimizer/scheduler plumbing

    3) **Algorithm wrapper** (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
       Provides the environment-facing training loop mechanics:
       - replay buffer (uniform or PER)
       - update scheduling (update_after / update_every / utd / gradient_steps)
       - PER sampling weights and priority update wiring (if enabled)
       - storage of behavior-policy metadata required by ACER

    Parameters
    ----------
    obs_dim : int
        Observation (state) vector dimension.
    n_actions : int
        Number of discrete actions.
    device : str or torch.device, default="cpu"
        Device to place the head and core on. (Note: Ray workers may override to CPU.)

    Network hyperparameters (head)
    ------------------------------
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        Hidden layer widths for actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation constructor used in MLP blocks.
    dueling_mode : bool, default=False
        If ``True`` and supported by your Q-network implementation, uses dueling Q.
    double_q : bool, default=True
        If ``True``, uses Double Q (two critics internally) and typically aggregates
        via min(Q1, Q2) (depending on your head implementation).
    init_type : str, default="orthogonal"
        Initialization scheme string forwarded to network constructors.
    gain : float, default=1.0
        Initialization gain forwarded to network constructors.
    bias : float, default=0.0
        Bias initialization forwarded to network constructors.

    ACER update hyperparameters (core)
    ---------------------------------
    gamma : float, default=0.99
        Discount factor. Must satisfy ``0 <= gamma < 1`` (enforced in core).
    tau : float, default=0.005
        Soft target update coefficient. ``tau=1`` approximates a hard copy.
    target_update_interval : int, default=1
        Target update cadence in optimizer steps. If 0, target updates are disabled.
    c_bar : float, default=10.0
        Truncation threshold for importance weights.
    entropy_coef : float, default=0.0
        Entropy regularization coefficient.
    critic_is : bool, default=False
        If ``True``, applies truncated IS weights to critic regression loss.
    max_grad_norm : float, default=0.0
        Global norm clipping threshold. If 0, clipping is disabled.
    use_amp : bool, default=False
        Enables automatic mixed precision inside the core.
    per_eps : float, default=1e-6
        Small epsilon used by PER implementations.

    Optimizers
    ----------
    actor_optim_name : str, default="adamw"
        Actor optimizer identifier resolved by :class:`ActorCriticCore`.
    actor_lr : float, default=3e-4
        Actor learning rate.
    actor_weight_decay : float, default=0.0
        Actor weight decay.
    critic_optim_name : str, default="adamw"
        Critic optimizer identifier.
    critic_lr : float, default=3e-4
        Critic learning rate.
    critic_weight_decay : float, default=0.0
        Critic weight decay.

    Schedulers (optional)
    ---------------------
    actor_sched_name : str, default="none"
        Scheduler identifier for actor optimizer.
    critic_sched_name : str, default="none"
        Scheduler identifier for critic optimizer.
    total_steps : int, default=0
        Total training steps for schedule parametrization (if scheduler uses it).
    warmup_steps : int, default=0
        Warmup steps for schedule (if enabled).
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for certain schedules.
    poly_power : float, default=1.0
        Polynomial decay power (if used).
    step_size : int, default=1000
        Step size for step-based schedules.
    sched_gamma : float, default=0.99
        Multiplicative factor for step schedules.
    milestones : Tuple[int, ...], default=()
        Milestones for multi-step schedules (if used).

    Off-policy replay / update scheduling
    -------------------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Batch size sampled from replay for each update.
    update_after : int, default=1_000
        First environment step at which updates are allowed.
    update_every : int, default=1
        Run updates every N environment steps (after ``update_after``).
    utd : float, default=1.0
        "Update-to-data" ratio. Depending on your wrapper semantics, this may
        scale how many updates occur per environment step.
    gradient_steps : int, default=1
        Gradient steps per update call (wrapper semantics).
    max_updates_per_call : int, default=1_000
        Safety cap on total gradient steps in a single call to avoid long stalls.

    PER configuration (optional)
    ----------------------------
    use_per : bool, default=True
        Enables prioritized replay sampling and importance weights.
    per_alpha : float, default=0.6
        Priority exponent.
    per_beta : float, default=0.4
        Initial importance-weight exponent.
    per_beta_final : float, default=1.0
        Final beta value after annealing.
    per_beta_anneal_steps : int, default=200_000
        Steps over which beta is annealed from ``per_beta`` to ``per_beta_final``.

    Returns
    -------
    OffPolicyAlgorithm
        A configured algorithm instance. Typical usage:

        - ``algo.setup(env)``
        - ``action = algo.act(obs)``
        - ``algo.on_env_step(transition)``
        - ``if algo.ready_to_update(): algo.update()``

    Important
    ---------
    ACER relies on importance sampling ratios:

    .. math::
        \\rho = \\frac{\\pi(a\\mid s)}{\\mu(a\\mid s)}

    Therefore, the replay buffer must store behavior policy metadata for each
    transition. This factory enables both:

    - ``store_behavior_logp=True``  (required; used for sampled action ratio)
    - ``store_behavior_probs=True`` (optional but recommended; enables bias correction)

    Notes
    -----
    - PER is optional for ACER. It primarily benefits critic regression by focusing
      updates on high-TD-error transitions and applying importance weights.
    - Scheduler arguments are forwarded into the :class:`ActorCriticCore` schedule
      builder used by :class:`ACERCore`.
    """

    # ---------------------------------------------------------------------
    # 1) Head: actor + critic + target critic
    # ---------------------------------------------------------------------
    head = ACERHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        double_q=bool(double_q),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # ---------------------------------------------------------------------
    # 2) Core: ACER update logic + optimizers/schedulers
    # ---------------------------------------------------------------------
    core = ACERCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        c_bar=float(c_bar),
        entropy_coef=float(entropy_coef),
        critic_is=bool(critic_is),
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
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        per_eps=float(per_eps),
    )

    # ---------------------------------------------------------------------
    # 3) Algorithm wrapper: replay + scheduling + PER + behavior-policy storage
    # ---------------------------------------------------------------------
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
        # PER (optional)
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        # ACER requires behavior policy info in replay batches
        store_behavior_logp=True,
        store_behavior_probs=True,
    )

    return algo
