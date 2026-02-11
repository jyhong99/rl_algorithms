from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm

from .core import QRDQNCore
from .head import QRDQNHead


# =============================================================================
# Builder (config-free)
# =============================================================================
def qrdqn(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    n_quantiles: int = 200,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # ---------------------------------------------------------------------
    # QR-DQN update (core) hyperparameters
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    tau: float = 0.0,
    double_dqn: bool = True,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) scheduler
    # ---------------------------------------------------------------------
    sched_name: str = "none",
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
    # ---------------------------------------------------------------------
    # PER (Prioritized Experience Replay)
    # ---------------------------------------------------------------------
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
    # ---------------------------------------------------------------------
    # Exploration (epsilon-greedy)
    # ---------------------------------------------------------------------
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
) -> OffPolicyAlgorithm:
    """
    Build a ready-to-run QR-DQN :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`.

    This is a **config-free builder** that composes three layers:

    1) Head (:class:`~.head.QRDQNHead`)
       - Owns the online and target **distributional** Q-networks.
       - Produces quantile outputs ``Z(s,a)`` with shape ``(B, N, A)``.
       - Provides expected Q-values ``Q(s,a) = E[Z(s,a)]`` for epsilon-greedy action selection.

    2) Core (:class:`~.core.QRDQNCore`)
       - Implements QR-DQN's distributional TD update via quantile regression loss.
       - Optionally applies Double DQN action selection to reduce overestimation bias.
       - Updates the target network periodically via hard/soft updates.

    3) Algorithm driver (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
       - Owns the replay buffer (uniform or PER).
       - Enforces warmup and update schedules (``update_after``, ``update_every``).
       - Executes repeated gradient steps using UTD settings (``utd``, ``gradient_steps``).
       - Manages PER beta annealing and epsilon-greedy exploration scheduling.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened state vector size).
    n_actions : int
        Number of discrete actions.

    device : Union[str, torch.device], default="cpu"
        Device used by the head/core and the algorithm driver.

    Network (head) hyperparameters
    ------------------------------
    n_quantiles : int, default=200
        Number of quantiles per action (``N``). The head outputs ``(B, N, A)``.
        Typical values range from ~50 to 200.
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        MLP hidden layer sizes for the quantile Q-network.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function **class** used by the underlying networks (e.g., ``nn.ReLU``).
    dueling_mode : bool, default=False
        If True, use a dueling architecture inside the quantile network (if supported
        by your :class:`~model_free.common.networks.q_networks.QuantileQNetwork`).
    init_type : str, default="orthogonal"
        Weight initialization scheme identifier passed through to the network builder.
    gain : float, default=1.0
        Optional initialization gain multiplier (meaning depends on ``init_type``).
    bias : float, default=0.0
        Optional bias initialization value.

    QR-DQN update (core) hyperparameters
    ------------------------------------
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    target_update_interval : int, default=1000
        Target update period measured in **core update calls**.
        - ``0`` disables target updates.
        - ``1`` updates every update.
    tau : float, default=0.0
        Soft-update coefficient:
        - ``tau <= 0`` is typically treated as a hard target copy at the update interval.
        - ``tau > 0`` performs Polyak averaging at the update interval.
    double_dqn : bool, default=True
        If True, uses Double DQN action selection:
        - ``a* = argmax_a Q_online(s',a)``
        - target quantiles taken from ``Z_target(s', a*)``.
    max_grad_norm : float, default=0.0
        Global gradient norm clipping threshold (0 disables clipping).
    use_amp : bool, default=False
        Enable torch AMP (mixed precision) in the core update.

    Optimizer / Scheduler
    ---------------------
    optim_name : str, default="adamw"
        Optimizer name for the online quantile network parameters.
    lr : float, default=3e-4
        Learning rate.
    weight_decay : float, default=0.0
        Weight decay.
    sched_name : str, default="none"
        Scheduler name. If "none", no scheduler is used.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Shared scheduler knobs passed through to the scheduler builder.

    OffPolicyAlgorithm schedule / replay
    ------------------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity (transitions).
    batch_size : int, default=256
        Batch size sampled per gradient step.
    update_after : int, default=1_000
        Start updating only after at least this many environment steps have been collected.
    update_every : int, default=1
        Attempt updates every N environment steps.
    utd : float, default=1.0
        Update-to-data ratio. Framework-dependent, but typically controls how many
        gradient updates are performed per environment step.
    gradient_steps : int, default=1
        Number of gradient steps per update call.
    max_updates_per_call : int, default=1_000
        Safety cap for updates in a single ``algo.update()`` call.

    PER (Prioritized Experience Replay)
    -----------------------------------
    use_per : bool, default=True
        Enable prioritized replay.
    per_alpha : float, default=0.6
        Priority exponent.
    per_beta : float, default=0.4
        Initial importance-sampling correction exponent.
    per_eps : float, default=1e-6
        Small epsilon used for PER stability (priorities and/or td-error clamping).
    per_beta_final : float, default=1.0
        Final beta after annealing.
    per_beta_anneal_steps : int, default=200_000
        Number of environment steps over which ``per_beta`` is annealed to ``per_beta_final``.

    Exploration (epsilon-greedy)
    ----------------------------
    exploration_eps : float, default=0.1
        Initial epsilon for epsilon-greedy exploration during training.
    exploration_eps_final : float, default=0.05
        Final epsilon after annealing.
    exploration_eps_anneal_steps : int, default=200_000
        Number of environment steps over which epsilon is annealed.
    exploration_eval_eps : float, default=0.0
        Epsilon used during evaluation (usually 0 for greedy evaluation).

    Returns
    -------
    algo : OffPolicyAlgorithm
        Fully constructed QR-DQN algorithm driver.

    Examples
    --------
    >>> algo = qrdqn(obs_dim=17, n_actions=6, device="cuda")
    >>> algo.setup(env)
    >>> obs = env.reset()
    >>> while True:
    ...     action = algo.act(obs)  # epsilon scheduling is managed by OffPolicyAlgorithm
    ...     next_obs, reward, done, info = env.step(action)
    ...     algo.on_env_step(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, info=info)
    ...     if algo.ready_to_update():
    ...         metrics = algo.update()
    ...     obs = next_obs
    """

    # ------------------------------------------------------------------
    # 1) Head: online + target quantile Q networks
    # ------------------------------------------------------------------
    head = QRDQNHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        n_quantiles=int(n_quantiles),
        hidden_sizes=tuple(int(x) for x in hidden_sizes),
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # ------------------------------------------------------------------
    # 2) Core: QR-DQN update engine
    # ------------------------------------------------------------------
    core = QRDQNCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        # optimizer
        optim_name=str(optim_name),
        lr=float(lr),
        weight_decay=float(weight_decay),
        # scheduler
        sched_name=str(sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # grad / amp
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER proxy epsilon (td_error clamp)
        per_eps=float(per_eps),
    )

    # ------------------------------------------------------------------
    # 3) Algorithm: replay buffer + update scheduling + exploration schedule
    # ------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # schedules
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
        # epsilon-greedy exploration scheduling (discrete)
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=int(exploration_eps_anneal_steps),
        exploration_eval_eps=float(exploration_eval_eps),
    )
    return algo
