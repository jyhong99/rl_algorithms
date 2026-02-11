from __future__ import annotations

from typing import Any, Tuple, Union

import torch as th

from .core import DQNCore
from .head import DQNHead
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def dqn(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -------------------------------------------------------------------------
    # Network (head) hyperparameters
    # -------------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -------------------------------------------------------------------------
    # DQN update (core) hyperparameters
    # -------------------------------------------------------------------------
    gamma: float = 0.99,
    double_dqn: bool = True,
    huber: bool = True,
    # Target update knobs
    target_update_interval: int = 1000,
    tau: float = 0.0,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    per_eps: float = 1e-6,
    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    # -------------------------------------------------------------------------
    # (Optional) scheduler
    # -------------------------------------------------------------------------
    sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
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
    # PER
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
    # -------------------------------------------------------------------------
    # Exploration (epsilon-greedy)
    # -------------------------------------------------------------------------
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
) -> OffPolicyAlgorithm:
    """
    Build a complete DQN algorithm (head + core + off-policy driver).

    This is a convenience builder that wires together:

    1) :class:`~.head.DQNHead`
        Owns the **online** and **target** Q-networks.

    2) :class:`~.core.DQNCore`
        Implements the TD update (DQN or Double DQN), loss selection (Huber/MSE),
        optimizer step, scheduler step, PER weighting, and target-network updates.

    3) :class:`model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
        Owns the replay buffer and update schedule, and calls ``core.update_from_batch``
        according to the off-policy training loop.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation/state vector (flattened).
    n_actions : int
        Number of discrete actions.
    device : Union[str, torch.device], default="cpu"
        Device used for training (e.g., ``"cpu"``, ``"cuda"``).

    Network hyperparameters
    -----------------------
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        Hidden layer sizes of the Q-network MLP.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function **class** used by the Q-network builder.
    dueling_mode : bool, default=False
        If True, enable dueling architecture in the Q-network.
    init_type : str, default="orthogonal"
        Initialization scheme identifier (interpreted by your network builder).
    gain : float, default=1.0
        Gain multiplier used by the initializer.
    bias : float, default=0.0
        Bias initialization constant.

    Core hyperparameters (TD update)
    --------------------------------
    gamma : float, default=0.99
        Discount factor.
    double_dqn : bool, default=True
        If True, use Double DQN target computation (online selects action,
        target evaluates it) to reduce overestimation bias.
    huber : bool, default=True
        If True, use Smooth L1 (Huber) TD loss. Else use MSE.
    target_update_interval : int, default=1000
        Target update interval measured in **core update calls**.
        Exact semantics depend on your ``QLearningCore._maybe_update_target``.
    tau : float, default=0.0
        Soft-update coefficient for target updates.
        - ``tau <= 0`` typically corresponds to hard updates only.
        - ``tau > 0`` enables Polyak averaging (if supported by base core).
    max_grad_norm : float, default=0.0
        Global gradient clipping threshold. ``0.0`` typically disables clipping.
    use_amp : bool, default=False
        Enable automatic mixed precision (AMP) for the update step.
    per_eps : float, default=1e-6
        Small epsilon used by PER pipelines when computing priorities from TD errors
        (often: ``priority = |td_error| + per_eps``). The core may store this as metadata.

    Optimizer / scheduler
    ---------------------
    optim_name : str, default="adamw"
        Optimizer name for online Q parameters.
    lr : float, default=3e-4
        Learning rate.
    weight_decay : float, default=0.0
        Weight decay.
    sched_name : str, default="none"
        Scheduler name (optional). If ``"none"``, no scheduler is used.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler knobs passed through to your scheduler builder implementation.

    Off-policy schedule / replay
    ----------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity (number of transitions).
    batch_size : int, default=256
        Batch size sampled from replay per gradient step.
    update_after : int, default=1000
        Updates are disabled until at least this many environment steps are collected.
    update_every : int, default=1
        Attempt updates every N environment steps.
    utd : float, default=1.0
        Update-to-data ratio. Interpretation is implementation-dependent; typically
        scales how many gradient updates occur per environment step.
    gradient_steps : int, default=1
        Number of gradient steps per update call (before caps).
    max_updates_per_call : int, default=1000
        Safety cap limiting the number of gradient updates performed by a single
        ``algo.update()`` call.

    Prioritized Experience Replay (PER)
    -----------------------------------
    use_per : bool, default=True
        Enable prioritized replay.
    per_alpha : float, default=0.6
        Priority exponent controlling how strongly prioritization is applied.
    per_beta : float, default=0.4
        Initial importance-sampling exponent.
    per_beta_final : float, default=1.0
        Final beta value after annealing (commonly 1.0).
    per_beta_anneal_steps : int, default=200000
        Number of environment steps over which beta is annealed from ``per_beta``
        to ``per_beta_final``.

    Exploration (epsilon-greedy)
    ----------------------------
    exploration_eps : float, default=0.1
        Initial epsilon for epsilon-greedy action selection during training.
    exploration_eps_final : float, default=0.05
        Final epsilon after annealing.
    exploration_eps_anneal_steps : int, default=200000
        Number of environment steps over which epsilon is annealed from
        ``exploration_eps`` to ``exploration_eps_final``.
    exploration_eval_eps : float, default=0.0
        Evaluation-time epsilon (typically 0.0 for greedy evaluation).

    Returns
    -------
    algo : OffPolicyAlgorithm
        Fully constructed algorithm instance. Typical usage:

        >>> algo = dqn(obs_dim=8, n_actions=4, device="cuda")
        >>> algo.setup(env)  # depends on your OffPolicyAlgorithm interface
        >>> obs = env.reset()
        >>> action = algo.act(obs)  # epsilon handled by algo/head depending on your design

    Notes
    -----
    - This builder does not validate environment compatibility. Ensure your environment
      provides discrete actions compatible with ``n_actions``.
    - Target update semantics (hard vs soft) are implemented in the base core
      (:class:`QLearningCore`) and/or your core; this builder only passes knobs through.
    """

    # -------------------------------------------------------------------------
    # 1) Head: online + target Q networks
    # -------------------------------------------------------------------------
    head = DQNHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # -------------------------------------------------------------------------
    # 2) Core: TD update engine (loss + optimization + target updates)
    # -------------------------------------------------------------------------
    core = DQNCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        huber=bool(huber),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        per_eps=float(per_eps),
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
    )

    # -------------------------------------------------------------------------
    # 3) Off-policy driver: replay buffer + scheduling + epsilon schedule
    # -------------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # update schedule
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
        # exploration (epsilon-greedy schedule)
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=int(exploration_eps_anneal_steps),
        exploration_eval_eps=float(exploration_eval_eps),
    )

    return algo
