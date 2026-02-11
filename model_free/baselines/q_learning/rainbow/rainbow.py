from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from .head import RainbowHead
from .core import RainbowCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def rainbow(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    atom_size: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    noisy_std_init: float = 0.5,
    # ---------------------------------------------------------------------
    # Rainbow update (core) hyperparameters
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    n_step: int = 1,
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
    Build a complete Rainbow :class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
    (config-free builder).

    This function is a high-level convenience factory that composes three layers:

    1) **Head** (:class:`~.head.RainbowHead`)
       Owns the neural networks and C51 support (atoms):
       - online distributional Q-network ``q``
       - target distributional Q-network ``q_target``
       - fixed categorical support ``support`` of size ``atom_size`` within ``[v_min, v_max]``
       - NoisyNet exploration hooks (best-effort via ``reset_noise()``)

    2) **Core** (:class:`~.core.RainbowCore`)
       Implements the Rainbow *C51* update rule:
       - select greedy next action (Double DQN option)
       - construct a distributional Bellman target
       - project the target distribution onto the fixed support (C51 projection)
       - minimize cross-entropy between target distribution and online distribution
       - update target network periodically (hard or Polyak)

    3) **Algorithm** (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
       Provides the training driver:
       - replay buffer (uniform or PER)
       - warmup / update scheduling (update_after, update_every, UTD)
       - PER sampling + beta annealing (if enabled)
       - epsilon schedule for discrete exploration (if integrated into your driver)
       - optional n-step return plumbing via ``n_step``

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened vector size).
    n_actions : int
        Number of discrete actions.
    device : Union[str, torch.device], default="cpu"
        Training device.

    Network (Head) hyperparameters
    ------------------------------
    atom_size : int, default=51
        Number of atoms ``K`` for the C51 support. Must be >= 2.
    v_min : float, default=-10.0
        Minimum support value.
    v_max : float, default=10.0
        Maximum support value. Must satisfy ``v_min < v_max``.
    hidden_sizes : Tuple[int, ...], default=(256, 256)
        Hidden layer sizes for the MLP trunk in the Rainbow network.
    activation_fn : Any, default=nn.ReLU
        Activation function class used in the MLP (e.g., ``nn.ReLU``, ``nn.SiLU``).
    init_type : str, default="orthogonal"
        Weight initialization strategy name forwarded to the network builder.
    gain : float, default=1.0
        Initialization gain multiplier (interpretation depends on ``init_type``).
    bias : float, default=0.0
        Bias initialization constant.
    noisy_std_init : float, default=0.5
        Initial standard deviation for NoisyNet layers (if present in the network).

    Rainbow update (Core) hyperparameters
    ------------------------------------
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    n_step : int, default=1
        N-step return horizon. If your replay buffer stores n-step fields,
        the core can use them; otherwise it falls back to 1-step.
        Effective backup discount becomes ``gamma**n_step``.
    target_update_interval : int, default=1000
        Target update period measured in **core update calls**.
        - 0 disables target updates
        - 1 updates every update call
    tau : float, default=0.0
        Polyak coefficient for target updates performed at ``target_update_interval``:
        - ``tau <= 0``: treated as hard update (copy parameters)
        - ``tau > 0`` : soft update (Polyak averaging)
    double_dqn : bool, default=True
        Whether to use Double DQN action selection (select action via online expected Q,
        evaluate distribution via target network).
    max_grad_norm : float, default=0.0
        Global gradient clipping threshold. ``0`` disables clipping.
    use_amp : bool, default=False
        Enable torch AMP (mixed precision) during the update step (CUDA only).

    Optimizer / Scheduler
    ---------------------
    optim_name : str, default="adamw"
        Optimizer name (resolved by your optimizer builder).
    lr : float, default=3e-4
        Learning rate.
    weight_decay : float, default=0.0
        Optimizer weight decay.
    sched_name : str, default="none"
        Scheduler name (resolved by your scheduler builder). Use "none" to disable.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration knobs (semantics depend on your scheduler builder).

    OffPolicyAlgorithm schedule / replay
    ------------------------------------
    buffer_size : int, default=1_000_000
        Replay buffer capacity in transitions.
    batch_size : int, default=256
        Batch size per gradient step.
    update_after : int, default=1_000
        Start gradient updates only after at least this many environment steps are collected.
    update_every : int, default=1
        Attempt updates every N environment steps.
    utd : float, default=1.0
        Update-to-data ratio. The driver may interpret this as a fractional update budget
        accumulated per environment step.
    gradient_steps : int, default=1
        Number of gradient steps per "update unit".
    max_updates_per_call : int, default=1_000
        Safety cap limiting how many gradient steps a single ``algo.update()`` call can perform.

    PER (Prioritized Experience Replay)
    ----------------------------------
    use_per : bool, default=True
        Enable prioritized replay sampling.
    per_alpha : float, default=0.6
        Priority exponent (how strongly priorities affect sampling).
    per_beta : float, default=0.4
        Initial importance-sampling correction exponent.
    per_eps : float, default=1e-6
        Small epsilon used to stabilize priorities / weights.
    per_beta_final : float, default=1.0
        Final beta after annealing (commonly 1.0).
    per_beta_anneal_steps : int, default=200_000
        Number of environment steps to anneal beta from ``per_beta`` to ``per_beta_final``.

    Exploration (epsilon-greedy)
    ----------------------------
    exploration_eps : float, default=0.1
        Initial epsilon value for epsilon-greedy exploration (if your driver uses it).
        Note: Rainbow often relies primarily on NoisyNet, so many setups keep epsilon near 0.
    exploration_eps_final : float, default=0.05
        Final epsilon after annealing.
    exploration_eps_anneal_steps : int, default=200_000
        Number of environment steps over which epsilon is annealed.
    exploration_eval_eps : float, default=0.0
        Epsilon used during evaluation (commonly 0).

    Returns
    -------
    algo : OffPolicyAlgorithm
        Fully constructed algorithm instance. Typical usage::

            algo.setup(env)

            # rollout
            a = algo.act(obs, deterministic=False)
            algo.on_env_step(transition)

            # learner
            if algo.ready_to_update():
                metrics = algo.update()

    Notes
    -----
    - Exploration strategy:
      Rainbow typically uses NoisyNet as the primary exploration mechanism; epsilon-greedy
      is optional and can be set to ~0. This builder still exposes epsilon scheduling via
      the algorithm driver for flexibility and consistent train/eval behavior.
    - N-step returns:
      ``n_step`` is forwarded to :class:`OffPolicyAlgorithm` so the replay buffer can
      optionally compute/store n-step fields. :class:`RainbowCore` will use those fields
      when present.
    """

    # ---------------------------------------------------------------------
    # Head: online + target distributional Q networks (C51 + NoisyNet)
    # ---------------------------------------------------------------------
    head = RainbowHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        atom_size=int(atom_size),
        v_min=float(v_min),
        v_max=float(v_max),
        hidden_sizes=tuple(int(x) for x in hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        noisy_std_init=float(noisy_std_init),
        device=device,
    )

    # ---------------------------------------------------------------------
    # Core: C51 distributional update + target updates
    # ---------------------------------------------------------------------
    core = RainbowCore(
        head=head,
        gamma=float(gamma),
        n_step=int(n_step),
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
        # PER proxy epsilon + numeric stability (RainbowCore may also own log_eps)
        per_eps=float(per_eps),
    )

    # ---------------------------------------------------------------------
    # Algorithm: replay + scheduling + PER + exploration schedule + n-step config
    # ---------------------------------------------------------------------
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
        # n-step configuration passed to replay/driver
        n_step=int(n_step),
        # epsilon-greedy exploration schedule (if OffPolicyAlgorithm implements it)
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=int(exploration_eps_anneal_steps),
        exploration_eval_eps=float(exploration_eval_eps),
    )
    return algo
