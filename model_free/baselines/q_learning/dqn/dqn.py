from __future__ import annotations

from typing import Any, Tuple, Union

import torch as th

from .head import DQNHead
from .core import DQNCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def dqn(
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
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # DQN update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    double_dqn: bool = True,
    huber: bool = True,
    # target update
    target_update_interval: int = 1000,  # hard update every N updates (if tau<=0)
    tau: float = 0.0,                    # if >0, Polyak at update_interval steps
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    per_eps: float = 1e-6,
    # -----------------------------
    # Optimizer
    # -----------------------------
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    # -----------------------------
    # (Optional) scheduler
    # -----------------------------
    sched_name: str = "none",
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
    warmup_env_steps: int = 10_000,
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
    # -----------------------------
    # Exploration (epsilon-greedy)
    # -----------------------------
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
    # ... existing ...
) -> OffPolicyAlgorithm:
    """
    Build a complete DQN OffPolicyAlgorithm (discrete action space).

    This is a high-level convenience factory that composes:
      1) DQNHead : the neural networks (online Q and target Q)
      2) DQNCore : the gradient update logic (TD loss, target updates, AMP, etc.)
      3) OffPolicyAlgorithm : replay buffer + scheduling + update loop wrapper

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened vector form).
    n_actions : int
        Number of discrete actions.
    device : Union[str, torch.device]
        Device used for training (e.g., "cpu", "cuda", torch.device(...)).

    Network (Head) hyperparams
    -------------------------
    hidden_sizes : Tuple[int, ...]
        MLP hidden layer sizes for Q networks.
    activation_fn : Any
        Activation function class (e.g., nn.ReLU, nn.SiLU).
    dueling_mode : bool
        If True, use dueling architecture in QNetwork.
    init_type : str
        Initialization strategy name (e.g., "orthogonal", "xavier", ...).
    gain : float
        Gain parameter for weight initialization.
    bias : float
        Bias initialization value.

    Core (Update) hyperparams
    ------------------------
    gamma : float
        Discount factor.
    double_dqn : bool
        If True, use Double DQN target:
          a* = argmax Q_online(s', a)
          Q_target(s', a*)
        This helps reduce overestimation bias.
    huber : bool
        If True, use smooth L1 (Huber) loss; else use MSE.
    target_update_interval : int
        How often to update target network (in core update calls).
    tau : float
        Target update blending factor:
          - tau <= 0: treated as hard update (copy parameters)
          - tau > 0 : Polyak averaging (soft update)
    max_grad_norm : float
        Max gradient norm for clipping (0 disables clipping).
    use_amp : bool
        Whether to use torch.cuda.amp mixed precision.
    per_eps : float
        Small epsilon used for PER priority stabilization.

    Optimizer / Scheduler
    ---------------------
    optim_name, lr, weight_decay:
        Optimizer selection + hyperparameters.
    sched_name, total_steps, warmup_steps, ...:
        Optional LR scheduler configuration.

    OffPolicyAlgorithm scheduling / replay
    --------------------------------------
    buffer_size : int
        Replay buffer capacity (number of transitions).
    batch_size : int
        Batch size sampled from replay per gradient step.
    warmup_env_steps : int
        Initial number of environment steps collected before training starts.
        (This is passed to OffPolicyAlgorithm as warmup_steps.)
    update_after : int
        Start updating only after at least this many env steps.
        (Often same or close to warmup_env_steps; you expose both for flexibility.)
    update_every : int
        Perform an update attempt every N environment steps.
    utd : float
        Update-To-Data ratio (how many gradient updates per env step).
        Some frameworks interpret this as multiplier.
    gradient_steps : int
        Number of gradient steps per update call.
    max_updates_per_call : int
        Safety cap for how many updates a single algo.update() may perform.

    PER (Prioritized Experience Replay)
    ----------------------------------
    use_per : bool
        Enable prioritized replay.
    per_alpha : float
        Priority exponent (how strongly prioritization is used).
    per_beta : float
        Initial importance-sampling correction exponent.
    per_beta_final : float
        Final beta value after annealing (typically 1.0).
    per_beta_anneal_steps : int
        Number of env steps over which beta is annealed from per_beta -> per_beta_final.

    Returns
    -------
    algo : OffPolicyAlgorithm
        A fully constructed algorithm instance ready for:
          algo.setup(env)
          action = algo.act(obs, epsilon=eps)   (if your wrapper forwards epsilon)
          algo.on_env_step(transition)
          if algo.ready_to_update(): algo.update()
    """

    # -----------------------------
    # Head: build online/target Q-networks
    # -----------------------------
    # The head is responsible for model definition + (optionally) action selection.
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

    # -----------------------------
    # Core: build update engine (loss + optimization step)
    # -----------------------------
    # DQNCore owns the TD target computation and optimizer steps.
    core = DQNCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        huber=bool(huber),
        # optimizer settings (handled by QLearningCore infrastructure)
        optim_name=str(optim_name),
        lr=float(lr),
        weight_decay=float(weight_decay),
        # scheduler settings (optional)
        sched_name=str(sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # gradient stability / performance
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER feedback epsilon (used for priority stability in some pipelines)
        per_eps=float(per_eps),
    )

    # -----------------------------
    # Algorithm: replay + scheduling wrapper
    # -----------------------------
    # OffPolicyAlgorithm coordinates:
    # - replay buffer storage/sampling
    # - warmup collection
    # - deciding when to call core.update_from_batch(...)
    # - PER beta annealing (if enabled)
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # warmup_steps: how many env steps before updates begin (buffer fill)
        warmup_steps=int(warmup_env_steps),
        # update schedule knobs
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER configuration
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        # NEW
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=int(exploration_eps_anneal_steps),
        exploration_eval_eps=float(exploration_eval_eps),
    )
    return algo
