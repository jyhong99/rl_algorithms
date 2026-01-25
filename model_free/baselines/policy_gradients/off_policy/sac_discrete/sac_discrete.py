from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th

from .head import SACDiscreteHead
from .core import SACDiscreteCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def sac_discrete(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    actor_hidden_sizes: Tuple[int, ...] = (256, 256),
    critic_hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # Discrete SAC update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
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
    alpha_optim_name: str = "adamw",
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 0.0,
    # -----------------------------
    # (Optional) schedulers
    # -----------------------------
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
    # PER (replay config)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Build a complete Discrete SAC OffPolicyAlgorithm (config-free).

    This is a convenience "factory" that wires together:
      1) SACDiscreteHead:  actor + twin critics + target critics (networks)
      2) SACDiscreteCore:  update rules (losses/optimizers/targets/alpha)
      3) OffPolicyAlgorithm: replay buffer + update scheduling + PER integration

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened). Head networks assume obs is shaped (B, obs_dim).
    n_actions : int
        Number of discrete actions. Actor outputs a categorical distribution over these.
    device : Union[str, torch.device]
        Device on which networks and updates will run (e.g. "cpu", "cuda").

    Network (head) hyperparams
    --------------------------
    actor_hidden_sizes : Tuple[int, ...]
        MLP hidden sizes for the actor (categorical policy).
    critic_hidden_sizes : Tuple[int, ...]
        MLP hidden sizes for the critics (twin Q networks over discrete actions).
    activation_fn : Any
        Nonlinearity constructor used in networks (e.g., torch.nn.ReLU).
    dueling_mode : bool
        If True, critics use dueling architecture (if supported by your DoubleQNetwork).
    init_type : str
        Weight initialization scheme name (e.g., "orthogonal").
    gain : float
        Initialization gain (passed to your network builders).
    bias : float
        Bias initialization value (passed to your network builders).

    Discrete SAC (core) hyperparams
    -------------------------------
    gamma : float
        Discount factor in [0, 1).
    tau : float
        Polyak coefficient for target critic update (0 < tau <= 1).
    target_update_interval : int
        Update target critic every N core updates (0 disables periodic update).
    auto_alpha : bool
        If True, learn entropy temperature alpha via log_alpha optimizer.
    alpha_init : float
        Initial alpha value (core stores log(alpha_init)).
    target_entropy : Optional[float]
        Entropy target. If None, core chooses a default (often log(|A|) for discrete).
        Note: the sign convention depends on your core's alpha_loss definition.
    max_grad_norm : float
        Global norm clipping threshold for gradients (0 disables clipping).
    use_amp : bool
        If True, use autocast + GradScaler for mixed precision updates.
    per_eps : float
        Small epsilon to clamp TD errors for PER priority update stability.

    Optimizers
    ----------
    actor_optim_name, actor_lr, actor_weight_decay :
        Optimizer config for actor parameters.
    critic_optim_name, critic_lr, critic_weight_decay :
        Optimizer config for critic parameters.
    alpha_optim_name, alpha_lr, alpha_weight_decay :
        Optimizer config for log_alpha (only used if auto_alpha=True).

    (Optional) schedulers
    ---------------------
    actor_sched_name, critic_sched_name, alpha_sched_name :
        Scheduler types ("none" disables).
    total_steps, warmup_steps, min_lr_ratio, poly_power :
        Common scheduler knobs (lambda/poly warmup etc., depending on your builder).
    step_size, sched_gamma, milestones :
        Step/MultiStep style scheduler knobs.

    OffPolicyAlgorithm schedule / replay
    ------------------------------------
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Batch size sampled from replay for each update.
    warmup_env_steps : int
        How many environment steps to collect before allowing updates.
        Passed to OffPolicyAlgorithm as `warmup_steps` (env-step based).
    update_after : int
        Earliest env step when updates are allowed (typical: >= warmup_env_steps).
    update_every : int
        Frequency of attempting updates in env steps (1 = every step).
    utd : float
        Update-to-data ratio multiplier (OffPolicyAlgorithm contract).
    gradient_steps : int
        Number of gradient steps per update call (OffPolicyAlgorithm contract).
    max_updates_per_call : int
        Safety cap on number of updates performed in a single call (prevents long stalls).

    PER (Prioritized Experience Replay)
    -----------------------------------
    use_per : bool
        Enable PER in OffPolicyAlgorithm replay.
    per_alpha : float
        Priority exponent (how strongly priorities affect sampling).
    per_beta : float
        Initial importance-sampling exponent.
    per_beta_final : float
        Final beta value for annealing.
    per_beta_anneal_steps : int
        Number of env steps over which beta is annealed.

    Returns
    -------
    algo : OffPolicyAlgorithm
        A complete algorithm object ready for:
          - algo.setup(env)
          - action = algo.act(obs)
          - algo.on_env_step(transition)
          - if algo.ready_to_update(): algo.update()
    """

    # -----------------------------
    # Head: networks (actor + critic + target critic)
    # -----------------------------
    # The head is responsible for forward passes / action selection primitives:
    # - Discrete policy distribution (categorical)
    # - Twin Q(s,·) and target twin Q_t(s,·)
    head = SACDiscreteHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        actor_hidden_sizes=tuple(actor_hidden_sizes),
        critic_hidden_sizes=tuple(critic_hidden_sizes),
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    # -----------------------------
    # Core: update engine
    # -----------------------------
    # The core owns the learning rule:
    # - Builds optimizers/schedulers (actor/critic via ActorCriticCore base)
    # - Computes target values using target critic
    # - Updates critic, actor, and optionally alpha (entropy temperature)
    # - Handles target critic polyak/hard updates
    core = SACDiscreteCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        auto_alpha=bool(auto_alpha),
        alpha_init=float(alpha_init),
        target_entropy=(None if target_entropy is None else float(target_entropy)),
        # optim
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        alpha_optim_name=str(alpha_optim_name),
        alpha_lr=float(alpha_lr),
        alpha_weight_decay=float(alpha_weight_decay),
        # sched
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
        # grad/amp
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER
        per_eps=float(per_eps),
    )

    # -----------------------------
    # Algorithm: replay + scheduling
    # -----------------------------
    # OffPolicyAlgorithm orchestrates:
    # - replay buffer creation (PER or uniform)
    # - collection scheduling (warmup, update_after, update_every)
    # - calling core.update_from_batch on sampled batches
    # - reporting common metrics (buffer size, env steps, PER beta, etc.)
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        warmup_steps=int(warmup_env_steps),      # env-step warmup gate
        update_after=int(update_after),          # first env step when updates allowed
        update_every=int(update_every),          # env-step update cadence
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
