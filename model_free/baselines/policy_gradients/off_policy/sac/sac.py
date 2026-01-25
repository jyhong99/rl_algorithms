from __future__ import annotations

from typing import Any, Tuple, Union, Optional

import torch as th

from .head import SACHead
from .core import SACCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def sac(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    # -----------------------------
    # SAC update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
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
    # PER (Prioritized Experience Replay)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Factory function: build a complete SAC OffPolicyAlgorithm (config-free).

    This function wires together:
      1) SACHead:  neural networks (actor + twin critics + target critics)
      2) SACCore:  update logic (critic/actor/alpha updates + target Polyak)
      3) OffPolicyAlgorithm: replay buffer + update scheduling + PER plumbing

    Parameters
    ----------
    obs_dim : int
        Observation dimension (flattened).
    action_dim : int
        Action dimension (continuous).
    device : Union[str, torch.device]
        Compute device for networks and updates.

    Head hyperparameters
    --------------------
    hidden_sizes : Tuple[int, ...]
        MLP hidden layer sizes for actor and critic networks.
    activation_fn : Any
        Activation function class (e.g., torch.nn.ReLU). May also be a string
        if your head supports resolving it (e.g., for Ray worker specs).
    init_type, gain, bias : str, float, float
        Weight initialization configuration forwarded to networks.
    log_std_mode : str
        How log-std is parameterized in the policy (e.g., "layer", "parameter", etc.).
    log_std_init : float
        Initial log-std value (stochasticity level at start of training).

    Core (SAC update) hyperparameters
    ---------------------------------
    gamma : float
        Discount factor.
    tau : float
        Polyak averaging coefficient for target critic updates.
    target_update_interval : int
        Perform target update every N update calls (1 = every update).
    auto_alpha : bool
        If True, learn entropy temperature alpha automatically.
    alpha_init : float
        Initial alpha value (used as exp(log_alpha_init)).
    target_entropy : Optional[float]
        If None, core uses a heuristic (typically -action_dim).
    max_grad_norm : float
        If > 0, applies gradient clipping (global norm) to actor/critic params.
    use_amp : bool
        If True, use torch.cuda.amp for mixed precision (useful on GPUs).

    Optimizer / scheduler hyperparameters
    -------------------------------------
    actor_optim_name, critic_optim_name, alpha_optim_name : str
        Optimizer names resolved by your build_optimizer() utility (e.g., "adamw").
    *_lr : float
        Learning rates for actor/critic/alpha optimizers.
    *_weight_decay : float
        Weight decay for actor/critic/alpha optimizers.
    *_sched_name : str
        Scheduler names resolved by build_scheduler() (e.g., "none", "linear", etc.).
    total_steps, warmup_steps, min_lr_ratio, poly_power : int, int, float, float
        Shared scheduler parameters (depends on your scheduler builder).

    OffPolicyAlgorithm schedule / replay
    ------------------------------------
    buffer_size : int
        Replay buffer capacity (transitions).
    batch_size : int
        Batch size sampled from replay per update step.
    warmup_env_steps : int
        Number of environment steps to collect before allowing updates.
        (OffPolicyAlgorithm uses this as warmup_steps)
    update_after : int
        Additional gate for updates (e.g., do not update before this many env steps).
    update_every : int
        Run update every N environment steps once ready.
    utd : float
        Update-to-data ratio multiplier (how many updates per env step, on average).
    gradient_steps : int
        Number of gradient steps per update call.
    max_updates_per_call : int
        Safety cap to prevent very large update bursts.

    PER parameters
    --------------
    use_per : bool
        Enable prioritized replay.
    per_alpha : float
        Priority exponent (0 = uniform, 1 = full prioritization).
    per_beta : float
        Initial importance-sampling correction exponent.
    per_eps : float
        Small constant added to TD errors to avoid zero priorities.
    per_beta_final : float
        Final beta value after annealing.
    per_beta_anneal_steps : int
        Number of environment steps over which beta anneals to per_beta_final.

    Returns
    -------
    algo : OffPolicyAlgorithm
        Fully constructed algorithm object.

        Typical usage:
            algo.setup(env)
            a = algo.act(obs)
            algo.on_env_step(transition)
            if algo.ready_to_update():
                metrics = algo.update()
    """

    # -----------------------------
    # Head: networks (actor + critics)
    # -----------------------------
    # SACHead owns:
    #  - actor: stochastic squashed Gaussian policy
    #  - critic: twin Q networks (Q1,Q2)
    #  - critic_target: target twin Q networks (frozen)
    head = SACHead(
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

    # -----------------------------
    # Core: update engine (SAC losses)
    # -----------------------------
    # SACCore owns the update logic:
    #  - critic loss: MSE(Q_i(s,a), y)
    #  - actor loss:  E[ alpha*logπ(a|s) - min(Q1,Q2)(s,a) ]
    #  - alpha loss:  -E[ log_alpha * (logπ(a|s) + target_entropy) ] (optional)
    #  - target critic Polyak updates
    core = SACCore(
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
        # grad/amp
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # -----------------------------
    # Algorithm: replay + scheduling
    # -----------------------------
    # OffPolicyAlgorithm owns:
    #  - replay buffer (uniform or PER)
    #  - warmup/update schedule gates
    #  - calling core.update_from_batch() repeatedly based on utd/gradient_steps
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        warmup_steps=int(warmup_env_steps),
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER config
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
    )
    return algo
