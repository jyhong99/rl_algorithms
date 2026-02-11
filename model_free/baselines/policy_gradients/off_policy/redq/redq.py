from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th

from .core import REDQCore
from .head import REDQHead
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def redq(
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
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    num_critics: int = 10,
    num_target_subset: int = 2,
    # -------------------------------------------------------------------------
    # REDQ update (core) hyperparameters
    # -------------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    # Core-side override (None -> use head default num_target_subset)
    num_target_subset_override: Optional[int] = None,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    per_eps: float = 1e-6,
    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    alpha_optim_name: str = "adamw",
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 0.0,
    # -------------------------------------------------------------------------
    # (Optional) schedulers
    # -------------------------------------------------------------------------
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
    # PER (replay config)
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
) -> OffPolicyAlgorithm:
    """
    Construct a complete REDQ algorithm instance (off-policy, continuous actions).

    This is a "config-free" factory that composes three layers:

    1) **Head** (:class:`~.head.REDQHead`)
       Owns neural networks and inference utilities:
       - Stochastic actor (SAC-style squashed Gaussian)
       - Online critic ensemble
       - Target critic ensemble
       - Action sampling and log-prob computation

    2) **Core** (:class:`~.core.REDQCore`)
       Owns learning logic:
       - Critic/actor/temperature losses (SAC-like with REDQ subset-min targets)
       - Optimizers and optional schedulers
       - Gradient clipping and optional AMP
       - Target updates (Polyak averaging)
       - TD-error computation for PER priority updates

    3) **Driver** (:class:`~model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`)
       Owns data collection / replay / update scheduling:
       - Replay buffer (uniform or prioritized)
       - Update cadence controls (update_after/update_every/UTD/gradient_steps)
       - Calls ``core.update_from_batch`` and propagates TD errors back to PER

    Parameters
    ----------
    obs_dim : int
        Observation dimension of the environment.
    action_dim : int
        Action dimension of the environment (continuous control).
    device : Union[str, torch.device], default="cpu"
        Compute device for training and inference (e.g., ``"cpu"``, ``"cuda"``).
        Note that Ray rollout workers may construct heads on CPU via a separate
        worker factory spec.

    hidden_sizes : Tuple[int, ...], default=(256, 256)
        MLP hidden layer sizes shared by actor and critics in the head.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class used by network modules (e.g. ``nn.ReLU``).
    init_type : str, default="orthogonal"
        Network initialization scheme identifier forwarded to your network builders.
    gain : float, default=1.0
        Initialization gain multiplier (network-implementation dependent).
    bias : float, default=0.0
        Initialization bias constant (network-implementation dependent).
    log_std_mode : str, default="layer"
        Actor log-std parameterization mode (e.g., ``"layer"``, ``"parameter"``).
    log_std_init : float, default=-0.5
        Initial log standard deviation value.
    num_critics : int, default=10
        Size of the critic ensemble for REDQ.
    num_target_subset : int, default=2
        Default subset size used by REDQ to compute target min-reduction.

    gamma : float, default=0.99
        Discount factor. Should satisfy ``0 <= gamma < 1``.
    tau : float, default=0.005
        Polyak coefficient for target critic updates. ``0 <= tau <= 1``.
    target_update_interval : int, default=1
        Target update cadence in *update calls*. If 0, disables target updates.
    num_target_subset_override : Optional[int], default=None
        If provided, overrides the subset size used by the core for REDQ target
        computation (otherwise core uses head defaults).
    auto_alpha : bool, default=True
        If True, automatically tune the entropy temperature alpha (SAC-style).
    alpha_init : float, default=0.2
        Initial alpha value (temperature). Internally optimized in log space.
    target_entropy : Optional[float], default=None
        Target entropy for alpha tuning. If None, core selects a default heuristic.
    max_grad_norm : float, default=0.0
        Global-norm gradient clipping threshold. If 0, disables clipping.
    use_amp : bool, default=False
        Enable mixed precision (torch.cuda.amp).
    per_eps : float, default=1e-6
        Small epsilon used for PER stability (e.g., TD-error clamping / non-zero priority).

    actor_optim_name, critic_optim_name, alpha_optim_name : str
        Optimizer identifiers forwarded to your optimizer builder.
    actor_lr, critic_lr, alpha_lr : float
        Learning rates for actor/critic/alpha optimizers.
    actor_weight_decay, critic_weight_decay, alpha_weight_decay : float
        Weight decay values.

    actor_sched_name, critic_sched_name, alpha_sched_name : str
        Scheduler identifiers forwarded to your scheduler builder.
    total_steps : int, default=0
        Total training steps for scheduler parameterization (if applicable).
    warmup_steps : int, default=0
        Warmup steps for scheduler parameterization (if applicable).
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for scheduler parameterization (if applicable).
    poly_power : float, default=1.0
        Polynomial decay exponent (if using polynomial scheduler).
    step_size : int, default=1000
        Step size for step-based schedulers (if applicable).
    sched_gamma : float, default=0.99
        Multiplicative factor for step schedulers (if applicable).
    milestones : Sequence[int], default=()
        Milestones for multi-step schedulers (if applicable).

    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Batch size sampled from replay for each update step.
    update_after : int, default=1_000
        Minimum environment steps before updates are allowed.
    update_every : int, default=1
        Perform an update every N environment steps (after ``update_after``).
    utd : float, default=1.0
        Update-to-data ratio. Controls how many gradient steps occur per env step
        (implementation-dependent; typically combines with ``gradient_steps``).
    gradient_steps : int, default=1
        Gradient steps executed per update call (in addition to UTD semantics).
    max_updates_per_call : int, default=1_000
        Safety cap to avoid long stalls in a single update call.

    use_per : bool, default=True
        If True, use prioritized experience replay.
    per_alpha : float, default=0.6
        Priority exponent (how strongly prioritization is applied).
    per_beta : float, default=0.4
        Importance sampling exponent (initial value).
    per_beta_final : float, default=1.0
        Final beta value for annealing.
    per_beta_anneal_steps : int, default=200_000
        Number of steps over which beta is annealed to ``per_beta_final``.

    Returns
    -------
    OffPolicyAlgorithm
        A fully wired algorithm instance that can be used as:

        - ``algo.setup(env)``
        - ``a = algo.act(obs)``
        - ``algo.on_env_step(transition)``
        - ``if algo.ready_to_update(): metrics = algo.update()``

    Notes
    -----
    - PER is optional. When enabled, the core typically returns TD-errors under
      ``"per/td_errors"`` so the driver can update priorities upstream.
    - The core may override critic optimizer/scheduler to cover *all* ensemble
      critics (depending on your ActorCriticCore defaults).
    """
    # ------------------------------------------------------------------
    # 1) Head: policy + critic ensembles (network construction only)
    # ------------------------------------------------------------------
    head = REDQHead(
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
        num_critics=int(num_critics),
        num_target_subset=int(num_target_subset),
    )

    # ------------------------------------------------------------------
    # 2) Core: update engine (losses + optimizers + target updates)
    # ------------------------------------------------------------------
    core = REDQCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        # subset-size override for REDQ target min-reduction
        num_target_subset=None if num_target_subset_override is None else int(num_target_subset_override),
        # entropy temperature (SAC-style)
        auto_alpha=bool(auto_alpha),
        alpha_init=float(alpha_init),
        target_entropy=None if target_entropy is None else float(target_entropy),
        # optimizers
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        alpha_optim_name=str(alpha_optim_name),
        alpha_lr=float(alpha_lr),
        alpha_weight_decay=float(alpha_weight_decay),
        # schedulers
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
        # stability / performance
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER stability
        per_eps=float(per_eps),
    )

    # ------------------------------------------------------------------
    # 3) Driver: replay buffer + update scheduling + PER integration
    # ------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay sizing / sampling
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # schedule controls
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        # PER controls
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
    )

    return algo
