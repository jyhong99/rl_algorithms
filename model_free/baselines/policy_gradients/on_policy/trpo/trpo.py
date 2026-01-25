from __future__ import annotations

from typing import Any, Sequence, Tuple, Union, Optional

import numpy as np
import torch as th

from .head import TRPOHead
from .core import TRPOCore

from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def trpo(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # -----------------------------
    # TRPO update (core) hyperparams
    # -----------------------------
    max_kl: float = 1e-2,
    cg_iters: int = 10,
    cg_damping: float = 1e-2,
    backtrack_iters: int = 10,
    backtrack_coeff: float = 0.8,
    accept_ratio: float = 0.1,
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    # -----------------------------
    # Critic optimizer / scheduler
    # -----------------------------
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    critic_sched_name: str = "none",
    # Scheduler knobs (shared across cores; optional)
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # -----------------------------
    # OnPolicyAlgorithm rollout / schedule
    # -----------------------------
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a complete TRPO OnPolicyAlgorithm (config-free).

    This is a *builder* function that wires together:
      1) TRPOHead : actor/critic neural networks (policy + value)
      2) TRPOCore : TRPO update engine (CG + KL-constrained line search + critic regression)
      3) OnPolicyAlgorithm : rollout collection + advantage/return computation + update scheduling

    Notes
    -----
    - TRPO commonly uses "one rollout -> one policy update", so `update_epochs=1` is recommended.
    - Mini-batching is not standard for classic TRPO; the typical implementation uses the full
      rollout batch for the natural gradient step. Therefore the default is `minibatch_size=None`.
      If your OnPolicyAlgorithm *requires* an integer, set `minibatch_size=rollout_steps`.

    Parameters
    ----------
    obs_dim : int
        Observation/state dimension.
    action_dim : int
        Continuous action dimension.
    device : Union[str, torch.device]
        Torch device used by head/core/algo.

    Network (head) hyperparams
    --------------------------
    hidden_sizes : Tuple[int, ...]
        Hidden layer sizes for both actor and critic MLPs.
    activation_fn : Any
        Activation function class (e.g., torch.nn.ReLU).
    init_type : str
        Weight initialization method (e.g., "orthogonal").
    gain : float
        Initialization gain multiplier (passed to your network init).
    bias : float
        Bias initialization value.
    log_std_mode : str
        Gaussian policy log-std parameterization mode (e.g., "param").
    log_std_init : float
        Initial log standard deviation for the Gaussian policy.

    TRPO update (core) hyperparams
    ------------------------------
    max_kl : float
        Maximum allowed mean KL divergence for each TRPO update (trust region size).
    cg_iters : int
        Conjugate gradient iterations for solving (F + damping I)x = g.
    cg_damping : float
        Damping term added to Fisher-vector product for numerical stability.
    backtrack_iters : int
        Backtracking line search iterations for enforcing KL constraint and improvement.
    backtrack_coeff : float
        Multiplicative factor applied to step size during backtracking (0 < coeff < 1).
    accept_ratio : float
        Minimum ratio of actual improvement to expected improvement (TRPO acceptance criterion).
    vf_coef : float
        Value-function loss scaling coefficient (if core uses it to weight critic regression).
    max_grad_norm : float
        Gradient clipping norm used for critic update (and any other clipped params in core).
    use_amp : bool
        Enable AMP for critic regression path (policy update still needs higher-order grads).

    Critic optimizer / scheduler
    ----------------------------
    critic_optim_name : str
        Optimizer name for critic (e.g., "adam", "adamw", "sgd").
    critic_lr : float
        Critic learning rate.
    critic_weight_decay : float
        Critic weight decay.
    critic_sched_name : str
        Scheduler name for critic (e.g., "none", "linear", "cosine", "onecycle", ...).
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Common scheduler knobs shared across cores/builders.

    OnPolicyAlgorithm rollout / schedule
    ------------------------------------
    rollout_steps : int
        Number of environment steps collected per update.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    update_epochs : int
        Number of update epochs per rollout (TRPO typically 1).
    minibatch_size : Optional[int]
        Minibatch size for updates. TRPO usually uses full-batch; keep None unless required.
    dtype_obs : Any
        Observation dtype used by rollout buffer / preprocessing.
    dtype_act : Any
        Action dtype used by rollout buffer / preprocessing.
    normalize_advantages : bool
        Whether to normalize advantages before policy update (can stabilize training).
    adv_eps : float
        Epsilon used in advantage normalization (std + eps).

    Returns
    -------
    algo : OnPolicyAlgorithm
        Fully constructed TRPO algorithm instance.

    Example
    -------
    algo = trpo(obs_dim=..., action_dim=..., device="cuda")
    algo.setup(env)
    obs = env.reset()
    while True:
        action = algo.act(obs)
        next_obs, reward, done, info = env.step(action)
        algo.on_env_step(...)
        if algo.ready_to_update():
            metrics = algo.update()
        obs = next_obs
    """

    # ---------------------------------------------------------------------
    # 1) Head: neural network container (actor Gaussian policy + critic V(s))
    # ---------------------------------------------------------------------
    head = TRPOHead(
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

    # ---------------------------------------------------------------------
    # 2) Core: TRPO update engine
    #    - critic regression optimizer/scheduler
    #    - policy natural gradient (CG + Fisher-vector product)
    #    - KL-constrained line search
    # ---------------------------------------------------------------------
    core = TRPOCore(
        head=head,
        # TRPO / natural gradient knobs
        max_kl=float(max_kl),
        cg_iters=int(cg_iters),
        cg_damping=float(cg_damping),
        # backtracking line search knobs
        backtrack_iters=int(backtrack_iters),
        backtrack_coeff=float(backtrack_coeff),
        accept_ratio=float(accept_ratio),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # critic optimizer / scheduler config
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        critic_sched_name=str(critic_sched_name),
        # scheduler shared knobs
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
    )

    # ---------------------------------------------------------------------
    # 3) Algorithm: rollout collection + GAE/returns + update scheduling
    # ---------------------------------------------------------------------
    # NOTE: TRPO is typically full-batch. If minibatch_size is None, ensure your
    # OnPolicyAlgorithm treats it as "use full batch"; otherwise pass rollout_steps.
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=(int(minibatch_size) if minibatch_size is not None else None),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
