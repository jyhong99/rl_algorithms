from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .head import VPGDiscreteHead
from .core import VPGDiscreteCore
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm


def vpg_discrete(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    use_baseline: bool = True,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # ---------------------------------------------------------------------
    # VPG update (core) hyperparameters
    # ---------------------------------------------------------------------
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
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
    # ---------------------------------------------------------------------
    # (Optional) schedulers
    # ---------------------------------------------------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # ---------------------------------------------------------------------
    # OnPolicyAlgorithm rollout / schedule
    # ---------------------------------------------------------------------
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.int64,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a discrete-action VPG :class:`OnPolicyAlgorithm` instance (builder function).

    This function wires together three components into a ready-to-use algorithm:

    1) **VPGDiscreteHead**
       - Actor: categorical policy :math:`\\pi(a\\mid s)` over ``n_actions``
       - Critic (optional): baseline :math:`V(s)` controlled by ``use_baseline``

    2) **VPGDiscreteCore**
       - Performs one update per call using:
         * policy-gradient loss (always),
         * optional entropy regularization (``ent_coef``),
         * optional value regression (``vf_coef``) when baseline is enabled.
       - Baseline behavior strictly follows the head configuration.

    3) **OnPolicyAlgorithm**
       - Collects rollouts from the environment,
       - Computes returns and advantages (typically via GAE),
       - Calls ``core.update_from_batch(...)`` according to scheduling parameters.

    Baseline contract
    -----------------
    ``use_baseline`` controls whether the head constructs a critic, and the core
    follows the head configuration:

    - baseline OFF:
        Actor-only REINFORCE-style updates (no value loss, no critic optimizer).
    - baseline ON:
        Actor + critic updates (recommended).

    Notes
    -----
    - Typical VPG settings are one update per rollout:
      ``update_epochs=1`` and ``minibatch_size=None`` (full-batch).
    - ``gae_lambda`` is included for consistency with a shared on-policy driver.
      Whether you use GAE for "pure VPG" is a design choice; in practice it is
      often used to reduce variance.
    - Discrete actions are typically stored as ``int64`` (PyTorch categorical
      distributions expect integer class indices).

    Parameters
    ----------
    obs_dim : int
        Observation (state) dimension.
    n_actions : int
        Number of discrete actions.
    device : Union[str, torch.device], default="cpu"
        Torch device used by head/core/algo.

    use_baseline : bool, default=True
        Whether to include a value baseline network :math:`V(s)`.
    hidden_sizes : Tuple[int, ...], default=(64, 64)
        Hidden layer sizes for actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function **class** used by the MLP builders.
    init_type : str, default="orthogonal"
        Weight initialization scheme identifier (primarily used by the critic builder).
    gain : float, default=1.0
        Initialization gain multiplier.
    bias : float, default=0.0
        Bias initialization value.

    vf_coef : float, default=0.5
        Value-loss coefficient (only applies when baseline is enabled).
    ent_coef : float, default=0.0
        Entropy regularization coefficient.
    max_grad_norm : float, default=0.5
        Global norm gradient clipping threshold (core-side).
    use_amp : bool, default=False
        Enable AMP for updates if supported by the core/BaseCore.

    actor_optim_name : str, default="adamw"
        Actor optimizer name (resolved by ``build_optimizer``).
    actor_lr : float, default=3e-4
        Actor learning rate.
    actor_weight_decay : float, default=0.0
        Actor weight decay.
    critic_optim_name : str, default="adamw"
        Critic optimizer name (only used if baseline is enabled).
    critic_lr : float, default=3e-4
        Critic learning rate (only used if baseline is enabled).
    critic_weight_decay : float, default=0.0
        Critic weight decay (only used if baseline is enabled).

    actor_sched_name : str, default="none"
        Actor LR scheduler name.
    critic_sched_name : str, default="none"
        Critic LR scheduler name (only used if baseline is enabled).
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler knobs passed through to the core and resolved by ``build_scheduler``.

    rollout_steps : int, default=2048
        Number of environment steps collected per update.
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE lambda for advantage estimation.
    update_epochs : int, default=1
        Number of update epochs per rollout (VPG typically uses 1).
    minibatch_size : Optional[int], default=None
        Minibatch size used by the on-policy algorithm update loop. ``None`` is
        interpreted as full-batch in most implementations.
    dtype_obs : Any, default=np.float32
        Observation dtype used by rollout buffer / preprocessing.
    dtype_act : Any, default=np.int64
        Action dtype used by rollout buffer / preprocessing (discrete indices).
    normalize_advantages : bool, default=False
        Whether to normalize advantages before the policy update (algorithm-side).
    adv_eps : float, default=1e-8
        Epsilon used for advantage normalization: ``std + adv_eps``.

    Returns
    -------
    algo : OnPolicyAlgorithm
        Fully constructed discrete-action VPG algorithm instance.

    Examples
    --------
    >>> algo = vpg_discrete(obs_dim=8, n_actions=4, device="cpu", use_baseline=True)
    >>> algo.setup(env)
    >>> obs = env.reset()
    >>> while True:
    ...     action = algo.act(obs)
    ...     next_obs, reward, done, info = env.step(action)
    ...     algo.on_env_step(obs=obs, action=action, reward=reward, done=done, info=info)
    ...     if algo.ready_to_update():
    ...         metrics = algo.update()
    ...     obs = next_obs
    """
    # ------------------------------------------------------------------
    # 1) Head: networks (actor + optional value baseline critic)
    # ------------------------------------------------------------------
    head = VPGDiscreteHead(
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(hidden_sizes),
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        use_baseline=bool(use_baseline),
    )

    # ------------------------------------------------------------------
    # 2) Core: update engine (baseline behavior follows head)
    # ------------------------------------------------------------------
    core = VPGDiscreteCore(
        head=head,
        vf_coef=float(vf_coef),
        ent_coef=float(ent_coef),
        # optimizers
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        # schedulers
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=tuple(int(m) for m in milestones),
        # misc
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
    )

    # ------------------------------------------------------------------
    # 3) Algorithm: rollout collection + update scheduling
    # ------------------------------------------------------------------
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=int(rollout_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=int(update_epochs),
        minibatch_size=None if minibatch_size is None else int(minibatch_size),
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
