from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from .head import QRDQNHead
from .core import QRDQNCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


# =============================================================================
# Builder (config-free)
# =============================================================================
def qrdqn(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    n_quantiles: int = 200,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # -----------------------------
    # QRDQN update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    tau: float = 0.0,
    double_dqn: bool = True,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
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
    # -----------------------------
    # PER (Prioritized Experience Replay)
    # -----------------------------
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
    # -----------------------------
    # Exploration (epsilon-greedy)
    # -----------------------------
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
) -> OffPolicyAlgorithm:
    """
    Build a complete QR-DQN OffPolicyAlgorithm (config-free).

    This builder wires together three components:
      1) Head (QRDQNHead)
         - owns online and target quantile Q-networks
         - provides quantiles() / q_values() API
         - provides epsilon-greedy discrete action selection

      2) Core (QRDQNCore)
         - implements the QR-DQN update rule using quantile regression loss
         - supports optional Double DQN target action selection
         - updates target network periodically (hard/soft)

      3) Algorithm driver (OffPolicyAlgorithm)
         - owns replay buffer (uniform or PER)
         - enforces warmup / update schedules
         - performs repeated gradient steps according to UTD configuration
         - handles exploration epsilon scheduling (if integrated into act())

    Returns
    -------
    algo : OffPolicyAlgorithm
        Typical usage:
            algo.setup(env)

            # rollout
            a = algo.act(obs)   # epsilon behavior depends on algorithm/head integration
            algo.on_env_step(transition)

            # learner
            if algo.ready_to_update():
                metrics = algo.update()

    Notes
    -----
    - This builder is "config-free": it returns a ready-to-run OffPolicyAlgorithm object.
    - Discrete exploration is epsilon-greedy. Depending on your framework design:
        (A) epsilon scheduling may live in OffPolicyAlgorithm (recommended),
        (B) or be passed into head.act(epsilon=...).
      Here we pass exploration parameters into OffPolicyAlgorithm so it can manage
      epsilon consistently across train/eval modes.
    """

    # ------------------------------------------------------------------
    # Head: quantile Q networks (online + target)
    #
    # QRDQNHead forward API:
    #   - quantiles(obs)        -> (B, N, A)
    #   - quantiles_target(obs) -> (B, N, A)
    #   - q_values(obs)         -> (B, A)  (mean over N)
    #   - act(obs, ...)         -> action index (int)
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
    # Core: update engine
    #
    # Implements distributional TD update:
    #   - quantile regression loss (quantile Huber)
    #   - optional Double DQN for action selection in target
    #   - target update every target_update_interval (hard/soft via tau)
    #
    # Also owns optimizer/scheduler for head.q parameters via QLearningCore.
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
    # Algorithm: replay buffer + update scheduling
    #
    # OffPolicyAlgorithm is responsible for:
    #   - storing transitions into replay buffer
    #   - warmup random action phase (warmup_env_steps)
    #   - enforcing update frequency (update_after, update_every)
    #   - applying UTD / gradient_steps loops
    #   - PER sampling / beta annealing (if enabled)
    #   - exploration epsilon schedule (NEW in your refactor)
    # ------------------------------------------------------------------
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        # replay
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        # schedules
        warmup_steps=int(warmup_env_steps),
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
