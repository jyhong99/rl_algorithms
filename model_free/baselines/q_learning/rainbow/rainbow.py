from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from .head import RainbowHead
from .core import RainbowCore
from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


# =============================================================================
# Builder (config-free)
# =============================================================================
def rainbow(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    # -----------------------------
    # Network (head) hyperparams
    # -----------------------------
    atom_size: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    noisy_std_init: float = 0.5,
    # -----------------------------
    # Rainbow update (core) hyperparams
    # -----------------------------
    gamma: float = 0.99,
    n_step: int = 1,
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
    # PER
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
    Build a complete Rainbow OffPolicyAlgorithm (config-free).

    This builder wires together:
      - RainbowHead: distributional Q network container (online + target) with NoisyNet support
      - RainbowCore: C51 update rule (distributional Bellman backup + projection)
      - OffPolicyAlgorithm: replay buffer + scheduling + PER + exploration schedule

    Notes on exploration
    --------------------
    - Rainbow typically relies primarily on NoisyNet for exploration.
      Therefore, in many setups you run with epsilon ~ 0 during training.
    - Nevertheless, this library exposes an epsilon schedule at the algorithm
      level (OffPolicyAlgorithm). This enables:
        * mixing NoisyNet + epsilon-greedy
        * turning epsilon on/off depending on environment
        * using epsilon=0 for evaluation via exploration_eval_eps

    Returns
    -------
    algo : OffPolicyAlgorithm
        Typical usage:
          algo.setup(env)
          a = algo.act(obs, deterministic=False)  # epsilon is handled by algorithm schedule (if implemented)
          algo.on_env_step(transition)
          if algo.ready_to_update(): algo.update()
    """

    # -------------------------------------------------------------------------
    # Head: online + target distributional Q networks
    #
    # - atom_size/v_min/v_max define the C51 support (K atoms).
    # - noisy_std_init controls initial parameter noise scale for NoisyLinear layers.
    # - The head exposes:
    #     * q.dist(obs) -> (B, A, K)
    #     * q(obs)      -> expected Q, (B, A)
    #     * act(...)    -> epsilon-greedy over expected Q (optionally with Noisy reset)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Core: C51 learning rule
    #
    # - gamma/n_step define the effective discount used in projection:
    #     gamma_n = gamma ** n_step
    # - double_dqn selects greedy action using online expected Q, then evaluates
    #   the target distribution using target network (standard Double DQN logic).
    # - target_update_interval/tau define hard vs Polyak updates:
    #     tau <= 0  -> hard copy every interval
    #     tau > 0   -> soft update every interval
    # -------------------------------------------------------------------------
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
        # grad/amp
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # PER proxy epsilon
        per_eps=float(per_eps),
    )

    # -------------------------------------------------------------------------
    # Algorithm: replay + update scheduling + PER + exploration schedule
    #
    # Replay / scheduling
    # -------------------
    # - warmup_steps: before this many env steps, act() samples random actions
    # - update_after: updates are disabled until env_steps >= update_after
    # - update_every: only update on env_steps % update_every == 0
    # - utd: updates-to-data; accumulates fractional update budget per env step
    # - gradient_steps: number of gradient steps per "update unit"
    #
    # PER
    # ---
    # - use_per + (alpha, beta schedule, eps) configure prioritized replay
    # - core returns "per/td_errors" as a proxy; algorithm can update priorities
    #
    # n-step plumbing
    # ---------------
    # - n_step and n_step_gamma are forwarded to the replay buffer so it can
    #   optionally compute/store n-step returns. RainbowCore will consume
    #   n-step fields if present; otherwise it falls back to 1-step.
    # -------------------------------------------------------------------------
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

        # PER configuration
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),

        # n-step configuration passed to replay
        n_step=int(n_step),

        # Exploration schedule (if OffPolicyAlgorithm implements epsilon scheduling)
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=int(exploration_eps_anneal_steps),
        exploration_eval_eps=float(exploration_eval_eps),
    )
    return algo
