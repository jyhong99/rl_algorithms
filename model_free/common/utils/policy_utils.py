from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn

from .common_utils import _polyak_update


# =============================================================================
# Distributional RL: Quantile Huber loss (QR-DQN / TQC)
# =============================================================================
def _quantile_huber_loss(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
    *,
    cum_prob: Optional[th.Tensor] = None,
    weights: Optional[th.Tensor] = None,
    huber_kappa: float = 1.0,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute the robust Quantile Huber loss used in QR-DQN / TQC-style critics.

    This implements the quantile regression loss with the Huber penalty:

    - For each pair (current quantile i, target quantile j), define
      :math:`\\delta = z_j^{target} - z_i^{current}`.
    - Apply Huber penalty `huber(delta)` with threshold `kappa`.
    - Weight the penalty by the quantile regression weighting term
      :math:`|\\tau_i - 1_{\\delta < 0}|`.

    Supported shapes
    ----------------
    Current quantiles:
      - (B, N)       : standard quantile critic
      - (B, C, N)    : multi-critic / ensemble critic (e.g., TQC with C critics)

    Target quantiles:
      - (B, Nt)
      - (B, C, Nt)   (broadcastable; if (B, 1, Nt) will be expanded to (B, C, Nt))

    Parameters
    ----------
    current_quantiles : torch.Tensor
        Current quantile estimates produced by the critic.
        Shape must be (B, N) or (B, C, N).
    target_quantiles : torch.Tensor
        Target quantile samples/estimates.
        Shape must be (B, Nt) or (B, C, Nt), and must match batch size B.
    cum_prob : torch.Tensor, optional
        Quantile midpoints :math:`\\tau_i` used by quantile regression, provided as:
          - if current is (B, N): shape (1, N, 1)
          - if current is (B, C, N): shape (1, 1, N, 1)
        If None, uses midpoints: (i + 0.5)/N.
    weights : torch.Tensor, optional
        Per-sample importance weights (PER). Accepts any shape broadcastable to (B,).
        If provided, loss is computed as mean(per_sample * weights).
    huber_kappa : float, default=1.0
        Huber threshold :math:`\\kappa > 0`.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss tensor.
    td_error : torch.Tensor
        A proxy TD-error for PER priorities, shape (B,). This is NOT the true
        max/mean quantile TD error; it uses a mean-quantile difference proxy.

    Raises
    ------
    ValueError
        If shapes are incompatible, batch sizes mismatch, or kappa <= 0.

    Notes
    -----
    - The returned `td_error` is detached and is intended as a stable priority proxy:
        |mean(target_quantiles) - mean(current_quantiles)|
      For more aggressive prioritization you might use quantile-wise L1 max, etc.
    - The reduction follows common practice:
        sum over current quantiles N, mean over target quantiles Nt,
        and additionally mean over critics C if present.
    """
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(f"current_quantiles must be 2D or 3D, got {current_quantiles.ndim}")
    if target_quantiles.ndim not in (2, 3):
        raise ValueError(f"target_quantiles must be 2D or 3D, got {target_quantiles.ndim}")
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Batch size mismatch: current {current_quantiles.shape[0]} vs target {target_quantiles.shape[0]}"
        )

    device = current_quantiles.device
    kappa = float(huber_kappa)
    if kappa <= 0.0:
        raise ValueError(f"huber_kappa must be > 0, got {kappa}")

    B = int(current_quantiles.shape[0])

    # ------------------------------------------------------------------
    # Case A: current (B, N), target (B, Nt)
    # ------------------------------------------------------------------
    if current_quantiles.ndim == 2:
        if target_quantiles.ndim != 2:
            raise ValueError("For 2D current_quantiles, target_quantiles must be 2D (B, Nt).")

        _, N = current_quantiles.shape

        if cum_prob is None:
            tau_hat = (th.arange(N, device=device, dtype=th.float32) + 0.5) / float(N)
            cum_prob = tau_hat.view(1, N, 1)  # (1, N, 1)

        # Pairwise deltas: (B, N, Nt)
        delta = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Huber penalty:
        #   if |d| > k: |d| - 0.5k
        #   else:       0.5 d^2 / k
        abs_delta = delta.abs()
        huber = th.where(
            abs_delta > kappa,
            abs_delta - 0.5 * kappa,
            0.5 * (delta ** 2) / kappa,
        )

        # Quantile regression weight term: |tau - 1_{delta < 0}|
        indicator = (delta.detach() < 0).to(th.float32)
        q_weight = (cum_prob.to(device=device) - indicator).abs()

        loss_mat = q_weight * huber  # (B, N, Nt)

        # Reduction: sum over N, mean over Nt -> (B,)
        per_sample = loss_mat.sum(dim=1).mean(dim=1)

        with th.no_grad():
            td_error = (target_quantiles.mean(dim=1) - current_quantiles.mean(dim=1)).abs()

    # ------------------------------------------------------------------
    # Case B: current (B, C, N), target (B, Nt) or (B, C, Nt)
    # ------------------------------------------------------------------
    else:
        _, C, N = current_quantiles.shape

        tq = target_quantiles
        if tq.ndim == 2:
            tq = tq.unsqueeze(1)  # (B, 1, Nt)
        elif tq.ndim != 3:
            raise ValueError("For 3D current_quantiles, target_quantiles must be 2D or 3D.")

        # If target has a single critic dimension, expand explicitly for clarity.
        if tq.shape[1] == 1 and C != 1:
            tq = tq.expand(-1, C, -1)  # (B, C, Nt)
        elif tq.shape[1] != C:
            raise ValueError(f"Critic dim mismatch: current C={C}, target has {tq.shape[1]}")

        if cum_prob is None:
            tau_hat = (th.arange(N, device=device, dtype=th.float32) + 0.5) / float(N)
            cum_prob = tau_hat.view(1, 1, N, 1)  # (1, 1, N, 1)

        # Pairwise deltas: (B, C, N, Nt)
        delta = tq.unsqueeze(2) - current_quantiles.unsqueeze(3)

        abs_delta = delta.abs()
        huber = th.where(
            abs_delta > kappa,
            abs_delta - 0.5 * kappa,
            0.5 * (delta ** 2) / kappa,
        )

        indicator = (delta.detach() < 0).to(th.float32)
        q_weight = (cum_prob.to(device=device) - indicator).abs()

        loss_mat = q_weight * huber  # (B, C, N, Nt)

        # Reduction: sum over N, mean over Nt, mean over C -> (B,)
        per_sample = loss_mat.sum(dim=-2).mean(dim=-1).mean(dim=1)

        with th.no_grad():
            tq_flat = tq.reshape(B, -1)
            cq_flat = current_quantiles.reshape(B, -1)
            td_error = (tq_flat.mean(dim=1) - cq_flat.mean(dim=1)).abs()

    # ------------------------------------------------------------------
    # PER weighting over batch
    # ------------------------------------------------------------------
    if weights is not None:
        w = weights.view(-1).to(device=device, dtype=per_sample.dtype)
        if int(w.shape[0]) != B:
            raise ValueError(f"PER weights batch mismatch: weights {tuple(w.shape)} vs B={B}")
        loss = (per_sample * w).mean()
    else:
        loss = per_sample.mean()

    return loss, td_error.detach()


# =============================================================================
# Distributional RL: C51 projection
# =============================================================================
def _distribution_projection(
    next_dist: th.Tensor,   # (B, K)
    rewards: th.Tensor,     # (B,) or (B, 1)
    dones: th.Tensor,       # (B,) or (B, 1) bool/{0,1}
    gamma: float,
    support: th.Tensor,     # (K,)
    v_min: float,
    v_max: float,
    eps: float = 1e-6,
) -> th.Tensor:
    """
    Project a Bellman-updated categorical distribution onto a fixed C51 support.

    This is the C51 projection operator used in distributional RL. Given a next-state
    distribution defined on a fixed discrete support z (atoms), it computes the
    projected distribution for the target:

        Tz = r + gamma * (1 - done) * z

    and then distributes probability mass from Tz back onto the fixed support via
    linear interpolation between neighboring atoms.

    Parameters
    ----------
    next_dist : torch.Tensor, shape (B, K)
        Next-state distribution (probabilities over K atoms). Assumed to be
        approximately normalized per batch element.
    rewards : torch.Tensor, shape (B,) or (B, 1)
        Rewards.
    dones : torch.Tensor, shape (B,) or (B, 1)
        Done flags (bool or {0,1}). `done=1` disables bootstrapping.
    gamma : float
        Discount factor.
    support : torch.Tensor, shape (K,)
        Fixed support atoms (z-values). Typically linearly spaced in [v_min, v_max].
    v_min : float
        Minimum support value.
    v_max : float
        Maximum support value.
    eps : float, default=1e-6
        Small constant for numerical stability (clamp and renormalization).

    Returns
    -------
    proj : torch.Tensor, shape (B, K)
        Projected distribution on the fixed support.

    Raises
    ------
    ValueError
        If shapes are incompatible, K < 2, v_max <= v_min, or eps <= 0.

    Notes
    -----
    - This implementation is defensive against tensor aliasing: it clones+detaches
      `support` so in-place ops cannot accidentally mutate a registered buffer.
    - The projection uses `index_put_(..., accumulate=True)` which is efficient and
      avoids explicit loops.
    - Output is clamped to `eps` and renormalized to ensure valid probabilities.
    """
    if next_dist.ndim != 2:
        raise ValueError(f"next_dist must have shape (B,K), got {tuple(next_dist.shape)}")
    if support.ndim != 1:
        raise ValueError(f"support must have shape (K,), got {tuple(support.shape)}")
    if next_dist.shape[1] != support.shape[0]:
        raise ValueError("K mismatch: next_dist.shape[1] must equal support.shape[0]")

    B, K = next_dist.shape
    gamma = float(gamma)
    v_min = float(v_min)
    v_max = float(v_max)
    eps = float(eps)

    if K < 2:
        raise ValueError("Support size K must be >= 2.")
    if not (v_max > v_min):
        raise ValueError(f"Require v_max > v_min. Got v_min={v_min}, v_max={v_max}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    device = next_dist.device
    dtype = next_dist.dtype

    # Break aliasing with the caller's support buffer (e.g., registered nn.Buffer)
    support_local = support.detach().clone().to(device=device, dtype=dtype)  # (K,)

    # Normalize shapes/dtypes
    rewards = rewards.view(-1, 1).to(device=device, dtype=dtype)  # (B, 1)
    dones = dones.view(-1, 1).to(device=device)                   # (B, 1)
    dones_f = dones.to(dtype=dtype)                               # (B, 1)

    if rewards.shape[0] != B or dones_f.shape[0] != B:
        raise ValueError(
            f"Batch mismatch: rewards/dones must have B={B}, got {rewards.shape[0]}, {dones_f.shape[0]}"
        )

    dz = (v_max - v_min) / float(K - 1)

    # Bellman-updated support: Tz = r + gamma*(1-done)*z
    tz = rewards + (1.0 - dones_f) * gamma * support_local.view(1, -1)  # (B, K)
    tz = tz.clamp(v_min, v_max)

    # Map to fractional indices in [0, K-1]
    b = (tz - v_min) / dz                                              # (B, K)
    l = b.floor().to(th.int64).clamp(0, K - 1)                         # (B, K)
    u = b.ceil().to(th.int64).clamp(0, K - 1)                          # (B, K)

    # Output distribution (fresh tensor; in-place ops are safe)
    proj = th.zeros_like(next_dist)                                    # (B, K)

    # Batch offsets for (B, K) advanced indexing
    offset = th.arange(B, device=device).view(-1, 1)                   # (B, 1)

    # Distribute probability mass (linear interpolation)
    proj.index_put_(
        (offset, l),
        next_dist * (u.to(dtype) - b),
        accumulate=True,
    )
    proj.index_put_(
        (offset, u),
        next_dist * (b - l.to(dtype)),
        accumulate=True,
    )

    # Numerical safety & renormalize
    proj = proj.clamp(min=eps)
    proj = proj / proj.sum(dim=1, keepdim=True).clamp(min=eps)
    return proj


# =============================================================================
# Target network utilities
# =============================================================================
@th.no_grad()
def _freeze_target(module: nn.Module) -> None:
    """
    Freeze a module for use as a target network.

    This function:
      - disables gradients (`requires_grad=False`)
      - sets the module to eval() mode

    Parameters
    ----------
    module : nn.Module
        Module to freeze.

    Returns
    -------
    None
    """
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


@th.no_grad()
def _unfreeze_target(module: nn.Module) -> None:
    """
    Re-enable gradients for a module.

    Parameters
    ----------
    module : nn.Module
        Module to unfreeze.

    Returns
    -------
    None

    Notes
    -----
    This does NOT call `train()`. Training/eval mode should be controlled by the
    training loop.
    """
    for p in module.parameters():
        p.requires_grad_(True)


@th.no_grad()
def _hard_update(target: nn.Module, source: nn.Module) -> None:
    """
    Hard update target parameters: target <- source.

    Parameters
    ----------
    target : nn.Module
        Target network to be updated.
    source : nn.Module
        Source network to copy from.

    Returns
    -------
    None
    """
    target.load_state_dict(source.state_dict())


@th.no_grad()
def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Soft update target module parameters (Polyak averaging):

        target <- (1 - tau) * target + tau * source

    Parameters
    ----------
    target : nn.Module
        Target network (updated in-place).
    source : nn.Module
        Source/online network.
    tau : float
        Source interpolation factor in (0, 1].
        Typical values: 0.005, 0.01.

    Raises
    ------
    ValueError
        If `tau` is outside (0, 1].

    Notes
    -----
    - Uses `_polyak_update` on parameter `.data` tensors to avoid autograd tracking.
    - Assumes `target` and `source` have identical parameter structure/order.
    """
    tau = float(tau)
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got: {tau}")

    for p_t, p_s in zip(target.parameters(), source.parameters()):
        _polyak_update(p_t.data, p_s.data, tau)


# =============================================================================
# PER helpers
# =============================================================================
def _get_per_weights(
    batch: Any,
    B: int,
    device: th.device | str,
) -> th.Tensor | None:
    """
    Extract PER importance weights from a batch container.

    Parameters
    ----------
    batch : Any
        Batch object that may contain attribute `weights`.
    B : int
        Expected batch size.
    device : torch.device or str
        Device to move the weights tensor to.

    Returns
    -------
    weights : torch.Tensor or None
        If present: tensor of shape (B, 1) suitable for broadcasting.
        If absent: None.

    Raises
    ------
    ValueError
        If `batch.weights` exists but has an incompatible batch dimension.

    Notes
    -----
    - Standardizes weights to shape (B, 1) so you can multiply with per-sample
      losses of shape (B,) or (B, 1) safely.
    - Dtype is preserved (no forced casting).
    """
    w = getattr(batch, "weights", None)
    if w is None:
        return None

    if not isinstance(w, th.Tensor):
        w = th.as_tensor(w)

    w = w.to(device=device)
    if w.dim() == 1:
        w = w.unsqueeze(1)  # (B, 1)

    if int(w.shape[0]) != int(B):
        raise ValueError(f"PER weights batch mismatch: weights {tuple(w.shape)} vs B={B}")

    return w


# =============================================================================
# Environment helpers
# =============================================================================
def _infer_n_actions_from_env(env: Any) -> int:
    """
    Infer the number of discrete actions from `env.action_space`.

    Supports
    --------
    - Discrete:      `action_space.n`
    - MultiDiscrete: product(action_space.nvec)

    Parameters
    ----------
    env : Any
        Environment instance with attribute `action_space`.

    Returns
    -------
    n_actions : int
        Total number of discrete actions.

    Raises
    ------
    ValueError
        If `env.action_space` is missing or unsupported (e.g., Box/Tuple),
        or if the inferred counts are invalid (<= 0).

    Notes
    -----
    For MultiDiscrete, this returns the *total* number of joint actions under a
    flat encoding. If you treat MultiDiscrete as factorized actions, you may
    want to keep `nvec` instead of collapsing to a single integer.
    """
    space = getattr(env, "action_space", None)
    if space is None:
        raise ValueError("env.action_space is missing; cannot infer n_actions.")

    # gymnasium/gym Discrete
    if hasattr(space, "n"):
        n = int(space.n)
        if n <= 0:
            raise ValueError(f"Invalid Discrete action_space.n: {n}")
        return n

    # gymnasium/gym MultiDiscrete
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        nvec = np.asarray(nvec, dtype=np.int64).reshape(-1)
        if np.any(nvec <= 0):
            raise ValueError(f"Invalid MultiDiscrete action_space.nvec: {nvec}")
        total = int(np.prod(nvec))
        if total <= 0:
            raise ValueError(f"Invalid MultiDiscrete total actions: {total}")
        return total

    raise ValueError(
        "Discrete action space required (Discrete or MultiDiscrete). "
        f"Got action_space={space}."
    )


# =============================================================================
# Small utilities
# =============================================================================
def _validate_action_bounds(
    *,
    action_dim: int,
    action_low: Optional[np.ndarray],
    action_high: Optional[np.ndarray],
) -> None:
    """
    Validate action bounds consistency and shape.

    Parameters
    ----------
    action_dim : int
        Expected action dimension.
    action_low : np.ndarray or None
        Lower bounds. Must be provided together with `action_high`.
        Expected shape: (action_dim,).
    action_high : np.ndarray or None
        Upper bounds. Must be provided together with `action_low`.
        Expected shape: (action_dim,).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If:
        - only one of (action_low, action_high) is provided,
        - bounds do not match the expected shape,
        - any element violates low <= high.

    Notes
    -----
    If both bounds are None, this function is a no-op.
    """
    if (action_low is None) ^ (action_high is None):
        raise ValueError("action_low and action_high must be provided together, or both be None.")
    if action_low is None:
        return

    if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
        raise ValueError(
            f"action_low/high must have shape ({action_dim},), got {action_low.shape}, {action_high.shape}"
        )
    if np.any(action_low > action_high):
        raise ValueError("Invalid action bounds: some action_low > action_high.")
