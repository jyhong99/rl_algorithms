from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn

from .common_utils import polyak_update


def quantile_huber_loss(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
    *,
    cum_prob: Optional[th.Tensor] = None,
    weights: Optional[th.Tensor] = None,
    huber_kappa: float = 1.0,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Robust Quantile Huber loss for QR-DQN / TQC style critics.

    Supports:
      - current: (B, N) or (B, C, N)
      - target : (B, Nt) or (B, C, Nt) (broadcastable)

    Returns:
      - loss: scalar
      - td_error: (B,) proxy for PER priorities
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
    # Case A: current (B, N)
    # ------------------------------------------------------------------
    if current_quantiles.ndim == 2:
        if target_quantiles.ndim != 2:
            raise ValueError("For 2D current_quantiles, target_quantiles must be 2D (B, Nt).")

        _, N = current_quantiles.shape
        if cum_prob is None:
            tau_hat = (th.arange(N, device=device, dtype=th.float32) + 0.5) / float(N)
            cum_prob = tau_hat.view(1, N, 1)  # (1, N, 1)

        # (B, N, Nt)
        delta = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        abs_delta = delta.abs()
        huber = th.where(abs_delta > kappa, abs_delta - 0.5 * kappa, 0.5 * (delta ** 2) / kappa)

        indicator = (delta.detach() < 0).to(th.float32)
        q_weight = (cum_prob.to(device=device) - indicator).abs()

        loss_mat = q_weight * huber  # (B, N, Nt)

        # sum over N, mean over Nt => (B,)
        per_sample = loss_mat.sum(dim=1).mean(dim=1)

        with th.no_grad():
            td_error = (target_quantiles.mean(dim=1) - current_quantiles.mean(dim=1)).abs()

    # ------------------------------------------------------------------
    # Case B: current (B, C, N)
    # ------------------------------------------------------------------
    else:
        _, C, N = current_quantiles.shape

        tq = target_quantiles
        if tq.ndim == 2:
            tq = tq.unsqueeze(1)  # (B,1,Nt)
        elif tq.ndim != 3:
            raise ValueError("For 3D current_quantiles, target_quantiles must be 2D or 3D.")

        # (선택이지만 강력 권장) C를 명시적으로 맞춰서 브로드캐스팅 애매함 제거
        if tq.shape[1] == 1 and C != 1:
            tq = tq.expand(-1, C, -1)  # (B,C,Nt)

        if cum_prob is None:
            tau_hat = (th.arange(N, device=device, dtype=th.float32) + 0.5) / float(N)
            cum_prob = tau_hat.view(1, 1, N, 1)  # (1,1,N,1)

        # 기대: (B, C, N, Nt)
        delta = tq.unsqueeze(2) - current_quantiles.unsqueeze(3)

        abs_delta = delta.abs()
        huber = th.where(abs_delta > kappa, abs_delta - 0.5 * kappa, 0.5 * (delta ** 2) / kappa)

        indicator = (delta.detach() < 0).to(th.float32)
        q_weight = (cum_prob.to(device=device) - indicator).abs()

        loss_mat = q_weight * huber

        # -----------------------------
        # Robust reduction (B, ...)
        # -----------------------------
        if loss_mat.ndim == 4:
            # (B, C, N, Nt) -> (B,)
            per_sample = loss_mat.sum(dim=-2).mean(dim=-1).mean(dim=1)
        elif loss_mat.ndim == 3:
            # Could be (B, C, N) or (B, N, Nt)
            if loss_mat.shape[1] == C:
                # (B, C, N) -> sum over N, mean over C
                per_sample = loss_mat.sum(dim=-1).mean(dim=1)
            else:
                # (B, N, Nt) -> sum over N, mean over Nt
                per_sample = loss_mat.sum(dim=1).mean(dim=-1)
        elif loss_mat.ndim == 2:
            # (B, N) -> sum over N
            per_sample = loss_mat.sum(dim=1)
        else:
            raise ValueError(f"Unexpected loss_mat rank: {loss_mat.ndim}, shape={tuple(loss_mat.shape)}")

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
def distribution_projection(
    next_dist: th.Tensor,   # (B, K)
    rewards: th.Tensor,     # (B,) or (B,1)
    dones: th.Tensor,       # (B,) or (B,1) with {0,1} or bool
    gamma: float,
    support: th.Tensor,     # (K,)
    v_min: float,
    v_max: float,
    eps: float = 1e-6,
) -> th.Tensor:
    """
    C51 distribution projection onto a fixed discrete support.

    This version is defensive against tensor aliasing:
    it clones+detaches support to avoid any inadvertent in-place modification
    on the original support buffer (e.g., a registered buffer in the head/network).

    Returns
    -------
    proj : torch.Tensor
        Projected distribution, shape (B, K).
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

    if K < 2:
        raise ValueError("Support size K must be >= 2.")
    if not (v_max > v_min):
        raise ValueError(f"Require v_max > v_min. Got v_min={v_min}, v_max={v_max}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    device = next_dist.device
    dtype = next_dist.dtype

    # ------------------------------------------------------------------
    # IMPORTANT: break any aliasing with caller's support buffer
    # ------------------------------------------------------------------
    support_local = support.detach().clone().to(device=device, dtype=dtype)  # (K,)

    # Normalize shapes/dtypes
    rewards = rewards.view(-1, 1).to(device=device, dtype=dtype)  # (B,1)
    dones = dones.view(-1, 1).to(device=device)                   # (B,1)
    dones_f = dones.to(dtype=dtype)                               # (B,1) float

    if rewards.shape[0] != B or dones_f.shape[0] != B:
        raise ValueError(f"Batch mismatch: rewards/dones must have B={B}, got {rewards.shape[0]}, {dones_f.shape[0]}")

    dz = (v_max - v_min) / float(K - 1)

    # ------------------------------------------------------------------
    # Bellman-updated support: Tz = r + gamma*(1-done)*z
    # ------------------------------------------------------------------
    tz = rewards + (1.0 - dones_f) * gamma * support_local.view(1, -1)  # (B,K)
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / dz                          # (B,K)
    l = b.floor().to(th.int64).clamp(0, K - 1)     # (B,K)
    u = b.ceil().to(th.int64).clamp(0, K - 1)      # (B,K)

    # Prepare output (fresh tensor; in-place ops here are safe)
    proj = th.zeros_like(next_dist)                # (B,K)

    # batch offsets for advanced indexing
    offset = th.arange(B, device=device).view(-1, 1)  # (B,1)

    # ------------------------------------------------------------------
    # Distribute probability mass
    # ------------------------------------------------------------------
    # Note:
    # - next_dist is expected to be (approximately) normalized.
    # - even if not, we renormalize at the end.
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
def freeze_target(module: nn.Module) -> None:
    """
    Freeze a module for use as a target network.

    This function:
      - disables gradients (requires_grad=False)
      - sets module to eval() mode

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
def unfreeze_target(module: nn.Module) -> None:
    """
    Re-enable gradients for a module.

    Notes
    -----
    This does NOT call train(). That should be controlled by the training loop.

    Parameters
    ----------
    module : nn.Module
        Module to unfreeze.

    Returns
    -------
    None
    """
    for p in module.parameters():
        p.requires_grad_(True)


@th.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
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
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Soft update target module parameters:
        target <- (1 - tau) * target + tau * source

    Parameters
    ----------
    target : nn.Module
        Target network (updated in-place).
    source : nn.Module
        Source/online network.
    tau : float
        Source interpolation factor in (0, 1].
        Typical: 0.005.
    """
    tau = float(tau)
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got: {tau}")

    for p_t, p_s in zip(target.parameters(), source.parameters()):
        polyak_update(p_t.data, p_s.data, tau)


# =============================================================================
# PER helpers
# =============================================================================
def get_per_weights(
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
    device : torch.device or str
        Device to move weights to.
    B : int
        Expected batch size.

    Returns
    -------
    weights : torch.Tensor or None
        If present: shape (B, 1), dtype preserved from batch.weights.
        If absent: None.

    Raises
    ------
    ValueError
        If batch.weights has an incompatible batch dimension.

    Notes
    -----
    This function standardizes weights to shape (B,1) for broadcasting against
    scalar per-sample losses of shape (B,).
    """
    w = getattr(batch, "weights", None)
    if w is None:
        return None

    if not isinstance(w, th.Tensor):
        w = th.as_tensor(w)

    w = w.to(device=device)
    if w.dim() == 1:
        w = w.unsqueeze(1)  # (B,1)

    if w.shape[0] != B:
        raise ValueError(f"PER weights batch mismatch: weights {tuple(w.shape)} vs B={B}")

    return w


# =============================================================================
# Environment helpers
# =============================================================================
def infer_n_actions_from_env(env: Any) -> int:
    """
    Infer number of discrete actions from env.action_space.

    Supports
    --------
    - Discrete:       action_space.n
    - MultiDiscrete:  product(action_space.nvec)

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
        If action_space is missing or unsupported (e.g., Box / Tuple),
        or if action counts are invalid.
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
def validate_action_bounds(
    *,
    action_dim: int,
    action_low: Optional[np.ndarray],
    action_high: Optional[np.ndarray],
) -> None:
    """Validate action bounds consistency and shape."""
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