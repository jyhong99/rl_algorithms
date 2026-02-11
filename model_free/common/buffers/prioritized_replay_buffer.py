from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from ..utils.buffer_utils import (
    MinSegmentTree,
    SumSegmentTree,
    _next_power_of_two,
    _stratified_prefixsum_indices,
)
from ..utils.common_utils import _to_tensor
from .replay_buffer import ReplayBatch, ReplayBuffer, make_replay_batch


# =============================================================================
# Batch: PrioritizedReplayBatch
# =============================================================================
@dataclass
class PrioritizedReplayBatch(ReplayBatch):
    """
    Replay batch returned by :class:`PrioritizedReplayBuffer`.

    Extends :class:`ReplayBatch` with PER-specific fields.

    Attributes
    ----------
    indices:
        Indices of sampled transitions in the underlying replay buffer,
        shape ``(B,)``.
    weights:
        Importance-sampling (IS) weights as torch tensor on buffer device,
        shape ``(B, 1)``.
    """
    indices: np.ndarray = None  # type: ignore[assignment]
    weights: th.Tensor = None   # type: ignore[assignment]


def make_prioritized_replay_batch(
    buf: object,
    idx: np.ndarray,
    *,
    weights: np.ndarray,
    device: Union[th.device, str],
) -> PrioritizedReplayBatch:
    """
    Build a :class:`PrioritizedReplayBatch` from buffer storage and indices.

    This function delegates the "base" fields (obs/action/reward/next_obs/done
    and optional behavior/n-step fields) to :func:`make_replay_batch`, then
    attaches PER-specific fields: ``indices`` and ``weights``.

    Parameters
    ----------
    buf:
        Replay-buffer-like object providing numpy storages used by
        :func:`make_replay_batch`.
    idx:
        Sampled indices, shape ``(B,)``.
    weights:
        IS weights as numpy array, shape ``(B, 1)`` or ``(B,)`` (will be reshaped
        by the caller or inside :func:`to_tensor` usage).
    device:
        Torch device where tensors should be placed.

    Returns
    -------
    PrioritizedReplayBatch
        Batch object containing both the base replay fields and PER extras.
    """
    base = make_replay_batch(buf, idx, device)
    return PrioritizedReplayBatch(
        observations=base.observations,
        actions=base.actions,
        rewards=base.rewards,
        next_observations=base.next_observations,
        dones=base.dones,
        behavior_logp=base.behavior_logp,
        behavior_probs=base.behavior_probs,
        n_step_returns=base.n_step_returns,
        n_step_dones=base.n_step_dones,
        n_step_next_observations=base.n_step_next_observations,
        indices=idx,
        weights=_to_tensor(weights, device=device),
    )


# =============================================================================
# Buffer: PrioritizedReplayBuffer (PER)
# =============================================================================
class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) replay buffer.

    This buffer implements **proportional prioritization** (Schaul et al., 2015)
    using segment trees for O(log N) sampling and updates:

    - :class:`~SumSegmentTree` stores transformed priorities and supports
      prefix-sum sampling (sampling by cumulative mass).
    - :class:`~MinSegmentTree` stores transformed priorities and supports
      computing the minimum sampling probability used to normalize IS weights.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    obs_shape:
        Observation shape (excluding batch dimension).
    action_shape:
        Action shape (excluding batch dimension).
    alpha:
        Priority exponent. ``alpha = 0`` reduces to uniform sampling.
    beta:
        Importance-sampling exponent. Commonly annealed toward 1.0 over training.
        You may override this per-sample call via :meth:`sample(beta=...)`.
    eps:
        Small positive constant to avoid zero priorities and numerical issues.
    device, dtype_obs, dtype_act, store_behavior_logp, store_behavior_probs, n_actions, n_step, gamma:
        Forwarded to :class:`ReplayBuffer`.

    Notes
    -----
    Priority and sampling
    ---------------------
    We store **transformed** priorities in the trees:

    .. math::

        p'_i = (\\max(p_i, \\varepsilon))^{\\alpha}

    Sampling probability is:

    .. math::

        P(i) = \\frac{p'_i}{\\sum_j p'_j}

    Importance-sampling (IS) weights (normalized) are:

    .. math::

        w_i = \\frac{(N \\cdot P(i))^{-\\beta}}{\\max_j (N \\cdot P(j))^{-\\beta}}

    where :math:`N = \\text{len(buffer)}` is the number of valid transitions.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        device: Union[str, th.device] = "cpu",
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
        store_behavior_logp: bool = False,
        store_behavior_probs: bool = False,
        n_actions: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            dtype_obs=dtype_obs,
            dtype_act=dtype_act,
            store_behavior_logp=store_behavior_logp,
            store_behavior_probs=store_behavior_probs,
            n_actions=n_actions,
            n_step=n_step,
            gamma=gamma,
        )

        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        # Segment trees require a power-of-two capacity.
        # We allocate trees with >= capacity and only use range [0, self.size).
        self.tree_capacity = _next_power_of_two(self.capacity)
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)

        # Track the maximum *raw* priority (before applying alpha) so that newly
        # added transitions are sampled at least once (common PER convention).
        self.max_priority = 1.0

    # -------------------------------------------------------------------------
    # Priority helpers
    # -------------------------------------------------------------------------
    def _priority_to_tree_value(self, priority: float) -> float:
        """
        Convert a raw priority into the stored tree value.

        Parameters
        ----------
        priority:
            Raw priority (e.g., absolute TD-error).

        Returns
        -------
        float
            Transformed priority:

            ``p' = (max(priority, eps)) ** alpha``
        """
        p = float(priority)
        p = max(p, self.eps)
        return p ** self.alpha

    def _set_priority(self, idx: int, priority: float) -> None:
        """
        Set the priority for a given transition index.

        Parameters
        ----------
        idx:
            Transition index in ``[0, capacity)``.
        priority:
            Raw priority value (e.g., abs TD-error).

        Notes
        -----
        - Both trees store the **transformed** priority value p' (priority^alpha).
        - ``self.max_priority`` tracks raw priority (before exponent) for the
          "new samples get max priority" heuristic.
        """
        tree_val = self._priority_to_tree_value(priority)
        self.sum_tree[idx] = tree_val
        self.min_tree[idx] = tree_val
        self.max_priority = max(self.max_priority, float(priority))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        priority: Optional[float] = None,
        behavior_logp: Optional[float] = None,
        behavior_probs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add one transition with an optional explicit priority.

        Parameters
        ----------
        obs:
            Observation s_t.
        action:
            Action a_t.
        reward:
            Reward r_t.
        next_obs:
            Next observation s_{t+1}.
        done:
            Episode termination flag after this transition.
        priority:
            Raw priority for this transition (e.g., abs TD-error). If None,
            uses ``self.max_priority`` so new samples are likely to be replayed.
        behavior_logp:
            Optional behavior-policy log-probability (if base ReplayBuffer stores it).
        behavior_probs:
            Optional behavior-policy action probabilities (discrete) (if stored).
        **kwargs:
            Extension point forwarded to base :meth:`ReplayBuffer.add`.

        Notes
        -----
        We capture the insertion index *before* calling ``super().add`` because
        the base implementation will advance the circular cursor.
        """
        idx = self.pos  # insertion slot before super().add advances pos
        super().add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            behavior_logp=behavior_logp,
            behavior_probs=behavior_probs,
            **kwargs,
        )

        raw_p = self.max_priority if priority is None else float(priority)
        self._set_priority(idx, raw_p)

    def sample(
        self,
        batch_size: int,
        *,
        beta: Optional[float] = None,
        **_: Any,
    ) -> PrioritizedReplayBatch:
        """
        Sample a PER mini-batch with importance-sampling (IS) weights.

        Parameters
        ----------
        batch_size:
            Number of samples to draw.
        beta:
            Optional override for the instance ``self.beta`` for this call.

        Returns
        -------
        PrioritizedReplayBatch
            Batch containing base replay fields plus:
            - ``indices``: sampled indices (np.ndarray, shape (B,))
            - ``weights``: IS weights (torch.Tensor, shape (B, 1))

        Raises
        ------
        ValueError
            If ``batch_size <= 0`` or if ``beta < 0``.
        RuntimeError
            If sampling is attempted from an empty buffer.

        Notes
        -----
        Implementation details:

        - We sample indices using *stratified* prefix-sum sampling for lower variance.
        - ``sum_tree[i]`` already stores transformed priority p'_i.
        - IS weights are normalized using the minimum probability over valid entries
          (from the min-tree) to ensure weights are in (0, 1].
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty PrioritizedReplayBuffer.")

        beta_val = float(self.beta) if beta is None else float(beta)
        if beta_val < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta_val}")

        # Total mass over valid range [0, self.size).
        total_p = float(self.sum_tree.sum(0, self.size))
        if total_p <= 0.0:
            # This should not happen if priorities are kept >= eps, but guard anyway.
            raise RuntimeError("Sum of priorities is non-positive; cannot sample.")

        indices = _stratified_prefixsum_indices(
            sum_tree=self.sum_tree,
            total_p=total_p,
            size=self.size,
            batch_size=batch_size,
        )

        # Compute min probability for normalization:
        #   p_min = min_i p'_i / sum_j p'_j
        # and max weight:
        #   max_w = (N * p_min)^(-beta)
        p_min = float(self.min_tree.min(0, self.size)) / total_p
        p_min = max(p_min, 1e-12)  # numerical guard (avoid div-by-zero)
        max_w = (self.size * p_min) ** (-beta_val)

        # Convert sampled p'_i -> P(i) = p'_i / total_p
        # (sum_tree stores p'_i already)
        p_samples = np.asarray([self.sum_tree[int(i)] for i in indices], dtype=np.float64) / total_p
        p_samples = np.maximum(p_samples, 1e-12)

        # Unnormalized IS weights: (N * P(i))^(-beta)
        weights = (self.size * p_samples) ** (-beta_val)
        # Normalize by max weight so weights are <= 1
        weights = (weights / max_w).astype(np.float32).reshape(-1, 1)

        return make_prioritized_replay_batch(self, indices, weights=weights, device=self.device)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for a set of transitions.

        Parameters
        ----------
        indices:
            Indices of transitions to update. Accepts numpy arrays or torch tensors.
            Shape ``(K,)`` (or anything broadcastable to 1D after reshape).
        priorities:
            New raw priorities (e.g., abs TD-errors). Accepts numpy arrays or torch tensors.
            Shape must match ``indices``.

        Raises
        ------
        ValueError
            If shapes mismatch after flattening.

        Notes
        -----
        We only update indices within the currently valid range ``[0, self.size)``.
        """
        if th.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        if th.is_tensor(priorities):
            priorities = priorities.detach().cpu().numpy()

        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        pr = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if idx.shape != pr.shape:
            raise ValueError(f"indices and priorities must match, got {idx.shape} vs {pr.shape}")

        for i, p in zip(idx.tolist(), pr.tolist()):
            i_int = int(i)
            if 0 <= i_int < self.size:
                self._set_priority(i_int, float(p))
