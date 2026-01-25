from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from ..utils.buffer_utils import (
    MinSegmentTree, 
    SumSegmentTree, 
    next_power_of_two, 
    stratified_prefixsum_indices
)

from ..utils.common_utils import to_tensor
from .replay_buffer import ReplayBatch, ReplayBuffer, make_replay_batch


@dataclass
class PrioritizedReplayBatch(ReplayBatch):
    indices: np.ndarray = None  # type: ignore[assignment]
    weights: th.Tensor = None   # type: ignore[assignment]


def make_prioritized_replay_batch(
    buf: object,
    idx: np.ndarray,
    *,
    weights: np.ndarray,
    device: th.device | str,
) -> PrioritizedReplayBatch:
    """
    Build PrioritizedReplayBatch. Uses make_replay_batch for base fields,
    then attaches indices + weights.
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
        weights=to_tensor(weights, device=device),
    )


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.

    Extends ReplayBuffer with prioritized sampling using segment trees:
      - SumSegmentTree: sampling by prefix-sum (proportional prioritization)
      - MinSegmentTree: min-prob for importance-sampling (IS) weight normalization

    Parameters
    ----------
    capacity : int
        Replay capacity (number of transitions stored).
    obs_shape : tuple of int
        Observation shape (excluding batch dimension).
    action_shape : tuple of int
        Action shape.
    alpha : float, default=0.6
        Priority exponent. alpha=0 reduces to uniform sampling.
    beta : float, default=0.4
        Importance sampling exponent. Typically annealed toward 1.0.
    eps : float, default=1e-6
        Small positive constant to avoid zero priorities and division issues.
    device, dtype_obs, dtype_act, store_behavior_logp, store_behavior_probs, n_actions, n_step, gamma
        Forwarded to ReplayBuffer.

    Notes
    -----
    Priority definition
    -------------------
    We store transformed priorities in trees:

        p_i' = (max(priority_i, eps)) ** alpha

    Sampling probability:

        P(i) = p_i' / sum_j p_j'

    Importance sampling weight (normalized):

        w_i = (N * P(i))^{-beta} / max_j (N * P(j))^{-beta}

    where N = len(buffer).
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

        # Segment tree capacity must be power-of-two
        self.tree_capacity = next_power_of_two(self.capacity)
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)

        # Track max raw priority (before alpha transform) for new samples
        self.max_priority = 1.0

    # -------------------------------------------------------------------------
    # Priority helpers
    # -------------------------------------------------------------------------
    def _priority_to_tree_value(self, priority: float) -> float:
        """Convert raw priority -> tree value p' = (max(p, eps)) ** alpha."""
        p = float(priority)
        p = max(p, self.eps)
        return p ** self.alpha

    def _set_priority(self, idx: int, priority: float) -> None:
        """
        Set priority for a given index.

        Parameters
        ----------
        idx : int
            Transition index in [0, capacity).
        priority : float
            Raw priority value (e.g., abs(TD-error)).
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
        Add one transition with optional explicit priority.

        If priority is None, uses the current max_priority (common PER convention).
        """
        idx = self.pos  # index before super().add advances pos
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

        p = self.max_priority if priority is None else float(priority)
        self._set_priority(idx, p)

    def sample(
        self,
        batch_size: int,
        *,
        beta: Optional[float] = None,
        **_: Any,
    ) -> PrioritizedReplayBatch:
        """
        Sample a prioritized batch with importance-sampling weights.

        Parameters
        ----------
        batch_size : int
            Number of samples.
        beta : Optional[float]
            If provided, overrides the instance beta for this sampling call.

        Returns
        -------
        batch : PrioritizedReplayBatch
            Batch containing:
              - indices (np.ndarray)
              - weights (torch.Tensor, shape (B,1))
              - plus base ReplayBatch fields (+ optional behavior fields)
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty PrioritizedReplayBuffer.")

        beta_val = float(self.beta) if beta is None else float(beta)
        if beta_val < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta_val}")

        total_p = float(self.sum_tree.sum(0, self.size))
        indices = stratified_prefixsum_indices(
            sum_tree=self.sum_tree,
            total_p=total_p,
            size=self.size,
            batch_size=batch_size,
        )

        # IS weights normalization using min probability over valid range
        p_min = float(self.min_tree.min(0, self.size)) / total_p
        p_min = max(p_min, 1e-12)  # numerical guard
        max_w = (self.size * p_min) ** (-beta_val)

        # Convert sampled tree values -> probabilities
        # NOTE: sum_tree[i] stores p_i' already (priority^alpha).
        p_samples = np.asarray([self.sum_tree[int(i)] for i in indices], dtype=np.float64) / total_p
        p_samples = np.maximum(p_samples, 1e-12)

        weights = (self.size * p_samples) ** (-beta_val)
        weights = (weights / max_w).astype(np.float32).reshape(-1, 1)

        return make_prioritized_replay_batch(self, indices, weights=weights, device=self.device)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for a set of indices.

        Parameters
        ----------
        indices : np.ndarray or torch.Tensor
            Indices of sampled transitions.
        priorities : np.ndarray or torch.Tensor
            New raw priorities (e.g., abs TD errors).
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