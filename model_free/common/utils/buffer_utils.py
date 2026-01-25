from __future__ import annotations

from typing import Any, Callable, Optional
import operator

import numpy as np


# =============================================================================
# GAE utility
# =============================================================================
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    last_value: float,
    last_done: bool,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE-λ).

    Parameters
    ----------
    rewards : np.ndarray
        Reward sequence, shape (T,).
    values : np.ndarray
        Value estimates V(s_t), shape (T,).
    dones : np.ndarray
        Done flags after each transition, shape (T,).
        Convention: dones[t] == 1 means the episode ended after step t.
    last_value : float
        Bootstrap value V(s_T) used for the boundary at t=T-1 if last_done is False.
        If last_done is True, bootstrap is ignored (treated as 0).
    last_done : bool
        Whether the rollout ended with a terminal transition at the last step.
    gamma : float
        Discount factor in [0, 1].
    gae_lambda : float
        GAE smoothing parameter λ in [0, 1].

    Returns
    -------
    advantages : np.ndarray
        Advantage estimates, shape (T,).

    Notes
    -----
    Implements:
        Schulman et al., "High-Dimensional Continuous Control Using GAE", 2016.

    Formula
    -------
    δ_t = r_t + γ (1 - done_t) V_{t+1} - V_t
    A_t = δ_t + γ λ (1 - done_t) A_{t+1}

    where V_{t+1} is values[t+1] for t<T-1 and last_value for t=T-1.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D (T,), got shape={rewards.shape}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D (T,), got shape={values.shape}")
    if dones.ndim != 1:
        raise ValueError(f"dones must be 1D (T,), got shape={dones.shape}")
    if rewards.shape[0] != values.shape[0] or rewards.shape[0] != dones.shape[0]:
        raise ValueError(
            f"Shape mismatch: rewards={rewards.shape}, values={values.shape}, dones={dones.shape}"
        )
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    T = rewards.shape[0]
    advantages = np.zeros((T,), dtype=np.float32)

    # Boundary bootstrap: V(s_T)
    bootstrap_v = 0.0 if bool(last_done) else float(last_value)

    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        v_next = bootstrap_v if (t == T - 1) else float(values[t + 1])

        delta = rewards[t] + gamma * nonterminal * v_next - float(values[t])
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae

    return advantages


def uniform_indices(size: int, batch_size: int) -> np.ndarray:
    """
    Uniform random indices in [0, size).

    Returns
    -------
    idx : np.ndarray, shape (batch_size,)
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if size <= 0:
        raise RuntimeError("Cannot sample from an empty buffer.")
    return np.random.randint(0, int(size), size=int(batch_size))


def stratified_prefixsum_indices(
    *,
    sum_tree: Any,
    total_p: float,
    size: int,
    batch_size: int,
) -> np.ndarray:
    """
    Stratified sampling over [0, total_p) using a SumSegmentTree-like object.

    Requirements on sum_tree:
      - sum_tree.retrieve(s: float) -> int  (returns index in [0, tree_capacity))
      - sum_tree[...] supports __getitem__ (for later weight calc; not required here)

    Notes
    -----
    Handles the common PER issue where retrieve() can return indices >= size
    when size < tree_capacity (tail leaves are zeros). We reject and resample.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if size <= 0:
        raise RuntimeError("Cannot sample from an empty buffer.")
    if total_p <= 0.0:
        raise RuntimeError("total_p must be positive for stratified sampling.")

    B = int(batch_size)
    seg = float(total_p) / float(B)
    idx = np.empty((B,), dtype=np.int64)

    for i in range(B):
        a = seg * i
        b = seg * (i + 1)
        s = float(np.random.uniform(a, b))
        j = int(sum_tree.retrieve(s))
        while j >= size:
            s = float(np.random.uniform(0.0, total_p))
            j = int(sum_tree.retrieve(s))
        idx[i] = j

    return idx


# =============================================================================
# Segment tree utilities (PER)
# =============================================================================
def is_power_of_two(n: int) -> bool:
    """
    Check whether an integer is a power of two.

    Parameters
    ----------
    n : int
        Input integer.

    Returns
    -------
    is_pow2 : bool
        True iff n is a positive power of two (1, 2, 4, 8, ...).
    """
    return (n > 0) and ((n & (n - 1)) == 0)


def next_power_of_two(n: int) -> int:
    """
    Return the smallest power-of-two integer >= n.

    Parameters
    ----------
    n : int
        Target integer.

    Returns
    -------
    p : int
        Smallest power of two >= n. For n <= 1, returns 1.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class SegmentTree:
    """
    Generic segment tree supporting range queries and point updates.

    Commonly used for Prioritized Experience Replay (PER) to support:
      - Sum tree: prefix-sum sampling
      - Min tree: min-prob computation for IS weight normalization

    Parameters
    ----------
    capacity : int
        Number of leaves. Must be a positive power of two.
    operation : Callable[[float, float], float]
        Associative binary operation (e.g., sum, min).
    init_value : float
        Initial fill value for all nodes.

    Notes
    -----
    Tree layout (1-indexed internal nodes):
      - Root: index 1
      - Leaves: indices [capacity, 2*capacity)
      - Index 0 is unused for simpler arithmetic.
    """

    def __init__(
        self,
        capacity: int,
        operation: Callable[[float, float], float],
        init_value: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not is_power_of_two(capacity):
            raise ValueError(
                f"capacity must be a power of two, got {capacity}. "
                "Use next_power_of_two(...) to round up."
            )

        self.capacity = int(capacity)
        self.operation = operation
        self.tree = [float(init_value) for _ in range(2 * self.capacity)]

    def _operate_inclusive(
        self,
        start: int,
        end: int,
        node: int,
        node_start: int,
        node_end: int,
    ) -> float:
        """Internal recursive helper for inclusive interval [start, end]."""
        if start == node_start and end == node_end:
            return self.tree[node]

        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_inclusive(start, end, 2 * node, node_start, mid)
        if start > mid:
            return self._operate_inclusive(start, end, 2 * node + 1, mid + 1, node_end)

        left_val = self._operate_inclusive(start, mid, 2 * node, node_start, mid)
        right_val = self._operate_inclusive(mid + 1, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(left_val, right_val)

    def operate(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Apply operation over the half-open interval [start, end).

        Parameters
        ----------
        start : int, default=0
            Start index (inclusive).
        end : Optional[int], default=None
            End index (exclusive). If None, uses capacity.

        Returns
        -------
        result : float
            Aggregated value over [start, end).

        Raises
        ------
        IndexError
            If indices are out of bounds.
        ValueError
            If start >= end.
        """
        if end is None:
            end = self.capacity

        if not (0 <= start < self.capacity):
            raise IndexError(f"start out of range: {start}")
        if not (0 < end <= self.capacity):
            raise IndexError(f"end out of range: {end}")
        if start >= end:
            raise ValueError(f"Invalid range: start={start} must be < end={end}")

        # Convert [start, end) -> [start, end-1] for inclusive helper.
        return self._operate_inclusive(start, end - 1, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        """
        Point update at leaf index.

        Parameters
        ----------
        idx : int
            Leaf index in [0, capacity).
        val : float
            New value for that leaf.
        """
        if not (0 <= idx < self.capacity):
            raise IndexError(f"idx out of range: {idx}")

        i = idx + self.capacity
        self.tree[i] = float(val)

        i //= 2
        while i >= 1:
            self.tree[i] = self.operation(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def __getitem__(self, idx: int) -> float:
        """
        Read a leaf value.

        Parameters
        ----------
        idx : int
            Leaf index in [0, capacity).

        Returns
        -------
        val : float
            Leaf value stored at idx.
        """
        if not (0 <= idx < self.capacity):
            raise IndexError(f"idx out of range: {idx}")
        return float(self.tree[self.capacity + idx])


class SumSegmentTree(SegmentTree):
    """
    Segment tree with sum aggregation (PER prefix-sum sampling).
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Sum over [start, end).

        Returns
        -------
        s : float
            Sum of values in the interval.
        """
        return super().operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """
        Return smallest index i such that prefix_sum(i) > upperbound.

        Parameters
        ----------
        upperbound : float
            Threshold in [0, total_sum).

        Returns
        -------
        idx : int
            Leaf index in [0, capacity).

        Raises
        ------
        ValueError
            If total_sum <= 0 or upperbound out of range.
        """
        total = float(self.tree[1])
        if total <= 0.0:
            raise ValueError("Cannot retrieve from an empty/non-positive sum tree.")
        if not (0.0 <= float(upperbound) < total):
            raise ValueError(f"upperbound must be in [0, total_sum), got {upperbound} with total_sum={total}")

        idx = 1
        ub = float(upperbound)
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > ub:
                idx = left
            else:
                ub -= self.tree[left]
                idx = left + 1

        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """
    Segment tree with min aggregation (PER min-prob / IS weight normalization).
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Min over [start, end).

        Returns
        -------
        m : float
            Minimum leaf value in the interval.
        """
        return super().operate(start, end)