from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Tuple, Union

import numpy as np
import torch as th


# =============================================================================
# Base: RolloutBuffer (on-policy)
# =============================================================================
class BaseRolloutBuffer(ABC):
    """
    Base class for on-policy rollout buffers.

    Contract
    --------
    - Fixed-length sequential storage (buffer_size transitions).
    - reset() clears storage and resets pos/full.
    - add(...) appends exactly one transition at a time.
    - sample(batch_size, shuffle=True) yields RolloutBatch on torch device.
    - compute_returns_and_advantage(last_value, last_done) fills self.advantages/self.returns.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        device: Union[str, th.device] = "cpu",
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.buffer_size = int(buffer_size)
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)

        self.device = device
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act

        self.reset()

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos
    
    def reset(self) -> None:
        """Clear storage and reset cursor state."""
        self.pos = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.dtype_obs)
        self.actions = np.zeros((self.buffer_size, *self.action_shape), dtype=self.dtype_act)

        # Scalars per timestep
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)      # store as 0/1
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)     # V(s_t)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Add one transition.

        Parameters
        ----------
        obs : np.ndarray
            Observation at time t, shape obs_shape.
        action : np.ndarray
            Action taken at time t, shape action_shape.
        reward : float
            Reward r_t.
        done : bool
            Episode done flag after the transition at time t.
        value : float
            Value estimate V(s_t).
        log_prob : float
            Log probability log π(a_t|s_t) used by policy gradient methods.
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("RolloutBuffer is full. Call reset() before starting a new rollout.")

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = 1.0 if bool(done) else 0.0
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)

        self.pos += 1
        self.full = (self.pos == self.buffer_size)

    @abstractmethod
    def compute_returns_and_advantage(self, last_value: float, last_done: bool) -> None:
        """Fill self.advantages and self.returns. Requires full buffer."""
        raise NotImplementedError

    def sample(self, batch_size: int, *, shuffle: bool = True) -> Any:
        """Yield mini-batches from the buffer."""
        raise NotImplementedError


# =============================================================================
# Base: ReplayBuffer (off-policy)
# =============================================================================
class BaseReplayBuffer(ABC):
    """
    Base class for off-policy replay buffers.

    Contract
    --------
    - Circular storage with capacity, pos/full.
    - __len__ returns number of valid transitions.
    - add(...) inserts one transition.
    - sample(batch_size) returns ReplayBatch (or subclass).
    - update_priorities(indices, priorities): default no-op (PER overrides).
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        device: Union[str, th.device] = "cpu",
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)

        self.device = device
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act

        self.pos = 0
        self.full = False

        self._init_storage()

    def _advance(self) -> None:
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def _init_storage(self) -> None:
        """Allocate numpy storage for transitions."""
        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=self.dtype_act)

        # Off-policy에서는 보통 reward/done을 (B,1)로 두는 편이 편합니다(브로드캐스팅 안정).
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)  # 0/1

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

    @abstractmethod
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs: Any,
    ) -> None:
        """Insert one transition into the buffer."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, **kwargs: Any) -> Any:
        """Sample a batch from the buffer."""
        raise NotImplementedError

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Optional hook for PER. Default: no-op."""
        return