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
   Abstract base class for **on-policy** rollout buffers.

    This buffer stores a *fixed-length* sequence of transitions collected under
    the current policy (e.g., PPO/A2C/TRPO-style rollouts). It is typically
    filled once per rollout horizon, then used to compute returns/advantages
    and to produce mini-batches for policy/value updates.

    Notes
    -----
    Contract (expected behavior of subclasses):

    - Storage is **sequential** and has a fixed length ``buffer_size``.
    - :meth:`reset` clears all arrays and resets internal cursor state.
    - :meth:`add` appends **exactly one** transition at a time.
    - :meth:`compute_returns_and_advantage` computes GAE/returns and fills
      ``self.advantages`` and ``self.returns``; it assumes the rollout is complete.
    - :meth:`sample` yields mini-batches on the configured torch ``device``.

    Parameters
    ----------
    buffer_size:
        Number of transitions to store (rollout horizon).
    obs_shape:
        Observation tensor shape (excluding batch dimension), e.g. ``(obs_dim,)``.
    action_shape:
        Action tensor shape (excluding batch dimension), e.g. ``(act_dim,)``.
    device:
        Torch device used by :meth:`sample` to place tensors (e.g., ``"cpu"``, ``"cuda"``).
    dtype_obs:
        Numpy dtype used for observation storage (default: ``np.float32``).
    dtype_act:
        Numpy dtype used for action storage (default: ``np.float32``).
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
        """
       Number of currently valid transitions.

        Returns
        -------
        int
            If the buffer has been fully filled, returns ``buffer_size``;
            otherwise returns the current cursor position ``pos``.
        """
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        """
       Reset the buffer to an empty state and reallocate storage.

        Notes
        -----
        - ``pos`` is set to 0 and ``full`` is set to False.
        - All arrays are (re)initialized to zeros.
        """
        self.pos = 0
        self.full = False

        # Vector fields (per step)
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.dtype_obs)
        self.actions = np.zeros((self.buffer_size, *self.action_shape), dtype=self.dtype_act)

        # Scalar fields (per step)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        # Store done flags as float {0.0, 1.0} for numerical convenience.
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        # State-value estimates V(s_t)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        # Log-probabilities log π(a_t|s_t)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)

        # Computed after rollout is collected
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
       Append a single transition (timestep) to the rollout.

        Parameters
        ----------
        obs:
            Observation at time t. Must have shape ``obs_shape``.
        action:
            Action taken at time t. Must have shape ``action_shape``.
        reward:
            Immediate reward r_t observed after taking ``action`` in ``obs``.
        done:
            Episode termination flag *after* this transition (i.e., for s_{t+1}).
        value:
            Value estimate V(s_t) produced by the critic at collection time.
        log_prob:
            Log probability log π(a_t|s_t) under the behavior/current policy
            at collection time.

        Raises
        ------
        RuntimeError
            If called when the buffer is already full.
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
        """
       Compute returns and advantages for the collected rollout.

        This method must populate:
        - ``self.advantages`` (e.g., via GAE(λ))
        - ``self.returns``    (e.g., advantages + values, or Monte Carlo returns)

        Parameters
        ----------
        last_value:
            Bootstrap value estimate V(s_T) for the state following the last stored
            transition. Used when the rollout ends due to horizon truncation rather
            than episode termination.
        last_done:
            Done flag for the *state after the last transition*. If True, the rollout
            ended because the episode terminated and bootstrapping should be disabled.

        Notes
        -----
        Subclasses commonly require that the buffer is fully filled
        (``self.full == True``) before calling this function.
        """
        raise NotImplementedError

    def sample(self, batch_size: int, *, shuffle: bool = True) -> Any:
        """
       Iterate over the stored rollout to produce mini-batches.

        Parameters
        ----------
        batch_size:
            Number of transitions per mini-batch.
        shuffle:
            If True, shuffle indices before batching (typical for SGD).
            If False, preserve time order (sometimes useful for debugging).

        Returns
        -------
        Any
            Subclasses should return an iterator/generator yielding a batch object
            (often a dataclass) containing torch tensors on ``self.device``.
        """
        raise NotImplementedError


# =============================================================================
# Base: ReplayBuffer (off-policy)
# =============================================================================
class BaseReplayBuffer(ABC):
    """
   Abstract base class for **off-policy** replay buffers.

    This buffer stores transitions in a *circular* (ring) structure up to a
    maximum ``capacity``. It is used by off-policy algorithms such as
    DQN/DDPG/TD3/SAC/REDQ/TQC, where samples are drawn uniformly or with
    prioritized sampling.

    Notes
    -----
    Contract (expected behavior of subclasses):

    - Storage is **circular**: new transitions overwrite old ones after capacity.
    - ``pos`` indicates the next insertion index.
    - ``full`` indicates whether we have wrapped around at least once.
    - :meth:`add` inserts exactly one transition.
    - :meth:`sample` returns a batch on torch ``device``.
    - :meth:`update_priorities` is an optional hook for PER (no-op by default).

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    obs_shape:
        Observation tensor shape (excluding batch dimension).
    action_shape:
        Action tensor shape (excluding batch dimension).
    device:
        Torch device used by :meth:`sample`.
    dtype_obs:
        Numpy dtype for observation arrays.
    dtype_act:
        Numpy dtype for action arrays.
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

    @property
    def size(self) -> int:
        """
       Number of valid transitions currently stored.

        Returns
        -------
        int
            If the buffer has wrapped around, returns ``capacity``;
            otherwise returns the current insertion index ``pos``.
        """
        return self.capacity if self.full else self.pos

    def _advance(self) -> None:
        """
       Advance the circular cursor by one step.

        Notes
        -----
        - When the cursor reaches ``capacity``, it wraps to 0 and sets ``full=True``.
        """
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def _init_storage(self) -> None:
        """
       Allocate numpy arrays for transition storage.

        Notes
        -----
        Off-policy code often benefits from storing reward/done as shape (N, 1),
        which avoids accidental broadcasting bugs when computing targets.
        """
        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=self.dtype_act)

        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        # Store done flags as float {0.0, 1.0} for easy target masking.
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

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
        """
       Insert a single transition into the replay buffer.

        Parameters
        ----------
        obs:
            Observation s_t.
        action:
            Action a_t taken in s_t.
        reward:
            Reward r_t observed after taking a_t.
        next_obs:
            Next observation s_{t+1}.
        done:
            Episode termination flag after the transition.
        **kwargs:
            Extension point for subclasses (e.g., storing log_probs, n-step info,
            env-specific metadata, PER priorities, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, **kwargs: Any) -> Any:
        """
       Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        **kwargs:
            Extension point for subclasses (e.g., PER parameters like beta,
            returning indices/weights, sampling with constraints, etc.).

        Returns
        -------
        Any
            Subclasses should return a batch object containing torch tensors on
            ``self.device`` (and optionally indices/weights for PER).
        """
        raise NotImplementedError

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
       Optional hook for prioritized replay buffers.

        Parameters
        ----------
        indices:
            Indices of sampled transitions to update.
        priorities:
            New priority values for those indices.

        Notes
        -----
        The default implementation is a no-op. PER-capable subclasses should
        override this method.
        """
        return
