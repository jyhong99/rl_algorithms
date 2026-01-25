from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .base_buffer import BaseReplayBuffer
from ..utils.buffer_utils import uniform_indices
from ..utils.common_utils import to_tensor


@dataclass
class ReplayBatch:
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor

    behavior_logp: Optional[th.Tensor] = None
    behavior_probs: Optional[th.Tensor] = None

    n_step_returns: Optional[th.Tensor] = None
    n_step_dones: Optional[th.Tensor] = None
    n_step_next_observations: Optional[th.Tensor] = None


def make_replay_batch(buf: object, idx: np.ndarray, device: th.device | str) -> ReplayBatch:
    """
    Build ReplayBatch from ReplayBuffer-like object.

    Required attributes on buf:
      observations, actions, rewards, next_observations, dones

    Optional attributes on buf (if present & enabled by flags):
      store_behavior_logp, behavior_logp
      store_behavior_probs, behavior_probs
      n_step, n_step_returns, n_step_dones, n_step_next_observations
    """
    obs_t = to_tensor(getattr(buf, "observations")[idx], device=device)
    act_t = to_tensor(getattr(buf, "actions")[idx], device=device)
    rew_t = to_tensor(getattr(buf, "rewards")[idx], device=device)
    nxt_t = to_tensor(getattr(buf, "next_observations")[idx], device=device)
    don_t = to_tensor(getattr(buf, "dones")[idx], device=device)

    beh_logp_t: Optional[th.Tensor] = None
    beh_probs_t: Optional[th.Tensor] = None

    if bool(getattr(buf, "store_behavior_logp", False)):
        beh = getattr(buf, "behavior_logp", None)
        if beh is None:
            raise RuntimeError("store_behavior_logp=True but behavior_logp storage is None.")
        beh_logp_t = to_tensor(beh[idx], device=device)

    if bool(getattr(buf, "store_behavior_probs", False)):
        behp = getattr(buf, "behavior_probs", None)
        if behp is None:
            raise RuntimeError("store_behavior_probs=True but behavior_probs storage is None.")
        beh_probs_t = to_tensor(behp[idx], device=device)

    n_step_returns_t: Optional[th.Tensor] = None
    n_step_dones_t: Optional[th.Tensor] = None
    n_step_next_obs_t: Optional[th.Tensor] = None

    if int(getattr(buf, "n_step", 1)) > 1:
        nsr = getattr(buf, "n_step_returns", None)
        nsd = getattr(buf, "n_step_dones", None)
        nsn = getattr(buf, "n_step_next_observations", None)
        if nsr is None or nsd is None or nsn is None:
            raise RuntimeError("n_step>1 but n-step storages are not allocated.")
        n_step_returns_t = to_tensor(nsr[idx], device=device)
        n_step_dones_t = to_tensor(nsd[idx], device=device)
        n_step_next_obs_t = to_tensor(nsn[idx], device=device)

    return ReplayBatch(
        observations=obs_t,
        actions=act_t,
        rewards=rew_t,
        next_observations=nxt_t,
        dones=don_t,
        behavior_logp=beh_logp_t,
        behavior_probs=beh_probs_t,
        n_step_returns=n_step_returns_t,
        n_step_dones=n_step_dones_t,
        n_step_next_observations=n_step_next_obs_t,
    )


class ReplayBuffer(BaseReplayBuffer):
    """
    Uniform replay buffer for off-policy algorithms.

    Supports optional storage of:
      - behavior policy log-probabilities (for off-policy corrections)
      - behavior policy action probabilities (for discrete actions)
      - n-step returns (for n-step TD targets)

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored in the circular buffer.
    obs_shape : tuple of int
        Observation shape (excluding batch dimension).
    action_shape : tuple of int
        Action shape.
    device : Union[str, torch.device], default="cpu"
        Device used when converting sampled batches to torch tensors.
    dtype_obs : Any, default=np.float32
        Numpy dtype for observations.
    dtype_act : Any, default=np.float32
        Numpy dtype for actions.
    store_behavior_logp : bool, default=False
        If True, store behavior_logp for each transition (shape (N, 1)).
    store_behavior_probs : bool, default=False
        If True, store behavior_probs for each transition (shape (N, A)).
        Requires n_actions to be provided.
    n_actions : Optional[int], default=None
        Number of discrete actions (required if store_behavior_probs=True).
    n_step : int, default=1
        If > 1, maintain an internal queue to compute n-step returns for each transition.
    gamma : float, default=0.99
        Discount factor used in n-step return computation.

    Notes
    -----
    Storage conventions
    -------------------
    - rewards: shape (N, 1), float32
    - dones:   shape (N, 1), float32 with {0.0, 1.0}
    - observations/next_observations: shape (N, *obs_shape)
    - actions: shape (N, *action_shape)

    n-step semantics
    ----------------
    For each base transition at time t, we compute:

        R_t^(n) = sum_{k=0}^{n-1} gamma^k r_{t+k}   (truncated if done occurs early)

    and store:
      - n_step_returns[t]
      - n_step_dones[t] (1 if a terminal was encountered within the n-step window)
      - n_step_next_observations[t] (the next_obs at the truncation boundary)

    This implementation records n-step results to the index of the oldest transition
    currently in the queue (idx0).
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
        store_behavior_logp: bool = False,
        store_behavior_probs: bool = False,
        n_actions: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        self.store_behavior_logp = bool(store_behavior_logp)
        self.store_behavior_probs = bool(store_behavior_probs)

        if self.store_behavior_probs:
            if n_actions is None or int(n_actions) <= 0:
                raise ValueError("store_behavior_probs=True requires n_actions (positive int).")
            self.n_actions = int(n_actions)
        else:
            self.n_actions = None

        self.n_step = int(n_step)
        if self.n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        self.gamma = float(gamma)
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        super().__init__(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            dtype_obs=dtype_obs,
            dtype_act=dtype_act,
        )

        # Optional behavior policy storage
        self.behavior_logp = np.zeros((self.capacity, 1), dtype=np.float32) if self.store_behavior_logp else None
        self.behavior_probs = (
            np.zeros((self.capacity, self.n_actions), dtype=np.float32) if self.store_behavior_probs else None
        )

        # Optional n-step storage
        self.n_step_returns: Optional[np.ndarray] = None
        self.n_step_dones: Optional[np.ndarray] = None
        self.n_step_next_observations: Optional[np.ndarray] = None
        self._nstep_queue: Optional[deque] = None

        if self.n_step > 1:
            self.n_step_returns = np.zeros((self.capacity, 1), dtype=np.float32)
            self.n_step_dones = np.zeros((self.capacity, 1), dtype=np.float32)
            self.n_step_next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)
            # Store only what we need to compute n-step for the oldest item:
            # (idx, reward, next_obs, done)
            self._nstep_queue = deque(maxlen=self.n_step)

    # -----------------------------
    # Internal: n-step computation
    # -----------------------------
    def _write_n_step_for_queue_front(self) -> None:
        """
        Compute n-step return for the oldest queued transition and write it to storage.

        Requires:
          - self._nstep_queue is not None
          - n_step storage arrays are allocated
        """
        assert self._nstep_queue is not None
        assert self.n_step_returns is not None
        assert self.n_step_dones is not None
        assert self.n_step_next_observations is not None

        idx0, _, _, _ = self._nstep_queue[0]

        R = 0.0
        done_n = False
        next_obs_n = None

        for k, (_, r_k, nxt_k, d_k) in enumerate(self._nstep_queue):
            R += (self.gamma ** k) * float(r_k)
            next_obs_n = nxt_k
            if bool(d_k):
                done_n = True
                break

        self.n_step_returns[idx0, 0] = float(R)
        self.n_step_dones[idx0, 0] = 1.0 if done_n else 0.0
        if next_obs_n is not None:
            self.n_step_next_observations[idx0] = next_obs_n

    # -----------------------------
    # Public API
    # -----------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        behavior_logp: Optional[float] = None,
        behavior_probs: Optional[np.ndarray] = None,
        **_: Any,
    ) -> None:
        """
        Insert one transition into the buffer.

        Parameters
        ----------
        obs : np.ndarray
            Observation s_t, shape obs_shape.
        action : np.ndarray
            Action a_t, shape action_shape.
        reward : float
            Reward r_t.
        next_obs : np.ndarray
            Next observation s_{t+1}, shape obs_shape.
        done : bool
            Episode termination after this transition.
        behavior_logp : Optional[float]
            Behavior policy log-prob for this action (required if store_behavior_logp=True).
        behavior_probs : Optional[np.ndarray]
            Behavior policy action probabilities (required if store_behavior_probs=True).
            Must be shape (n_actions,).
        """
        # Minimal shape sanity (fast fail). Keep lightweight for performance.
        obs = np.asarray(obs, dtype=self.dtype_obs)
        next_obs = np.asarray(next_obs, dtype=self.dtype_obs)
        action = np.asarray(action, dtype=self.dtype_act)

        if obs.shape != self.obs_shape:
            raise ValueError(f"obs must have shape {self.obs_shape}, got {obs.shape}")
        if next_obs.shape != self.obs_shape:
            raise ValueError(f"next_obs must have shape {self.obs_shape}, got {next_obs.shape}")
        if action.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}, got {action.shape}")

        # Write base 1-step transition at current pos
        idx_cur = self.pos
        self.observations[idx_cur] = obs
        self.actions[idx_cur] = action
        self.rewards[idx_cur, 0] = float(reward)
        self.next_observations[idx_cur] = next_obs
        self.dones[idx_cur, 0] = 1.0 if bool(done) else 0.0

        # Optional behavior policy fields
        if self.store_behavior_logp:
            if behavior_logp is None:
                raise ValueError("store_behavior_logp=True requires behavior_logp.")
            assert self.behavior_logp is not None
            self.behavior_logp[idx_cur, 0] = float(behavior_logp)

        if self.store_behavior_probs:
            if behavior_probs is None:
                raise ValueError("store_behavior_probs=True requires behavior_probs.")
            assert self.behavior_probs is not None
            bp = np.asarray(behavior_probs, dtype=np.float32).reshape(-1)
            if bp.shape[0] != self.behavior_probs.shape[1]:
                raise ValueError(f"behavior_probs must have shape (A,), got {bp.shape}")
            self.behavior_probs[idx_cur] = bp

        # n-step queue update
        if self.n_step > 1:
            assert self._nstep_queue is not None
            self._nstep_queue.append((idx_cur, float(reward), next_obs, bool(done)))

            # If queue is full, compute n-step for the oldest element.
            if len(self._nstep_queue) == self.n_step:
                self._write_n_step_for_queue_front()

            # If episode ended, flush remaining partial n-step windows.
            if bool(done):
                while len(self._nstep_queue) > 0:
                    self._write_n_step_for_queue_front()
                    self._nstep_queue.popleft()

        # Circular advance
        self._advance()

    def sample(self, batch_size: int, **_: Any) -> ReplayBatch:
        """
        Uniformly sample a batch.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        batch : ReplayBatch
            Torch tensors on `self.device`.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        idx = uniform_indices(self.size, batch_size)
        return make_replay_batch(self, idx, self.device)