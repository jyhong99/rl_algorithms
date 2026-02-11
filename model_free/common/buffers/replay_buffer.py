from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch as th

from .base_buffer import BaseReplayBuffer
from ..utils.buffer_utils import _uniform_indices
from ..utils.common_utils import _to_tensor


# =============================================================================
# Batch: ReplayBatch
# =============================================================================
@dataclass
class ReplayBatch:
    """
    A mini-batch of replay transitions returned by :class:`ReplayBuffer`.

    All required fields are torch tensors on the target device.

    Attributes
    ----------
    observations:
        Batch of observations s_t, shape ``(B, *obs_shape)``.
    actions:
        Batch of actions a_t, shape ``(B, *action_shape)``.
    rewards:
        Batch of rewards r_t, shape ``(B, 1)``.
    next_observations:
        Batch of next observations s_{t+1}, shape ``(B, *obs_shape)``.
    dones:
        Batch of done flags, shape ``(B, 1)`` with values in {0.0, 1.0}.

    behavior_logp:
        Optional behavior-policy log-probabilities, shape ``(B, 1)``.
        Present only if the buffer was configured with ``store_behavior_logp=True``.
    behavior_probs:
        Optional behavior-policy action probabilities (discrete actions),
        shape ``(B, A)``. Present only if ``store_behavior_probs=True``.

    n_step_returns:
        Optional n-step returns, shape ``(B, 1)``. Present only if ``n_step > 1``.
    n_step_dones:
        Optional n-step done flags, shape ``(B, 1)``. Present only if ``n_step > 1``.
    n_step_next_observations:
        Optional n-step next observations, shape ``(B, *obs_shape)``.
        Present only if ``n_step > 1``.
    """
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


def make_replay_batch(buf: object, idx: np.ndarray, device: Union[th.device, str]) -> ReplayBatch:
    """
    Build a :class:`ReplayBatch` from a replay-buffer-like object.

    This function is intentionally "duck-typed": it expects ``buf`` to expose
    specific numpy storages and configuration flags.

    Parameters
    ----------
    buf:
        Replay-buffer-like object that provides numpy storages. Required
        attributes:

        - ``observations``          : np.ndarray, shape ``(N, *obs_shape)``
        - ``actions``               : np.ndarray, shape ``(N, *action_shape)``
        - ``rewards``               : np.ndarray, shape ``(N, 1)``
        - ``next_observations``     : np.ndarray, shape ``(N, *obs_shape)``
        - ``dones``                 : np.ndarray, shape ``(N, 1)``

        Optional attributes are used if present and enabled by flags:

        - ``store_behavior_logp`` and ``behavior_logp`` (shape ``(N, 1)``)
        - ``store_behavior_probs`` and ``behavior_probs`` (shape ``(N, A)``)
        - ``n_step`` and n-step storages:
            ``n_step_returns`` (N,1), ``n_step_dones`` (N,1),
            ``n_step_next_observations`` (N,*obs_shape)

    idx:
        Indices of sampled transitions, shape ``(B,)``.
    device:
        Target torch device for returned tensors.

    Returns
    -------
    ReplayBatch
        Batch of torch tensors on ``device``.

    Raises
    ------
    RuntimeError
        If an optional feature flag is enabled but the required storage is missing.
    """
    # Base required fields
    obs_t = _to_tensor(getattr(buf, "observations")[idx], device=device)
    act_t = _to_tensor(getattr(buf, "actions")[idx], device=device)
    rew_t = _to_tensor(getattr(buf, "rewards")[idx], device=device)
    nxt_t = _to_tensor(getattr(buf, "next_observations")[idx], device=device)
    don_t = _to_tensor(getattr(buf, "dones")[idx], device=device)

    # Optional: behavior policy corrections
    beh_logp_t: Optional[th.Tensor] = None
    beh_probs_t: Optional[th.Tensor] = None

    if bool(getattr(buf, "store_behavior_logp", False)):
        beh = getattr(buf, "behavior_logp", None)
        if beh is None:
            raise RuntimeError("store_behavior_logp=True but behavior_logp storage is None.")
        beh_logp_t = _to_tensor(beh[idx], device=device)

    if bool(getattr(buf, "store_behavior_probs", False)):
        behp = getattr(buf, "behavior_probs", None)
        if behp is None:
            raise RuntimeError("store_behavior_probs=True but behavior_probs storage is None.")
        beh_probs_t = _to_tensor(behp[idx], device=device)

    # Optional: n-step targets
    n_step_returns_t: Optional[th.Tensor] = None
    n_step_dones_t: Optional[th.Tensor] = None
    n_step_next_obs_t: Optional[th.Tensor] = None

    if int(getattr(buf, "n_step", 1)) > 1:
        nsr = getattr(buf, "n_step_returns", None)
        nsd = getattr(buf, "n_step_dones", None)
        nsn = getattr(buf, "n_step_next_observations", None)
        if nsr is None or nsd is None or nsn is None:
            raise RuntimeError("n_step>1 but n-step storages are not allocated.")
        n_step_returns_t = _to_tensor(nsr[idx], device=device)
        n_step_dones_t = _to_tensor(nsd[idx], device=device)
        n_step_next_obs_t = _to_tensor(nsn[idx], device=device)

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


# =============================================================================
# Buffer: ReplayBuffer (uniform, off-policy)
# =============================================================================
class ReplayBuffer(BaseReplayBuffer):
    """
    Uniform replay buffer for off-policy algorithms.

    This buffer stores transitions in a fixed-capacity circular array and
    supports uniform sampling. It also optionally stores:

    - Behavior policy log-probabilities (for off-policy corrections)
    - Behavior policy action probabilities (for discrete behavior policies)
    - n-step returns (for n-step TD targets)

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored in the circular buffer.
    obs_shape:
        Observation shape (excluding batch dimension), e.g. ``(obs_dim,)``.
    action_shape:
        Action shape (excluding batch dimension), e.g. ``(act_dim,)``.
    device:
        Target torch device used when converting sampled batches to tensors.
    dtype_obs:
        Numpy dtype for observations (default: ``np.float32``).
    dtype_act:
        Numpy dtype for actions (default: ``np.float32``).
    store_behavior_logp:
        If True, store behavior-policy log-probabilities per transition
        in ``behavior_logp`` with shape ``(N, 1)``.
    store_behavior_probs:
        If True, store behavior-policy action probabilities per transition
        in ``behavior_probs`` with shape ``(N, A)``. Requires ``n_actions``.
    n_actions:
        Number of discrete actions (required if ``store_behavior_probs=True``).
    n_step:
        If > 1, compute and store n-step returns and associated fields for each
        transition as soon as enough future steps are available (or when an
        episode terminates).
    gamma:
        Discount factor in [0, 1] used for n-step return computation.

    Notes
    -----
    Storage layout
    --------------
    - ``observations`` / ``next_observations``: shape ``(N, *obs_shape)``
    - ``actions``: shape ``(N, *action_shape)``
    - ``rewards``: shape ``(N, 1)``, float32
    - ``dones``: shape ``(N, 1)``, float32 with values in {0.0, 1.0}

    n-step semantics
    ----------------
    For a base transition at time t, the n-step return is:

    .. math::

        R_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k}

    truncated early if a terminal state is encountered within the window.
    We store the results at the index of the oldest transition in the internal
    queue (the transition being "finalized"):

    - ``n_step_returns[idx0]``: the computed n-step return
    - ``n_step_dones[idx0]``: 1.0 if a terminal was encountered within the window
    - ``n_step_next_observations[idx0]``: the boundary next observation

    Implementation detail
    ---------------------
    This implementation uses an internal deque of length ``n_step`` storing:

    ``(idx, reward, next_obs, done)``

    and finalizes the n-step quantities for the front element once the deque is
    full, and flushes remaining partial windows when ``done=True`` is observed.
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

        # Optional storages: behavior policy info
        self.behavior_logp = np.zeros((self.capacity, 1), dtype=np.float32) if self.store_behavior_logp else None
        self.behavior_probs = (
            np.zeros((self.capacity, self.n_actions), dtype=np.float32) if self.store_behavior_probs else None
        )

        # Optional storages: n-step info
        self.n_step_returns: Optional[np.ndarray] = None
        self.n_step_dones: Optional[np.ndarray] = None
        self.n_step_next_observations: Optional[np.ndarray] = None
        self._nstep_queue: Optional[deque] = None

        if self.n_step > 1:
            self.n_step_returns = np.zeros((self.capacity, 1), dtype=np.float32)
            self.n_step_dones = np.zeros((self.capacity, 1), dtype=np.float32)
            self.n_step_next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=self.dtype_obs)

            # We only store what is necessary to finalize the n-step return
            # for the oldest queued transition.
            self._nstep_queue = deque(maxlen=self.n_step)

    # -------------------------------------------------------------------------
    # Internal: n-step computation
    # -------------------------------------------------------------------------
    def _write_n_step_for_queue_front(self) -> None:
        """
        Compute and write n-step quantities for the oldest queued transition.

        This method finalizes the n-step return for the transition stored at the
        front of ``self._nstep_queue`` and writes results into:

        - ``self.n_step_returns[idx0]``
        - ``self.n_step_dones[idx0]``
        - ``self.n_step_next_observations[idx0]``

        Notes
        -----
        Preconditions:
        - ``self.n_step > 1``
        - ``self._nstep_queue`` is not None
        - n-step storage arrays are allocated
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
        behavior_logp: Optional[float] = None,
        behavior_probs: Optional[np.ndarray] = None,
        **_: Any,
    ) -> None:
        """
        Insert one transition into the replay buffer.

        Parameters
        ----------
        obs:
            Observation s_t, shape ``obs_shape``.
        action:
            Action a_t, shape ``action_shape``.
        reward:
            Reward r_t.
        next_obs:
            Next observation s_{t+1}, shape ``obs_shape``.
        done:
            Episode termination flag after this transition.
        behavior_logp:
            Behavior-policy log-probability for this action. Required if
            ``store_behavior_logp=True``.
        behavior_probs:
            Behavior-policy action probabilities for discrete actions. Required if
            ``store_behavior_probs=True``. Must have shape ``(n_actions,)``.
        **_:
            Ignored extra keyword arguments (kept for API compatibility).

        Raises
        ------
        ValueError
            If required behavior fields are missing when the corresponding
            storage option is enabled, or if shapes mismatch.
        """
        # Lightweight shape/dtype normalization (fast-fail for common mistakes).
        obs = np.asarray(obs, dtype=self.dtype_obs)
        next_obs = np.asarray(next_obs, dtype=self.dtype_obs)
        action = np.asarray(action, dtype=self.dtype_act)

        if obs.shape != self.obs_shape:
            raise ValueError(f"obs must have shape {self.obs_shape}, got {obs.shape}")
        if next_obs.shape != self.obs_shape:
            raise ValueError(f"next_obs must have shape {self.obs_shape}, got {next_obs.shape}")
        if action.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}, got {action.shape}")

        # Write base 1-step transition at current cursor.
        idx_cur = self.pos
        self.observations[idx_cur] = obs
        self.actions[idx_cur] = action
        self.rewards[idx_cur, 0] = float(reward)
        self.next_observations[idx_cur] = next_obs
        self.dones[idx_cur, 0] = 1.0 if bool(done) else 0.0

        # Optional: behavior policy info
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

        # Optional: n-step bookkeeping.
        if self.n_step > 1:
            assert self._nstep_queue is not None
            self._nstep_queue.append((idx_cur, float(reward), next_obs, bool(done)))

            # Once we have n items, we can finalize the oldest one.
            if len(self._nstep_queue) == self.n_step:
                self._write_n_step_for_queue_front()

            # If episode terminates, flush remaining partial windows.
            if bool(done):
                while len(self._nstep_queue) > 0:
                    self._write_n_step_for_queue_front()
                    self._nstep_queue.popleft()

        # Move insertion cursor forward (circularly).
        self._advance()

    def sample(self, batch_size: int, **_: Any) -> ReplayBatch:
        """
        Uniformly sample a mini-batch from the replay buffer.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.

        Returns
        -------
        ReplayBatch
            Batch of torch tensors on ``self.device``.

        Raises
        ------
        ValueError
            If ``batch_size <= 0``.
        RuntimeError
            If sampling is attempted from an empty buffer.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        idx = _uniform_indices(self.size, batch_size)
        return make_replay_batch(self, idx, self.device)
