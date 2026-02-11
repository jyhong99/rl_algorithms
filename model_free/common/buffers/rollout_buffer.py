from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Tuple, Union

import numpy as np
import torch as th

from .base_buffer import BaseRolloutBuffer
from ..utils.buffer_utils import _compute_gae
from ..utils.common_utils import _to_tensor


# =============================================================================
# Batch: RolloutBatch
# =============================================================================
@dataclass
class RolloutBatch:
    """
    A mini-batch of rollout transitions returned by :class:`RolloutBuffer`.

    Attributes
    ----------
    observations:
        Observations s_t, shape ``(B, *obs_shape)``.
    actions:
        Actions a_t, shape ``(B, *action_shape)``.
    log_probs:
        Log-probabilities log π(a_t|s_t) under the behavior/current policy at
        collection time, shape ``(B,)`` or ``(B, 1)`` depending on storage.
    values:
        Value estimates V(s_t) at collection time, shape ``(B,)`` or ``(B, 1)``.
    returns:
        Computed returns, shape ``(B,)`` or ``(B, 1)``.
    advantages:
        Computed advantages (e.g., GAE), shape ``(B,)`` or ``(B, 1)``.
    dones:
        Done flags after the transition, shape ``(B,)`` or ``(B, 1)``
        with values in {0.0, 1.0}.
    """
    observations: th.Tensor
    actions: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    advantages: th.Tensor
    dones: th.Tensor


def make_rollout_batch(buf: object, idx: np.ndarray, device: Union[th.device, str]) -> RolloutBatch:
    """
    Build a :class:`RolloutBatch` from a rollout-buffer-like object.

    This function is intentionally "duck-typed": it only requires that ``buf``
    exposes certain numpy storages.

    Parameters
    ----------
    buf:
        Rollout-buffer-like object. Required attributes:

        - ``observations`` : np.ndarray, shape ``(T, *obs_shape)``
        - ``actions``      : np.ndarray, shape ``(T, *action_shape)``
        - ``log_probs``    : np.ndarray, shape ``(T,)`` or ``(T, 1)``
        - ``values``       : np.ndarray, shape ``(T,)`` or ``(T, 1)``
        - ``returns``      : np.ndarray, shape ``(T,)`` or ``(T, 1)``
        - ``advantages``   : np.ndarray, shape ``(T,)`` or ``(T, 1)``
        - ``dones``        : np.ndarray, shape ``(T,)`` or ``(T, 1)``

    idx:
        Indices of sampled timesteps, shape ``(B,)``.
    device:
        Torch device where tensors should be placed.

    Returns
    -------
    RolloutBatch
        Batch of torch tensors on ``device``.
    """
    return RolloutBatch(
        observations=_to_tensor(getattr(buf, "observations")[idx], device=device),
        actions=_to_tensor(getattr(buf, "actions")[idx], device=device),
        log_probs=_to_tensor(getattr(buf, "log_probs")[idx], device=device),
        values=_to_tensor(getattr(buf, "values")[idx], device=device),
        returns=_to_tensor(getattr(buf, "returns")[idx], device=device),
        advantages=_to_tensor(getattr(buf, "advantages")[idx], device=device),
        dones=_to_tensor(getattr(buf, "dones")[idx], device=device),
    )


# =============================================================================
# Concrete: RolloutBuffer with GAE(λ)
# =============================================================================
class RolloutBuffer(BaseRolloutBuffer):
    """
    Rollout buffer for on-policy algorithms (PPO/A2C/TRPO) with GAE(λ).

    The buffer stores a fixed-length rollout of T transitions and then computes:
    - Advantages using Generalized Advantage Estimation (GAE)
    - Returns as ``returns = advantages + values``

    Parameters
    ----------
    buffer_size:
        Number of transitions per rollout (T).
    obs_shape:
        Observation shape (excluding batch dimension).
    action_shape:
        Action shape (excluding batch dimension).
    gamma:
        Discount factor in [0, 1].
    gae_lambda:
        GAE λ parameter in [0, 1]. Smaller values bias toward TD(0), larger values
        approach Monte-Carlo-style advantages.
    normalize_advantages:
        If True, normalize advantages to zero mean and unit variance.
        (Useful for PPO stability; optional depending on implementation.)
    adv_eps:
        Small constant added to the advantage standard deviation to avoid division
        by zero during normalization.
    device:
        Target torch device used when producing batches.
    dtype_obs:
        Numpy dtype used to store observations.
    dtype_act:
        Numpy dtype used to store actions.

    Notes
    -----
    Expected usage pattern
    ----------------------
    1) Call :meth:`reset`.
    2) Call :meth:`add` exactly ``buffer_size`` times (fills the rollout).
    3) Call :meth:`compute_returns_and_advantage(last_value, last_done)`.
    4) Iterate :meth:`sample(batch_size)` to obtain training mini-batches.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = False,
        adv_eps: float = 1e-8,
        device: Union[str, th.device] = "cpu",
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            dtype_obs=dtype_obs,
            dtype_act=dtype_act,
        )

        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
        if adv_eps <= 0.0:
            raise ValueError(f"adv_eps must be > 0, got {adv_eps}")

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = bool(normalize_advantages)
        self.adv_eps = float(adv_eps)

    def compute_returns_and_advantage(self, last_value: float, last_done: bool) -> None:
        """
        Compute advantages (GAE) and returns for the current rollout.

        Parameters
        ----------
        last_value:
            Bootstrap value estimate V(s_T) for the observation following the last
            stored transition (i.e., at time step T). Used when the rollout ends
            due to horizon truncation rather than episode termination.
        last_done:
            Done flag for the state after the final stored transition. If True,
            the episode terminated at the end of the rollout and bootstrapping
            should be disabled.

        Raises
        ------
        RuntimeError
            If called before the rollout is fully collected.

        Notes
        -----
        This method requires the buffer to be full (``self.full == True``).
        """
        if not self.full:
            raise RuntimeError("compute_returns_and_advantage() requires a full buffer.")

        adv = _compute_gae(
            rewards=self.rewards,
            values=self.values,
            dones=self.dones,
            last_value=float(last_value),
            last_done=bool(last_done),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        ret = adv + self.values

        if self.normalize_advantages:
            mean = float(adv.mean())
            std = float(adv.std()) + self.adv_eps
            adv = (adv - mean) / std

        self.advantages[:] = adv
        self.returns[:] = ret

    def sample(self, batch_size: int, *, shuffle: bool = True) -> Iterator[RolloutBatch]:
        """
        Yield mini-batches from a completed rollout.

        Parameters
        ----------
        batch_size:
            Number of timesteps per mini-batch.
        shuffle:
            If True, shuffle timestep indices before batching (standard SGD).
            If False, keep chronological order (sometimes useful for debugging).

        Yields
        ------
        RolloutBatch
            A mini-batch of tensors placed on ``self.device``.

        Raises
        ------
        RuntimeError
            If called before the rollout is fully collected.
        ValueError
            If ``batch_size <= 0``.
        """
        if not self.full:
            raise RuntimeError("sample() requires a full buffer (full rollout).")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        indices = np.arange(self.buffer_size, dtype=np.int64)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            yield make_rollout_batch(self, idx, self.device)
