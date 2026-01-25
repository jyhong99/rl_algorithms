from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Union, Iterator

import numpy as np
import torch as th

from .base_buffer import BaseRolloutBuffer
from ..utils.buffer_utils import compute_gae
from ..utils.common_utils import to_tensor


@dataclass
class RolloutBatch:
    observations: th.Tensor
    actions: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    advantages: th.Tensor
    dones: th.Tensor


def make_rollout_batch(buf: object, idx: np.ndarray, device: th.device | str) -> RolloutBatch:
    """
    Build RolloutBatch from BaseRolloutBuffer-like object.

    Required attributes on buf:
      observations, actions, log_probs, values, returns, advantages, dones
    """
    return RolloutBatch(
        observations=to_tensor(getattr(buf, "observations")[idx], device=device),
        actions=to_tensor(getattr(buf, "actions")[idx], device=device),
        log_probs=to_tensor(getattr(buf, "log_probs")[idx], device=device),
        values=to_tensor(getattr(buf, "values")[idx], device=device),
        returns=to_tensor(getattr(buf, "returns")[idx], device=device),
        advantages=to_tensor(getattr(buf, "advantages")[idx], device=device),
        dones=to_tensor(getattr(buf, "dones")[idx], device=device),
    )


# =============================================================================
# Concrete: RolloutBuffer with GAE
# =============================================================================
class RolloutBuffer(BaseRolloutBuffer):
    """
    Rollout buffer for on-policy algorithms (PPO/A2C/TRPO) with GAE.

    This buffer stores a fixed-length rollout and computes:
      - advantages via GAE(λ)
      - returns = advantages + values

    Parameters
    ----------
    buffer_size : int
        Number of transitions per rollout (T).
    obs_shape : tuple of int
        Observation shape (excluding batch dimension).
    action_shape : tuple of int
        Action shape.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter.
    normalize_adv : bool, default=False
        If True, normalizes advantages to zero mean and unit variance.
    device : torch.device or str
        Target device used at sampling time.
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
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0,1], got {gae_lambda}")

        self.adv_eps = float(adv_eps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = bool(normalize_advantages)

    def compute_returns_and_advantage(self, last_value: float, last_done: bool) -> None:
        """
        Compute advantages (GAE) and returns for the current rollout.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_T) for the state after the last stored transition.
        last_done : bool
            Whether the last stored transition ended the episode.

        Notes
        -----
        Requires the buffer to be full (pos == buffer_size).
        """
        if not self.full:
            raise RuntimeError("compute_returns_and_advantage() requires a full buffer.")

        adv = compute_gae(
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

    def sample(self, batch_size: int, *, shuffle: bool = True) -> Iterator:
        """
        Yield mini-batches from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        shuffle : bool
            Whether to shuffle indices before batching.

        Yields
        ------
        RolloutBatch
            A batch of tensors placed on `self.device`.
        """
        if not self.full:
            raise RuntimeError("sample() requires a full buffer (full rollout).")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        indices = np.arange(self.buffer_size)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            yield make_rollout_batch(self, idx, self.device)