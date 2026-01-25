from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np


# =============================================================================
# Minimal wrapper base (gym/gymnasium optional)
# =============================================================================

class MinimalWrapper:
    """
    Minimal environment wrapper base (gym/gymnasium-agnostic).

    Purpose
    -------
    Some projects want to keep core modules importable even when neither
    `gym` nor `gymnasium` is installed (e.g., unit tests that do not touch
    actual environments). This wrapper provides a tiny, dependency-free
    compatibility layer.

    Behavior
    --------
    - Delegates most attributes to `env` via __getattr__.
    - Defines reset/step passthrough methods.
    - Does NOT attempt to emulate full Gym API surface.

    Parameters
    ----------
    env : Any
        Wrapped environment object.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        # Delegate missing attributes to the wrapped env.
        return getattr(self.env, name)

    def reset(self, **kwargs) -> Any:  # pragma: no cover
        """Pass-through reset."""
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:  # pragma: no cover
        """Pass-through step."""
        return self.env.step(action)


# =============================================================================
# Running mean / variance (online, mergeable; Chan et al.-style)
# =============================================================================

@dataclass
class RunningMeanStdState:
    """
    Serializable state for RunningMeanStd.

    Attributes
    ----------
    mean : np.ndarray
        Running mean, shape = `shape`.
    var : np.ndarray
        Running variance, shape = `shape`.
    count : float
        Effective sample count.
    """
    mean: np.ndarray
    var: np.ndarray
    count: float


class RunningMeanStd:
    """
    Running mean/variance estimator (online, mergeable).

    This implements a numerically stable parallel update rule (Chan et al.)
    and supports combining stats from multiple workers/batches.

    Typical RL uses
    ---------------
    - Observation normalization (VecNormalize-style)
    - Reward normalization / scaling

    Parameters
    ----------
    epsilon : float, optional
        Initial count to avoid division-by-zero and stabilize early updates.
        Interpretable as a small prior with mean=0 and var=1.
    shape : Tuple[int, ...], optional
        Shape of a single sample (excluding batch dimension).

    Notes
    -----
    - Internally uses float64 for numerical stability.
    - `var` is the population variance estimate consistent with the update rule.
    """

    def __init__(self, *, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got: {epsilon}")
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of a single sample (excluding batch dimension)."""
        return tuple(self.mean.shape)

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics from a batch.

        Parameters
        ----------
        x : np.ndarray
            Samples with shape (B, *shape) or (*shape,) (treated as B=1).

        Raises
        ------
        ValueError
            If `x` is not compatible with the configured `shape`.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.shape == self.mean.shape:
            x = x[None, ...]  # (1, *shape)

        if x.ndim != self.mean.ndim + 1 or x.shape[1:] != self.mean.shape:
            raise ValueError(
                "Invalid shape for update(). "
                f"Expected (B, {self.mean.shape}) or ({self.mean.shape},), got: {x.shape}"
            )

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = int(x.shape[0])
        self.update_from_moments(batch_mean=batch_mean, batch_var=batch_var, batch_count=batch_count)

    def update_from_moments(self, *, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """
        Merge running statistics with externally computed batch moments.

        This is useful for multi-process/vectorized collection where each worker
        computes (mean, var, count) locally and you then merge them centrally.

        Parameters
        ----------
        batch_mean : np.ndarray
            Mean of the batch, shape = `shape`.
        batch_var : np.ndarray
            Variance of the batch, shape = `shape`.
        batch_count : int
            Number of samples in the batch (B).

        Raises
        ------
        ValueError
            If shapes mismatch or batch_count is non-positive.
        """
        if batch_count <= 0:
            raise ValueError(f"batch_count must be > 0, got: {batch_count}")

        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)

        if batch_mean.shape != self.mean.shape or batch_var.shape != self.var.shape:
            raise ValueError(
                "Moment shapes mismatch. "
                f"Expected mean/var shape {self.mean.shape}, got mean {batch_mean.shape}, var {batch_var.shape}."
            )

        batch_count_f = float(batch_count)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count_f

        new_mean = self.mean + delta * (batch_count_f / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count_f
        m2 = m_a + m_b + np.square(delta) * (self.count * batch_count_f / tot_count)
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def std(self, *, eps: float = 1e-8) -> np.ndarray:
        """
        Compute standard deviation.

        Parameters
        ----------
        eps : float, optional
            Small constant to avoid sqrt(0).

        Returns
        -------
        np.ndarray
            Standard deviation, shape = `shape`.
        """
        return np.sqrt(self.var + float(eps))

    def normalize(self, x: np.ndarray, *, clip: Optional[float] = None, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize input using running mean/std.

        Parameters
        ----------
        x : np.ndarray
            Input array. Accepts any shape broadcastable with `shape`.
            Commonly: (*, *shape).
        clip : float, optional
            If provided and > 0, output is clipped to [-clip, +clip].
        eps : float, optional
            Numerical stability term for std.

        Returns
        -------
        np.ndarray
            Normalized array in float64 by default.

        Notes
        -----
        - This function intentionally returns float64 for stability/consistency
          with internal statistics. If you need float32, cast at the call site.
        """
        x = np.asarray(x, dtype=np.float64)
        y = (x - self.mean) / self.std(eps=eps)

        if clip is not None and float(clip) > 0.0:
            c = float(clip)
            y = np.clip(y, -c, c)

        return y

    # -------------------------------------------------------------------------
    # Serialization helpers (checkpointing)
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        Get a serializable snapshot of internal statistics.

        Returns
        -------
        Dict[str, Any]
            Dict with keys: 'mean', 'var', 'count'.
        """
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore internal statistics from a state dict.

        Parameters
        ----------
        state : Dict[str, Any]
            Output of `state_dict()`.

        Raises
        ------
        ValueError
            If shape mismatch occurs.
        """
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])

        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError(
                "State shape mismatch. "
                f"Expected {self.mean.shape}, got mean {mean.shape}, var {var.shape}."
            )
        if count <= 0:
            raise ValueError(f"Invalid count in state_dict: {count}")

        self.mean = mean
        self.var = var
        self.count = count