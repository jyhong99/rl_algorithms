from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Minimal wrapper base (gym/gymnasium optional)
# =============================================================================
class MinimalWrapper:
    """
    Minimal environment wrapper base (gym/gymnasium-agnostic).

    Purpose
    -------
    Some projects want core modules to remain importable even when neither
    `gym` nor `gymnasium` is installed (e.g., unit tests that do not touch real
    environments, or lightweight tooling like log parsers).

    This wrapper provides a dependency-free compatibility layer that:
      - stores a single wrapped object `env`
      - delegates missing attributes to `env`
      - exposes `reset()` and `step()` pass-through methods

    Parameters
    ----------
    env : Any
        Wrapped environment-like object.

    Notes
    -----
    - This is **not** a full Gym wrapper. It does not attempt to mirror Gym's
      metadata/spec/spaces semantics. It only provides the small subset most
      training loops rely on.
    - Attribute delegation uses `__getattr__` (called only when normal lookup fails).
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """
        Delegate missing attribute access to the wrapped environment.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Attribute resolved from `self.env`.

        Raises
        ------
        AttributeError
            If the wrapped env does not have the attribute either.
        """
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> Any:  # pragma: no cover
        """
        Reset the wrapped environment (pass-through).

        Parameters
        ----------
        **kwargs : Any
            Forwarded to `env.reset(**kwargs)`.

        Returns
        -------
        Any
            Whatever `env.reset()` returns.
        """
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:  # pragma: no cover
        """
        Step the wrapped environment (pass-through).

        Parameters
        ----------
        action : Any
            Action passed to `env.step(action)`.

        Returns
        -------
        Any
            Whatever `env.step()` returns.
        """
        return self.env.step(action)


# =============================================================================
# Running mean / variance (online, mergeable; Chan et al.-style)
# =============================================================================
@dataclass
class RunningMeanStdState:
    """
    Serializable state container for RunningMeanStd.

    This dataclass is useful when you want a typed, explicit representation of the
    state (instead of a loose dict). It is optional: `RunningMeanStd.state_dict()`
    and `.load_state_dict()` still use plain dicts for maximum interoperability.

    Parameters
    ----------
    mean : np.ndarray
        Running mean, shape = `shape`.
    var : np.ndarray
        Running (population) variance, shape = `shape`.
    count : float
        Effective sample count (can be fractional due to epsilon initialization).

    Notes
    -----
    - `count` is a float because the initialization uses an `epsilon` prior.
    - Arrays are expected to be float-like.
    """
    mean: np.ndarray
    var: np.ndarray
    count: float


class RunningMeanStd:
    """
    Running mean/variance estimator (online, mergeable).

    This implements a numerically stable parallel update rule (Chan et al.-style)
    that supports combining statistics computed from multiple batches/workers.

    Typical RL uses
    ---------------
    - Observation normalization (VecNormalize-style)
    - Reward/return normalization (running return statistics)

    Parameters
    ----------
    epsilon : float, default=1e-4
        Initial pseudo-count to avoid division-by-zero and stabilize early updates.
        This can be interpreted as a weak prior with:
          - mean = 0
          - var  = 1
        and total "count" = epsilon.
    shape : Tuple[int, ...], default=()
        Shape of a single sample (excluding the batch dimension).
        Examples:
          - scalar statistics: shape=()
          - vector obs:       shape=(obs_dim,)
          - image obs:        shape=(C,H,W)

    Attributes
    ----------
    mean : np.ndarray
        Running mean, dtype float64, shape=`shape`.
    var : np.ndarray
        Running variance (population), dtype float64, shape=`shape`.
    count : float
        Effective sample count.

    Notes
    -----
    - Internally uses float64 for stability (especially important when counts grow).
    - Variance is a *population* variance consistent with the merge rule. If you
      need an unbiased sample variance (ddof=1), compute it separately.
    - Merging rule corresponds to maintaining the second central moment in an
      aggregation-friendly way.

    References
    ----------
    Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979).
    "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
    """

    def __init__(self, *, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        epsilon = float(epsilon)
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got: {epsilon}")

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of a single sample (excluding batch dimension).

        Returns
        -------
        shape : Tuple[int, ...]
            Current configured sample shape.
        """
        return tuple(self.mean.shape)

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics from raw samples.

        Parameters
        ----------
        x : np.ndarray
            Samples with shape:
              - (B, *shape)  (batch update)
              - (*shape,)    (single sample, treated as B=1)

        Raises
        ------
        ValueError
            If `x` is not compatible with the configured `shape`.

        Notes
        -----
        This method computes batch moments (mean/var/count) and delegates to
        `update_from_moments(...)` for the numerically stable merge.
        """
        x = np.asarray(x, dtype=np.float64)

        # Allow single-sample update: (*shape,) -> (1, *shape)
        if x.shape == self.mean.shape:
            x = x[None, ...]

        if x.ndim != self.mean.ndim + 1 or x.shape[1:] != self.mean.shape:
            raise ValueError(
                "Invalid shape for update(). "
                f"Expected (B, {self.mean.shape}) or ({self.mean.shape},), got: {x.shape}"
            )

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)  # population variance (ddof=0)
        batch_count = int(x.shape[0])

        self.update_from_moments(batch_mean=batch_mean, batch_var=batch_var, batch_count=batch_count)

    def update_from_moments(
        self,
        *,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """
        Merge running statistics with externally computed batch moments.

        This is useful in vectorized / multi-process collection:
        each worker computes local moments and the learner merges them.

        Parameters
        ----------
        batch_mean : np.ndarray
            Batch mean, shape = `shape`.
        batch_var : np.ndarray
            Batch variance (population, ddof=0), shape = `shape`.
        batch_count : int
            Number of samples in the batch, B (> 0).

        Raises
        ------
        ValueError
            If shapes mismatch or `batch_count` is non-positive.

        Notes
        -----
        Let current stats be (mean_a, var_a, count_a) and batch stats be
        (mean_b, var_b, count_b). The merged stats are computed via:

            delta = mean_b - mean_a
            tot   = count_a + count_b
            mean  = mean_a + delta * count_b / tot

        and the merged second moment:

            M2 = var_a*count_a + var_b*count_b + delta^2 * count_a*count_b/tot
            var = M2 / tot

        This is stable for large counts and supports associative merging.
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
        Compute standard deviation from the running variance.

        Parameters
        ----------
        eps : float, default=1e-8
            Small constant added inside sqrt to avoid sqrt(0) and improve numerical
            stability for near-zero variance components.

        Returns
        -------
        std : np.ndarray
            Standard deviation, shape=`shape`, dtype float64.
        """
        return np.sqrt(self.var + float(eps))

    def normalize(
        self,
        x: np.ndarray,
        *,
        clip: Optional[float] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize input using current running mean and standard deviation.

        Parameters
        ----------
        x : np.ndarray
            Input array. Any shape broadcastable with `shape`.
            Common forms:
              - (*shape,)          (single sample)
              - (B, *shape)        (batch)
              - (..., *shape)      (arbitrary leading dims)
        clip : Optional[float], default=None
            If provided and > 0, clip normalized output to [-clip, +clip].
        eps : float, default=1e-8
            Numerical stability constant used in standard deviation.

        Returns
        -------
        y : np.ndarray
            Normalized array, dtype float64.

        Notes
        -----
        - Returns float64 to be consistent with internal statistics and avoid
          precision loss in long-running training. Cast to float32 at the call
          site if required (e.g., before feeding to a network).
        - Normalization uses broadcasting against `self.mean` / `self.std()`.
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
        Return a serializable snapshot of internal statistics.

        Returns
        -------
        state : Dict[str, Any]
            Dictionary containing:
            - "mean": np.ndarray, shape=`shape`, dtype float64
            - "var" : np.ndarray, shape=`shape`, dtype float64
            - "count": float

        Notes
        -----
        - Arrays are copied to avoid external mutation of internal state.
        - This is intentionally compatible with typical PyTorch-style checkpointing.
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
            Dictionary produced by `state_dict()`.

        Raises
        ------
        KeyError
            If required keys are missing from `state`.
        ValueError
            If shapes mismatch or count is non-positive.

        Notes
        -----
        - This method is strict about shape compatibility to avoid silently
          applying wrong statistics to different observation shapes.
        """
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])

        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError(
                "State shape mismatch. "
                f"Expected {self.mean.shape}, got mean {mean.shape}, var {var.shape}."
            )
        if count <= 0.0:
            raise ValueError(f"Invalid count in state_dict: {count}")

        self.mean = mean
        self.var = var
        self.count = count
