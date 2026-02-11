from __future__ import annotations

from abc import ABC, abstractmethod

import torch as th


class NoiseProcess(ABC):
    """
    Base interface for exploration-noise processes.

    A noise process provides stochastic perturbations used for exploration
    in reinforcement learning. This base class defines a common lifecycle hook
    (`reset`) that can be used at episode boundaries.

    The design separates *stateless* noise (i.i.d. sampling) from *stateful*
    noise (temporal correlation, internal state), while keeping a shared reset API.

    Methods
    -------
    reset() -> None
        Reset internal state of the noise process (no-op by default).

    Notes
    -----
    - Stateless noise (e.g., i.i.d. Gaussian) may keep `reset()` as a no-op.
    - Stateful noise (e.g., Ornstein-Uhlenbeck) should override `reset()` to
      reinitialize its internal state, typically when an environment episode ends.
    - This interface does not prescribe shape/device/dtype; those are specified
      by subclasses such as :class:`BaseNoise` and :class:`BaseActionNoise`.
    """

    def reset(self) -> None:
        """
        Reset internal state of the noise process.

        Notes
        -----
        Default implementation is a no-op, appropriate for stateless noise.
        Stateful noise implementations should override this method.
        """
        return None


class BaseNoise(NoiseProcess):
    """
    Action-independent noise process.

    This interface is used when a consumer (e.g., an agent) requests a noise
    sample without conditioning on the current action. Typical usage includes
    parameter noise or exploration noise applied in state space rather than
    action space.

    Methods
    -------
    sample() -> torch.Tensor
        Draw one noise sample.

    Returns
    -------
    torch.Tensor
        A noise tensor. Shape/device/dtype are implementation-defined.

    Notes
    -----
    Implementations should document:
    - Output shape (e.g., (act_dim,), (B, act_dim), or scalar).
    - Device/dtype semantics (e.g., always CPU float32, or inferred from module
      parameters).
    """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Draw one noise sample.

        Returns
        -------
        noise : torch.Tensor
            Noise sample tensor.

        Notes
        -----
        - Implementations should avoid surprising device/dtype behavior.
          If the consumer expects to add noise to model outputs, returning a tensor
          on the correct device (and usually matching dtype) is strongly preferred.
        """
        raise NotImplementedError


class BaseActionNoise(NoiseProcess):
    """
    Action-dependent noise process.

    This interface is used when noise depends on the current deterministic action:

        a_noisy = action + noise(action)

    Common examples:
    - multiplicative noise: sigma * action * N(0, 1)
    - scale-aware noise: sigma * max(|action|, eps) * N(0, 1)
    - clipped additive noise: clamp(action + sigma * N(0, 1), low, high) - action

    Methods
    -------
    sample(action: torch.Tensor) -> torch.Tensor
        Draw noise conditioned on the provided action.

    Notes
    -----
    Output contract (recommended):
    - `noise.shape == action.shape`
    - `noise.device == action.device`
    - `noise.dtype == action.dtype` (or safely castable to it)

    If your implementation intentionally deviates (e.g., always float32),
    document it explicitly.
    """

    @abstractmethod
    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw one noise sample conditioned on the given action.

        Parameters
        ----------
        action : torch.Tensor
            Deterministic action tensor. Typical shapes:
            - (act_dim,)
            - (B, act_dim)

        Returns
        -------
        noise : torch.Tensor
            Noise tensor with the same shape as `action`.

        Notes
        -----
        - Implementations should preserve `action.shape`.
        - Implementations should return noise on `action.device`.
        - Implementations should typically match `action.dtype` to avoid implicit
          casts when computing `action + noise`.
        """
        raise NotImplementedError
