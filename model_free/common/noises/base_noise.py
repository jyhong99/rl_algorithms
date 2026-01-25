from __future__ import annotations

from abc import ABC, abstractmethod

import torch as th


class NoiseProcess(ABC):
    """
    Base interface for exploration-noise processes.

    This class defines the minimal lifecycle hook (`reset`) shared by both
    stateless and stateful noise processes used in continuous-control RL.

    Notes
    -----
    - Stateless noise (e.g., i.i.d. Gaussian) can keep `reset()` as a no-op.
    - Stateful noise (e.g., Ornstein-Uhlenbeck) should reset its internal state
      at episode boundaries.
    """

    def reset(self) -> None:
        """
        Reset internal state of the noise process.

        Notes
        -----
        Default implementation is a no-op for stateless noise.
        """
        return None


class BaseNoise(NoiseProcess):
    """
    Action-independent noise process.

    This interface is suitable when the algorithm requests a noise sample
    without conditioning on the current action.

    Methods
    -------
    sample() -> torch.Tensor
        Draw a noise sample.
    """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Draw one noise sample.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor.

        Notes
        -----
        Implementations should ensure the returned tensor's device/dtype are
        consistent with the consuming policy/action tensors when applicable.
        """
        raise NotImplementedError


class BaseActionNoise(NoiseProcess):
    """
    Action-dependent noise process.

    This interface is suitable when the noise is a function of the current action:
        noisy_action = action + noise(action)

    Methods
    -------
    sample(action: torch.Tensor) -> torch.Tensor
        Draw a noise sample conditioned on the given action.
    """

    @abstractmethod
    def sample(self, action: th.Tensor) -> th.Tensor:
        """
        Draw one noise sample conditioned on the given action.

        Parameters
        ----------
        action : torch.Tensor
            Deterministic action tensor.
            Shape: (B, act_dim) or (act_dim,).

        Returns
        -------
        noise : torch.Tensor
            Noise tensor with the same shape as `action`.

        Notes
        -----
        - Implementations should preserve `action.shape`.
        - Implementations should place the output on `action.device` and
          typically match `action.dtype`.
        """
        raise NotImplementedError