from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch as th
import torch.nn as nn

from .distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    CategoricalDistribution,
)
from .base_networks import (
    BasePolicyNetwork,
    BaseContinuousStochasticPolicy,
    BaseDiscreteStochasticPolicy,
)


# =============================================================================
# Deterministic continuous policy (DDPG/TD3-style)
# =============================================================================
class DeterministicPolicyNetwork(BasePolicyNetwork):
    """
    Deterministic actor network for continuous actions (DDPG/TD3-style).

    Computes:
        u = mu(trunk(obs))
        a = tanh(u)                       in (-1, 1)
        a_env = bias + scale * a          if bounds provided (per-dimension)

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : list[int]
        Trunk MLP hidden sizes.
    activation_fn : type[nn.Module], optional
        Activation class for trunk, by default nn.ReLU.
    action_low : Optional[np.ndarray], optional
        Environment action lower bounds, shape (A,), by default None.
    action_high : Optional[np.ndarray], optional
        Environment action upper bounds, shape (A,), by default None.
    init_type : str, optional
        Weight init type, by default "orthogonal".
    gain : float, optional
        Init gain, by default 1.0.
    bias : float, optional
        Bias init constant, by default 0.0.

    Notes
    -----
    - If bounds are not provided, outputs are in (-1, 1) due to tanh.
    - `act()` optionally adds Gaussian exploration noise in action space.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.action_dim = int(action_dim)

        super().__init__(
            obs_dim=int(obs_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.mu = nn.Linear(self.trunk_dim, self.action_dim)
        self._setup_action_scaling(action_low=action_low, action_high=action_high)

    def _setup_action_scaling(
        self,
        *,
        action_low: Optional[np.ndarray],
        action_high: Optional[np.ndarray],
    ) -> None:
        """
        Setup per-dimension affine mapping from (-1, 1) to [low, high].

        If bounds are missing, use identity scaling.
        """
        if action_low is None or action_high is None:
            scale = th.ones(self.action_dim, dtype=th.float32)
            bias = th.zeros(self.action_dim, dtype=th.float32)
            self._has_bounds = False
        else:
            low = th.as_tensor(action_low, dtype=th.float32).view(-1)
            high = th.as_tensor(action_high, dtype=th.float32).view(-1)
            if low.numel() != self.action_dim or high.numel() != self.action_dim:
                raise ValueError(
                    f"action_low/high must be shape ({self.action_dim},), "
                    f"got {tuple(low.shape)} and {tuple(high.shape)}"
                )
            scale = (high - low) / 2.0
            bias = (high + low) / 2.0
            self._has_bounds = True

        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor
            Observations, shape (B, obs_dim) or (obs_dim,).

        Returns
        -------
        action : torch.Tensor
            Scaled action, shape (B, action_dim).
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)
        u = self.mu(feat)
        a = th.tanh(u)  # (-1,1)
        return self.action_bias + self.action_scale * a

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        *,
        deterministic: bool = True,
        noise_std: float = 0.0,
        clip: bool = True,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Rollout-time action selection.

        Parameters
        ----------
        obs : torch.Tensor
            Observation, shape (B, obs_dim) or (obs_dim,).
        deterministic : bool, optional
            If False and noise_std>0, add exploration noise, by default True.
        noise_std : float, optional
            Gaussian noise std in action space, by default 0.0.
        clip : bool, optional
            If True and bounds exist, clamp to [low, high], by default True.

        Returns
        -------
        action : torch.Tensor
            Action, shape (B, action_dim).
        info : dict[str, torch.Tensor]
            Contains "noise" used for exploration.
        """
        obs = self._ensure_batch(obs)
        action = self.forward(obs)

        noise = th.zeros_like(action)
        if (not deterministic) and float(noise_std) > 0.0:
            noise = float(noise_std) * th.randn_like(action)
            action = action + noise

        if clip and self._has_bounds:
            low = self.action_bias - self.action_scale
            high = self.action_bias + self.action_scale
            action = th.clamp(action, min=low, max=high)

        return action, {"noise": noise}


# =============================================================================
# Stochastic continuous policy (Gaussian; PPO/SAC-style)
# =============================================================================
class ContinuousPolicyNetwork(BaseContinuousStochasticPolicy):
    """
    Gaussian policy network for continuous actions.

    Supports:
    - Unsquashed diagonal Gaussian (PPO/A2C style)
    - Squashed diagonal Gaussian via tanh (SAC style)

    Parameters
    ----------
    squash : bool, optional
        If True, returns a squashed Gaussian distribution, by default False.

    Notes
    -----
    - When squash=True, stochastic actions use rsample() for pathwise gradients
      (useful for SAC). Deterministic action uses tanh(mean).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash: bool = False,
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.squash = bool(squash)

        super().__init__(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            log_std_mode=log_std_mode,
            log_std_init=log_std_init,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

    def get_dist(self, obs: th.Tensor) -> DiagGaussianDistribution | SquashedDiagGaussianDistribution:
        """
        Build a distribution object for given observations.
        """
        mean, log_std = self._dist_params(obs)
        return SquashedDiagGaussianDistribution(mean, log_std) if self.squash else DiagGaussianDistribution(mean, log_std)

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        *,
        deterministic: bool = False,
        return_logp: bool = True,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Rollout-time action selection.

        Parameters
        ----------
        obs : torch.Tensor
            Observations, shape (B, obs_dim) or (obs_dim,).
        deterministic : bool, optional
            If True, take mode action, by default False.
        return_logp : bool, optional
            If True, also compute log π(a|s), by default True.

        Returns
        -------
        action : torch.Tensor
            Action tensor, shape (B, action_dim).
        info : dict[str, torch.Tensor]
            Contains "logp" when requested.
        """
        obs = self._ensure_batch(obs)
        dist = self.get_dist(obs)

        info: Dict[str, th.Tensor] = {}

        if deterministic:
            # For squashed Gaussian, mode() is tanh(mean) already.
            action = dist.mode()

            if return_logp:
                if self.squash:
                    # Most stable: pre_tanh corresponding to mode is mean.
                    pre_tanh = dist.mean  # type: ignore[attr-defined]
                    logp = dist.log_prob(action, pre_tanh=pre_tanh)
                else:
                    logp = dist.log_prob(action)
                info["logp"] = logp
            return action, info

        # Stochastic
        if self.squash:
            # Prefer rsample + pre_tanh for stability (SAC)
            action, pre_tanh = dist.rsample(return_pre_tanh=True)  # type: ignore[assignment]
            if return_logp:
                info["logp"] = dist.log_prob(action, pre_tanh=pre_tanh)
        else:
            action = dist.sample()
            if return_logp:
                info["logp"] = dist.log_prob(action)

        return action, info


# =============================================================================
# Discrete policy (Categorical)
# =============================================================================
class DiscretePolicyNetwork(BaseDiscreteStochasticPolicy):
    """
    Categorical policy network for discrete actions.

    Notes
    -----
    - `forward(obs)` should return logits of shape (B, n_actions).
    - `CategoricalDistribution` wrapper handles sampling/log_prob/entropy.
    """

    def get_dist(self, obs: th.Tensor) -> CategoricalDistribution:
        obs = self._ensure_batch(obs)
        logits = self.forward(obs)
        return CategoricalDistribution(logits)

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        *,
        deterministic: bool = False,
        return_logp: bool = True,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        obs = self._ensure_batch(obs)
        dist = self.get_dist(obs)

        action = dist.mode() if deterministic else dist.sample()

        info: Dict[str, th.Tensor] = {}
        if return_logp:
            info["logp"] = dist.log_prob(action)

        return action, info