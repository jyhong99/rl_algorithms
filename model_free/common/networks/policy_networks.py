from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch as th
import torch.nn as nn

from .distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from .base_networks import (
    BaseContinuousStochasticPolicy,
    BaseDiscreteStochasticPolicy,
    BasePolicyNetwork,
)


# =============================================================================
# Deterministic continuous policy (DDPG/TD3-style)
# =============================================================================

class DeterministicPolicyNetwork(BasePolicyNetwork):
    """
    Deterministic actor network for continuous actions (DDPG/TD3-style).

    This actor produces actions by:
        1) encoding observations with a shared trunk MLP
        2) mapping features to an unconstrained action `u`
        3) applying tanh squashing to obtain `a` in (-1, 1)
        4) optionally applying an affine mapping to match environment bounds

    Specifically:
        u = mu(trunk(obs))                    (B, A)
        a = tanh(u)                           (B, A) in (-1, 1)
        a_env = action_bias + action_scale*a  (B, A) in [low, high] if bounds provided

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : list[int]
        Hidden sizes for the trunk MLP.
    activation_fn : type[nn.Module], optional
        Activation class used in the trunk (default: ``nn.ReLU``).
    action_low : np.ndarray, optional
        Per-dimension lower bounds from environment, shape (A,).
        If None, output remains in (-1, 1) with identity scaling.
    action_high : np.ndarray, optional
        Per-dimension upper bounds from environment, shape (A,).
        If None, output remains in (-1, 1) with identity scaling.
    init_type : str, optional
        Weight initialization scheme name (default: ``"orthogonal"``).
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    action_dim : int
        Action dimension.
    mu : nn.Linear
        Linear head mapping trunk features to pre-tanh action `u`.
    action_scale : torch.Tensor
        Registered buffer. If bounds exist:
            (high - low) / 2
        else:
            ones(A)
    action_bias : torch.Tensor
        Registered buffer. If bounds exist:
            (high + low) / 2
        else:
            zeros(A)
    _has_bounds : bool
        Whether bounds were provided.

    Notes
    -----
    - This is a deterministic policy, so the training algorithm typically
      adds exploration noise externally (TD3/DDPG). Here, `act()` optionally
      adds Gaussian exploration noise in *action space* for convenience.
    - If bounds are provided, action clipping uses [low, high] computed from
      (action_bias ± action_scale) to avoid storing raw numpy arrays.
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
        self._init_module(self.mu)  # keep init consistent with trunk

        self._setup_action_scaling(action_low=action_low, action_high=action_high)

    def _setup_action_scaling(
        self,
        *,
        action_low: Optional[np.ndarray],
        action_high: Optional[np.ndarray],
    ) -> None:
        """
        Configure affine mapping from (-1, 1) to [low, high] (per dimension).

        Parameters
        ----------
        action_low : np.ndarray or None
            Lower bounds, shape (A,). If None, use identity scaling.
        action_high : np.ndarray or None
            Upper bounds, shape (A,). If None, use identity scaling.

        Notes
        -----
        If either bound is missing, this policy behaves as an unbounded actor
        *after tanh*, i.e. outputs are in (-1, 1) with:
            scale = 1, bias = 0
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
                    f"action_low/high must have shape ({self.action_dim},), "
                    f"got {tuple(low.shape)} and {tuple(high.shape)}"
                )

            scale = (high - low) / 2.0
            bias = (high + low) / 2.0
            self._has_bounds = True

        # Buffers are moved with .to(device) and saved in state_dict.
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Compute deterministic scaled action.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (obs_dim,) or (B, obs_dim).

        Returns
        -------
        torch.Tensor
            Action tensor of shape (B, action_dim).
            If bounds provided: in [low, high]
            else: in (-1, 1)
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)

        u = self.mu(feat)
        a = th.tanh(u)  # (-1, 1)

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
        Rollout-time action selection (optionally with exploration noise).

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (obs_dim,) or (B, obs_dim).
        deterministic : bool, optional
            If False and `noise_std > 0`, add Gaussian exploration noise
            in action space (default: True).
        noise_std : float, optional
            Standard deviation of Gaussian exploration noise (default: 0.0).
        clip : bool, optional
            If True and bounds exist, clamp actions to [low, high]
            (default: True).

        Returns
        -------
        action : torch.Tensor
            Action tensor of shape (B, action_dim).
        info : dict[str, torch.Tensor]
            Dictionary containing:
            - "noise": the noise tensor added to the action (zeros if none)

        Notes
        -----
        - Noise is added *after* scaling, i.e. in environment action space.
        - If bounds exist and `clip=True`, action is clamped after adding noise.
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

    This policy builds a distribution from:
        mean, log_std = _dist_params(obs)
    and returns either:
    - DiagGaussianDistribution (unsquashed; PPO/A2C style)
    - SquashedDiagGaussianDistribution (tanh-squashed; SAC style)

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : list[int]
        Trunk hidden sizes.
    activation_fn : type[nn.Module], optional
        Activation class used in trunk (default: ``nn.ReLU``).
    squash : bool, optional
        If True, use tanh-squashed Gaussian distribution (default: False).
    log_std_mode : {"param", "layer"}, optional
        Log-std parameterization mode (default: "param").
    log_std_init : float, optional
        Initial log-std value (default: -0.5).
    init_type : str, optional
        Weight initialization scheme name (default: "orthogonal").
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    squash : bool
        Whether the distribution is squashed by tanh.

    Notes
    -----
    - If `squash=True`, stochastic sampling uses `rsample()` with `pre_tanh`
      to enable pathwise gradients and improve numerical stability (SAC).
    - Deterministic action is `mode()`:
        * unsquashed: mean
        * squashed  : tanh(mean)
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
        Build an action distribution for the given observations.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (obs_dim,) or (B, obs_dim).

        Returns
        -------
        DiagGaussianDistribution or SquashedDiagGaussianDistribution
            Distribution instance configured for the batch.
        """
        mean, log_std = self._dist_params(obs)
        if self.squash:
            return SquashedDiagGaussianDistribution(mean, log_std)
        return DiagGaussianDistribution(mean, log_std)

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
            Observation tensor of shape (obs_dim,) or (B, obs_dim).
        deterministic : bool, optional
            If True, use `mode()`; otherwise sample (default: False).
        return_logp : bool, optional
            If True, compute log π(a|s) and include it in info (default: True).

        Returns
        -------
        action : torch.Tensor
            Action tensor of shape (B, action_dim).
        info : dict[str, torch.Tensor]
            If `return_logp=True`, contains:
            - "logp": log π(a|s), shape (B, 1)

        Notes
        -----
        Squashed case (SAC-style):
        - For stochastic actions, use `rsample(return_pre_tanh=True)` and pass
          `pre_tanh` into `log_prob` to avoid inverse tanh.
        - For deterministic action, the most stable pre_tanh corresponding to
          `mode()` is the mean itself (z = mean).
        """
        obs = self._ensure_batch(obs)
        dist = self.get_dist(obs)

        info: Dict[str, th.Tensor] = {}

        if deterministic:
            action = dist.mode()
            if return_logp:
                if self.squash:
                    # mode() = tanh(mean); the associated pre_tanh is mean.
                    pre_tanh = dist.mean  # type: ignore[attr-defined]
                    info["logp"] = dist.log_prob(action, pre_tanh=pre_tanh)
                else:
                    info["logp"] = dist.log_prob(action)
            return action, info

        # Stochastic sampling
        if self.squash:
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

    This policy uses the base trunk + logits head from `BaseDiscreteStochasticPolicy`,
    and wraps logits into `CategoricalDistribution`.

    Notes
    -----
    - `forward(obs)` returns logits of shape (B, n_actions).
    - `CategoricalDistribution` handles:
        sampling / log_prob / entropy / mode.
    """

    def get_dist(self, obs: th.Tensor) -> CategoricalDistribution:
        """
        Build a categorical distribution for the given observations.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (obs_dim,) or (B, obs_dim).

        Returns
        -------
        CategoricalDistribution
            Categorical distribution wrapper.
        """
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
        """
        Rollout-time action selection.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (obs_dim,) or (B, obs_dim).
        deterministic : bool, optional
            If True, use `mode()`; otherwise sample (default: False).
        return_logp : bool, optional
            If True, compute log π(a|s) (default: True).

        Returns
        -------
        action : torch.Tensor
            Discrete action indices, shape (B, 1).
        info : dict[str, torch.Tensor]
            If `return_logp=True`, contains:
            - "logp": log π(a|s), shape (B, 1)
        """
        obs = self._ensure_batch(obs)
        dist = self.get_dist(obs)

        action = dist.mode() if deterministic else dist.sample()

        info: Dict[str, th.Tensor] = {}
        if return_logp:
            info["logp"] = dist.log_prob(action)

        return action, info
