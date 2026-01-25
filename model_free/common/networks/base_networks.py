from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.network_utils import validate_hidden_sizes, make_weights_init, ensure_batch


# =============================================================================
# MLPFeaturesExtractor
# =============================================================================
class MLPFeaturesExtractor(nn.Module):
    """
    MLP feature extractor (shared trunk).

    Parameters
    ----------
    input_dim : int
        Input dimensionality (observation/state dimension).
    hidden_sizes : list[int]
        Hidden layer sizes. Output feature dim is `hidden_sizes[-1]`.
    activation_fn : type[nn.Module], optional
        Activation module class inserted after each Linear, by default nn.ReLU.

    Returns
    -------
    features : torch.Tensor
        Feature tensor of shape (B, hidden_sizes[-1]).

    Notes
    -----
    - This module is deliberately minimal and typically used as a shared encoder
      for actor/critic heads.
    - Each hidden layer is Linear + activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one layer size.")

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_sizes:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(activation_fn())
            prev = h

        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_sizes[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, input_dim).

        Returns
        -------
        features : torch.Tensor
            Latent features, shape (B, out_dim).
        """
        return self.net(x)


# =============================================================================
# Base Policy
# =============================================================================
class BasePolicyNetwork(nn.Module, ABC):
    """
    Base policy network for RL policies (shared trunk + init + act contract).

    Contract
    --------
    - Subclasses implement `act(obs, ...) -> (action, info)` for rollout-time use.
    - Shared trunk MLP is provided via `self.trunk`.
    - Initialization helpers are provided; heads should be initialized after creation.

    Notes
    -----
    - Deterministic / Gaussian / Categorical policies differ materially,
      so this base does not enforce a single distribution interface.
    - IMPORTANT: trunk is created here, but heads are created in subclasses.
      Therefore we keep an init function and provide `_init_module(...)` to
      initialize newly created heads without renaming your public API.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        hs = validate_hidden_sizes(hidden_sizes)

        self.obs_dim = int(obs_dim)
        self.hidden_sizes = list(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(self.obs_dim, self.hidden_sizes, self.activation_fn)
        self.trunk_dim = int(self.trunk.out_dim)

        # keep init function for later head init
        self._init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)

        # init trunk now
        self.trunk.apply(self._init_fn)

    def _init_module(self, module: nn.Module) -> None:
        """Initialize (sub)module weights (Linear only) using the stored init_fn."""
        module.apply(self._init_fn)

    @abstractmethod
    @th.no_grad()
    def act(self, obs: th.Tensor, *args: Any, **kwargs: Any) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """Rollout/eval-time action selection."""
        raise NotImplementedError

    def _ensure_batch(self, x: Any) -> th.Tensor:
        device = next(self.parameters()).device
        return ensure_batch(x, device=device)


# =============================================================================
# Continuous Gaussian Policy (PPO/SAC-style)
# =============================================================================
class BaseContinuousStochasticPolicy(BasePolicyNetwork, ABC):
    """
    Base class for continuous stochastic policies that expose `get_dist(obs)`.

    Provides:
    - mean head (mu)
    - log_std parameterization ("param" or "layer")

    Notes
    -----
    Subclasses implement `get_dist(obs)` and typically return one of:
    - DiagGaussianDistribution
    - SquashedDiagGaussianDistribution
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
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
        self._init_module(self.mu)

        self.log_std_mode = str(log_std_mode).lower().strip()
        if self.log_std_mode == "param":
            self.log_std = nn.Parameter(th.ones(self.action_dim) * float(log_std_init))
        elif self.log_std_mode == "layer":
            self.log_std = nn.Linear(self.trunk_dim, self.action_dim)
            self._init_module(self.log_std)
        else:
            raise ValueError(f"Unknown log_std_mode: {log_std_mode!r}")

    def _dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute distribution parameters.

        Parameters
        ----------
        obs : torch.Tensor
            Observations, shape (B, obs_dim) or (obs_dim,).

        Returns
        -------
        mean : torch.Tensor
            Mean, shape (B, action_dim).
        log_std : torch.Tensor
            Log std, shape (B, action_dim).
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)
        mean = self.mu(feat)
        if self.log_std_mode == "param":
            log_std = self.log_std.expand_as(mean)  # type: ignore[union-attr]
        else:
            log_std = self.log_std(feat)  # type: ignore[operator]
        return mean, log_std

    @abstractmethod
    def get_dist(self, obs: th.Tensor):
        raise NotImplementedError


# =============================================================================
# Discrete Categorical Policy
# =============================================================================
class BaseDiscreteStochasticPolicy(BasePolicyNetwork, ABC):
    """
    Base class for discrete stochastic policies that expose `get_dist(obs)`.

    Provides:
    - logits head
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.n_actions = int(n_actions)

        super().__init__(
            obs_dim=int(obs_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.logits = nn.Linear(self.trunk_dim, self.n_actions)
        self._init_module(self.logits)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor
            Observations, shape (B, obs_dim) or (obs_dim,).

        Returns
        -------
        logits : torch.Tensor
            Action logits, shape (B, n_actions).
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)
        return self.logits(feat)

    @abstractmethod
    def get_dist(self, obs: th.Tensor):
        raise NotImplementedError


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)

        self.weight_mu = nn.Parameter(th.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(th.empty(self.out_features, self.in_features))
        self.register_buffer("weight_epsilon", th.empty(self.out_features, self.in_features))

        self.bias_mu = nn.Parameter(th.empty(self.out_features))
        self.bias_sigma = nn.Parameter(th.empty(self.out_features))
        self.register_buffer("bias_epsilon", th.empty(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        with th.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.fill_(self.std_init / math.sqrt(self.in_features))

            self.bias_mu.uniform_(-mu_range, mu_range)
            self.bias_sigma.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: th.device, dtype: th.dtype) -> th.Tensor:
        x = th.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Resample factorized noise for weights and biases.

        IMPORTANT
        ---------
        This function mutates epsilon buffers in-place. That is fine as long as
        forward() does not keep references to the same storage in the autograd graph.
        forward() below uses cloned epsilons to avoid version-counter errors.
        """
        device = self.weight_epsilon.device
        dtype = self.weight_epsilon.dtype

        eps_in = self._scale_noise(self.in_features, device=device, dtype=dtype)
        eps_out = self._scale_noise(self.out_features, device=device, dtype=dtype)

        with th.no_grad():
            self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))  # (out,in)
            self.bias_epsilon.copy_(eps_out)                                       # (out,)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Notes
        -----
        We clone epsilon buffers so that later reset_noise() (in-place) does not
        invalidate autograd graphs that captured previous epsilons.
        """
        # clone() is critical: detach() alone shares storage and can still trigger
        # "modified by an inplace operation" version-counter errors.
        w_eps = self.weight_epsilon.detach().clone()
        b_eps = self.bias_epsilon.detach().clone()

        w = self.weight_mu + self.weight_sigma * w_eps
        b = self.bias_mu + self.bias_sigma * b_eps
        return F.linear(x, w, b)


class NoisyMLPFeaturesExtractor(nn.Module):
    """
    Noisy MLP feature extractor.

    A trunk MLP that uses:
      - deterministic first layer (nn.Linear)
      - NoisyLinear for subsequent layers

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    hidden_sizes : tuple[int, ...], optional
        Hidden sizes, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    init_type : str, optional
        Initializer name for deterministic layers, by default "orthogonal".
    gain : float, optional
        Init gain, by default 1.0.
    bias : float, optional
        Bias init constant, by default 0.0.
    noisy_std_init : float, optional
        Initial sigma for NoisyLinear, by default 0.5.

    Notes
    -----
    Call `reset_noise()` to resample all NoisyLinear noise buffers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.5,
    ) -> None:
        super().__init__()

        hs = validate_hidden_sizes(hidden_sizes)
        self.activation = activation_fn()

        self.input_layer = nn.Linear(int(input_dim), int(hs[0]))
        self.hidden_layers = nn.ModuleList(
            [NoisyLinear(int(hs[i]), int(hs[i + 1]), std_init=noisy_std_init) for i in range(len(hs) - 1)]
        )

        init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.input_layer.apply(init_fn)

        self.out_dim = int(hs[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, input_dim).

        Returns
        -------
        feat : torch.Tensor
            Features, shape (B, out_dim).
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        for layer in self.hidden_layers:
            layer.reset_noise()


# =============================================================================
# BaseValueNetwork (DQN 계열에서 쓰는 베이스)
# =============================================================================
class BaseValueNetwork(nn.Module, ABC):
    """
    Base for value/Q networks (single trunk).

    Provides
    --------
    - trunk construction
    - optional init application (Linear only)
    - batch-shape helper

    Notes
    -----
    - In many patterns, `super().__init__()` is called first while the head modules have not been created yet.
    As a result, subclasses often need to apply initialization once more after constructing the heads.
    - To support this, we provide `apply_init=True/False`.
    * For backward compatibility, the default is `True`.
    * However, for networks that create heads later (e.g., `QNetwork`), it is often cleaner to set
        `apply_init=False` and then call `self.apply(init_fn)` after the heads are created.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        apply_init: bool = True,
    ) -> None:
        super().__init__()
        hs = validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(self.state_dim, list(self.hidden_sizes), self.activation_fn)
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))

        self._init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if bool(apply_init):
            self.apply(self._init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        device = next(self.parameters()).device
        return ensure_batch(x, device=device)


# =============================================================================
# Critic base (SAC/TD3 계열에서 쓰는 베이스)
# =============================================================================
class BaseCriticNet(nn.Module, ABC):
    """
    Shared base for critic/value networks.

    Provides
    --------
    - batch helper
    - init finalization: call `_finalize_init()` AFTER creating heads
    """

    def __init__(self, *, init_type: str, gain: float, bias: float) -> None:
        super().__init__()
        self._init_type = str(init_type)
        self._gain = float(gain)
        self._bias = float(bias)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        device = next(self.parameters()).device
        return ensure_batch(x, device=device)

    def _finalize_init(self) -> None:
        """Apply init to all modules (trunk + heads). Call once after building heads."""
        init_fn = make_weights_init(init_type=self._init_type, gain=self._gain, bias=self._bias)
        self.apply(init_fn)


class BaseStateCritic(BaseCriticNet):
    """
    Base for networks that take only state: f(s).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__(init_type=init_type, gain=gain, bias=bias)
        hs = validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(self.state_dim, list(self.hidden_sizes), activation_fn)
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))


class BaseStateActionCritic(BaseCriticNet):
    """
    Base for networks that take (state, action): f(s,a).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__(init_type=init_type, gain=gain, bias=bias)
        hs = validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.input_dim = self.state_dim + self.action_dim

        self.trunk = MLPFeaturesExtractor(self.input_dim, list(self.hidden_sizes), activation_fn)
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))