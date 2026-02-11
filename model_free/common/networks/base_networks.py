from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.network_utils import _validate_hidden_sizes, _make_weights_init, _ensure_batch


# =============================================================================
# Feature Extractors
# =============================================================================

class MLPFeaturesExtractor(nn.Module):
    """
    Standard MLP feature extractor (shared trunk / encoder).

    This module is intentionally minimal and is meant to be reused as a
    *shared trunk* for multiple heads (actor/critic, value/Q, etc.).

    Parameters
    ----------
    input_dim : int
        Input dimensionality (e.g., observation/state dimension).
    hidden_sizes : list[int]
        Hidden layer widths. The output feature dimension is ``hidden_sizes[-1]``.
        Must contain at least one element.
    activation_fn : type[nn.Module], optional
        Activation module class inserted after each ``nn.Linear`` layer
        (default: ``nn.ReLU``).

    Attributes
    ----------
    net : nn.Sequential
        Sequential stack: (Linear -> Activation) repeated.
    out_dim : int
        Output feature dimensionality (= ``hidden_sizes[-1]``).

    Notes
    -----
    - Architecture: (Linear → Activation) × N
    - No normalization/residual connections are included by design.
    - This is a pure feature extractor; distribution/value heads are built elsewhere.
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
        prev_dim = int(input_dim)

        for h in hidden_sizes:
            h = int(h)
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_sizes[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, out_dim)``.
        """
        return self.net(x)


# =============================================================================
# Policy Bases
# =============================================================================

class BasePolicyNetwork(nn.Module, ABC):
    """
    Base class for policy networks with a shared trunk (encoder) and a rollout API.

    This class defines the *common skeleton* for actor/policy networks:
    - builds a trunk MLP feature extractor
    - stores a weight initialization function
    - provides batch/device normalization helper
    - enforces an `act()` interface for rollout/evaluation usage

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    hidden_sizes : list[int]
        Trunk MLP hidden sizes. Must be non-empty.
    activation_fn : type[nn.Module], optional
        Activation class used inside the trunk (default: ``nn.ReLU``).
    init_type : str, optional
        Initialization scheme name forwarded to `_make_weights_init`
        (default: ``"orthogonal"``).
    gain : float, optional
        Gain argument for the initializer (default: 1.0).
    bias : float, optional
        Bias constant for the initializer (default: 0.0).

    Attributes
    ----------
    obs_dim : int
        Observation dimension.
    hidden_sizes : list[int]
        Validated hidden sizes for trunk.
    activation_fn : type[nn.Module]
        Activation function class for trunk.
    trunk : MLPFeaturesExtractor
        Shared feature extractor.
    trunk_dim : int
        Feature dimension output by `trunk`.
    _init_fn : callable
        Initialization function created by `_make_weights_init`.

    Contract
    --------
    Subclasses MUST implement:
    - `act(obs, *args, **kwargs) -> (action, info)`
      where `action` is a tensor and `info` is a dict of tensors.

    Notes
    -----
    - This base does NOT enforce a specific distribution interface because
      deterministic policies, Gaussian policies, categorical policies, and
      squashed distributions differ materially.
    - Heads are usually created in subclasses; therefore `_init_module()` exists
      to initialize newly created head modules consistently with the trunk.
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
        hs = _validate_hidden_sizes(hidden_sizes)

        self.obs_dim = int(obs_dim)
        self.hidden_sizes = list(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(
            self.obs_dim,
            self.hidden_sizes,
            self.activation_fn,
        )
        self.trunk_dim = int(self.trunk.out_dim)

        # Keep init function for later head initialization.
        self._init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)

        # Initialize trunk immediately.
        self.trunk.apply(self._init_fn)

    def _init_module(self, module: nn.Module) -> None:
        """
        Initialize a (sub)module using the stored initializer.

        Parameters
        ----------
        module : nn.Module
            Module to initialize. Typically a head created after `super().__init__()`.

        Notes
        -----
        - `_make_weights_init` usually targets `nn.Linear` parameters; applying it
          over the full module is convenient and consistent.
        """
        module.apply(self._init_fn)

    @abstractmethod
    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Select action for rollout/evaluation (no grad).

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor. Can be shape ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns
        -------
        action : torch.Tensor
            Action tensor.
        info : dict[str, torch.Tensor]
            Additional tensors (e.g., log_prob, entropy, mean, std, logits).

        Notes
        -----
        - This is the *deployment/rollout-time* API. Training-time methods may
          return richer objects (distributions, etc.) in subclasses.
        """
        raise NotImplementedError

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has a batch dimension and is placed on the module's device.

        Parameters
        ----------
        x : Any
            Input that `_ensure_batch` knows how to convert (tensor/np/sequence).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, dim)`` on the correct device.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)


class BaseContinuousStochasticPolicy(BasePolicyNetwork, ABC):
    """
    Base class for continuous stochastic policies (Gaussian-style).

    This base provides:
    - mean head (`mu`)
    - log standard deviation parameterization:
        * "param": global trainable vector (state-independent)
        * "layer": linear head from features (state-dependent)

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : list[int]
        Trunk hidden sizes.
    activation_fn : type[nn.Module], optional
        Activation class (default: ``nn.ReLU``).
    log_std_mode : {"param", "layer"}, optional
        Log-std parameterization mode (default: "param").
    log_std_init : float, optional
        Initial value for log-std (default: -0.5).
    init_type, gain, bias
        Passed to base initializer.

    Attributes
    ----------
    action_dim : int
        Action dimension.
    mu : nn.Linear
        Mean head mapping trunk features -> action mean.
    log_std_mode : str
        Lowercased mode string.
    log_std : nn.Parameter or nn.Linear
        Either a parameter vector (param mode) or a linear layer (layer mode).

    Notes
    -----
    Subclasses typically implement `get_dist(obs)` returning a distribution object:
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

        # Mean head
        self.mu = nn.Linear(self.trunk_dim, self.action_dim)
        self._init_module(self.mu)

        # Log-std head / parameter
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
        Compute Gaussian distribution parameters (mean and log-std).

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns
        -------
        mean : torch.Tensor
            Mean tensor of shape ``(B, action_dim)``.
        log_std : torch.Tensor
            Log-std tensor of shape ``(B, action_dim)``.

        Notes
        -----
        - In "param" mode, log-std is broadcast to match the mean shape.
        - In "layer" mode, log-std is computed from features (state-dependent).
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
        """
        Build and return a distribution object for the given observations.
        """
        raise NotImplementedError


class BaseDiscreteStochasticPolicy(BasePolicyNetwork, ABC):
    """
    Base class for discrete stochastic policies (categorical-style).

    Provides:
    - logits head mapping trunk features -> action logits

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    n_actions : int
        Number of discrete actions.
    hidden_sizes : list[int]
        Trunk hidden sizes.
    activation_fn : type[nn.Module], optional
        Activation class (default: ``nn.ReLU``).
    init_type, gain, bias
        Passed to base initializer.

    Attributes
    ----------
    n_actions : int
        Number of discrete actions.
    logits : nn.Linear
        Logits head producing shape ``(B, n_actions)``.
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
        Compute action logits.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, n_actions)``.
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)
        return self.logits(feat)

    @abstractmethod
    def get_dist(self, obs: th.Tensor):
        """
        Build and return a categorical distribution object.
        """
        raise NotImplementedError


# =============================================================================
# Noisy Networks (Rainbow / Exploration-friendly)
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al.).

    This layer replaces deterministic Linear weights with:
        W = W_mu + W_sigma ⊙ eps_W
        b = b_mu + b_sigma ⊙ eps_b
    where eps is sampled from a factorized noise distribution.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    std_init : float, optional
        Initial value for sigma parameters (default: 0.5).

    Attributes
    ----------
    weight_mu : nn.Parameter
        Mean of weights, shape ``(out_features, in_features)``.
    weight_sigma : nn.Parameter
        Std (scale) of weights, same shape.
    weight_epsilon : torch.Tensor (buffer)
        Sampled noise for weights, same shape.
    bias_mu : nn.Parameter
        Mean of bias, shape ``(out_features,)``.
    bias_sigma : nn.Parameter
        Std (scale) of bias, shape ``(out_features,)``.
    bias_epsilon : torch.Tensor (buffer)
        Sampled noise for bias, shape ``(out_features,)``.

    Notes
    -----
    - Call `reset_noise()` to resample epsilons.
    - `reset_noise()` mutates epsilon buffers in-place; `forward()` clones epsilons
      to avoid autograd version-counter errors if noise is reset between forwards.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
    ) -> None:
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
        """
        Initialize mu and sigma parameters.

        Notes
        -----
        - mu is uniform in [-1/sqrt(in), 1/sqrt(in)]
        - sigma is constant scaled by std_init
        """
        mu_range = 1.0 / math.sqrt(self.in_features)
        with th.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.fill_(self.std_init / math.sqrt(self.in_features))

            self.bias_mu.uniform_(-mu_range, mu_range)
            self.bias_sigma.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: th.device, dtype: th.dtype) -> th.Tensor:
        """
        Sample noise vector using factorized form: f(x)=sign(x)*sqrt(|x|).

        Returns
        -------
        torch.Tensor
            Noise vector of shape ``(size,)``.
        """
        x = th.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Resample factorized noise for weights and biases.

        Notes
        -----
        - Epsilon buffers are updated in-place.
        - Safe with autograd as long as forward does not keep references to the same
          storage; forward clones epsilons to avoid version-counter issues.
        """
        device = self.weight_epsilon.device
        dtype = self.weight_epsilon.dtype

        eps_in = self._scale_noise(self.in_features, device=device, dtype=dtype)
        eps_out = self._scale_noise(self.out_features, device=device, dtype=dtype)

        with th.no_grad():
            self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))  # (out, in)
            self.bias_epsilon.copy_(eps_out)  # (out,)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass with noisy parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, out_features)``.

        Notes
        -----
        - `clone()` is critical: `detach()` alone shares storage and may still trigger
          "modified by an inplace operation" errors if `reset_noise()` happens later.
        """
        w_eps = self.weight_epsilon.detach().clone()
        b_eps = self.bias_epsilon.detach().clone()

        w = self.weight_mu + self.weight_sigma * w_eps
        b = self.bias_mu + self.bias_sigma * b_eps
        return F.linear(x, w, b)


class NoisyMLPFeaturesExtractor(nn.Module):
    """
    MLP feature extractor with NoisyLinear layers.

    Architecture
    ------------
    - First layer: deterministic `nn.Linear` (often stabilizes early learning)
    - Remaining layers: `NoisyLinear` (exploration via parameter noise)

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    init_type : str, optional
        Initializer name for deterministic layers (default: "orthogonal").
    gain : float, optional
        Init gain (default: 1.0).
    bias : float, optional
        Bias init constant (default: 0.0).
    noisy_std_init : float, optional
        Initial sigma for `NoisyLinear` (default: 0.5).

    Attributes
    ----------
    input_layer : nn.Linear
        Deterministic first layer.
    hidden_layers : nn.ModuleList
        List of `NoisyLinear` layers.
    activation : nn.Module
        Activation instance.
    out_dim : int
        Output feature dimensionality.

    Notes
    -----
    Call `reset_noise()` to resample noise in all NoisyLinear layers.
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

        hs = _validate_hidden_sizes(hidden_sizes)
        self.activation = activation_fn()

        self.input_layer = nn.Linear(int(input_dim), int(hs[0]))
        self.hidden_layers = nn.ModuleList(
            [
                NoisyLinear(int(hs[i]), int(hs[i + 1]), std_init=noisy_std_init)
                for i in range(len(hs) - 1)
            ]
        )

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.input_layer.apply(init_fn)

        self.out_dim = int(hs[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, out_dim)``.
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        """
        Resample noise in all NoisyLinear layers.
        """
        for layer in self.hidden_layers:
            layer.reset_noise()


# =============================================================================
# Value/Q Network Base (DQN-family)
# =============================================================================

class BaseValueNetwork(nn.Module, ABC):
    """
    Base class for value/Q networks with a single trunk.

    This base is designed for DQN-family networks where:
    - you may build trunk first, then build head(s) later
    - initialization timing matters (heads may not exist at `super().__init__()`)

    Parameters
    ----------
    state_dim : int
        State (observation) dimension.
    hidden_sizes : tuple[int, ...]
        Trunk hidden layer sizes.
    activation_fn : type[nn.Module], optional
        Activation class used in trunk (default: ``nn.ReLU``).
    init_type : str, optional
        Initializer scheme name (default: "orthogonal").
    gain : float, optional
        Init gain (default: 1.0).
    bias : float, optional
        Bias init constant (default: 0.0).
    apply_init : bool, optional
        If True, apply initializer immediately to all modules constructed so far
        (default: True). If heads are created later, consider False and apply init
        once after head creation.

    Attributes
    ----------
    trunk : MLPFeaturesExtractor
        Trunk feature extractor.
    trunk_dim : int
        Output dimension of trunk.
    _init_fn : callable
        Stored initializer for reuse.

    Notes
    -----
    Recommended patterns:
    - If your subclass creates ALL layers before calling `super().__init__()`:
        keep `apply_init=True`.
    - If your subclass creates heads AFTER calling `super().__init__()`:
        set `apply_init=False`, then call `self.apply(self._init_fn)` after building heads.
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
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(
            self.state_dim,
            list(self.hidden_sizes),
            self.activation_fn,
        )
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))

        self._init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if bool(apply_init):
            self.apply(self._init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has batch dim and correct device.

        Returns
        -------
        torch.Tensor
            Tensor on module device, shape ``(B, state_dim)``.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)


# =============================================================================
# Critic Bases (SAC/TD3-family)
# =============================================================================

class BaseCriticNet(nn.Module, ABC):
    """
    Shared base class for critic/value networks (SAC/TD3-style).

    This base is designed for the common pattern:
    - create trunk (and sometimes multiple heads) in subclass
    - then call `_finalize_init()` once when everything exists

    Parameters
    ----------
    init_type : str
        Initializer scheme name.
    gain : float
        Init gain.
    bias : float
        Bias init constant.

    Notes
    -----
    `_finalize_init()` applies init to *all* current submodules (trunk + heads).
    Call it exactly once after building all layers in the subclass.
    """

    def __init__(self, *, init_type: str, gain: float, bias: float) -> None:
        super().__init__()
        self._init_type = str(init_type)
        self._gain = float(gain)
        self._bias = float(bias)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has batch dim and correct device.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    def _finalize_init(self) -> None:
        """
        Apply initializer to all modules (trunk + heads).

        Notes
        -----
        Call this AFTER constructing all layers in the subclass.
        """
        init_fn = _make_weights_init(
            init_type=self._init_type,
            gain=self._gain,
            bias=self._bias,
        )
        self.apply(init_fn)


class BaseStateCritic(BaseCriticNet):
    """
    Base class for critics that take only state: f(s).

    Parameters
    ----------
    state_dim : int
        State dimension.
    hidden_sizes : tuple[int, ...]
        Trunk hidden sizes.
    activation_fn : type[nn.Module], optional
        Activation class (default: ``nn.ReLU``).
    init_type, gain, bias
        Initialization settings for `_finalize_init()`.

    Attributes
    ----------
    trunk : MLPFeaturesExtractor
        Feature extractor operating on state only.
    trunk_dim : int
        Output feature dimension from trunk.
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
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.trunk = MLPFeaturesExtractor(
            self.state_dim,
            list(self.hidden_sizes),
            activation_fn,
        )
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))


class BaseStateActionCritic(BaseCriticNet):
    """
    Base class for critics that take (state, action): f(s, a).

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : tuple[int, ...]
        Trunk hidden sizes for the concatenated input [s, a].
    activation_fn : type[nn.Module], optional
        Activation class (default: ``nn.ReLU``).
    init_type, gain, bias
        Initialization settings for `_finalize_init()`.

    Attributes
    ----------
    input_dim : int
        Concatenated input dimension (= state_dim + action_dim).
    trunk : MLPFeaturesExtractor
        Feature extractor operating on concatenated [state, action].
    trunk_dim : int
        Output feature dimension from trunk.
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
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        self.input_dim = self.state_dim + self.action_dim

        self.trunk = MLPFeaturesExtractor(
            self.input_dim,
            list(self.hidden_sizes),
            activation_fn,
        )
        self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))
