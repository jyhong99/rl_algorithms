from __future__ import annotations

from typing import Any, Tuple

import torch as th
import torch.nn as nn

from .base_networks import BaseValueNetwork, NoisyLinear, NoisyMLPFeaturesExtractor
from ..utils.network_utils import DuelingMixin, _ensure_batch, _make_weights_init


# =============================================================================
# Constants
# =============================================================================

PROB_EPS = 1e-6
"""
Small constant used to avoid exact zeros in categorical distributions.

Notes
-----
For distributional RL (C51), probabilities close to 0 can cause numerical issues
(e.g., when taking log or when projecting distributions). Clamping to PROB_EPS
is a common practical stabilization trick.
"""


# =============================================================================
# Standard DQN Q-network
# =============================================================================

class QNetwork(BaseValueNetwork, DuelingMixin):
    """
    Discrete-action Q-network (DQN-style), optionally with dueling decomposition.

    This network maps states to Q-values over discrete actions:
        Q(s) ∈ R^{A}

    If dueling is enabled, Q-values are computed via:
        Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class for trunk (default: ``nn.ReLU``).
    dueling_mode : bool, optional
        If True, use dueling heads (default: False).
    init_type : str, optional
        Weight initializer scheme name passed to `_make_weights_init`
        (default: "orthogonal").
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Returns
    -------
    torch.Tensor
        Q-values of shape (B, action_dim).

    Attributes
    ----------
    action_dim : int
        Number of actions.
    dueling_mode : bool
        Whether dueling decomposition is enabled.
    trunk : MLPFeaturesExtractor
        Shared trunk feature extractor (from BaseValueNetwork).
    trunk_dim : int
        Output dimension of trunk features.
    q_head : nn.Linear
        Q head producing (B, A), present iff dueling_mode=False.
    value_head : nn.Linear
        Value head producing (B, 1), present iff dueling_mode=True.
    adv_head : nn.Linear
        Advantage head producing (B, A), present iff dueling_mode=True.

    Notes
    -----
    Initialization:
    - We intentionally apply initialization AFTER creating heads so that heads
      are included in initialization. This avoids subtle bugs where `super().__init__`
      would initialize only the trunk (or only modules that exist at that time).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        dueling_mode: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.dueling_mode = bool(dueling_mode)

        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        if self.dueling_mode:
            self.value_head = nn.Linear(self.trunk_dim, 1)             # (B, 1)
            self.adv_head = nn.Linear(self.trunk_dim, self.action_dim) # (B, A)
        else:
            self.q_head = nn.Linear(self.trunk_dim, self.action_dim)   # (B, A)

        # Apply init after heads exist (trunk + heads).
        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute Q-values.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Q-values tensor of shape (B, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat)   # (B, 1)
            a = self.adv_head(feat)     # (B, A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat)


class DoubleQNetwork(nn.Module):
    """
    Twin Q-network wrapper for discrete actions (independent networks).

    This module is commonly used for Double DQN-style target computation
    (or for algorithms that maintain two estimators to reduce overestimation).

    Parameters
    ----------
    *args, **kwargs
        Forwarded to `QNetwork`.

    Attributes
    ----------
    q1 : QNetwork
        First Q network.
    q2 : QNetwork
        Second Q network.

    Returns
    -------
    q1, q2 : torch.Tensor
        Each tensor has shape (B, action_dim).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.q1 = QNetwork(*args, **kwargs)
        self.q2 = QNetwork(*args, **kwargs)

    def forward(self, state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        q1 : torch.Tensor
            Q-values from first network, shape (B, action_dim).
        q2 : torch.Tensor
            Q-values from second network, shape (B, action_dim).
        """
        return self.q1(state), self.q2(state)


# =============================================================================
# Quantile Regression DQN (QR-DQN-style)
# =============================================================================

class QuantileQNetwork(BaseValueNetwork, DuelingMixin):
    """
    Quantile Q-network (QR-DQN style), optionally with dueling decomposition.

    This network outputs quantile estimates for each action:
        Z_θ(s, a) ∈ R^{N}
    returned as a tensor of shape:
        (B, N, A)

    where:
    - B: batch size
    - N: number of quantiles
    - A: number of actions

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    n_quantiles : int, optional
        Number of quantiles (N), default: 200.
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    dueling_mode : bool, optional
        If True, apply dueling decomposition at the quantile level (default: False).
    init_type : str, optional
        Weight initializer (default: "orthogonal").
    gain : float, optional
        Init gain (default: 1.0).
    bias : float, optional
        Bias init constant (default: 0.0).

    Returns
    -------
    torch.Tensor
        Quantile tensor of shape (B, n_quantiles, action_dim).

    Notes
    -----
    Dueling shapes:
    - V(s): (B, N, 1)
    - A(s,a): (B, N, A)
    - Combine using mean over action dimension (mean_dim=-1).

    Initialization:
    - Applied after heads are created (same rationale as `QNetwork`).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 200,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        dueling_mode: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.dueling_mode = bool(dueling_mode)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got: {self.n_quantiles}")

        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        out_dim = self.action_dim * self.n_quantiles

        if self.dueling_mode:
            self.value_head = nn.Linear(self.trunk_dim, self.n_quantiles)  # (B, N)
            self.adv_head = nn.Linear(self.trunk_dim, out_dim)             # (B, A*N)
        else:
            self.q_head = nn.Linear(self.trunk_dim, out_dim)               # (B, A*N)

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute quantile values for each action.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Quantile values of shape (B, n_quantiles, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat).view(-1, self.n_quantiles, 1)               # (B, N, 1)
            a = self.adv_head(feat).view(-1, self.n_quantiles, self.action_dim)   # (B, N, A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat).view(-1, self.n_quantiles, self.action_dim)


# =============================================================================
# Rainbow / C51 with Noisy + Dueling (Distributional)
# =============================================================================

class RainbowQNetwork(nn.Module, DuelingMixin):
    """
    Rainbow-style C51 Q-network (NoisyNet + Dueling + Distributional head).

    This network models the return distribution Z(s, a) as a categorical
    distribution over a fixed support (atoms).

    Outputs
    -------
    - dist(state):
        Probability distribution over atoms for each action:
        shape (B, action_dim, atom_size), sums to 1 over last dimension.
    - forward(state):
        Expected Q-values:
        shape (B, action_dim).
    - reset_noise():
        Resample all NoisyLinear noise buffers (typically called every step).

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    atom_size : int
        Number of atoms (K).
    support : torch.Tensor
        Support values of shape (K,). Will be registered as a buffer.
    hidden_sizes : tuple[int, ...], optional
        Noisy trunk hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class for trunk (default: ``nn.ReLU``).
    init_type : str, optional
        Initializer name for deterministic parts (default: "orthogonal").
    gain : float, optional
        Init gain for deterministic parts (default: 1.0).
    bias : float, optional
        Bias init constant for deterministic parts (default: 0.0).
    noisy_std_init : float, optional
        Initial sigma for NoisyLinear layers (default: 0.5).

    Attributes
    ----------
    action_dim : int
        Number of actions.
    atom_size : int
        Number of atoms.
    support : torch.Tensor
        Registered buffer of shape (K,).
    trunk : NoisyMLPFeaturesExtractor
        Noisy feature extractor.
    value_layer : NoisyLinear
        Value head producing logits for atoms, shape (B, K) before reshaping.
    adv_layer : NoisyLinear
        Advantage head producing logits, shape (B, A*K) before reshaping.

    Notes
    -----
    Dueling combination (logits space):
        logits = V + (A - mean_a A)
    where mean is taken over action dimension (dim=1) for logits shaped (B, A, K).

    Probability stabilization:
    - `softmax(logits)` produces probabilities; we clamp to `PROB_EPS` and renormalize
      to avoid exact zeros.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_size: int,
        support: th.Tensor,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.action_dim = int(action_dim)
        self.atom_size = int(atom_size)

        if self.atom_size <= 0:
            raise ValueError(f"atom_size must be > 0, got: {self.atom_size}")
        if support.numel() != self.atom_size:
            raise ValueError(
                f"support must have numel()==atom_size ({self.atom_size}), got: {support.numel()}"
            )

        self.trunk = NoisyMLPFeaturesExtractor(
            input_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
            noisy_std_init=noisy_std_init,
        )

        # Keep support on the same device and in state_dict.
        self.register_buffer("support", support.view(-1))

        feat_dim = int(self.trunk.out_dim)
        self.value_layer = NoisyLinear(feat_dim, self.atom_size, std_init=noisy_std_init)                  # (B, K)
        self.adv_layer = NoisyLinear(feat_dim, self.action_dim * self.atom_size, std_init=noisy_std_init)  # (B, A*K)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has batch dimension and is moved to module device.

        Parameters
        ----------
        x : Any
            Tensor/ndarray/sequence input convertible by `_ensure_batch`.

        Returns
        -------
        torch.Tensor
            Tensor on module device with a batch dimension.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    def dist(self, state: th.Tensor) -> th.Tensor:
        """
        Compute categorical distribution over atoms for each action.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Probability distribution over atoms, shape (B, action_dim, atom_size).
            Each (B, action_dim, :) slice sums to 1 over the last dimension.

        Notes
        -----
        - The dueling combination is performed in logits space.
        - Probabilities are clamped and renormalized for numerical stability.
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        adv = self.adv_layer(feat).view(-1, self.action_dim, self.atom_size)  # (B, A, K)
        val = self.value_layer(feat).view(-1, 1, self.atom_size)              # (B, 1, K)

        # Mean over action dimension (dim=1) for logits in (B, A, K).
        logits = self.combine_dueling(val, adv, mean_dim=1)

        prob = th.softmax(logits, dim=-1)
        prob = prob.clamp(min=PROB_EPS)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        return prob

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute expected Q-values from the categorical distribution.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Expected Q-values, shape (B, action_dim).

        Notes
        -----
        Expectation is computed as:
            Q(s,a) = sum_k p_k(s,a) * support_k
        """
        prob = self.dist(state)  # (B, A, K)
        return th.sum(prob * self.support.view(1, 1, -1), dim=-1)

    def reset_noise(self) -> None:
        """
        Resample noise in trunk and heads (NoisyNet exploration).

        Notes
        -----
        Typical usage is to call this once per environment step so that the
        policy induced by the network changes over time via parameter noise.
        """
        self.trunk.reset_noise()
        self.value_layer.reset_noise()
        self.adv_layer.reset_noise()
