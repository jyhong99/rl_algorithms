from __future__ import annotations

from typing import Any, Tuple

import torch as th
import torch.nn as nn

from .base_networks import NoisyMLPFeaturesExtractor, NoisyLinear, BaseValueNetwork
from ..utils.network_utils import make_weights_init, DuelingMixin, ensure_batch


# =============================================================================
# Standard DQN Q-network
# =============================================================================
class QNetwork(BaseValueNetwork, DuelingMixin):
    """
    Discrete-action Q-network (DQN-style), optionally with dueling heads.

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions.
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    dueling : bool, optional
        If True, use dueling decomposition Q = V + (A - mean(A)), by default False.
    init_type : str, optional
        Weight initializer name, by default "orthogonal".
    gain : float, optional
        Init gain, by default 1.0.
    bias : float, optional
        Bias init constant, by default 0.0.

    Returns
    -------
    q : torch.Tensor
        Q-values, shape (B, action_dim).

    Notes
    -----
    Initialization is applied AFTER creating heads to ensure heads are initialized.
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
            self.value_head = nn.Linear(self.trunk_dim, 1)                 # (B,1)
            self.adv_head = nn.Linear(self.trunk_dim, self.action_dim)     # (B,A)
        else:
            self.q_head = nn.Linear(self.trunk_dim, self.action_dim)       # (B,A)

        # IMPORTANT: apply init after creating heads (avoid missing head init)
        init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).

        Returns
        -------
        q : torch.Tensor
            Q-values, shape (B, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat)    # (B,1)
            a = self.adv_head(feat)      # (B,A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat)


class DoubleQNetwork(nn.Module):
    """
    Twin Q-networks wrapper for discrete actions (independent networks).

    Returns
    -------
    q1, q2 : torch.Tensor
        Each is shape (B, action_dim).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.q1 = QNetwork(*args, **kwargs)
        self.q2 = QNetwork(*args, **kwargs)

    def forward(self, state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1(state), self.q2(state)


# =============================================================================
# Quantile Regression DQN (QR-DQN-style)
# =============================================================================
class QuantileQNetwork(BaseValueNetwork, DuelingMixin):
    """
    Quantile Q-network (QR-DQN style), optionally with dueling heads.

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions.
    n_quantiles : int, optional
        Number of quantiles, by default 200.
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    dueling : bool, optional
        If True, use dueling decomposition on quantile outputs, by default False.

    Returns
    -------
    quantiles : torch.Tensor
        Quantile values, shape (B, n_quantiles, action_dim).

    Notes
    -----
    Dueling shapes:
      - V(s): (B, n_quantiles, 1)
      - A(s,a): (B, n_quantiles, action_dim)
      - Combine over action dimension (mean_dim=-1)
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
            self.value_head = nn.Linear(self.trunk_dim, self.n_quantiles)  # (B,N)
            self.adv_head = nn.Linear(self.trunk_dim, out_dim)             # (B,A*N)
        else:
            self.q_head = nn.Linear(self.trunk_dim, out_dim)               # (B,A*N)

        init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).

        Returns
        -------
        quantiles : torch.Tensor
            Quantile values, shape (B, n_quantiles, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat).view(-1, self.n_quantiles, 1)                    # (B,N,1)
            a = self.adv_head(feat).view(-1, self.n_quantiles, self.action_dim)        # (B,N,A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat).view(-1, self.n_quantiles, self.action_dim)


# =============================================================================
# Rainbow / C51 with Noisy + Dueling
# =============================================================================
class RainbowQNetwork(nn.Module, DuelingMixin):
    """
    Rainbow-style C51 Q-network (Noisy + Dueling + Distributional).

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of actions.
    atom_size : int
        Number of support atoms (K).
    support : torch.Tensor
        Support values, shape (K,) (will be registered as buffer).
    hidden_sizes : tuple[int, ...], optional
        Trunk hidden sizes, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    noisy_std_init : float, optional
        Initial sigma for NoisyLinear, by default 0.5.

    Methods
    -------
    dist(state) -> torch.Tensor
        Returns categorical distribution over atoms:
        shape (B, action_dim, atom_size), sums to 1 over last dim.
    forward(state) -> torch.Tensor
        Returns expected Q-values:
        shape (B, action_dim).
    reset_noise()
        Resample all NoisyLinear noise buffers (call per step for exploration).

    Notes
    -----
    - This is C51-style distributional head. The logits are dueling-combined as:
        logits = V + (A - mean_a A)
      where mean is taken over action dimension (dim=1) for logits shaped (B, A, K).
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

        self.register_buffer("support", support.view(-1))

        feat_dim = int(self.trunk.out_dim)  # prefer trunk-provided out_dim
        self.value_layer = NoisyLinear(feat_dim, self.atom_size, std_init=noisy_std_init)                 # (B,K)
        self.adv_layer = NoisyLinear(feat_dim, self.action_dim * self.atom_size, std_init=noisy_std_init) # (B,A*K)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        device = next(self.parameters()).device
        return ensure_batch(x, device=device)
    
    def dist(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).

        Returns
        -------
        dist : torch.Tensor
            Probability distribution over atoms, shape (B, action_dim, atom_size).
            Sums to 1 over last dimension.
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        adv = self.adv_layer(feat).view(-1, self.action_dim, self.atom_size)  # (B,A,K)
        val = self.value_layer(feat).view(-1, 1, self.atom_size)              # (B,1,K)

        logits = self.combine_dueling(val, adv, mean_dim=1)  # mean over action dim=1

        dist = th.softmax(logits, dim=-1)
        dist = dist.clamp(min=1e-6)
        dist = dist / dist.sum(dim=-1, keepdim=True)
        return dist

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).

        Returns
        -------
        q : torch.Tensor
            Expected Q-values, shape (B, action_dim).
        """
        dist = self.dist(state)  # (B,A,K)
        return th.sum(dist * self.support.view(1, 1, -1), dim=-1)

    def reset_noise(self) -> None:
        """Resample noise in trunk and heads (call per step for NoisyNet exploration)."""
        self.trunk.reset_noise()
        self.value_layer.reset_noise()
        self.adv_layer.reset_noise()