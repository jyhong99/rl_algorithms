from __future__ import annotations

from typing import Any, Tuple, Type

import torch as th
import torch.nn as nn

from .base_networks import BaseStateCritic, BaseStateActionCritic, MLPFeaturesExtractor
from ..utils.network_utils import validate_hidden_sizes, make_weights_init, ensure_batch


# =============================================================================
# V(s)
# =============================================================================
class StateValueNetwork(BaseStateCritic):
    """
    State-value function network V(s).

    Parameters
    ----------
    state_dim : int
        State dimension.
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    init_type : str, optional
        Weight initializer name, by default "orthogonal".
    gain : float, optional
        Init gain, by default 1.0.
    bias : float, optional
        Bias init constant, by default 0.0.

    Returns
    -------
    v : torch.Tensor
        Value estimates, shape (B, 1).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.head = nn.Linear(self.trunk_dim, 1)

        # BaseStateCritic uses BaseCriticNet; finalize init after heads exist
        self._finalize_init()

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).

        Returns
        -------
        v : torch.Tensor
            Value estimates, shape (B, 1).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)
        return self.head(feat)


# =============================================================================
# Q(s,a)
# =============================================================================
class StateActionValueNetwork(BaseStateActionCritic):
    """
    State-action value function network Q(s,a).

    Returns
    -------
    q : torch.Tensor
        Q-value estimates, shape (B, 1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.head = nn.Linear(self.trunk_dim, 1)

        # BaseStateActionCritic -> BaseCriticNet pattern: call finalize after heads
        self._finalize_init()

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).
        action : torch.Tensor
            Action tensor, shape (B, action_dim) or (action_dim,).

        Returns
        -------
        q : torch.Tensor
            Q-value estimates, shape (B, 1).
        """
        state = self._ensure_batch(state)
        action = self._ensure_batch(action)

        x = th.cat([state, action], dim=-1)
        feat = self.trunk(x)
        return self.head(feat)


class DoubleStateActionValueNetwork(nn.Module):
    """
    Twin critics Q1(s,a), Q2(s,a).

    Returns
    -------
    q1, q2 : torch.Tensor
        Each is shape (B, 1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.q1 = StateActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )
        self.q2 = StateActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

    def forward(self, state: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1(state, action), self.q2(state, action)


# =============================================================================
# Quantile critic ensemble (TQC/QR variants)
# =============================================================================
class QuantileStateActionValueNetwork(nn.Module):
    """
    Quantile critic ensemble with independent trunks/heads.

    Outputs a stack of quantile values per critic:
        (B, n_nets, n_quantiles)

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    n_quantiles : int, optional
        Number of quantiles per critic, by default 25.
    n_nets : int, optional
        Number of critic networks in the ensemble, by default 2.
    hidden_sizes : tuple[int, ...], optional
        Hidden sizes for each critic trunk, by default (64, 64).
    activation_fn : type[nn.Module], optional
        Activation module class, by default nn.ReLU.
    init_type : str, optional
        Weight initializer name, by default "orthogonal".
    gain : float, optional
        Init gain, by default 1.0.
    bias : float, optional
        Bias init constant, by default 0.0.

    Notes
    -----
    - This class does NOT inherit BaseStateActionCritic because it builds multiple
      independent trunks; inheriting a base that assumes a single trunk complicates
      initialization and creates unused modules.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 25,
        n_nets: int = 2,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()

        hs = validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.state_dim + self.action_dim

        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be positive, got: {self.n_quantiles}")
        if self.n_nets <= 0:
            raise ValueError(f"n_nets must be positive, got: {self.n_nets}")

        self.trunks = nn.ModuleList(
            [MLPFeaturesExtractor(self.input_dim, list(hs), activation_fn) for _ in range(self.n_nets)]
        )
        trunk_dim = int(getattr(self.trunks[0], "out_dim", hs[-1]))

        self.heads = nn.ModuleList([nn.Linear(trunk_dim, self.n_quantiles) for _ in range(self.n_nets)])

        init_fn = make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        device = next(self.parameters()).device
        return ensure_batch(x, device=device)
    
    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor, shape (B, state_dim) or (state_dim,).
        action : torch.Tensor
            Action tensor, shape (B, action_dim) or (action_dim,).

        Returns
        -------
        quantiles : torch.Tensor
            Quantile values, shape (B, n_nets, n_quantiles).
        """
        state = self._ensure_batch(state)
        action = self._ensure_batch(action)

        x = th.cat([state, action], dim=-1)

        qs = []
        for trunk, head in zip(self.trunks, self.heads):
            feat = trunk(x)
            qs.append(head(feat))  # (B, n_quantiles)

        return th.stack(qs, dim=1)  # (B, n_nets, n_quantiles)