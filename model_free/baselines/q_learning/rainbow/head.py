from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.q_networks import RainbowQNetwork
from model_free.common.policies.base_head import QLearningHead
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn


# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_rainbow_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build `RainbowHead` on CPU.

    This function must live at module scope so that Ray can serialize/deserialize it.

    Notes
    -----
    - `device` is forced to "cpu" on the worker for safety / portability.
    - `activation_fn` is stored as a string (or None) and resolved here.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = RainbowHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


class RainbowHead(QLearningHead):
    """
    Rainbow Head (Online + Target) [config-free + Ray + persistence]

    Purpose
    -------
    `RainbowHead` is a discrete Q-learning head that owns:
      - an online distributional Q-network `q`
      - a target distributional Q-network `q_target`
      - the fixed C51 support (atoms) buffer

    Rainbow component coverage
    --------------------------
    This head is responsible for the following Rainbow building blocks:
      - C51 distributional Q-values (categorical distribution over atoms)
      - NoisyNet exploration (by resetting noise before action selection)

    Interface expected by Q-learning style algorithms
    -------------------------------------------------
    - q_values(obs) / q_values_target(obs) -> (B, A)
        Expected Q-values used for greedy or epsilon-greedy action selection.
        In C51, these are computed as:
            Q(s,a) = sum_{k} support[k] * p_k(s,a)

    - dist(obs) / dist_target(obs) -> (B, A, K)
        Categorical distribution (pmf) over atoms for each action.

    - act(obs, epsilon=..., deterministic=...) -> action index
        Overrides base epsilon-greedy to refresh NoisyNet noise each step.

    - reset_noise()
        Resamples NoisyNet parameters in q and q_target.

    - save(path) / load(path)
        Saves constructor kwargs and both networks' weights.

    - get_ray_policy_factory_spec()
        Provides a Ray-safe entrypoint + kwargs to recreate this head on workers.

    Notes
    -----
    - `support` is registered as a buffer so it:
        * moves with `.to(device)`
        * is saved/restored in state_dict (if needed)
        * is not trainable (no gradients)
    - Target net is frozen via `freeze_target()` to prevent accidental gradient flow.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        # -----------------------------
        # C51 distributional settings
        # -----------------------------
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # -----------------------------
        # MLP trunk
        # -----------------------------
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        # -----------------------------
        # Initialization
        # -----------------------------
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        # -----------------------------
        # NoisyNet
        # -----------------------------
        noisy_std_init: float = 0.5,
        # -----------------------------
        # Device
        # -----------------------------
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        # QLearningHead provides:
        #   - self.device
        #   - _to_tensor_batched(...)
        #   - epsilon-greedy act(...) using q_values(...)
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        # C51 atom / support configuration
        self.atom_size = int(atom_size)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        # Network hyperparameters
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn

        # Initialization hyperparameters (forwarded to network builder)
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # NoisyNet exploration hyperparameter
        self.noisy_std_init = float(noisy_std_init)

        # ---------------------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------------------
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got: {self.n_actions}")
        if self.atom_size <= 1:
            raise ValueError(f"atom_size must be >= 2, got: {self.atom_size}")
        if not (self.v_min < self.v_max):
            raise ValueError(f"Require v_min < v_max, got: v_min={self.v_min}, v_max={self.v_max}")

        # ---------------------------------------------------------------------
        # C51 support atoms (fixed values)
        # - Detach+clone to prevent any shared-storage / inplace surprises.
        # - Register as buffer so it moves with .to(device) and is not trainable.
        # ---------------------------------------------------------------------
        support = th.linspace(self.v_min, self.v_max, self.atom_size, dtype=th.float32)
        support = support.detach().clone()  # break view/storage ties proactively
        self.register_buffer("support", support)  # (K,)

        # ---------------------------------------------------------------------
        # IMPORTANT: DO NOT pass self.support directly into both networks.
        # Give each network its own detached+cloned support tensor.
        #
        # This prevents autograd "version counter" failures if any component
        # accidentally touches support in-place (even though it "shouldn't").
        # ---------------------------------------------------------------------
        support_q = self.support.detach().clone()
        support_t = self.support.detach().clone()

        self.q = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_q,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        self.q_target = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_t,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    # =============================================================================
    # NoisyNet
    # =============================================================================
    def reset_noise(self) -> None:
        """
        Resample parameter noise for NoisyNet layers.

        In Rainbow implementations, it is common to reset the online network's
        noise at every action selection step. Resetting target noise is not
        strictly required for correctness, but kept symmetric here.

        Notes
        -----
        This is a best-effort call:
          - only executed if the underlying network exposes `reset_noise()`.
        """
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()
        if hasattr(self.q_target, "reset_noise"):
            self.q_target.reset_noise()

    # =============================================================================
    # Acting (inject Noisy reset)
    # =============================================================================
    @th.no_grad()
    def act(self, obs: Any, *, epsilon: float = 0.0, deterministic: bool = True) -> th.Tensor:
        """
        Select an action using epsilon-greedy policy on expected Q-values.

        Parameters
        ----------
        obs : Any
            Single observation or batch of observations.
        epsilon : float
            Exploration probability (random action with prob epsilon).
            Even if epsilon=0, NoisyNet still provides implicit exploration.
        deterministic : bool
            If True, act greedily.
            If False, allows epsilon-greedy sampling (and/or Noisy exploration).

        Returns
        -------
        action : torch.Tensor
            Discrete action index tensor (shape depends on QLearningHead.act()):
              - usually (B,) or (1,)
        """
        # Refresh NoisyNet noise before action selection.
        # This makes the policy stochastic even with epsilon=0.0.
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()

        # QLearningHead.act(...) uses self.q_values(obs) internally.
        return super().act(obs, epsilon=epsilon, deterministic=deterministic)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in JSON-safe form.

        This is used for:
          - save/load round-trip reproducibility
          - Ray worker construction
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "atom_size": int(self.atom_size),
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "noisy_std_init": float(self.noisy_std_init),
            "device": str(self.device),  # Ray worker will override to "cpu"
        }

    def save(self, path: str) -> None:
        """
        Save head checkpoint (weights + ctor kwargs).

        Stored fields
        -------------
        - kwargs   : JSON-safe constructor configuration
        - support  : C51 atoms buffer (CPU)
        - q        : online network state_dict
        - q_target : target network state_dict
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "support": self.support.detach().cpu(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load checkpoint into existing head instance.

        Notes
        -----
        - This function loads only weights into the current instance.
        - It assumes the current instance was constructed with compatible shapes.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        # restore online weights
        self.q.load_state_dict(ckpt["q"])

        # restore target weights if present
        if ckpt.get("q_target", None) is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
            self.freeze_target(self.q_target)
            self.q_target.eval()
        else:
            self.hard_update(self.q_target, self.q)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return Ray factory specification.

        Ray will recreate this head by calling:
          entrypoint(**kwargs)
        on the remote worker process.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_rainbow_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )