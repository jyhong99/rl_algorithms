from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    make_entrypoint,
    resolve_activation_fn,
)
from model_free.common.policies.base_head import OnPolicyDiscreteActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_a2c_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build a DISCRETE A2C head on CPU.

    Why this function exists
    ------------------------
    Ray workers often reconstruct policies from a pickled entrypoint + kwargs.
    The safest pattern is:
      - keep the factory function at module scope (pickle-friendly)
      - pass JSON-safe kwargs (so kwargs can be serialized easily)
      - force worker policies onto CPU (rollout workers should not depend on GPU)

    Behavior
    --------
    - Forces `device="cpu"` for portability/safety on remote workers.
    - `activation_fn` is stored as a string (or None) in kwargs; we resolve it back
      to a callable here.
    - Puts the returned head into evaluation mode via `set_training(False)`.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = A2CDiscreteHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# A2C (DISCRETE)
# =============================================================================
class A2CDiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    A2C network container for DISCRETE action spaces.

    Components
    ----------
    Actor
      - `DiscretePolicyNetwork`: categorical policy π(a|s) over `n_actions`.

    Critic
      - `StateValueNetwork`: state-value baseline V(s).

    Inherited contract (from OnPolicyDiscreteActorCriticHead)
    ---------------------------------------------------------
    - act(obs, deterministic=False) -> action indices, shape (B,)
    - value_only(obs) -> value tensor, shape (B, 1)
    - evaluate_actions(obs, action) -> dict with:
        value    : (B, 1)
        log_prob : (B, 1)   (categorical log_prob standardized)
        entropy  : (B, 1)

    Constructor compatibility note
    ------------------------------
    In your codebase you may have a unified dispatcher/builder that forwards a shared
    kwargs set for both continuous and discrete heads. To keep checkpoint/Ray
    reconstruction robust, this discrete head accepts continuous-only kwargs
    (e.g., `log_std_*`) but ignores them.

    This prevents failures like:
      - "unexpected keyword argument log_std_mode" during load/Ray worker creation
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,  # DISCRETE: number of actions
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension.
        n_actions : int
            Number of discrete actions (size of categorical distribution).
        hidden_sizes : Sequence[int]
            MLP hidden sizes for actor and critic.
        activation_fn : Any
            Activation function class (e.g., nn.ReLU).
        init_type : str
            Weight initialization scheme name used by your network builders.
        gain : float
            Optional initialization gain.
        bias : float
            Optional initialization bias.
        device : str | torch.device
            Torch device for this head.
        log_std_mode, log_std_init
            Accepted only for compatibility with a unified builder; unused in discrete.
        """
        super().__init__(device=device)

        # -----------------------------
        # Store configuration
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.action_space = "discrete"

        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # -----------------------------
        # Actor: categorical policy π(a|s)
        # -----------------------------
        # Expected behavior:
        # - actor.get_dist(obs) returns a Categorical-like distribution
        # - distribution.log_prob(action_idx) returns shape (B,)
        # - distribution.entropy() returns shape (B,)
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Critic: V(s)
        # -----------------------------
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Persistence / Ray
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe form.

        Why include log_std_* here?
        ---------------------------
        If you have a unified constructor path for continuous and discrete heads
        (e.g., a single builder that forwards a shared kwargs set), storing these
        fields avoids mismatch during:
          - checkpoint load/rebuild
          - Ray worker reconstruction

        They are harmless for discrete and simply ignored by the implementation.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_space": "discrete",
            # NOTE: if your codebase standardizes the key name as `n_actions`,
            # consider exporting "n_actions" instead of "action_dim".
            "action_dim": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save head checkpoint.

        Stored fields
        -------------
        - kwargs : JSON-safe config for reconstruction/debugging
        - actor  : actor state_dict
        - critic : critic state_dict
        """
        if not path.endswith(".pt"):
            path += ".pt"
        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the existing instance.

        Notes
        -----
        - Loads weights only; does not reconstruct the object.
        - Assumes this instance was created with compatible shapes.
        """
        if not path.endswith(".pt"):
            path += ".pt"
        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {path}")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-safe construction spec (entrypoint + JSON-safe kwargs).
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_a2c_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
