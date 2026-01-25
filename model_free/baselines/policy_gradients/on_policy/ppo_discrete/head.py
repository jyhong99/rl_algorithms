from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

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
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_ppo_discrete_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build PPODiscreteHead on CPU.

    Why this exists
    ---------------
    Ray must be able to serialize/deserialize the policy builder function.
    Therefore, this function MUST live at the module level (not inside a class).

    Behavior
    --------
    - Force `device="cpu"` on the worker side for portability/safety.
    - `activation_fn` is stored in a JSON/pickle-safe representation (string/None),
      so we resolve it back to an actual callable/class here.

    Parameters
    ----------
    **kwargs : Any
        Constructor kwargs for PPODiscreteHead.

    Returns
    -------
    head : nn.Module
        A PPODiscreteHead instance on CPU, set to eval-like mode via set_training(False).
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"

    # activation_fn is serialized (string/None) => resolve to nn.Module/callable
    act = kwargs.get("activation_fn", None)
    kwargs["activation_fn"] = resolve_activation_fn(act)

    head = PPODiscreteHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# PPODiscreteHead (config-free)
# =============================================================================
class PPODiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    PPO Discrete Network Container (Actor + Critic)
    ----------------------------------------------
    Config-free PPO head implementation for DISCRETE action spaces.

    This head bundles:
      - actor: categorical policy π(a|s) implemented via DiscretePolicyNetwork
      - critic: state-value baseline V(s) implemented via StateValueNetwork

    Architecture
    ------------
    Actor:
      - DiscretePolicyNetwork -> produces logits/probabilities for a Categorical dist

    Critic:
      - StateValueNetwork -> predicts scalar V(s)

    Head Contract (expected by OnPolicyAlgorithm)
    ---------------------------------------------
    Inherited APIs (typical):
      - device: torch.device
      - set_training(training: bool) -> None
      - act(obs, deterministic=False) -> torch.Tensor
      - value_only(obs) -> torch.Tensor
      - evaluate_actions(obs, action, as_scalar=False) -> Dict[str, Any]
      - save(path) / load(path)
      - get_ray_policy_factory_spec() -> PolicyFactorySpec

    Shape conventions
    -----------------
    observations:
      - (obs_dim,) or (B, obs_dim)

    actions (discrete indices):
      - scalar integer
      - (B,) LongTensor

    Output conventions
    ------------------
    - act(...) typically returns (B,) of dtype long (action indices)
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        """
        Construct PPO discrete head.

        Parameters
        ----------
        obs_dim : int
            Observation vector dimension.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : Sequence[int]
            MLP hidden layer sizes used by both actor and critic.
        activation_fn : Any
            Activation function (e.g., nn.ReLU).
        init_type : str
            Network initialization strategy string (e.g., "orthogonal").
        gain : float
            Initialization gain multiplier.
        bias : float
            Bias initialization constant.
        device : Union[str, torch.device]
            Torch device for networks and computation.
        """
        # NOTE:
        # If your base class expects device, you'd normally call:
        #   super().__init__(device=device)
        # Here you used `super().__init__()` and manually assign self.device.
        # That is OK if OnPolicyDiscreteActorCriticHead does not manage device itself.
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # Normalize device to torch.device
        self.device = th.device(device)

        # ---------------------------------------------------------------------
        # Actor network: categorical policy π(a|s)
        # ---------------------------------------------------------------------
        # DiscretePolicyNetwork should expose get_dist(obs) to produce a
        # torch.distributions.Categorical-like distribution.
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic network: baseline value V(s)
        # ---------------------------------------------------------------------
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Internal helpers
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in JSON/pickle-safe format.

        This is used for:
          - save/load payload reproducibility
          - Ray worker reconstruction (factory spec)

        Key point
        ---------
        `activation_fn` cannot be pickled reliably across processes, so we store a
        string name (or None) and resolve it with resolve_activation_fn(...) later.
        """
        act_name: Optional[str] = None
        if self.activation_fn is not None:
            # Prefer a stable callable name, fallback to string representation
            act_name = getattr(self.activation_fn, "__name__", None) or str(self.activation_fn)

        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": act_name,
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def save(self, path: str) -> None:
        """
        Save actor/critic weights to disk (config-free checkpoint).

        Stored payload
        --------------
        - kwargs : JSON-safe constructor config
        - actor  : actor.state_dict()
        - critic : critic.state_dict()

        Notes
        -----
        - This is a *weights checkpoint* + *constructor config*.
        - The module is NOT automatically rebuilt on load(); load() only restores weights.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load weights from disk into the existing module instance.

        Behavior
        --------
        - Loads ONLY actor/critic state_dict into the current instance.
        - Does NOT rebuild the module architecture (shapes must match).

        Raises
        ------
        ValueError
            If checkpoint payload structure is not recognized.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized PPODiscreteHead checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Provide a Ray-safe policy construction spec.

        Returns
        -------
        PolicyFactorySpec
            - entrypoint : module-level function pointer for Ray workers
            - kwargs      : JSON/pickle-safe constructor arguments

        Important
        ---------
        - entrypoint MUST be module-level for Ray serialization.
        - activation_fn is serialized as a string and resolved on worker via
          build_ppo_discrete_head_worker_policy().
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_ppo_discrete_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
