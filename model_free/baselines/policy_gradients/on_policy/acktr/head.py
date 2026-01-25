from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    make_entrypoint,
    resolve_activation_fn,
)
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_acktr_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build ACKTRHead on CPU.

    Why this exists
    ---------------
    When using Ray, policy modules are often constructed on remote workers.
    Ray requires a module-level callable as an "entrypoint" so it can serialize
    and recreate the object deterministically on each worker process.

    Notes
    -----
    - device is forced to "cpu" on the worker for safety/portability.
      (Workers can still run rollout on CPU even if the learner is on GPU.)
    - activation_fn is serialized as a string/None, so we resolve it here.
    """
    kwargs = dict(kwargs)

    # Always build worker-side policy on CPU
    kwargs["device"] = "cpu"

    # Resolve activation function from serialized representation (e.g., "relu")
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Construct the head
    head = ACKTRHead(**kwargs).to("cpu")

    # Workers usually run inference only (no gradients)
    head.set_training(False)
    return head


# =============================================================================
# ACKTRHead (refactored: BaseHead + OnPolicy actor-critic)
# =============================================================================
class ACKTRHead(OnPolicyContinuousActorCriticHead):
    """
    ACKTR network container (Actor + Critic) for continuous control.

    Overview
    --------
    ACKTR (Actor Critic using Kronecker-Factored Trust Region) uses the same
    actor/critic *network architecture* as PPO/A2C for continuous actions:
      - Actor : Gaussian policy π(a|s) (unsquashed)
      - Critic: State-value function V(s)

    Important
    ---------
    The "ACKTR-specific" part is NOT the head.
    The algorithmic difference (K-FAC / natural gradient approximation) belongs
    to the optimizer/core implementation. Therefore, this head is intentionally
    a standard on-policy actor-critic container.

    Head contract (for OnPolicyAlgorithm)
    -------------------------------------
    Inherited from OnPolicyContinuousActorCriticHead:
      - device: torch.device
      - set_training(training: bool) -> None
      - act(obs, deterministic=False) -> torch.Tensor
      - evaluate_actions(obs, action, as_scalar=False) -> Dict[str, Any]
      - value_only(obs) -> torch.Tensor
      - save(path) / load(path)
      - get_ray_policy_factory_spec() -> PolicyFactorySpec

    Returned tensors follow your base head conventions:
      - value    : (B, 1)
      - log_prob : (B, 1) or per-dim depending on distribution
      - entropy  : (B, 1) or per-dim depending on distribution
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
        # Gaussian log-std parameterization options
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int]
            MLP hidden layer sizes for actor/critic.
        activation_fn : Any
            Activation function class (e.g., nn.ReLU).
        init_type : str
            Weight initialization scheme (library-specific).
        gain : float
            Initialization gain (library-specific).
        bias : float
            Initialization bias (library-specific).
        device : Union[str, torch.device]
            Device for this module ("cpu" or "cuda").
        log_std_mode : str
            How to parameterize the Gaussian log standard deviation.
            Example: "param" means learnable parameter vector.
        log_std_init : float
            Initial value for log_std parameters.
        """
        super().__init__(device=device)

        # Store config for checkpoint/Ray reconstruction
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---------------------------------------------------------------------
        # Actor network: unsquashed diagonal Gaussian policy
        # ---------------------------------------------------------------------
        # squash=False: no tanh squashing; ACKTR/A2C-style usually uses raw Gaussian.
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic network: state-value function V(s)
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
    # Persistence (PPOHead-style checkpointing)
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs into a JSON-safe dict.

        This is used for:
        - checkpoint reproducibility
        - Ray worker reconstruction (PolicyFactorySpec kwargs)
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "log_std_mode": str(self.log_std_mode),
            "log_std_init": float(self.log_std_init),
        }

    def _state_payload(self) -> Dict[str, Any]:
        """
        Return the checkpoint payload (ready to be passed into torch.save()).

        Contents
        --------
        - kwargs : JSON-safe constructor config
        - actor  : actor state_dict
        - critic : critic state_dict
        """
        return {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def _load_state_payload(self, ckpt: Dict[str, Any]) -> None:
        """
        Load weights from a checkpoint payload dict into this instance.

        Notes
        -----
        - This does NOT reconstruct the module; it only loads parameters.
        - The existing instance must have compatible shapes.
        """
        if "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError("Unrecognized ACKTRHead checkpoint payload (missing actor/critic).")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def save(self, path: str) -> None:
        """
        Save a checkpoint to disk.

        The saved file contains only:
        - ctor kwargs (JSON-safe)
        - actor weights
        - critic weights
        """
        if not path.endswith(".pt"):
            path += ".pt"
        th.save(self._state_payload(), path)

    def load(self, path: str) -> None:
        """
        Load a checkpoint from disk into the existing instance.

        This loads weights only (no reconstruction).
        """
        if not path.endswith(".pt"):
            path += ".pt"
        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format: {path}")
        self._load_state_payload(ckpt)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Provide a Ray-safe construction spec:
          - entrypoint : module-level function pointer
          - kwargs      : JSON-safe ctor args

        Ray workers can call:
          build_acktr_head_worker_policy(**kwargs)
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_acktr_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
