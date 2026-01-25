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
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_ppo_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build `PPOHead` on CPU.

    Why this exists
    ---------------
    Ray needs a *module-level* callable (picklable entrypoint) to reconstruct
    policy objects inside worker processes.

    Behavior
    --------
    - Forces `device="cpu"` for safety and portability.
      (Workers usually only need CPU inference for rollout collection.)
    - `activation_fn` is stored as a JSON-safe string/None in checkpoints/specs,
      so we resolve it back into a callable here via `resolve_activation_fn(...)`.

    Returns
    -------
    nn.Module
        A `PPOHead` instance placed on CPU and set to evaluation mode.
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = PPOHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# PPOHead (continuous-only)
# =============================================================================
class PPOHead(OnPolicyContinuousActorCriticHead):
    """
    PPO network container (Actor + Critic) for continuous actions.

    Overview
    --------
    This head bundles:
      - Actor: Gaussian policy network π(a|s)
      - Critic: state-value network V(s)

    PPO-specific details (ratio clipping, etc.) belong to the *core*, not the head.
    The head only provides forward interfaces used by the algorithm core.

    Expected usage
    --------------
    The OnPolicyAlgorithm expects the head to provide:
      - act(obs, deterministic=False) -> action
      - evaluate_actions(obs, action) -> dict(log_prob, entropy, value, ...)
      - value_only(obs) -> value
      - set_training(training: bool)
      - get_ray_policy_factory_spec() for Ray rollout workers

    Notes
    -----
    - ContinuousPolicyNetwork is configured with `squash=False` because PPO typically
      operates on an *unsquashed* Gaussian distribution and lets environment wrappers
      handle action scaling/clipping.
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
        # Gaussian std parameterization
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation vector dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int]
            MLP hidden layer sizes shared by actor and critic.
        activation_fn : Any
            Activation function for MLP layers (callable).
        init_type : str
            Network weight init strategy (passed to your network implementations).
        gain : float
            Optional init scaling gain.
        bias : float
            Optional bias init.
        device : Union[str, torch.device]
            Target device ("cpu" or "cuda").
        log_std_mode : str
            How to parameterize Gaussian log-std (e.g., "param", "mlp").
        log_std_init : float
            Initial value for log-std if `log_std_mode="param"`.
        """
        super().__init__(device=device)

        # Store hyperparameters for reproducibility / checkpoint reconstruction
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
        # Actor: Gaussian policy π(a|s)
        # ---------------------------------------------------------------------
        # - squash=False: no tanh-squash; action bounding can be handled outside
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
        # Critic: state-value function V(s)
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
    # Persistence / Ray kwargs export
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe form.

        This is used for:
          - checkpoint metadata ("kwargs")
          - Ray worker reconstruction via PolicyFactorySpec

        Notes
        -----
        `activation_fn` must be serialized into a string form, because callables
        are not JSON-serializable and are often not safe to pickle across machines.
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

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-safe construction spec for this head.

        Ray will call:
          entrypoint(**kwargs)
        inside worker processes.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_ppo_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
