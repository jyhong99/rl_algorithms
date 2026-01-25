from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OnPolicyDiscreteActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_vpg_discrete_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build VPGDiscreteHead on CPU.

    Why module-level?
    -----------------
    Ray (and your entrypoint resolver) generally require an importable top-level
    callable for remote construction.

    Notes
    -----
    - `device` is forced to "cpu" on rollout workers:
        workers typically do inference only, so CPU avoids GPU contention.
    - `activation_fn` may be serialized (e.g., "ReLU") and is resolved back into
      an actual torch.nn.Module class here.
    """
    # Copy to avoid mutating the caller's dictionary
    kwargs = dict(kwargs)

    # Force CPU on rollout workers
    kwargs["device"] = "cpu"

    # Resolve serialized activation function -> actual torch activation class/module
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Construct the head and put it into inference mode
    head = VPGDiscreteHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# VPGDiscreteHead
# =============================================================================
class VPGDiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    VPG head for discrete-action environments.

    Components
    ----------
    - Actor:
        DiscretePolicyNetwork producing logits/probabilities for a categorical policy π(a|s).
    - Critic (optional baseline):
        StateValueNetwork estimating V(s).

    Baseline modes
    --------------
    - use_baseline=False:
        REINFORCE-style (no critic). Value baseline is not used.
    - use_baseline=True:
        Actor-critic VPG with a value baseline (variance reduction).

    Inherited API
    -------------
    This class inherits from OnPolicyDiscreteActorCriticHead, which is expected to provide:
      - act(obs, deterministic=False)
      - evaluate_actions(obs, action, as_scalar=False)
      - value_only(obs)

    Important
    ---------
    If your base head implementation assumes critic always exists, you should ensure:
      - evaluate_actions() returns a safe zero baseline when critic is None
      - value_only() returns zeros((B,1)) when critic is None
    (You already fixed this in base_head, so this class is compatible.)
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
        use_baseline: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation/state dimension.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : Sequence[int]
            Hidden layer sizes for actor and critic MLPs.
        activation_fn : Any
            Activation function class (e.g., nn.ReLU). May be serialized for Ray.
        init_type : str
            Weight initialization method (used for critic; actor may ignore depending on your network code).
        gain : float
            Initialization gain multiplier.
        bias : float
            Bias initialization value.
        device : Union[str, torch.device]
            Device for actor/critic networks.
        use_baseline : bool
            Whether to include a value baseline network V(s).
        """
        super().__init__(device=device)

        # ----- dimensions -----
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        # ----- MLP configuration -----
        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ----- baseline flag -----
        self.use_baseline = bool(use_baseline)

        # ---------------------------------------------------------------------
        # Actor: categorical policy network for discrete actions
        # ---------------------------------------------------------------------
        # The DiscretePolicyNetwork is expected to provide get_dist(obs) returning
        # a torch.distributions.Categorical-like object with:
        #   - sample()
        #   - log_prob(action)
        #   - entropy()
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic (optional): value baseline V(s)
        # ---------------------------------------------------------------------
        self.critic: Optional[StateValueNetwork] = None
        if self.use_baseline:
            self.critic = StateValueNetwork(
                state_dim=self.obs_dim,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

    # =============================================================================
    # Persistence + Ray spec
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export JSON-safe constructor kwargs.

        Uses
        ----
        - checkpoint metadata (for debugging/reconstruction)
        - Ray worker construction (kwargs must be serializable)
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "use_baseline": bool(self.use_baseline),
        }

    def save(self, path: str) -> None:
        """
        Save checkpoint to disk.

        Stored payload
        --------------
        - kwargs : JSON-safe config used to build this head (for reproducibility/debugging)
        - actor  : actor state_dict
        - critic : critic state_dict or None (if baseline disabled)

        Notes
        -----
        - Optimizer state belongs to the core/algorithm, not the head.
        - Adds ".pt" suffix if missing.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": None if self.critic is None else self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint from disk into this instance.

        Notes
        -----
        - Loads weights only; does not reconstruct a new object.
        - Uses map_location=self.device for CPU/GPU portability.
        - Enforces baseline compatibility:
            * baseline OFF instance cannot load baseline ON checkpoint (critic mismatch)
            * baseline ON  instance cannot load baseline OFF checkpoint
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        if "actor" not in ckpt:
            raise ValueError("Unrecognized VPGDiscreteHead checkpoint payload (missing actor).")

        # Always load actor weights
        self.actor.load_state_dict(ckpt["actor"])

        ckpt_critic = ckpt.get("critic", None)

        # Baseline compatibility checks
        if self.critic is None:
            # current instance baseline OFF
            if ckpt_critic is not None:
                raise ValueError("Checkpoint contains critic weights but this VPGDiscreteHead has use_baseline=False.")
        else:
            # current instance baseline ON
            if ckpt_critic is None:
                raise ValueError("Checkpoint has no critic weights but this VPGDiscreteHead has use_baseline=True.")
            self.critic.load_state_dict(ckpt_critic)

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly factory spec for constructing this head on workers.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_vpg_discrete_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
