from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DiscretePolicyNetwork
from model_free.common.networks.q_networks import QNetwork, DoubleQNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OffPolicyDiscreteActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_acer_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build an ACERHead instance on CPU.

    Why this exists
    ---------------
    In Ray, remote worker processes often reconstruct policies from a serialized
    "factory spec" (entrypoint + kwargs). The entrypoint must be module-level so
    it is importable by name in a separate process.

    Notes
    -----
    - Forces `device="cpu"` to keep Ray workers lightweight and avoid GPU contention.
    - `activation_fn` may be provided as string/None in kwargs; it is resolved here.
    - Returned module is put into inference mode via set_training(False) (best-effort).
    """
    kwargs = dict(kwargs)

    # Force CPU on Ray worker side (avoid accidental GPU allocation).
    kwargs["device"] = "cpu"

    # activation_fn can be a name/string; resolve to actual nn.Module class.
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = ACERHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# ACERHead (config-free)
# =============================================================================
class ACERHead(OffPolicyDiscreteActorCriticHead):
    """
    ACER Head (Discrete Actor + Q Critic + Target Critic)

    Overview
    --------
    This head bundles:
      - A discrete stochastic actor π(a|s) (categorical policy)
      - An action-value critic Q(s,a) (either single Q or Double Q)
      - A target critic Q_target(s,a) for stable off-policy targets

    Inherited base functionality (from OffPolicyActorCriticHead)
    ------------------------------------------------------------
    The base head typically provides:
      - `device` management
      - `_to_tensor_batched(x)` utilities that ensure (B, dim) tensor shapes
      - `set_training(training: bool)` to switch train/eval behavior
      - `act(...)` helper (depending on your base class)
      - target-network utilities like:
          * hard_update(dst, src)
          * freeze_target(module)
          * hard_update_target() / soft_update_target(tau)
        (Exact names depend on your base implementation.)

    ACER-specific helpers
    ---------------------
    - dist(obs)  -> returns a categorical distribution object
    - logp(obs, action) -> log π(a|s), shape (B, 1)
    - probs(obs) -> π(a|s), shape (B, A)
    - save/load for persistence
    - get_ray_policy_factory_spec() for Ray reconstruction
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        double_q: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        # Base head stores the device and provides shared utilities.
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Store constructor args for introspection / persistence
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn

        self.dueling_mode = bool(dueling_mode)
        self.double_q = bool(double_q)

        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---------------------------------------------------------------------
        # Actor: discrete policy network π(a|s)
        #
        # Expected output:
        # - logits over actions or a distribution wrapper internally
        # ---------------------------------------------------------------------
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
        # Critic: Q(s,a) and target Q_target(s,a)
        #
        # double_q=True:
        #   - use DoubleQNetwork (two independent Q heads)
        #   - reduces positive bias in TD targets (typical Double Q motivation)
        #
        # dueling_mode=True:
        #   - if supported by your QNetwork implementation, uses dueling architecture:
        #       Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        # ---------------------------------------------------------------------
        if self.double_q:
            self.critic = DoubleQNetwork(
                state_dim=self.obs_dim,
                action_dim=self.n_actions,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                dueling_mode=self.dueling_mode,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

            self.critic_target = DoubleQNetwork(
                state_dim=self.obs_dim,
                action_dim=self.n_actions,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                dueling_mode=self.dueling_mode,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)
        else:
            self.critic = QNetwork(
                state_dim=self.obs_dim,
                action_dim=self.n_actions,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                dueling_mode=self.dueling_mode,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

            self.critic_target = QNetwork(
                state_dim=self.obs_dim,
                action_dim=self.n_actions,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                dueling_mode=self.dueling_mode,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

        # ---------------------------------------------------------------------
        # Initialize target critic weights from online critic, then freeze it.
        # This ensures Q_target starts identical and is updated only via
        # hard/soft updates (not by gradient descent).
        # ---------------------------------------------------------------------
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)
    
    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe format.

        Notes
        -----
        - activation_fn must be converted to a stable string name.
        - device is stored as string, but worker-side factory will override to "cpu".
        """
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "double_q": bool(self.double_q),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),  # overridden to "cpu" on Ray worker anyway
        }

    def save(self, path: str) -> None:
        """
        Save actor/critic/target weights plus minimal constructor kwargs.

        Format
        ------
        Uses torch.save with a dict payload:
          - kwargs         : json-safe ctor args (enough to reconstruct)
          - actor          : actor.state_dict()
          - critic         : critic.state_dict()
          - critic_target  : target critic state_dict()
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load actor/critic/target weights from a checkpoint.

        Notes
        -----
        - Loads onto current self.device (map_location=self.device).
        - If q_target is missing in checkpoint, it is refreshed from online critic.
        - Target critic is frozen after loading (to prevent optimizer updates).
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # If target weights exist, load them; otherwise sync from online critic.
        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.freeze_target(self.critic_target)
        else:
            self.hard_update_target()

        # Ensure target is in eval mode.
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly policy factory spec (entrypoint + kwargs).

        Notes
        -----
        - entrypoint must be importable by Ray workers (module-level function).
        - kwargs must be JSON-safe, since it is typically serialized across processes.
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_acer_head_worker_policy),
            kwargs=self._export_ctor_kwargs_json_safe(),
        )
