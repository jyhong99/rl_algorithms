from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.q_networks import QNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import QLearningHead


# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================
def build_dqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build a DQNHead instance on CPU.

    Why this exists
    ---------------
    Ray serializes callables (entrypoints) and arguments. The worker process
    reconstructs the policy/head object from JSON-safe kwargs, avoiding
    pickling issues with lambdas/classes/closures.

    Notes
    -----
    - `device` is forced to "cpu" on workers (common pattern for rollout workers).
    - `activation_fn` is typically serialized as a string/name and resolved here
      back into a torch.nn.Module class/function (e.g., "ReLU" -> nn.ReLU).
    - The head is put into eval/frozen mode via `set_training(False)` because
      workers should not do gradient updates (only inference/rollouts).
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = DQNHead(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# DQNHead
# =============================================================================
class DQNHead(QLearningHead):
    """
    DQN Network Container (Online Q + Target Q)

    Purpose
    -------
    Thin “head” wrapper that owns:
      - online Q-network (self.q)
      - target Q-network (self.q_target)

    This matches the expectations of DQN-style cores / algorithms:
      - compute Q(s, ·) from online network for action selection
      - compute Q_target(s', ·) from target network for bootstrapped targets

    Head Contract (for OffPolicyAlgorithm / DQN-style cores)
    -------------------------------------------------------
    Expected methods/properties (some are inherited from QLearningHead):
      - device: torch.device
      - set_training(training) -> None
      - act(obs, epsilon=0.0, deterministic=True) -> actions (B,) long
      - q_values(obs) -> (B, A)
      - q_values_target(obs) -> (B, A)
      - save(path), load(path)
      - get_ray_policy_factory_spec()

    Design Notes
    ------------
    - Target net is conventionally frozen (requires_grad=False) and in eval() mode.
      This prevents accidental gradient flow and ensures deterministic behavior
      (e.g., dropout/bn not used in training mode).
    - The initial target parameters are copied from the online network via
      `hard_update`, then frozen via `freeze_target`.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        # Base head sets `self.device` and typically provides:
        # - set_training()
        # - hard_update(), freeze_target()
        # - act(), q_values(), q_values_target() (depending on your base class)
        super().__init__(device=device)

        # Store hyperparameters (useful for saving/rebuilding and Ray kwargs).
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.dueling_mode = bool(dueling_mode)
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ----- online Q -----
        # QNetwork returns Q-values for all discrete actions: shape (B, A).
        # If dueling_mode=True, QNetwork internally uses advantage/value streams.
        self.q = QNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ----- target Q -----
        # Structurally identical to online Q. Updated periodically/slowly from self.q
        # (hard update for DQN, or Polyak for variants if your core uses it).
        self.q_target = QNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target parameters = online parameters.
        # `hard_update(dst, src)` should copy src.state_dict() into dst.
        self.hard_update(self.q_target, self.q)

        # Freeze + eval target network by convention to:
        # - avoid gradients through target
        # - keep deterministic behavior
        self.freeze_target(self.q_target)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        Why
        ---
        - Needed for Ray worker reconstruction (PolicyFactorySpec kwargs).
        - Helpful for checkpoint metadata / reproducibility.

        Implementation detail
        ---------------------
        activation_fn is converted to a name string because function/class objects
        are not JSON-serializable. The worker resolves it back using
        `resolve_activation_fn`.
        """
        act_name: Optional[str] = None
        if self.activation_fn is not None:
            # Prefer the python name for canonical activations (ReLU, SiLU, etc.)
            act_name = getattr(self.activation_fn, "__name__", None) or str(self.activation_fn)

        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": act_name,  # resolved on worker
            "dueling_mode": bool(self.dueling_mode),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            # Keep device string mostly for convenience; worker overrides to CPU.
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save a checkpoint (.pt) containing:
          - JSON-safe kwargs to rebuild the head
          - online network weights
          - target network weights
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load a checkpoint saved by `save()`.

        Notes
        -----
        - We load onto `self.device` via map_location.
        - After loading the target weights, we re-freeze/eval the target net to
          enforce invariants even if checkpoint was created differently.
        - If `q_target` is missing, we fall back to syncing target from online.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        # Restore online network parameters.
        self.q.load_state_dict(ckpt["q"])

        # Restore target network if available; otherwise rebuild from online.
        if "q_target" in ckpt and ckpt["q_target"] is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
            self.freeze_target(self.q_target)  # enforce frozen/eval invariant
        else:
            self.hard_update(self.q_target, self.q)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly factory specification.

        The spec contains:
          - entrypoint: module-level function to build the policy on workers
          - kwargs: JSON-safe kwargs required to reconstruct the head

        The worker will call:
          head = build_dqn_head_worker_policy(**kwargs)
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_dqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
