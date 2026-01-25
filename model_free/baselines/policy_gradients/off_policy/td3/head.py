from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DeterministicPolicyNetwork
from model_free.common.networks.value_networks import DoubleStateActionValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import DeterministicActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_td3_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build TD3Head on CPU.

    Why this exists
    ---------------
    In Ray multi-process rollouts, you typically construct policies inside worker
    processes. Those workers should avoid GPU ownership and should receive only
    JSON-serializable kwargs.

    What this function enforces
    ---------------------------
    - device is forced to "cpu" (workers stay CPU-only)
    - activation_fn is resolved from a string/name to an nn.Module class
    - action_low/action_high are converted from list -> np.ndarray (float32)

    Returns
    -------
    head : nn.Module
        A TD3Head instance on CPU, set to eval/inference mode.
    """
    kwargs = dict(kwargs)

    # Force CPU on workers to avoid GPU contention / CUDA context init in subprocesses.
    kwargs["device"] = "cpu"

    # activation_fn might be serialized as a string; map it back to nn.Module class.
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    # Action bounds are often stored as JSON-safe lists; convert back to float32 arrays.
    if kwargs.get("action_low", None) is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    if kwargs.get("action_high", None) is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)

    # Build the head and disable training behaviors (dropout/bn, etc.) for rollouts.
    head = TD3Head(**kwargs).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# TD3Head
# =============================================================================
class TD3Head(DeterministicActorCriticHead):
    """
    TD3 Head (Actor + Twin Critics + Target Networks)

    Overview
    --------
    TD3 uses:
      - a deterministic actor π(s)
      - twin Q critics Q1(s,a), Q2(s,a)
      - target networks π'(s), Q1'(s,a), Q2'(s,a) for stable bootstrapping
      - target policy smoothing for critic target computation

    Inheritance
    -----------
    Inherits from DeterministicActorCriticHead (your base head for DDPG/TD3-style):
      - set_training(training): toggles train/eval modes; targets stay eval
      - act(obs, deterministic=False): returns action; may add exploration noise
      - reset_exploration_noise(): if noise process needs reset (e.g., OU)
      - hard_update(target, source) / soft_update(target, source, tau)
      - freeze_target(module): disable grads for target nets
      - _to_tensor_batched(x): convert obs/action to (B,dim) tensor on device
      - _clamp_action(a): clamp to bounds if action_low/high are provided

    TD3-specific additions
    ----------------------
    - target_action(next_obs, noise_std, noise_clip):
        Implements TD3 target policy smoothing:
          a' = clip( π'(s') + clip(ε, -c, c), bounds )

    Required by downstream (OffPolicyAlgorithm / TD3Core)
    -----------------------------------------------------
    - self.actor: nn.Module producing actions
    - self.critic: nn.Module returning (q1,q2)
    - self.actor_target, self.critic_target: frozen target networks
    - save(path), load(path) for checkpointing
    - get_ray_policy_factory_spec() for Ray worker creation
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
        action_low: Optional[Union[np.ndarray, Sequence[float]]] = None,
        action_high: Optional[Union[np.ndarray, Sequence[float]]] = None,
        noise: Optional[Any] = None,
        noise_clip: Optional[float] = None,
    ) -> None:
        # BaseHead initialization sets self.device and common helpers.
        super().__init__(device=device)

        # Cache core hyperparams for export/repro.
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---------------------------------------------------------------------
        # Action bounds handling
        # ---------------------------------------------------------------------
        # Keep bounds as numpy arrays (float32) for easy JSON-safe export and
        # consistent shape checks. The actor network will use these to scale/clamp.
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32).reshape(-1)

        # Require both bounds or neither. TD3 can work without bounds, but if you
        # provide them, it must be a matched pair.
        if (self.action_low is None) ^ (self.action_high is None):
            raise ValueError("action_low and action_high must be provided together, or both be None.")

        # Validate bounds dimensionality matches action_dim.
        if self.action_low is not None:
            if self.action_low.shape[0] != self.action_dim or self.action_high.shape[0] != self.action_dim:
                raise ValueError(
                    f"action_low/high must have shape ({self.action_dim},), "
                    f"got {self.action_low.shape}, {self.action_high.shape}"
                )

        # Exploration noise config:
        # - noise: external noise object/process (e.g., OU or Gaussian) handled by base act()
        # - noise_clip: optional clamp applied to the noise (common in TD3 target smoothing too)
        self.noise = noise
        self.noise_clip = None if noise_clip is None else float(noise_clip)

        # ---------------------------------------------------------------------
        # Actor network (deterministic policy)
        # ---------------------------------------------------------------------
        # DeterministicPolicyNetwork typically outputs in bounds directly if bounds are provided.
        self.actor = DeterministicPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            action_low=self.action_low,
            action_high=self.action_high,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Twin critic network
        # ---------------------------------------------------------------------
        # DoubleStateActionValueNetwork returns (q1, q2), each shaped (B,1).
        self.critic = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Target networks
        # ---------------------------------------------------------------------
        # TD3 uses target actor and target critics for bootstrapped targets.
        self.actor_target = DeterministicPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            action_low=self.action_low,
            action_high=self.action_high,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.critic_target = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize targets to match online networks, then freeze targets (no grads).
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # TD3 target policy smoothing
    # =============================================================================
    @th.no_grad()
    def target_action(
        self,
        next_obs: Any,
        *,
        noise_std: float,
        noise_clip: float,
    ) -> th.Tensor:
        """
        Compute smoothed target action for TD3 critic target computation.

        TD3 uses "target policy smoothing" to reduce overestimation by adding
        clipped Gaussian noise to the target actor output:
          a' = clip( π'(s') + clip(ε, -c, c), action_bounds )
        where ε ~ N(0, noise_std^2).

        Parameters
        ----------
        next_obs : Any
            Next observation(s). Accepts numpy arrays, lists, tensors, etc.
        noise_std : float
            Standard deviation of Gaussian noise added to target action.
        noise_clip : float
            Clip range for the noise term ε (applied elementwise).

        Returns
        -------
        next_action : torch.Tensor
            Smoothed target action tensor of shape (B, action_dim), clamped to bounds
            if action_low/action_high are configured.
        """
        # Convert to (B, obs_dim) tensor on the head device.
        s2 = self._to_tensor_batched(next_obs)

        # Base target action from target actor.
        a2 = self.actor_target(s2)  # (B, action_dim)

        ns = float(noise_std)
        nc = float(noise_clip)

        # Add clipped Gaussian noise (policy smoothing).
        if ns > 0.0:
            eps = ns * th.randn_like(a2)
            if nc > 0.0:
                eps = eps.clamp(-nc, nc)
            a2 = a2 + eps

        # Clamp to valid action bounds (if bounds are defined).
        return self._clamp_action(a2)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in JSON-safe form.

        Notes
        -----
        - activation_fn is converted to a name/string (resolved back on load/worker)
        - device is stored as string for portability
        - action bounds are stored as Python lists
        """
        low = None if self.action_low is None else [float(x) for x in self.action_low.reshape(-1)]
        high = None if self.action_high is None else [float(x) for x in self.action_high.reshape(-1)]
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "action_low": low,
            "action_high": high,
        }

    def save(self, path: str) -> None:
        """
        Save actor/critic + target networks into a single checkpoint file.

        Parameters
        ----------
        path : str
            Target path prefix. If not ending with ".pt", it will be appended.

        Saved contents
        --------------
        - kwargs: json-safe constructor args (for Ray/spec reconstruction)
        - actor, critic: online networks
        - actor_target, critic_target: target networks
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load actor/critic + target networks from a checkpoint file.

        Parameters
        ----------
        path : str
            Checkpoint path prefix. If not ending with ".pt", it will be appended.

        Notes
        -----
        - If a target network state is missing, targets are hard-copied from online.
        - Targets are frozen after loading to ensure no accidental gradient updates.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized TD3Head checkpoint format at: {path}")

        # Restore online networks.
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Restore targets if present; otherwise re-init from online.
        if ckpt.get("actor_target", None) is not None:
            self.actor_target.load_state_dict(ckpt["actor_target"])
        else:
            self.hard_update(self.actor_target, self.actor)

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        # Targets should not receive gradients.
        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-serializable spec that reconstructs this head inside workers.

        The entrypoint must be a module-level function (Ray cloudpickle-safe),
        and kwargs must be JSON-safe (no tensors, no classes).
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_td3_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
