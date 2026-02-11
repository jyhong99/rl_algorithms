from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import DeterministicPolicyNetwork
from model_free.common.networks.value_networks import DoubleStateActionValueNetwork
from model_free.common.policies.base_head import DeterministicActorCriticHead
from model_free.common.utils.ray_utils import PolicyFactorySpec, _make_entrypoint, _resolve_activation_fn

# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_td3_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a TD3Head instance for Ray rollout workers (CPU-only).

    Ray background
    --------------
    In Ray multi-process/multi-node rollouts, workers typically reconstruct the
    policy module from a `(entrypoint, kwargs)` specification. For this to work,
    the factory must be defined at module scope (importable / picklable).

    This factory enforces rollout-safe conventions
    ----------------------------------------------
    - Forces ``device="cpu"`` to avoid accidental GPU allocation on workers.
    - Resolves ``activation_fn`` from a serialized identifier (e.g., ``"relu"``)
      into a concrete ``torch.nn`` activation class.
    - Converts action bounds (often JSON lists) into ``np.ndarray`` for consistent
      shape checks and downstream clamping.

    Parameters
    ----------
    **kwargs : Any
        JSON-serializable constructor keyword arguments for :class:`TD3Head`.
        Typical fields include:
        ``obs_dim``, ``action_dim``, ``hidden_sizes``, ``activation_fn``,
        ``init_type``, ``gain``, ``bias``, ``action_low``, ``action_high``.

    Returns
    -------
    head : torch.nn.Module
        A CPU-resident :class:`TD3Head` configured for inference/rollout
        (``set_training(False)``).
    """
    kwargs = dict(kwargs)

    # Force CPU on rollout workers to avoid CUDA context initialization in subprocesses.
    kwargs["device"] = "cpu"

    # activation_fn may arrive as a string; resolve into a torch.nn activation class.
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))

    # Bounds commonly arrive as JSON-safe lists; convert to float32 numpy arrays.
    if kwargs.get("action_low", None) is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    if kwargs.get("action_high", None) is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)

    head = TD3Head(**kwargs).to("cpu")

    # Rollout workers are inference-only; keep networks in eval-like behavior.
    head.set_training(False)
    return head


# =============================================================================
# TD3Head
# =============================================================================
class TD3Head(DeterministicActorCriticHead):
    """
    TD3 policy head: deterministic actor + twin critics + target networks.

    Overview
    --------
    TD3 (Twin Delayed DDPG) uses:
    - Deterministic actor: :math:`a = \\pi(s)`
    - Twin critics: :math:`Q_1(s,a)`, :math:`Q_2(s,a)`
    - Target networks: :math:`\\pi'`, :math:`Q_1'`, :math:`Q_2'`
    - Target policy smoothing (for critic targets) by adding clipped Gaussian noise
      to the target actor output.

    Separation of concerns
    ----------------------
    - This *head* owns network modules and I/O (save/load, Ray spec).
    - The *core* owns training logic (losses, optimizers, delayed policy update, etc.).
      The core may call:
        - ``head.target_action(...)`` for TD3 target smoothing
        - ``head.actor``, ``head.critic`` for forward passes
        - ``head.actor_target``, ``head.critic_target`` for target value computation

    Inherited utilities
    -------------------
    The base :class:`~model_free.common.policies.base_head.DeterministicActorCriticHead`
    is expected to provide:
    - ``self.device`` device management
    - ``set_training(training)`` mode toggle
    - ``hard_update(target, source)``, ``soft_update(target, source, tau)``
    - ``freeze_target(module)`` (disable gradients for target nets)
    - ``_to_tensor_batched(x)`` to normalize inputs to batched tensors
    - ``_clamp_action(a)`` to clamp to ``action_low/high`` if configured
    - Optional exploration noise handling in ``act(..., deterministic=False)``

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for actor and critics.
    activation_fn : Any, default=torch.nn.ReLU
        Torch activation class used inside MLPs (e.g., ``nn.ReLU``).
    init_type : str, default="orthogonal"
        Initialization scheme identifier forwarded to your network modules.
    gain : float, default=1.0
        Initialization gain multiplier forwarded to your network modules.
    bias : float, default=0.0
        Initialization bias constant forwarded to your network modules.
    device : Union[str, torch.device], default=("cuda" if available else "cpu")
        Device for online and target networks.
    action_low : Optional[Union[np.ndarray, Sequence[float]]], default=None
        Lower bounds for actions (shape ``(action_dim,)``). If provided, ``action_high``
        must also be provided.
    action_high : Optional[Union[np.ndarray, Sequence[float]]], default=None
        Upper bounds for actions (shape ``(action_dim,)``). If provided, ``action_low``
        must also be provided.
    noise : Optional[Any], default=None
        Optional exploration noise object (typically used by ``act()`` when
        ``deterministic=False``). This is treated as runtime-only and is not serialized.
    noise_clip : Optional[float], default=None
        Optional clamp range for exploration noise (runtime-only).

    Attributes
    ----------
    actor : torch.nn.Module
        Deterministic policy network :math:`\\pi(s)`.
    critic : torch.nn.Module
        Twin critic network returning ``(q1, q2)`` each shaped ``(B,1)``.
    actor_target : torch.nn.Module
        Target actor network.
    critic_target : torch.nn.Module
        Target twin critic network.
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
        super().__init__(device=device)

        # -----------------------------
        # Configuration (for export/repro)
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # -----------------------------
        # Action bounds (optional)
        # -----------------------------
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32).reshape(-1)

        if (self.action_low is None) ^ (self.action_high is None):
            raise ValueError("action_low and action_high must be provided together, or both be None.")

        if self.action_low is not None:
            if self.action_low.shape[0] != self.action_dim or self.action_high.shape[0] != self.action_dim:
                raise ValueError(
                    f"action_low/high must have shape ({self.action_dim},), "
                    f"got {self.action_low.shape}, {self.action_high.shape}"
                )

        # Runtime-only exploration noise configuration (not serialized).
        self.noise = noise
        self.noise_clip = None if noise_clip is None else float(noise_clip)

        # -----------------------------
        # Actor (deterministic)
        # -----------------------------
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

        # -----------------------------
        # Critic (twin Q)
        # -----------------------------
        self.critic = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Target networks
        # -----------------------------
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

        # Sync targets once (hard copy), then freeze to prevent gradient updates.
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
        Compute the TD3 "smoothed" target action :math:`a'` for critic targets.

        TD3 target policy smoothing
        ---------------------------
        For critic target computation, TD3 adds clipped Gaussian noise to the
        *target actor* output:
        :math:`a' = \\mathrm{clip}(\\pi'(s') + \\mathrm{clip}(\\epsilon, -c, c))`,
        with :math:`\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)`.

        Parameters
        ----------
        next_obs : Any
            Next observations. Any input accepted by ``_to_tensor_batched``.
            Expected final tensor shape is ``(B, obs_dim)`` on ``self.device``.
        noise_std : float
            Standard deviation :math:`\\sigma` for Gaussian noise.
        noise_clip : float
            Clip range :math:`c` applied elementwise to the noise term.

        Returns
        -------
        next_action : torch.Tensor
            Smoothed action tensor of shape ``(B, action_dim)``. If action bounds
            are configured, the result is clamped to ``[action_low, action_high]``.
        """
        s2 = self._to_tensor_batched(next_obs)
        a2 = self.actor_target(s2)

        ns = float(noise_std)
        nc = float(noise_clip)

        if ns > 0.0:
            eps = ns * th.randn_like(a2)
            if nc > 0.0:
                eps = eps.clamp(-nc, nc)
            a2 = a2 + eps

        return self._clamp_action(a2)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor configuration in a JSON-serializable format.

        Returns
        -------
        kwargs : Dict[str, Any]
            JSON-friendly kwargs used for:
            - checkpoint metadata (debugging / reproducibility)
            - Ray policy reconstruction (``PolicyFactorySpec``)

        Notes
        -----
        - ``activation_fn`` is exported as a string name and later resolved via
          :func:`~model_free.common.utils.ray_utils._resolve_activation_fn`.
        - ``noise`` and ``noise_clip`` are intentionally excluded since they are
          typically runtime-only and may not be serializable.
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
        Save TD3Head weights and minimal configuration to a checkpoint.

        Parameters
        ----------
        path : str
            Output path. If ``path`` does not end with ``.pt``, the extension is appended.

        Saved payload
        -------------
        kwargs : Dict[str, Any]
            JSON-safe constructor kwargs (see ``_export_kwargs_json_safe``).
        actor : Dict[str, torch.Tensor]
            State dict of the online actor.
        critic : Dict[str, torch.Tensor]
            State dict of the online twin critics.
        actor_target : Dict[str, torch.Tensor]
            State dict of the target actor.
        critic_target : Dict[str, torch.Tensor]
            State dict of the target twin critics.
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
        Load TD3Head weights from a checkpoint produced by :meth:`save`.

        Parameters
        ----------
        path : str
            Checkpoint path. If ``path`` does not end with ``.pt``, the extension is appended.

        Raises
        ------
        ValueError
            If the checkpoint does not contain the minimal expected keys.

        Notes
        -----
        - If target networks are missing (older checkpoints), they are reconstructed
          via hard copy from the corresponding online networks.
        - Target networks are frozen after loading to avoid accidental optimization.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized TD3Head checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("actor_target", None) is not None:
            self.actor_target.load_state_dict(ckpt["actor_target"])
        else:
            self.hard_update(self.actor_target, self.actor)

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        self.freeze_target(self.actor_target)
        self.freeze_target(self.critic_target)

        # Targets should always stay in eval mode.
        self.actor_target.eval()
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-serializable specification to reconstruct this head on workers.

        Returns
        -------
        spec : PolicyFactorySpec
            A spec containing:
            - ``entrypoint``: importable module-level factory function
            - ``kwargs``: JSON-safe constructor kwargs

        Notes
        -----
        Ray requires the entrypoint to be module-level (import path resolution),
        and kwargs must avoid non-serializable objects (tensors, callables, etc.).
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_td3_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
