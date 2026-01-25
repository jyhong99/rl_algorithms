from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import QuantileStateActionValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_tqc_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build TQCHead on CPU.

    Why this exists
    ---------------
    In Ray multi-worker rollouts, you generally want:
      - worker-side policies forced onto CPU (GPU reserved for learner)
      - JSON-safe kwargs (activation_fn stored as string) re-resolved on worker

    Notes
    -----
    - device is overridden to "cpu" for worker stability and lower GPU contention
    - activation_fn is resolved from a string/name into a torch.nn module class
      (because kwargs are serialized through JSON / Ray plasma)
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = TQCHead(**kwargs).to("cpu")
    head.set_training(False)  # ensure eval-mode behavior on workers
    return head


# =============================================================================
# TQCHead (config-free)
# =============================================================================
class TQCHead(OffPolicyContinuousActorCriticHead):
    """
    TQC Head (Actor + Quantile Critic Ensemble + Target Critic Ensemble)

    What it contains
    ----------------
    - actor: squashed Gaussian policy (SAC-style)
    - critic: quantile critic ensemble producing Z(s,a) (distributional Q)
    - critic_target: target copy of critic (Polyak / hard update)

    Contract (for OffPolicyAlgorithm)
    --------------------------------
    - device
    - set_training(training)
    - act(obs, deterministic=False)
    - sample_action_and_logp(obs) -> (action, logp)
    - quantiles(obs, action) -> (B, C, N)
    - quantiles_target(obs, action) -> (B, C, N)
    - hard_update_target() / soft_update_target(tau) (inherited helpers)
    - save(path), load(path)
    - get_ray_policy_factory_spec()
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
        # actor distribution params (SAC-like)
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
        # quantile critic params
        n_quantiles: int = 25,
        n_nets: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension.
        action_dim : int
            Action dimension.
        hidden_sizes : Sequence[int]
            Shared MLP hidden layer sizes for both actor and critic networks.
        activation_fn : Any
            Activation class (e.g., nn.ReLU). If serialized for Ray, store name and
            re-resolve via resolve_activation_fn on the worker.
        init_type : str
            Weight initialization strategy used by your network builders.
        gain : float
            Gain parameter forwarded to init logic (commonly orthogonal gain).
        bias : float
            Bias initialization value forwarded to init logic.
        device : str or torch.device
            Target device for learner-side head (workers override to CPU).
        log_std_mode : str
            Actor log-std parameterization mode (e.g., "layer", "state_dependent", ...),
            depends on your ContinuousPolicyNetwork implementation.
        log_std_init : float
            Initial log standard deviation for the Gaussian policy.
        n_quantiles : int
            Number of quantiles per critic head (N).
        n_nets : int
            Number of critic ensemble members (C).
        """
        super().__init__(device=device)

        # ---- basic shapes / config ----
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # ---- actor distribution hyperparams ----
        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---- distributional critic hyperparams ----
        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be positive, got {self.n_quantiles}")
        if self.n_nets <= 0:
            raise ValueError(f"n_nets must be positive, got {self.n_nets}")

        # ---------------------------------------------------------------------
        # Actor: Squashed Gaussian policy (SAC-style)
        #
        # - outputs distribution parameters (mean + log_std)
        # - sampling uses rsample + tanh bijector correction (usually implemented in base head)
        # ---------------------------------------------------------------------
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=True,  # tanh-squash to keep actions in [-1, 1] before any rescale
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic: Quantile ensemble
        #
        # Returns: Z(s,a) with shape (B, C, N)
        #   B = batch size
        #   C = n_nets (ensemble members)
        #   N = n_quantiles (atoms)
        # ---------------------------------------------------------------------
        self.critic = QuantileStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_quantiles=self.n_quantiles,
            n_nets=self.n_nets,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Target critic: same architecture, updated from critic by hard/soft update.
        #
        # NOTE:
        # - your existing line: bias=self.gain if False else self.bias
        #   is a no-op "explicitness" trick. It's harmless but confusing.
        #   Recommended: just pass bias=self.bias for clarity.
        # ---------------------------------------------------------------------
        self.critic_target = QuantileStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_quantiles=self.n_quantiles,
            n_nets=self.n_nets,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target = online critic, then freeze to avoid accidental grads.
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Quantiles (TQC-specific)
    # =============================================================================
    def quantiles(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute critic quantiles Z(s,a).

        Parameters
        ----------
        obs : Any
            Observation(s). Accepts numpy arrays, tensors, or single obs depending
            on your _to_tensor_batched implementation.
        action : Any
            Action(s). Same batching rules as obs.

        Returns
        -------
        z : torch.Tensor
            Quantiles with shape (B, C, N) where:
              B=batch size, C=n_nets, N=n_quantiles
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def quantiles_target(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute target critic quantiles Z_t(s,a).

        Returns
        -------
        z_t : torch.Tensor
            Target quantiles with shape (B, C, N).
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe form.

        Why:
        - Ray/serialization prefers primitive types.
        - activation_fn is exported as a string and re-resolved on worker.
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
            "n_quantiles": int(self.n_quantiles),
            "n_nets": int(self.n_nets),
        }

    def save(self, path: str) -> None:
        """
        Save head weights and JSON-safe kwargs.

        File format
        -----------
        torch.save({
          "kwargs": {...},
          "actor": actor_state_dict,
          "critic": critic_state_dict,
          "critic_target": critic_target_state_dict,
        }, path + ".pt")
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
        Load head weights from a checkpoint.

        Behavior
        --------
        - Loads actor + critic.
        - Loads critic_target if present; otherwise syncs from critic.
        - Always re-freezes critic_target afterwards.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized TQCHead checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        # Ensure target is frozen even if checkpoint was created differently.
        self.freeze_target(self.critic_target)
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a JSON-serializable spec used by Ray workers to reconstruct this head.

        - entrypoint: a module-level function (required by your resolver)
        - kwargs: JSON-safe exported kwargs
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_tqc_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
