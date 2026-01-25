from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateActionValueNetwork
from model_free.common.utils.ray_utils import PolicyFactorySpec, make_entrypoint, resolve_activation_fn
from model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_redq_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Ray worker-side factory: build a REDQHead instance on CPU.

    Why this exists
    ---------------
    In Ray multi-process/multi-node rollouts, each worker must be able to
    reconstruct the policy module from an "entrypoint + kwargs" specification.
    Therefore this factory must be defined at module-level (picklable).

    Behavior / conventions
    ----------------------
    - Forces device="cpu" regardless of incoming kwargs to avoid GPU contention.
    - Converts activation_fn from a serialized form (string/name) into a torch.nn class.
    - Returns a fully constructed REDQHead with training mode set to eval-like behavior
      for rollout usage (set_training(False)).
    """
    kwargs = dict(kwargs)

    # Force CPU for rollout workers (stable, cheap, avoids accidental GPU usage)
    kwargs["device"] = "cpu"

    # Convert activation spec (e.g. "relu") -> nn.ReLU
    kwargs["activation_fn"] = resolve_activation_fn(kwargs.get("activation_fn", None))

    head = REDQHead(**kwargs).to("cpu")

    # Rollout workers typically should not keep dropout/bn in train mode.
    # Also target nets should always remain eval mode anyway.
    head.set_training(False)
    return head


# =============================================================================
# REDQHead
# =============================================================================
class REDQHead(OffPolicyContinuousActorCriticHead):
    """
    REDQ Head (Actor + Critic Ensemble + Target Critic Ensemble).

    REDQ summary
    ------------
    REDQ uses:
      - A stochastic actor (SAC-style squashed Gaussian policy)
      - An ensemble of Q critics {Q_i} (online critics)
      - A corresponding ensemble of target critics {Q_i^t}

    A key REDQ detail is the target value computation:
      - randomly sample a subset of target critics
      - take the minimum over the subset to reduce overestimation bias

    Contract (expected by OffPolicyAlgorithm)
    ----------------------------------------
    This head exposes the standard OffPolicyActorCritic interface:
      - device
      - set_training(training)
      - act(obs, deterministic=False)
      - sample_action_and_logp(obs)
      - q_values_all(obs, action) -> List[Tensor(B,1)]
      - q_values_target_subset_min(obs, action, subset_size=None) -> Tensor(B,1)
      - hard_update_target()
      - soft_update_target(tau)
      - save(path), load(path)
      - get_ray_policy_factory_spec()

    Shapes
    ------
    - obs:    (B, obs_dim)
    - action: (B, action_dim)
    - Q(s,a): (B, 1)
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
        # REDQ ensemble params
        num_critics: int = 10,
        num_target_subset: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension.
        action_dim : int
            Action dimension.
        hidden_sizes : Sequence[int]
            Shared MLP hidden sizes for actor and critics.
        activation_fn : Any
            Torch activation class (e.g., nn.ReLU).
        init_type : str
            Network init scheme used by your network modules.
        gain : float
            Init gain multiplier.
        bias : float
            Init bias constant.
        device : Union[str, torch.device]
            Device for online nets. Target nets are also allocated on this device.
        log_std_mode : str
            Policy log-std parameterization (e.g. "layer", "parameter").
        log_std_init : float
            Initial value for log standard deviation.
        num_critics : int
            Number of critic networks in the ensemble.
        num_target_subset : int
            Subset size used for REDQ target min-reduction.
            Must satisfy: 1 <= num_target_subset <= num_critics
        """
        super().__init__(device=device)

        # Store architecture/meta parameters
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        # Store init / activation config
        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        # Actor distribution config (SAC-style)
        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # REDQ ensemble hyperparameters
        self.num_critics = int(num_critics)
        self.num_target_subset = int(num_target_subset)

        # Defensive validation (fail fast)
        if self.num_critics <= 0:
            raise ValueError(f"num_critics must be positive, got {self.num_critics}")
        if self.num_target_subset <= 0 or self.num_target_subset > self.num_critics:
            raise ValueError(
                f"num_target_subset must be in [1, {self.num_critics}], got {self.num_target_subset}"
            )

        # ---------------------------------------------------------------------
        # Actor: Squashed Gaussian policy (like SAC)
        # ---------------------------------------------------------------------
        # - outputs actions in [-1,1] if squash=True
        # - supports rsample() for reparameterization trick
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=True,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic ensemble (online): {Q_i}
        # ---------------------------------------------------------------------
        # Each critic estimates Q(s, a) -> (B,1)
        self.critics = nn.ModuleList(
            [
                StateActionValueNetwork(
                    state_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_sizes=self.hidden_sizes,
                    activation_fn=self.activation_fn,
                    init_type=self.init_type,
                    gain=self.gain,
                    bias=self.bias,
                ).to(self.device)
                for _ in range(self.num_critics)
            ]
        )

        # ---------------------------------------------------------------------
        # Target critic ensemble: {Q_i^t}
        # ---------------------------------------------------------------------
        # These are lagged copies of online critics.
        self.critics_target = nn.ModuleList(
            [
                StateActionValueNetwork(
                    state_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_sizes=self.hidden_sizes,
                    activation_fn=self.activation_fn,
                    init_type=self.init_type,
                    gain=self.gain,
                    bias=self.bias,
                ).to(self.device)
                for _ in range(self.num_critics)
            ]
        )

        self.critic = self.critics[0]
        self.critic_target = self.critics_target[0]

        # Initialize targets = online critics (hard copy)
        for q_t, q in zip(self.critics_target, self.critics):
            self.hard_update(q_t, q)

        # Freeze target params so:
        # - they are not updated by optimizer
        # - they are safe from accidental gradient flows
        for q_t in self.critics_target:
            self.freeze_target(q_t)

    # =============================================================================
    # Modes
    # =============================================================================
    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for online networks.

        Convention
        ----------
        - Online actor/critics follow `training` flag.
        - Target critics always remain in eval mode (and frozen).

        Notes
        -----
        Even if the user calls set_training(True), target critics are kept in eval.
        """
        # Online actor follows training flag
        self.actor.train(training)

        # Online critics follow training flag
        for q in self.critics:
            q.train(training)

        # Targets remain eval always
        for q_t in self.critics_target:
            q_t.eval()

    # =============================================================================
    # Acting / sampling
    # =============================================================================
    @th.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> th.Tensor:
        """
        Compute an action from the current policy.

        Parameters
        ----------
        obs : Any
            Observation (np array / torch tensor / list) convertible via _to_tensor_batched.
        deterministic : bool
            - True: use mean action (or deterministic path) for evaluation
            - False: sample from policy distribution (stochastic)

        Returns
        -------
        action : torch.Tensor
            Shape (B, action_dim), on self.device.
        """
        s = self._to_tensor_batched(obs)
        action, _info = self.actor.act(s, deterministic=deterministic)
        return action

    # =============================================================================
    # Q interfaces (REDQ-specific)
    # =============================================================================
    @th.no_grad()
    def q_values_all(self, obs: Any, action: Any) -> List[th.Tensor]:
        """
        Compute all online ensemble Q-values.

        Returns
        -------
        qs : List[torch.Tensor]
            List length = num_critics
            Each tensor has shape (B,1)
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return [q(s, a) for q in self.critics]

    @th.no_grad()
    def q_values_target_all(self, obs: Any, action: Any) -> List[th.Tensor]:
        """
        Compute all target ensemble Q-values.

        Returns
        -------
        qs_t : List[torch.Tensor]
            List length = num_critics
            Each tensor has shape (B,1)
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return [q_t(s, a) for q_t in self.critics_target]

    @th.no_grad()
    def q_values_target_subset_min(
        self,
        obs: Any,
        action: Any,
        *,
        subset_size: Optional[int] = None,
    ) -> th.Tensor:
        """
        REDQ target: sample a random subset of target critics and return the minimum.

        This implements the REDQ "min over random subset" trick:
          Q_target(s,a) = min_{i in subset} Q_i^t(s,a)

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Action batch.
        subset_size : Optional[int]
            If None, uses self.num_target_subset.
            Must satisfy 1 <= subset_size <= num_critics.

        Returns
        -------
        q_min : torch.Tensor
            Shape (B,1)
        """
        k = int(self.num_target_subset if subset_size is None else subset_size)
        if k <= 0 or k > self.num_critics:
            raise ValueError(f"subset_size must be in [1, {self.num_critics}], got {k}")

        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)

        # We sample indices on the same device to keep behavior consistent between CPU/GPU.
        # randperm is deterministic under torch RNG state if seeds are controlled.
        idx = th.randperm(self.num_critics, device=self.device)[:k].tolist()

        # Collect Q-values from subset -> stack -> min
        qs = [self.critics_target[i](s, a) for i in idx]   # each: (B,1)
        q_stack = th.stack(qs, dim=0)                      # (k, B, 1)
        q_min = th.min(q_stack, dim=0).values              # (B,1)
        return q_min

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe format.

        Motivation
        ----------
        This enables:
        - clean checkpointing with enough metadata to rebuild the head
        - Ray factory specs (kwargs must be serializable)
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
            "num_critics": int(self.num_critics),
            "num_target_subset": int(self.num_target_subset),
        }

    def save(self, path: str) -> None:
        """
        Save actor + ensemble critics + ensemble targets into a single checkpoint.

        Format
        ------
        {
          "kwargs": {...json safe...},
          "actor": state_dict,
          "critics": [state_dict, ...] length = num_critics,
          "critics_target": [state_dict, ...] length = num_critics,
        }
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critics": [q.state_dict() for q in self.critics],
            "critics_target": [q_t.state_dict() for q_t in self.critics_target],
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load actor + critic ensembles from a checkpoint.

        Notes
        -----
        - If target critics are missing in the checkpoint, targets are rebuilt via hard_update_target().
        - After loading, targets are frozen + set to eval.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critics" not in ckpt:
            raise ValueError(f"Unrecognized REDQHead checkpoint format at: {path}")

        # Load actor weights
        self.actor.load_state_dict(ckpt["actor"])

        # Load online critics weights
        critics_sd = ckpt["critics"]
        if len(critics_sd) != len(self.critics):
            raise ValueError(
                f"Critic ensemble size mismatch: ckpt={len(critics_sd)} vs model={len(self.critics)}"
            )
        for q, sd in zip(self.critics, critics_sd):
            q.load_state_dict(sd)

        # Load target ensemble if present; otherwise reconstruct from online critics
        critics_t_sd = ckpt.get("critics_target", None)
        if critics_t_sd is not None:
            if len(critics_t_sd) != len(self.critics_target):
                raise ValueError(
                    f"Target ensemble size mismatch: ckpt={len(critics_t_sd)} vs model={len(self.critics_target)}"
                )
            for q_t, sd in zip(self.critics_target, critics_t_sd):
                q_t.load_state_dict(sd)
        else:
            # If targets were not saved, sync them directly from online critics
            for q_t, q in zip(self.critics_target, self.critics):
                self.hard_update(q_t, q)

        for q_t in self.critics_target:
            self.freeze_target(q_t)

        for q_t in self.critics_target:
            q_t.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-serializable spec to reconstruct this REDQHead on workers.

        The spec contains:
        - entrypoint: module-level function pointer
        - kwargs: JSON-safe constructor args
        """
        return PolicyFactorySpec(
            entrypoint=make_entrypoint(build_redq_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )