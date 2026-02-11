from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateValueNetwork
from model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)

# =============================================================================
# Ray worker factory (MUST be module-level)
# =============================================================================


def build_a2c_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build an :class:`A2CHead` inside a Ray rollout worker (CPU-only).

    Ray pattern and rationale
    -------------------------
    Ray often constructs policies through a pickled "entrypoint" callable plus
    JSON-serializable kwargs. The most robust pattern is therefore:

    - Define the factory at **module scope** (pickle-friendly).
    - Ensure kwargs are **JSON-safe** (strings, numbers, lists, dicts).
    - Force the resulting module onto **CPU** to avoid GPU/driver dependencies in
      remote rollout workers.

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments for :class:`A2CHead`. In Ray settings this
        dict is typically produced by :meth:`A2CHead._export_kwargs_json_safe`.

        Notes
        -----
        - ``activation_fn`` is expected to be serialized as a string name (or ``None``).
          This factory resolves it back into an activation class via
          :func:`_resolve_activation_fn`.
        - ``device`` is overwritten to ``"cpu"`` unconditionally.

    Returns
    -------
    torch.nn.Module
        An :class:`A2CHead` instance on CPU with training disabled
        (i.e., set to evaluation/inference mode via :meth:`set_training(False)`).

    See Also
    --------
    A2CHead.get_ray_policy_factory_spec
        Produces the (entrypoint, kwargs) pair used by Ray to reconstruct workers.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))

    head = A2CHead(**cfg).to("cpu")
    head.set_training(False)
    return head


# =============================================================================
# A2C head (CONTINUOUS ONLY)
# =============================================================================


class A2CHead(OnPolicyContinuousActorCriticHead):
    """
    Advantage Actor-Critic (A2C) network head for **continuous** action spaces.

    This module is a *network container* (actor + critic) used by on-policy
    algorithms (A2C/PPO-style). It intentionally does **not** implement the loss
    or update rule; those are handled by the algorithm/core.

    Architecture
    ------------
    Actor
        A diagonal Gaussian policy implemented by :class:`ContinuousPolicyNetwork`.

        - Uses ``squash=False`` to produce an **unsquashed** Gaussian.
        - Action bounding (e.g., tanh) is typically unnecessary for A2C/PPO if the
          environment handles action scaling/clipping, but you can change this by
          swapping the policy network configuration.

    Critic
        A state-value baseline :math:`V(s)` implemented by :class:`StateValueNetwork`.

    Inherited Interface
    -------------------
    The parent class :class:`OnPolicyContinuousActorCriticHead` is expected to provide
    (or require) the following common API:

    - ``act(obs, deterministic=False) -> action``
    - ``value_only(obs) -> value`` with shape ``(B, 1)``
    - ``evaluate_actions(obs, action) -> Mapping[str, Tensor]`` with keys like:

      - ``"value"``    : ``(B, 1)``
      - ``"log_prob"`` : ``(B, 1)`` or ``(B, action_dim)`` depending on distribution impl
      - ``"entropy"``  : ``(B, 1)`` or ``(B, action_dim)``

    Notes
    -----
    - This head stores enough constructor configuration to reconstruct itself
      in Ray workers and to emit informative checkpoint metadata.
    - The exported kwargs are kept JSON-safe (e.g., activation function is stored
      by name).
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
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
    ) -> None:
        """
        Initialize actor and critic networks for continuous-control A2C.

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation vector.
        action_dim : int
            Dimension of the continuous action vector.
        hidden_sizes : Sequence[int], default=(64, 64)
            Hidden layer sizes used for both actor and critic MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class (not an instance), e.g. ``nn.ReLU``, ``nn.Tanh``.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier understood by your network builders.
        gain : float, default=1.0
            Optional initialization gain passed to network builders.
        bias : float, default=0.0
            Optional initialization bias passed to network builders.
        device : str | torch.device, default="cpu"
            Torch device on which this head will live (e.g. ``"cpu"``, ``"cuda:0"``).
        log_std_mode : str, default="param"
            Gaussian log-standard-deviation parameterization mode used by the actor.

            Common options (implementation-dependent)
            - ``"param"``: a trainable parameter vector (state-independent).
            - other modes may exist in your codebase (e.g., state-dependent heads).
        log_std_init : float, default=-0.5
            Initial value for log standard deviation when ``log_std_mode="param"``.

        Raises
        ------
        ValueError
            If dimensions are invalid (may be raised downstream by network constructors).

        Notes
        -----
        - The actor uses an *unsquashed* Gaussian (``squash=False``).
        - The parent head owns ``self.device`` and the training/eval toggle via
          :meth:`set_training`.
        """
        super().__init__(device=device)

        # ---- store configuration (useful for save/load and Ray reconstruction)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # -----------------------------
        # Actor: Gaussian policy π(a|s)
        # -----------------------------
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,  # unsquashed Gaussian policy (A2C/PPO-style)
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # -----------------------------
        # Critic: state-value V(s)
        # -----------------------------
        self.critic = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

    # =============================================================================
    # Export / persistence / Ray integration
    # =============================================================================

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        This is used for
        -----------------
        1) Checkpoint metadata (so runs are reproducible and debuggable)
        2) Ray worker reconstruction (kwargs must be serializable)

        Returns
        -------
        dict
            JSON-safe kwargs that can be passed back into :class:`A2CHead` after
            resolving non-JSON fields (e.g., activation functions).

            Keys
            ----
            obs_dim : int
            action_space : str
                Always ``"continuous"`` for this head.
            action_dim : int
            hidden_sizes : list[int]
            activation_fn : str | None
                Activation function name as produced by ``_activation_to_name``.
            init_type : str
            gain : float
            bias : float
            device : str
            log_std_mode : str
            log_std_init : float
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_space": "continuous",
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
        Save a checkpoint to disk.

        Parameters
        ----------
        path : str
            Output path. If ``.pt`` is missing, it is appended automatically.

        Saved contents
        --------------
        kwargs : dict
            JSON-safe constructor config (useful for reconstruction and debugging).
        actor : dict
            ``state_dict`` for the actor network.
        critic : dict
            ``state_dict`` for the critic network.

        Notes
        -----
        - This method stores *weights + metadata*.
        - Reconstruction is typically done by creating a compatible instance
          (possibly from ``kwargs``) and then calling :meth:`load`.
        """
        out = path if path.endswith(".pt") else (path + ".pt")
        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            out,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the *existing* instance.

        Parameters
        ----------
        path : str
            Checkpoint path. If ``.pt`` is missing, it is appended automatically.

        Raises
        ------
        ValueError
            If the checkpoint format does not match the expected structure.

        Notes
        -----
        - This only loads weights into the current object.
        - It does **not** reconstruct architecture; dimensions must be compatible.
        - ``map_location`` uses ``self.device`` (as configured at construction time).
        """
        ckpt_path = path if path.endswith(".pt") else (path + ".pt")
        ckpt = th.load(ckpt_path, map_location=self.device)

        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-safe construction spec (entrypoint + JSON-safe kwargs).

        Returns
        -------
        PolicyFactorySpec
            A lightweight spec containing:
            - ``entrypoint``: picklable reference to :func:`build_a2c_head_worker_policy`
            - ``kwargs``: JSON-safe constructor arguments for worker-side reconstruction

        Notes
        -----
        - The worker factory forces CPU and resolves the activation function name.
        - This method should be called on the driver side (learner process).
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_a2c_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
