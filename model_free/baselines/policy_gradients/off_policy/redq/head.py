from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch as th
import torch.nn as nn

from model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from model_free.common.networks.value_networks import StateActionValueNetwork
from model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead
from model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
)


# =============================================================================
# Ray worker factory (MUST be module-level for your entrypoint resolver)
# =============================================================================
def build_redq_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a :class:`REDQHead` instance on a Ray worker (CPU-only).

    In Ray rollouts (multi-process / multi-node), worker processes frequently
    reconstruct policy modules from a serialized "factory spec" containing:

    - an importable *module-level* entrypoint (this function)
    - a JSON-serializable ``kwargs`` mapping

    This function enforces worker-safe defaults and converts serialized fields
    into their runtime forms.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`REDQHead`.

        Notes
        -----
        - ``device`` is forcibly overridden to ``"cpu"`` to avoid GPU contention
          and accidental GPU allocation on rollout workers.
        - ``activation_fn`` may arrive as a string (e.g., ``"relu"``); it is
          resolved into an activation constructor via :func:`_resolve_activation_fn`.

    Returns
    -------
    torch.nn.Module
        Constructed :class:`REDQHead` placed on CPU and set to inference behavior
        via ``set_training(False)`` (best-effort; depends on base class semantics).

    See Also
    --------
    REDQHead.get_ray_policy_factory_spec :
        Produces the factory spec that references this entrypoint.
    """
    kwargs = dict(kwargs)

    # Force CPU for rollout workers (stable, cheap, avoids accidental GPU usage).
    kwargs["device"] = "cpu"

    # Convert activation spec (e.g. "relu") -> nn.ReLU.
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))

    head = REDQHead(**kwargs).to("cpu")

    # Rollout workers should not keep dropout/bn in train mode; targets are eval anyway.
    head.set_training(False)
    return head


# =============================================================================
# REDQHead
# =============================================================================
class REDQHead(OffPolicyContinuousActorCriticHead):
    r"""
    REDQ head: stochastic actor + critic ensemble + target critic ensemble.

    REDQ overview
    -------------
    REDQ (Randomized Ensembled Double Q-learning) is an off-policy actor-critic
    method that combines a SAC-style stochastic actor with an ensemble of Q
    critics. For target value estimation, REDQ samples a random subset of
    target critics and takes the minimum over that subset to reduce
    overestimation bias.

    This head wires:
    - a squashed Gaussian actor :math:`\pi_\theta(a\mid s)` (SAC-like)
    - an ensemble of critics :math:`\{Q_{\phi_i}(s,a)\}_{i=1}^N`
    - an ensemble of target critics :math:`\{Q_{\bar{\phi}_i}(s,a)\}_{i=1}^N`

    Target reduction (REDQ trick)
    -----------------------------
    Let :math:`\mathcal{I}` be a random subset of size :math:`K` drawn uniformly
    from ``{1, ..., N}``. REDQ forms a target critic value as:

    .. math::
        Q_{\text{targ}}(s,a) = \min_{i \in \mathcal{I}} Q_{\bar{\phi}_i}(s,a)

    This differs from classic Double-Q (min over 2) by using a larger ensemble and
    randomized subsets.

    Expected interface
    ------------------
    This head follows the project's off-policy continuous actor-critic interface.
    Typical consumers (e.g., OffPolicyAlgorithm and REDQCore-like update engines)
    rely on the following:

    Attributes
    ----------
    actor : torch.nn.Module
        Stochastic policy network (squashed Gaussian).
    critics : torch.nn.ModuleList
        Online critic ensemble, each mapping (s,a) -> (B,1).
    critics_target : torch.nn.ModuleList
        Target critic ensemble, frozen and eval-only.
    device : torch.device
        Device used for computation.

    Methods
    -------
    set_training(training)
        Toggle training/eval for online networks while keeping targets eval.
    act(obs, deterministic=False)
        Compute actions for rollout/evaluation.
    q_values_all(obs, action)
        Evaluate all online critics.
    q_values_target_all(obs, action)
        Evaluate all target critics.
    q_values_target_subset_min(obs, action, subset_size=None)
        Sample subset of target critics and return min value (REDQ reduction).
    save(path), load(path)
        Persist actor and ensembles.
    get_ray_policy_factory_spec()
        Provide Ray reconstruction spec (entrypoint + JSON-safe kwargs).

    Shapes
    ------
    - obs:    ``(B, obs_dim)``
    - action: ``(B, action_dim)``
    - Q(s,a): ``(B, 1)``
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
        # Actor distribution params (SAC-like).
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
        # REDQ ensemble params.
        num_critics: int = 10,
        num_target_subset: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation (state) vector dimension.
        action_dim : int
            Action vector dimension.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer widths used for both actor and critics.
        activation_fn : Any, default=torch.nn.ReLU
            Activation constructor used in MLP blocks.

            Notes
            -----
            - When reconstructed via Ray, this may be provided as a string name and
              should be resolved by the worker factory.
        init_type : str, default="orthogonal"
            Initialization scheme forwarded to network constructors.
        gain : float, default=1.0
            Initialization gain forwarded to network constructors.
        bias : float, default=0.0
            Bias initialization forwarded to network constructors.
        device : str or torch.device, default=("cuda" if available else "cpu")
            Device for online and target networks.
        log_std_mode : str, default="layer"
            Policy log-standard-deviation parameterization mode used by
            :class:`~model_free.common.networks.policy_networks.ContinuousPolicyNetwork`.
        log_std_init : float, default=-0.5
            Initial value for the log standard deviation.
        num_critics : int, default=10
            Number of online critics in the ensemble.
        num_target_subset : int, default=2
            Subset size ``K`` used for REDQ target min-reduction.

            Must satisfy
            ------------
            ``1 <= num_target_subset <= num_critics``.

        Raises
        ------
        ValueError
            If ``num_critics <= 0`` or ``num_target_subset`` violates constraints.
        """
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Store architecture/meta parameters
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        # Init / activation config
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
        # Actor: squashed Gaussian policy (SAC-like)
        # ---------------------------------------------------------------------
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

        # Optional compatibility aliases: some generic cores expect single critic attrs.
        self.critic = self.critics[0]
        self.critic_target = self.critics_target[0]

        # Initialize targets from online critics (hard copy).
        for q_t, q in zip(self.critics_target, self.critics):
            self.hard_update(q_t, q)

        # Freeze target params to prevent optimizer updates and accidental grads.
        for q_t in self.critics_target:
            self.freeze_target(q_t)

    # =============================================================================
    # Modes
    # =============================================================================
    def set_training(self, training: bool) -> None:
        """
        Set training/eval mode for online networks.

        Parameters
        ----------
        training : bool
            If ``True``, sets online networks to training mode.
            If ``False``, sets online networks to evaluation mode.

        Notes
        -----
        - Online actor and online critics follow the provided ``training`` flag.
        - Target critics remain in evaluation mode regardless of ``training``
          and are expected to remain frozen.
        """
        self.actor.train(training)
        for q in self.critics:
            q.train(training)

        # Targets remain eval always.
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
            Observation batch. Accepts numpy arrays / torch tensors / lists that are
            convertible via the base head's ``_to_tensor_batched`` helper.
        deterministic : bool, default=False
            Action selection mode:

            - ``True``: use the deterministic action (typically the mean of the policy).
            - ``False``: sample an action from the policy distribution.

        Returns
        -------
        torch.Tensor
            Action tensor of shape ``(B, action_dim)`` on ``self.device``.
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
        Evaluate all online critics :math:`\\{Q_i(s,a)\\}`.

        Parameters
        ----------
        obs : Any
            Observation batch convertible via ``_to_tensor_batched``.
        action : Any
            Action batch convertible via ``_to_tensor_batched``.

        Returns
        -------
        List[torch.Tensor]
            List of length ``num_critics``.
            Each element is a tensor of shape ``(B, 1)``.
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return [q(s, a) for q in self.critics]

    @th.no_grad()
    def q_values_target_all(self, obs: Any, action: Any) -> List[th.Tensor]:
        """
        Evaluate all target critics :math:`\\{Q_i^t(s,a)\\}`.

        Parameters
        ----------
        obs : Any
            Observation batch convertible via ``_to_tensor_batched``.
        action : Any
            Action batch convertible via ``_to_tensor_batched``.

        Returns
        -------
        List[torch.Tensor]
            List of length ``num_critics``.
            Each element is a tensor of shape ``(B, 1)``.
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
        Compute the REDQ target value: min over a random subset of target critics.

        This implements the REDQ randomized min reduction:

        .. math::
            Q_{\\text{targ}}(s,a) = \\min_{i \\in \\mathcal{I}} Q_i^t(s,a)

        where :math:`\\mathcal{I}` is a uniformly sampled subset of size ``K``.

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Action batch.
        subset_size : int, optional
            Subset size ``K``. If ``None``, uses ``self.num_target_subset``.

            Constraints
            -----------
            ``1 <= subset_size <= num_critics``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, 1)`` containing the minimum Q-values over the
            sampled subset.

        Raises
        ------
        ValueError
            If ``subset_size`` violates constraints.
        """
        k = int(self.num_target_subset if subset_size is None else subset_size)
        if k <= 0 or k > self.num_critics:
            raise ValueError(f"subset_size must be in [1, {self.num_critics}], got {k}")

        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)

        # Sample subset indices on the same device for consistent CPU/GPU behavior.
        idx = th.randperm(self.num_critics, device=self.device)[:k].tolist()

        qs = [self.critics_target[i](s, a) for i in idx]  # each: (B,1)
        q_stack = th.stack(qs, dim=0)  # (k, B, 1)
        q_min = th.min(q_stack, dim=0).values  # (B,1)
        return q_min

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable format.

        This metadata supports:
        - checkpoints with sufficient configuration context for debugging
        - Ray worker reconstruction (kwargs must be JSON-safe)

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor kwargs. Activation functions are exported as a
            stable string name; all numeric fields are cast to Python scalars.
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
        Save actor + critic ensembles into a single ``.pt`` checkpoint.

        Parameters
        ----------
        path : str
            Output checkpoint path. If the path does not end with ``.pt``, the
            suffix is appended.

        Stored Payload
        --------------
        The checkpoint is stored via ``torch.save`` as a dict with keys:

        - ``"kwargs"`` : JSON-safe constructor metadata (see ``_export_kwargs_json_safe``)
        - ``"actor"`` : ``actor.state_dict()``
        - ``"critics"`` : list of ``state_dict`` (length = ``num_critics``)
        - ``"critics_target"`` : list of ``state_dict`` (length = ``num_critics``)

        Notes
        -----
        Optimizer state is not stored here; that is typically owned by the core/algorithm.
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
        Load actor + critic ensembles from a ``.pt`` checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path produced by :meth:`save`. If the path does not end
            with ``.pt``, the suffix is appended.

        Raises
        ------
        ValueError
            If the checkpoint format is not recognized or ensemble sizes mismatch.

        Notes
        -----
        - Loads tensors onto ``self.device`` via ``map_location=self.device``.
        - If target critics are missing in the checkpoint, targets are reconstructed
          by hard-copying online critics.
        - After loading, target critics are frozen and set to eval.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critics" not in ckpt:
            raise ValueError(f"Unrecognized REDQHead checkpoint format at: {path}")

        # Actor
        self.actor.load_state_dict(ckpt["actor"])

        # Online critics
        critics_sd = ckpt["critics"]
        if len(critics_sd) != len(self.critics):
            raise ValueError(
                f"Critic ensemble size mismatch: ckpt={len(critics_sd)} vs model={len(self.critics)}"
            )
        for q, sd in zip(self.critics, critics_sd):
            q.load_state_dict(sd)

        # Target critics
        critics_t_sd = ckpt.get("critics_target", None)
        if critics_t_sd is not None:
            if len(critics_t_sd) != len(self.critics_target):
                raise ValueError(
                    f"Target ensemble size mismatch: ckpt={len(critics_t_sd)} vs model={len(self.critics_target)}"
                )
            for q_t, sd in zip(self.critics_target, critics_t_sd):
                q_t.load_state_dict(sd)
        else:
            # If targets were not saved, sync directly from online critics.
            for q_t, q in zip(self.critics_target, self.critics):
                self.hard_update(q_t, q)

        for q_t in self.critics_target:
            self.freeze_target(q_t)
            q_t.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Create a Ray-serializable spec for reconstructing this head on workers.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint``: module-level function used by Ray workers
            - ``kwargs``: JSON-safe constructor kwargs

        Notes
        -----
        - The worker entrypoint overrides ``device`` to CPU.
        - ``kwargs`` must remain JSON-serializable.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_redq_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
