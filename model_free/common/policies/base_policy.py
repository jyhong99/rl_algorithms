from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch as th

from ..utils.common_utils import _is_scalar_like, _to_scalar
from ..utils.ray_utils import PolicyFactorySpec


class BaseAlgorithm:
    """
    Shared base class for algorithm *drivers* (e.g., on-policy/off-policy trainers).

    This class is intentionally minimal: it does not interact with environments
    and does not implement rollout/replay logic. Instead, it glues together two
    duck-typed components:

    - **head**: inference-facing object (actor/critic/q head) responsible for
      action selection and (optionally) providing a Ray policy factory spec.

    - **core**: update engine responsible for optimization (losses, optimizer,
      scheduler, gradient scaling/clipping, target updates, etc.).

    Responsibilities
    ----------------
    - Owns `head` and `core`.
    - Normalizes `device` for consistent checkpoint `torch.load(map_location=...)`.
    - Provides common `set_training()` and `act()` passthrough.
    - Implements `save()` / `load()` for checkpointing head/core state.
    - Filters/normalizes scalar metrics for logging.

    Assumed interfaces (duck-typed)
    -------------------------------
    head:
      - device: torch.device or str (optional)
      - set_training(training: bool) -> None
      - act(obs, deterministic: bool = False) -> Any
      - state_dict() / load_state_dict(...) (optional)
      - get_ray_policy_factory_spec() -> PolicyFactorySpec (optional)

    core:
      - state_dict() / load_state_dict(...) (optional)

    Notes
    -----
    - This class does **not** define the environment loop. See BasePolicyAlgorithm
      for an env-facing protocol.
    - `act()` is tolerant to heads that do not accept `deterministic` as a keyword:
      it tries keyword first, then falls back to positional call.
    """

    #: Whether the algorithm is off-policy (for logging/UI purposes).
    is_off_policy: bool = False

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        device: Optional[Union[str, th.device]] = None,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Head object (duck-typed). Expected to expose `set_training` and `act`.
        core : Any
            Core/update engine object (duck-typed).
        device : Optional[Union[str, torch.device]], optional
            Device used for checkpoint loading (`map_location`) and any algorithm-
            level device bookkeeping. If None, falls back to `head.device` if
            present; otherwise "cpu".
        """
        self.head = head
        self.core = core

        # Normalize device (important for torch.load map_location consistency)
        if device is None:
            device = getattr(head, "device", "cpu")
        self.device: th.device = device if isinstance(device, th.device) else th.device(str(device))

    # ------------------------------------------------------------------
    # Modes / action selection
    # ------------------------------------------------------------------
    def set_training(self, training: bool) -> None:
        """
        Set train/eval mode for the head modules.

        Parameters
        ----------
        training : bool
            If True, set modules to train mode; if False, set to eval mode.

        Raises
        ------
        AttributeError
            If `head` does not expose `set_training(training: bool)`.

        Notes
        -----
        - Optimizer/scheduler state and AMP/scaler are typically core-owned.
          This method only toggles the head networks' train/eval mode.
        """
        fn = getattr(self.head, "set_training", None)
        if not callable(fn):
            raise AttributeError("head has no set_training(training: bool).")
        fn(bool(training))

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """
        Select an action using the underlying head.

        Parameters
        ----------
        obs : Any
            Observation(s) in environment-native format (tensor/ndarray/list/etc.).
        deterministic : bool, default=False
            If True, disables exploration where applicable (e.g., mean action,
            greedy action).

        Returns
        -------
        action : Any
            Action in env-native format or tensor/ndarray depending on the head.

        Raises
        ------
        AttributeError
            If `head` does not expose `act`.

        Notes
        -----
        Some heads implement `act(obs, deterministic=...)`, while others accept
        `deterministic` positionally only. We try keyword first, then fallback
        to positional to be robust across implementations.
        """
        fn = getattr(self.head, "act", None)
        if not callable(fn):
            raise AttributeError("head has no act(obs, deterministic=...).")

        try:
            return fn(obs, deterministic=bool(deterministic))
        except TypeError:
            return fn(obs, bool(deterministic))

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_scalar_metrics(
        metrics_any: Any,
        *,
        drop_non_finite: bool = True,
    ) -> Dict[str, float]:
        """
        Filter a metrics mapping to a float-only dict (for logging).

        Parameters
        ----------
        metrics_any : Any
            Typically a Mapping[str, Any] produced by a core update step.
            If not a Mapping, returns an empty dict.
        drop_non_finite : bool, default=True
            If True, drops NaN/Inf values.

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of `{name: float_value}` containing only scalar-like metrics.

        Notes
        -----
        Relies on project utilities:
        - `_is_scalar_like(x)` identifies scalar-like values (Python numbers, 0-d tensors,
          numpy scalars, etc.).
        - `_to_scalar(x)` converts scalar-like objects to a Python float-like value.
        """
        metrics: Dict[str, Any] = dict(metrics_any) if isinstance(metrics_any, Mapping) else {}
        out: Dict[str, float] = {}

        for k, v in metrics.items():
            if not _is_scalar_like(v):
                continue

            sv = _to_scalar(v)
            if sv is None:
                continue

            fv = float(sv)
            if drop_non_finite and (not th.isfinite(th.tensor(fv)).item()):
                continue

            out[str(k)] = fv

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save an algorithm checkpoint to disk.

        The saved payload is a dictionary that may include:
        - "meta": metadata describing the checkpoint format and component classes
        - "head_state_dict": `head.state_dict()` if available
        - "core_state": `core.state_dict()` if available

        Parameters
        ----------
        path : str
            Output file path. If the suffix is not ".pt", ".pt" is appended.

        Notes
        -----
        - Checkpoints are meant to be loaded into an algorithm instance with the
          same head/core structure.
        - Device info is stored in metadata for debugging and reproducibility.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        payload: Dict[str, Any] = {
            "meta": {
                "format_version": 1,
                "algorithm_class": self.__class__.__name__,
                "head_class": getattr(self.head, "__class__", type(self.head)).__name__,
                "core_class": getattr(self.core, "__class__", type(self.core)).__name__,
                "device": str(self.device),
            }
        }

        if callable(getattr(self.head, "state_dict", None)):
            payload["head_state_dict"] = self.head.state_dict()
        if callable(getattr(self.core, "state_dict", None)):
            payload["core_state"] = self.core.state_dict()

        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load an algorithm checkpoint from disk and restore head/core states.

        Parameters
        ----------
        path : str
            Checkpoint file path. If the suffix is not ".pt", ".pt" is appended.

        Raises
        ------
        ValueError
            If the checkpoint format is invalid, or required load methods are missing.

        Notes
        -----
        - Uses `map_location=self.device` to ensure compatibility across devices.
        - Restores head before core (typical), but there is no strict dependency.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        if "head_state_dict" in ckpt:
            if not callable(getattr(self.head, "load_state_dict", None)):
                raise ValueError("Checkpoint has head_state_dict but head has no load_state_dict().")
            self.head.load_state_dict(ckpt["head_state_dict"])

        core_state = ckpt.get("core_state", None)
        if core_state is not None:
            if not callable(getattr(self.core, "load_state_dict", None)):
                raise ValueError("Checkpoint has core_state but core has no load_state_dict().")
            self.core.load_state_dict(core_state)

    # ------------------------------------------------------------------
    # Ray hook passthrough
    # ------------------------------------------------------------------
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return `PolicyFactorySpec` used to construct Ray remote policies.

        Returns
        -------
        spec : PolicyFactorySpec
            Factory spec returned by `head.get_ray_policy_factory_spec()`.

        Raises
        ------
        ValueError
            If the head does not expose `get_ray_policy_factory_spec()`.
        """
        fn = getattr(self.head, "get_ray_policy_factory_spec", None)
        if not callable(fn):
            raise ValueError("head has no get_ray_policy_factory_spec().")
        return fn()


class BasePolicyAlgorithm(BaseAlgorithm):
    """
    Base class for algorithms that interact with environments (env-facing).

    This class extends BaseAlgorithm with a minimal environment-facing lifecycle
    protocol, without prescribing replay/rollout storage formats.

    Adds
    ----
    - Environment step counter (`env_steps`).

    Subclass contract
    -----------------
    Subclasses must implement:
    - setup(env): initialize any env-dependent buffers/state
    - on_env_step(transition): consume one transition
    - ready_to_update(): whether an update can be performed
    - update(): perform one training update and return scalar metrics

    Notes
    -----
    - The base class does not increment `_env_steps` automatically because the
      definition of "step" can vary (vector envs, frame-skip, multi-agent). A
      typical implementation increments it inside `on_env_step`.
    """

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        device: Optional[Union[str, th.device]] = None,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Head object used for action selection.
        core : Any
            Core/update engine.
        device : Optional[Union[str, torch.device]], optional
            See BaseAlgorithm.
        """
        super().__init__(head=head, core=core, device=device)
        self._env_steps: int = 0

    @property
    def env_steps(self) -> int:
        """
        Number of environment steps processed by this algorithm.

        Returns
        -------
        steps : int
            Step counter. Semantics are defined by subclasses (e.g., vectorized env
            may count transitions rather than raw env.step calls).
        """
        return int(self._env_steps)

    # ------------------------------------------------------------------
    # Minimal protocol
    # ------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        """
        Perform any environment-dependent setup.

        Parameters
        ----------
        env : Any
            Environment object (Gym/Gymnasium-like or custom).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Consume one environment transition.

        Parameters
        ----------
        transition : Dict[str, Any]
            One step of data. Recommended keys (convention):
            - "observations"
            - "actions"
            - "rewards"
            - "next_observations"
            - "dones"
            - "infos" (optional)

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Notes
        -----
        This method is typically where subclasses:
        - push into rollout buffer / replay buffer
        - update internal step counters
        - trigger target network updates (if core/head handles them here)
        """
        raise NotImplementedError

    def ready_to_update(self) -> bool:
        """
        Return True if the algorithm has enough data to perform `update()`.

        Returns
        -------
        ready : bool
            Whether `update()` can be called safely.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update(self) -> Dict[str, float]:
        """
        Perform one training update.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar training metrics suitable for logging (losses, KL, entropy, etc.).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError
