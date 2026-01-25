from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch as th

from ..utils.common_utils import to_scalar, is_scalar_like
from ..utils.ray_utils import PolicyFactorySpec


class BaseAlgorithm:
    """
    Shared base class for algorithm drivers (OnPolicyAlgorithm / OffPolicyAlgorithm).

    Responsibilities
    ----------------
    - Owns `head` and `core` objects (duck-typed)
    - Provides common `save()` / `load()` checkpointing
    - Passes through Ray policy factory hook (if available)
    - Filters / normalizes scalar metrics for logging

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
    - This class does NOT perform environment interaction by itself.
      See BasePolicyAlgorithm for env-facing protocol.
    """

    is_off_policy: bool = False

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        device: Optional[Union[str, th.device]] = None,
    ) -> None:
        self.head = head
        self.core = core

        # Normalize device (important for torch.load map_location consistency)
        if device is None:
            device = getattr(head, "device", "cpu")
        self.device: th.device = device if isinstance(device, th.device) else th.device(str(device))

    # ------------------------------------------------------------------
    # Modes / action
    # ------------------------------------------------------------------
    def set_training(self, training: bool) -> None:
        """
        Set train/eval mode for the head modules.

        Notes
        -----
        - Optimizer/scheduler modes are core-owned (if relevant).
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
            Observation(s) in env-native format.
        deterministic : bool, optional
            If True, disable exploration (where applicable).

        Returns
        -------
        action : Any
            Action in env-native format or tensor/ndarray depending on head.

        Notes
        -----
        Head implementations are not always consistent in whether they accept
        `deterministic` as a keyword. We try keyword call first, then fallback.
        """
        fn = getattr(self.head, "act", None)
        if not callable(fn):
            raise AttributeError("head has no act(obs, deterministic=...).")

        try:
            return fn(obs, deterministic=bool(deterministic))
        except TypeError:
            # fallback: positional
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
        Keep only scalar-like metrics for logging.

        Parameters
        ----------
        metrics_any : Any
            Typically a Mapping[str, Any]. Non-mappings yield empty dict.
        drop_non_finite : bool, optional
            If True, drops NaN/Inf values.

        Returns
        -------
        metrics : Dict[str, float]
            Only scalar-like values that can be converted to float.

        Notes
        -----
        Relies on project utilities:
          - is_scalar_like(x) -> bool
          - to_scalar(x) -> Optional[float]
        """
        metrics: Dict[str, Any] = dict(metrics_any) if isinstance(metrics_any, Mapping) else {}
        out: Dict[str, float] = {}

        for k, v in metrics.items():
            if not is_scalar_like(v):
                continue

            sv = to_scalar(v)
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
        Save algorithm checkpoint.

        The checkpoint is a dictionary that may contain:
          - head_state_dict
          - core_state
          - meta

        Parameters
        ----------
        path : str
            File path. If missing ".pt" suffix, it will be appended.
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
        Load algorithm checkpoint.

        Parameters
        ----------
        path : str
            File path. If missing ".pt" suffix, it will be appended.

        Raises
        ------
        ValueError
            If the checkpoint format is invalid or required methods are missing.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        # torch.load signature differs across versions; keep it simple & compatible.
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
        Return PolicyFactorySpec used to construct Ray remote policies.

        Raises
        ------
        ValueError
            If head does not expose get_ray_policy_factory_spec().
        """
        fn = getattr(self.head, "get_ray_policy_factory_spec", None)
        if not callable(fn):
            raise ValueError("head has no get_ray_policy_factory_spec().")
        return fn()


class BasePolicyAlgorithm(BaseAlgorithm):
    """
    Base class for algorithms that interact with environments (env-facing).

    Adds:
      - env step counter

    Subclasses must implement:
      - setup(env)
      - on_env_step(transition)
      - ready_to_update()
      - update()

    Notes
    -----
    - This base does not prescribe replay/rollout storage format; it only defines
      the minimal lifecycle hooks.
    """

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        device: Optional[Union[str, th.device]] = None,
    ) -> None:
        super().__init__(head=head, core=core, device=device)
        self._env_steps: int = 0

    @property
    def env_steps(self) -> int:
        """Number of environment steps processed by this algorithm."""
        return int(self._env_steps)

    # ------------------------------------------------------------------
    # Minimal protocol
    # ------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        raise NotImplementedError

    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Consume one environment transition.

        Recommended transition keys (convention)
        ---------------------------------------
        - observations
        - actions
        - rewards
        - next_observations
        - dones
        - infos (optional)
        """
        raise NotImplementedError

    def ready_to_update(self) -> bool:
        """Return True if the algorithm has enough data to perform update()."""
        raise NotImplementedError

    def update(self) -> Dict[str, float]:
        """
        Perform one training update.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar training metrics suitable for logging.
        """
        raise NotImplementedError