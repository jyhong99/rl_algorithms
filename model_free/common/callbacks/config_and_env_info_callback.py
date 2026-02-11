from __future__ import annotations

import math
import platform
import sys
import time
from typing import Any, Dict, Mapping, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import _infer_step


class ConfigAndEnvInfoCallback(BaseCallback):
    """
    Log basic run metadata at training start (best-effort, framework-agnostic).

    This callback collects a JSON-friendly snapshot of:
    - Trainer configuration knobs (if present)
    - Algorithm / head / core identities (if exposed by the trainer)
    - Environment identity hints (Gym/Gymnasium style) and vectorization info
    - Runtime/system info (Python version, platform string, timestamp)

    Design goals
    ------------
    - Never raise: all introspection is best-effort.
    - JSON-friendly payload:
        - scalars (bool/int/finite float), strings (truncated),
          small dict/list summaries (bounded size, recursive).
    - Minimal assumptions about Trainer/Env: everything is accessed duck-typed.

    Parameters
    ----------
    log_prefix:
        Prefix used by the logger (passed to :meth:`BaseCallback.log`).
        Typical values include ``"sys/"`` or ``"meta/"``.
    max_collection_items:
        Maximum number of items/keys to keep when normalizing lists/dicts.
        Extra items are summarized with an ellipsis marker.
    max_string_len:
        Maximum string length. Longer strings are truncated with ellipsis.
    log_once:
        If True, log only once per callback instance, even if
        :meth:`on_train_start` is called multiple times (e.g., resume/restart).

    Notes
    -----
    Trainer contract (duck-typed)
    -----------------------------
    The callback may read some subset of:
    - Common trainer fields: ``seed``, ``batch_size``, ``gamma``, ``device``, etc.
    - Optional identity fields: ``algo``, ``core``, ``head``, ``logger``
    - Environment handle: ``train_env``

    Step selection
    --------------
    The log step is determined by :func:`infer_step`, which typically prefers
    an environment-step counter when available, falling back to update-step
    counters otherwise.
    """

    def __init__(
        self,
        *,
        log_prefix: str = "sys/",
        max_collection_items: int = 32,
        max_string_len: int = 200,
        log_once: bool = True,
    ) -> None:
        self.log_prefix = str(log_prefix)
        self.max_collection_items = int(max_collection_items)
        self.max_string_len = int(max_string_len)
        self.log_once = bool(log_once)

        self._did_log = False

    # =========================================================================
    # Internal helpers
    # =========================================================================
    @staticmethod
    def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
        """
        Best-effort getattr that never raises.

        Parameters
        ----------
        obj:
            Object from which to read an attribute.
        name:
            Attribute name to read.
        default:
            Value to return if getattr fails.

        Returns
        -------
        Any
            Attribute value if accessible; otherwise ``default``.
        """
        try:
            return getattr(obj, name)
        except Exception:
            return default

    def _truncate_str(self, s: str, *, max_len: Optional[int] = None) -> str:
        """
        Truncate a string to a maximum length with ellipsis.

        Parameters
        ----------
        s:
            Input string.
        max_len:
            Maximum length. If None, uses ``self.max_string_len``.

        Returns
        -------
        str
            Truncated string. If ``max_len <= 0``, returns an empty string.
        """
        ml = self.max_string_len if max_len is None else int(max_len)
        if ml <= 0:
            return ""
        if len(s) <= ml:
            return s
        return s[:ml] + "..."

    def _norm_jsonish(self, x: Any) -> Any:
        """
        Normalize arbitrary Python objects into JSON-friendly types (best-effort).

        Conversion rules
        ---------------
        Preserved types:
        - None
        - bool, int
        - float (finite only; NaN/inf -> None)
        - str (truncated)

        Best-effort scalar extraction:
        - numpy/torch scalars that implement ``.item()`` are converted to a Python scalar.

        Summaries:
        - Mapping: keep up to ``max_collection_items`` key/value pairs (recursive).
        - list/tuple: keep up to ``max_collection_items`` items (recursive), then append a
          summary marker (e.g., ``"... +12 more"``).

        Fallback:
        - If an object cannot be normalized, returns a string ``"<ClassName>"`` (truncated).

        Parameters
        ----------
        x:
            Arbitrary input value.

        Returns
        -------
        Any
            JSON-friendly representation.
        """
        if x is None:
            return None

        if isinstance(x, (bool, int)):
            return x

        if isinstance(x, float):
            return x if math.isfinite(x) else None

        if isinstance(x, str):
            return self._truncate_str(x)

        # numpy/torch scalar best-effort: x.item()
        try:
            item = getattr(x, "item", None)
            if callable(item):
                v = item()
                if isinstance(v, (bool, int)):
                    return v
                if isinstance(v, float):
                    return v if math.isfinite(v) else None
        except Exception:
            pass

        # Mapping: keep up to N keys
        if isinstance(x, Mapping):
            out: Dict[str, Any] = {}
            try:
                n_total = len(x)
            except Exception:
                n_total = None

            for i, (k, v) in enumerate(x.items()):
                if i >= self.max_collection_items:
                    if n_total is not None:
                        out["..."] = f"+{max(0, n_total - self.max_collection_items)} more keys"
                    else:
                        out["..."] = "+more keys"
                    break
                out[str(k)] = self._norm_jsonish(v)
            return out

        # Sequence: list/tuple
        if isinstance(x, (list, tuple)):
            n = len(x)
            if n <= self.max_collection_items:
                return [self._norm_jsonish(v) for v in x]
            head = [self._norm_jsonish(v) for v in x[: self.max_collection_items]]
            head.append(f"... +{n - self.max_collection_items} more")
            return head

        return self._truncate_str(f"<{type(x).__name__}>")

    def _infer_env_id(self, env: Any) -> Optional[str]:
        """
        Infer an environment id string from common Gym/Gymnasium patterns.

        Attempts the following in order (best-effort):
        - ``env.spec.id``
        - ``env.unwrapped.spec.id``
        - ``env.envs[0].spec.id`` (VecEnv-like containers)

        Parameters
        ----------
        env:
            Environment object (duck-typed).

        Returns
        -------
        Optional[str]
            Environment id string if inferred; otherwise None.
        """
        try:
            if env is None:
                return None

            spec = getattr(env, "spec", None)
            if spec is None and hasattr(env, "unwrapped"):
                spec = getattr(env.unwrapped, "spec", None)

            if spec is not None:
                env_id = getattr(spec, "id", None)
                if isinstance(env_id, str) and env_id:
                    return env_id

            envs = getattr(env, "envs", None)
            if isinstance(envs, (list, tuple)) and len(envs) > 0:
                return self._infer_env_id(envs[0])

        except Exception:
            return None

        return None

    def _infer_env_num(self, env: Any) -> Optional[int]:
        """
        Infer the number of environments (vectorization degree) best-effort.

        Attempts:
        - ``env.num_envs`` (common VecEnv API)
        - ``len(env.envs)`` (list/tuple container)

        Parameters
        ----------
        env:
            Environment object (duck-typed).

        Returns
        -------
        Optional[int]
            Number of environments if inferred; otherwise None.
        """
        try:
            if env is None:
                return None

            n = getattr(env, "num_envs", None)
            if isinstance(n, int) and n > 0:
                return n

            envs = getattr(env, "envs", None)
            if isinstance(envs, (list, tuple)):
                return len(envs)
        except Exception:
            pass

        return None

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_train_start(self, trainer: Any) -> bool:
        """
        Collect and log trainer/environment metadata at training start.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). The callback introspects commonly used
            configuration fields if present, and uses :func:`infer_step` to
            choose a logging step.

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        - If ``log_once=True``, this method logs at most once per instance.
        - All failures are swallowed (best-effort introspection).
        """
        if self.log_once and self._did_log:
            return True
        self._did_log = True

        payload: Dict[str, Any] = {}

        # ---- trainer knobs (common fields) ----
        trainer_keys = (
            "seed",
            "total_env_steps",
            "total_timesteps",
            "n_envs",
            "rollout_steps",
            "rollout_steps_per_env",
            "batch_size",
            "minibatch_size",
            "update_epochs",
            "utd",
            "gamma",
            "gae_lambda",
            "max_episode_steps",
            "deterministic",
            "device",
            "dtype_obs",
            "dtype_act",
        )
        for k in trainer_keys:
            if hasattr(trainer, k):
                payload[k] = self._norm_jsonish(self._safe_getattr(trainer, k))

        # ---- algorithm / core / head identity (if exposed) ----
        algo = self._safe_getattr(trainer, "algo")
        if algo is not None:
            payload["algo.class"] = type(algo).__name__
            for k in ("name", "algo_name"):
                v = self._safe_getattr(algo, k)
                if isinstance(v, str) and v:
                    payload[f"algo.{k}"] = self._truncate_str(v)

        head = self._safe_getattr(trainer, "head")
        if head is not None:
            payload["head.class"] = type(head).__name__

        core = self._safe_getattr(trainer, "core")
        if core is not None:
            payload["core.class"] = type(core).__name__

        # ---- logger identity ----
        logger = self._safe_getattr(trainer, "logger")
        if logger is not None:
            payload["logger.class"] = type(logger).__name__

        # ---- environment info ----
        env = self._safe_getattr(trainer, "train_env")
        if env is not None:
            payload["env.class"] = type(env).__name__

            env_id = self._infer_env_id(env)
            if env_id is not None:
                payload["env_id"] = env_id

            n_env = self._infer_env_num(env)
            if n_env is not None and "n_envs" not in payload:
                payload["n_envs"] = int(n_env)

            # Wrapper hints (trainer-level flags)
            for k in ("_normalize_enabled", "normalize", "norm_obs", "norm_reward", "clip_obs", "clip_reward"):
                if hasattr(trainer, k):
                    payload[k] = self._norm_jsonish(self._safe_getattr(trainer, k))

            # Wrapper hints (env-level flags)
            env_keys = (
                "norm_obs",
                "norm_reward",
                "clip_obs",
                "clip_reward",
                "gamma",
                "epsilon",
                "action_rescale",
                "clip_action",
            )
            for k in env_keys:
                if hasattr(env, k):
                    payload[f"env.{k}"] = self._norm_jsonish(self._safe_getattr(env, k))

        # ---- runtime/system info ----
        payload["python.version"] = self._truncate_str(sys.version.replace("\n", " "))
        payload["platform"] = self._truncate_str(platform.platform())
        payload["time.unix"] = int(time.time())

        if payload:
            step = _infer_step(trainer)
            self.log(trainer, payload, step=step, prefix=self.log_prefix)

        return True
