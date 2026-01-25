from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import math
import platform
import sys
import time

from .base_callback import BaseCallback
from ..utils.log_utils import log
from ..utils.callback_utils import infer_step


class ConfigAndEnvInfoCallback(BaseCallback):
    """
    Log basic run metadata at train start (best-effort, framework-agnostic).

    Design goals
    ------------
    - Never raise (best-effort introspection)
    - JSON-friendly payload (scalars, small dict/list summaries, strings truncated)
    - Minimal assumptions about Trainer/Env shape (duck-typed)

    Parameters
    ----------
    log_prefix : str, default="sys/"
        Prefix used by your logger's `log(...)` function.
    max_collection_items : int, default=32
        Max number of items/keys to keep when normalizing lists/dicts.
    max_string_len : int, default=200
        Max string length (ellipsis truncation).
    log_once : bool, default=True
        If True, log only once per callback instance (even if on_train_start called again).
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
    # Internal helpers (merged from standalone utilities)
    # =========================================================================
    def _safe_getattr(self, obj: Any, name: str, default: Any = None) -> Any:
        """Best-effort getattr that never raises."""
        try:
            return getattr(obj, name)
        except Exception:
            return default

    def _truncate_str(self, s: str, *, max_len: Optional[int] = None) -> str:
        """Truncate a string to max_len with ellipsis."""
        ml = self.max_string_len if max_len is None else int(max_len)
        if ml <= 0:
            return ""
        if len(s) <= ml:
            return s
        return s[:ml] + "..."

    def _norm_jsonish(self, x: Any) -> Any:
        """
        Normalize arbitrary values into JSON-friendly types (best-effort).

        Keeps:
          - None, bool, int, float (finite only), str (truncated)
        Summarizes:
          - Mapping: keep up to max_collection_items keys (recursive)
          - list/tuple: keep up to max_collection_items items (recursive)
        Fallback:
          - "<ClassName>" string (truncated)
        """
        if x is None:
            return None

        # Basic scalars
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

        # Fallback: class name
        return self._truncate_str(f"<{type(x).__name__}>")

    def _infer_env_id(self, env: Any) -> Optional[str]:
        """
        Try to infer environment id string from common patterns:
          - env.spec.id (gym/gymnasium)
          - env.unwrapped.spec.id
          - env.envs[0].spec.id (VecEnv-like)
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
        """Best-effort env count."""
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
        if self.log_once and self._did_log:
            return True
        self._did_log = True

        payload: Dict[str, Any] = {}

        # ---- trainer knobs (common) ----
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
            step = infer_step(trainer)
            log(trainer, payload, step=step, prefix=self.log_prefix)

        return True