from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover
    # Ray is an optional dependency. This module must remain importable without it.
    ray = None  # type: ignore

import torch as th

from ..utils.common_utils import (
    to_action_np,
    to_scalar,
    to_cpu_state_dict,
    obs_to_cpu_tensor,
)

from ..utils.train_utils import(
    set_random_seed,
    env_reset,
    unpack_step,
)

from ..utils.ray_utils import (
    PolicyFactorySpec,
    build_policy_from_spec,
    require_ray
)


class RayEnvRunner:
    """
    CPU environment runner for collecting rollouts.

    Contract
    --------
    The policy must implement:
        policy.act(obs_t, deterministic=..., return_info=...) -> action
        policy.act(...) -> (action, info)   # optional

    If want_onpolicy_extras=True, the worker will request return_info=True and
    will record, when available:
        - "log_prob" (from "logp" or "log_prob")
        - "value"

    Notes
    -----
    - This runner is designed to be robust across Gym/Gymnasium variants.
    - Environment state sync (state_dict/load_state_dict) is supported best-effort,
      typically for NormalizeWrapper running statistics.
    """

    def __init__(
        self,
        env_make_fn: Callable[[], Any],
        policy_spec: PolicyFactorySpec,
        *,
        seed: int = 0,
        max_episode_steps: Optional[int] = None,
        want_onpolicy_extras: bool = False,
        # step unpacking
        flatten_obs: bool = False,
        obs_dtype: Any = np.float32,
    ) -> None:
        """
        Parameters
        ----------
        env_make_fn : Callable[[], Any]
            Callable that builds a fresh environment instance.
        policy_spec : PolicyFactorySpec
            Serializable spec for constructing a policy on the worker.
        seed : int
            Base seed for worker RNG and env reset seeding (best-effort).
        max_episode_steps : int, optional
            Fallback episode length cap when env/wrappers do not provide truncation.
        want_onpolicy_extras : bool
            If True, request policy info and record log_prob/value when available.
        flatten_obs : bool
            Forwarded to unpack_step(...). If True, flattens observations.
        obs_dtype : Any
            Forwarded to unpack_step(...). Cast/convert observation dtype.
        """
        set_random_seed(int(seed), deterministic=True, verbose=False)

        self.env = env_make_fn()
        self._seed_env_best_effort(self.env, int(seed))

        self.policy = build_policy_from_spec(policy_spec)
        self.policy.eval()

        self.want_onpolicy_extras = bool(want_onpolicy_extras)

        # Prefer letting env/wrappers handle time limits; keep as fallback only.
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)

        self.flatten_obs = bool(flatten_obs)
        self.obs_dtype = obs_dtype

        self.obs = env_reset(self.env)
        self.ep_len = 0

    # ------------------------------------------------------------------
    # Public API (called by Ray actor messages)
    # ------------------------------------------------------------------
    def set_policy_weights(self, state_dict: Mapping[str, Any]) -> None:
        """
        Load policy weights (CPU) and set eval mode.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            Policy state dict. Will be converted to CPU tensors.
        """
        sd = to_cpu_state_dict(state_dict)
        self.policy.load_state_dict(sd, strict=True)
        self.policy.eval()

    def set_env_state(self, state: Mapping[str, Any]) -> None:
        """
        Load env/wrapper state (e.g., NormalizeWrapper running stats) if supported.

        Parameters
        ----------
        state : Mapping[str, Any]
            Environment state dict payload (typically produced by env.state_dict()).
        """
        env = self.env

        load_fn = getattr(env, "load_state_dict", None)
        if callable(load_fn):
            try:
                load_fn(dict(state))
            except Exception:
                pass

        # Workers should not update running stats if we are syncing from learner.
        set_train_fn = getattr(env, "set_training", None)
        if callable(set_train_fn):
            try:
                set_train_fn(False)
            except Exception:
                pass

    def get_env_state(self) -> Optional[Dict[str, Any]]:
        """
        Export env/wrapper state if supported.

        Returns
        -------
        state : Optional[Dict[str, Any]]
            Dict from env.state_dict() if available; otherwise None.
        """
        state_fn = getattr(self.env, "state_dict", None)
        if callable(state_fn):
            try:
                return dict(state_fn())
            except Exception:
                return None
        return None

    @th.no_grad()
    def rollout(self, n_steps: int, deterministic: bool = False) -> List[Dict[str, Any]]:
        """
        Collect a fixed-length rollout chunk.

        Parameters
        ----------
        n_steps : int
            Number of environment steps to collect.
        deterministic : bool
            If True, request deterministic actions from policy.

        Returns
        -------
        traj : List[Dict[str, Any]]
            A list of transitions with keys:
              - obs, action, reward, next_obs, done, info
            Plus optional extras if want_onpolicy_extras:
              - log_prob, value
        """
        traj: List[Dict[str, Any]] = []
        env = self.env
        action_space = getattr(env, "action_space", None)

        for _ in range(int(n_steps)):
            obs_t = obs_to_cpu_tensor(self.obs)

            action_t, info_pol = self._policy_act(obs_t, deterministic=bool(deterministic))
            action_env = self._format_action_for_env(action_t, action_space)

            next_obs, reward, done, info_out = self._env_step(action_env)

            tr: Dict[str, Any] = {
                "obs": self.obs,
                "action": action_env,
                "reward": float(reward),
                "next_obs": next_obs,
                "done": bool(done),
                "info": info_out,
            }

            extras = self._extract_onpolicy_extras(info_pol)
            if extras:
                tr.update(extras)

            traj.append(tr)

            self.obs = next_obs
            if done:
                self.obs = env_reset(env)
                self.ep_len = 0

        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seed_env_best_effort(env: Any, seed: int) -> None:
        """Best-effort environment seeding for Gym/Gymnasium variants."""
        try:
            env.reset(seed=int(seed))
            return
        except Exception:
            pass
        try:
            env.seed(int(seed))  # older gym
        except Exception:
            pass

    def _policy_act(self, obs_t: th.Tensor, *, deterministic: bool) -> tuple[Any, Dict[str, Any]]:
        """
        Call policy.act and normalize output to (action, info_dict).
        """
        if self.want_onpolicy_extras:
            out = self.policy.act(obs_t, deterministic=deterministic, return_info=True)
        else:
            out = self.policy.act(obs_t, deterministic=deterministic)

        if isinstance(out, tuple) and len(out) == 2:
            action_t, info_pol = out
            return action_t, info_pol if isinstance(info_pol, dict) else {}

        return out, {}

    @staticmethod
    def _format_action_for_env(action_t: Any, action_space: Any) -> Any:
        """
        Convert policy output into an env-compatible action.

        Discrete
        --------
        If action_space.n exists, coerce to int using the first element.

        Continuous / Unknown
        --------------------
        Use action_space.shape if available; convert to float32 numpy; drop a leading
        batch dimension if present (shape[0] == 1).
        """
        is_discrete = bool(action_space is not None and hasattr(action_space, "n"))

        if is_discrete:
            a = to_action_np(action_t, action_shape=None)
            a = np.asarray(a).reshape(-1)
            return int(a[0])

        action_shape = getattr(action_space, "shape", None) if action_space is not None else None
        if not isinstance(action_shape, tuple):
            action_shape = None

        a = to_action_np(action_t, action_shape=action_shape).astype(np.float32, copy=False)
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == 1:
            a = a[0]
        return a

    def _env_step(self, action_env: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """
        Step the environment and apply optional fallback TimeLimit synthesis.

        Returns
        -------
        next_obs : Any
        reward : float
        done : bool
        info : Dict[str, Any]
        """
        step_out = self.env.step(action_env)
        next_obs, reward, done, info_env = unpack_step(
            step_out,
            flatten_obs=self.flatten_obs,
            obs_dtype=self.obs_dtype,
        )

        self.ep_len += 1
        info_out: Dict[str, Any] = dict(info_env) if isinstance(info_env, Mapping) else {}

        # Fallback TimeLimit synthesis ONLY if env didn't provide it.
        if self.max_episode_steps is not None and self.ep_len >= int(self.max_episode_steps):
            if not bool(done):
                done = True
            if "TimeLimit.truncated" not in info_out:
                info_out["TimeLimit.truncated"] = True

        return next_obs, float(reward), bool(done), info_out

    def _extract_onpolicy_extras(self, info_pol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract on-policy extras from policy info dict.

        Returns
        -------
        extras : Dict[str, Any]
            Possibly empty dict containing "log_prob" and/or "value".
        """
        if not self.want_onpolicy_extras:
            return {}
        if not isinstance(info_pol, dict):
            return {}

        extras: Dict[str, Any] = {}

        lp = None
        if "logp" in info_pol:
            lp = to_scalar(info_pol["logp"])
        elif "log_prob" in info_pol:
            lp = to_scalar(info_pol["log_prob"])
        if lp is not None:
            extras["log_prob"] = float(lp)

        v = None
        if "value" in info_pol:
            v = to_scalar(info_pol["value"])
        if v is not None:
            extras["value"] = float(v)

        return extras


# =============================================================================
# Learner-side driver (non-remote orchestrator)
# =============================================================================

# Expose a Ray actor class only when Ray is available.
if ray is not None:
    RayEnvWorker = ray.remote(RayEnvRunner)  # type: ignore[attr-defined]
else:  # pragma: no cover
    RayEnvWorker = None  # type: ignore


class RayLearner:
    """
    Learner-side orchestrator that manages RayEnvWorker actors.

    This class:
      - creates N RayEnvRunner actors
      - broadcasts policy weights (and optional env state) to all workers
      - collects rollout chunks and flattens them into a single list
    """

    def __init__(
        self,
        *,
        env_make_fn: Callable[[], Any],
        policy_spec: PolicyFactorySpec,
        n_workers: int,
        steps_per_worker: int,
        base_seed: int = 0,
        max_episode_steps: Optional[int] = None,
        want_onpolicy_extras: bool = False,
        # step unpacking
        flatten_obs: bool = False,
        obs_dtype: Any = np.float32,
    ) -> None:
        """
        Parameters
        ----------
        env_make_fn : Callable[[], Any]
            Environment factory. Must be serializable by Ray.
        policy_spec : PolicyFactorySpec
            Serializable policy construction spec. Sent to workers via ray.put.
        n_workers : int
            Number of Ray workers.
        steps_per_worker : int
            Steps collected per worker per collect() call.
        base_seed : int
            Base seed for per-worker seeding (seed = base_seed + i).
        max_episode_steps : int, optional
            Fallback episode length cap when env/wrappers do not provide truncation.
        want_onpolicy_extras : bool
            If True, workers request policy extras (log_prob/value).
        flatten_obs : bool
            Forwarded to worker unpack_step(...).
        obs_dtype : Any
            Forwarded to worker unpack_step(...).
        """
        require_ray()
        if RayEnvWorker is None:  # pragma: no cover
            raise RuntimeError("RayEnvWorker is unavailable because Ray is not installed.")

        self.n_workers = int(n_workers)
        self.steps_per_worker = int(steps_per_worker)

        spec_ref = ray.put(policy_spec)

        self.workers = [
            RayEnvWorker.remote(
                env_make_fn,
                spec_ref,
                seed=int(base_seed + i),
                max_episode_steps=max_episode_steps,
                want_onpolicy_extras=bool(want_onpolicy_extras),
                flatten_obs=bool(flatten_obs),
                obs_dtype=obs_dtype,
            )
            for i in range(self.n_workers)
        ]

    def broadcast_policy(self, policy_state_dict_cpu: Mapping[str, Any]) -> None:
        """
        Broadcast latest policy weights to all workers.

        Parameters
        ----------
        policy_state_dict_cpu : Mapping[str, Any]
            CPU state dict. Will be normalized to CPU tensors before ray.put.
        """
        sd = to_cpu_state_dict(policy_state_dict_cpu)
        sd_ref = ray.put(sd)
        ray.get([w.set_policy_weights.remote(sd_ref) for w in self.workers])

    def broadcast_env_state(self, env_state: Mapping[str, Any]) -> None:
        """
        Broadcast env/wrapper state (e.g., NormalizeWrapper running stats) to all workers.

        Parameters
        ----------
        env_state : Mapping[str, Any]
            Environment state dict payload from learner environment.
        """
        st = dict(env_state)
        st_ref = ray.put(st)
        ray.get([w.set_env_state.remote(st_ref) for w in self.workers])

    def collect(self, deterministic: bool = False, n_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Collect rollout chunks from all workers and flatten them into one list.

        Parameters
        ----------
        deterministic : bool
            If True, collect deterministic actions from workers.
        n_steps : int, optional
            Override steps_per_worker for this call only.

        Returns
        -------
        transitions : List[Dict[str, Any]]
            Flattened list of transitions from all workers.
        """
        steps = self.steps_per_worker if n_steps is None else int(n_steps)
        futs = [w.rollout.remote(steps, deterministic=bool(deterministic)) for w in self.workers]
        chunks = ray.get(futs)

        out: List[Dict[str, Any]] = []
        for c in chunks:
            out.extend(c)
        return out