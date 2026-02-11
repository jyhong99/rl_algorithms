from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover
    # Ray is an optional dependency. This module must remain importable without it.
    ray = None  # type: ignore

import torch as th

from ..utils.common_utils import (
    _to_action_np,
    _to_scalar,
    _to_cpu_state_dict,
    _obs_to_cpu_tensor,
)

from ..utils.train_utils import (
    _set_random_seed,
    _env_reset,
    _unpack_step,
)

from ..utils.ray_utils import (
    PolicyFactorySpec,
    _build_policy_from_spec,
    _require_ray,
)


class RayEnvRunner:
    """
    CPU environment runner for collecting rollout transitions.

    This class is intended to run *inside* a Ray actor (worker process) and
    provides a simple message-driven API:
      - set_policy_weights(...)
      - set_env_state(...), get_env_state(...)
      - rollout(n_steps, deterministic)

    Policy contract (duck-typed)
    ----------------------------
    The policy object must implement:
        act(obs_t, deterministic=..., return_info=...) -> action
    and may optionally return:
        (action, info_dict)

    If `want_onpolicy_extras=True`, the runner will request `return_info=True`
    and will record (when present in `info_dict`):
      - log_prob  : from "logp" or "log_prob"
      - value     : from "value"

    Environment contract (duck-typed)
    ---------------------------------
    The environment must implement:
      - reset(...) and step(action)
    and may optionally implement:
      - state_dict() / load_state_dict(...)  (e.g., normalization running stats)
      - set_training(bool)                   (freeze running stats updates)
      - action_space with Gym-like attributes: `.n` for discrete, `.shape` for Box

    Robustness notes
    ----------------
    - Gym/Gymnasium step/reset variants are handled via `_env_reset` and `_unpack_step`.
    - All optional interfaces are called best-effort: failures are ignored to keep
      rollout collection robust in heterogeneous environments/wrappers.
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
            Factory that builds a fresh environment instance on this worker.
            Must be Ray-serializable (picklable).
        policy_spec : PolicyFactorySpec
            Serializable spec for constructing a policy on the worker.
        seed : int, default=0
            Base seed for this worker:
              - seeds Python/NumPy/Torch RNGs via `_set_random_seed`
              - seeds environment reset best-effort
        max_episode_steps : int, optional
            Fallback episode-length cap used ONLY when the env does not signal
            termination/truncation. This runner synthesizes:
                done = True
                info["TimeLimit.truncated"] = True
            when the step count hits the cap.
        want_onpolicy_extras : bool, default=False
            If True, request policy info (`return_info=True`) and attempt to record
            "log_prob" and "value" from the returned policy info dict.
        flatten_obs : bool, default=False
            Forwarded to `_unpack_step(...)`. If True, flattens observations.
        obs_dtype : Any, default=np.float32
            Forwarded to `_unpack_step(...)`. Cast/convert observation dtype.

        Notes
        -----
        - The policy is built locally on the worker and forced into eval mode.
        - The environment is created locally; seeding is best-effort to cover
          different Gym/Gymnasium versions.
        """
        _set_random_seed(int(seed), deterministic=True, verbose=False)

        self.env = env_make_fn()
        self._seed_env_best_effort(self.env, int(seed))

        self.policy = _build_policy_from_spec(policy_spec)
        self.policy.eval()

        self.want_onpolicy_extras = bool(want_onpolicy_extras)

        # Prefer letting env/wrappers handle time limits; keep as fallback only.
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)

        self.flatten_obs = bool(flatten_obs)
        self.obs_dtype = obs_dtype

        # Initialize rollout state.
        self.obs = _env_reset(self.env)
        self.ep_len = 0

    # ------------------------------------------------------------------
    # Public API (Ray actor methods)
    # ------------------------------------------------------------------
    def set_policy_weights(self, state_dict: Mapping[str, Any]) -> None:
        """
        Load policy weights (CPU) and set eval mode.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            Policy state dict. Can contain CPU/GPU tensors; will be normalized to
            CPU tensors via `_to_cpu_state_dict` before loading.
        """
        sd = _to_cpu_state_dict(state_dict)
        self.policy.load_state_dict(sd, strict=True)
        self.policy.eval()

    def set_env_state(self, state: Mapping[str, Any]) -> None:
        """
        Load env/wrapper state (e.g., NormalizeWrapper running stats) if supported.

        Parameters
        ----------
        state : Mapping[str, Any]
            Environment state dict payload (typically produced by env.state_dict()).

        Notes
        -----
        - If `env.load_state_dict` exists, it is called best-effort.
        - If `env.set_training` exists, we set it to False to avoid drifting
          running statistics while the learner is controlling the state.
        """
        env = self.env

        load_fn = getattr(env, "load_state_dict", None)
        if callable(load_fn):
            try:
                load_fn(dict(state))
            except Exception:
                pass

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
        state : dict[str, Any] or None
            A dict from env.state_dict() if available and successful; otherwise None.
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
        deterministic : bool, default=False
            If True, request deterministic actions from policy.

        Returns
        -------
        traj : list[dict[str, Any]]
            A list of transition dicts with keys:
              - "obs"      : observation *before* the action
              - "action"   : action passed to env.step(...)
              - "reward"   : float reward
              - "next_obs" : observation after the step
              - "done"     : episode termination flag
              - "info"     : info dict returned from env (normalized to dict)
            If `want_onpolicy_extras=True`, may also include:
              - "log_prob" : float (from policy info)
              - "value"    : float (from policy info)

        Notes
        -----
        - This runner maintains internal state (`self.obs`, `self.ep_len`) across calls.
          When `done=True`, it resets the env and continues collecting until n_steps.
        - Action formatting uses env.action_space when available to coerce discrete/box.
        """
        traj: List[Dict[str, Any]] = []
        env = self.env
        action_space = getattr(env, "action_space", None)

        for _ in range(int(n_steps)):
            obs_prev = self.obs  # keep the "pre-step" obs for the transition
            obs_t = _obs_to_cpu_tensor(obs_prev)

            action_t, info_pol = self._policy_act(obs_t, deterministic=bool(deterministic))
            action_env = self._format_action_for_env(action_t, action_space)

            next_obs, reward, done, info_out = self._env_step(action_env)

            tr: Dict[str, Any] = {
                "obs": obs_prev,
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
                self.obs = _env_reset(env)
                self.ep_len = 0

        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seed_env_best_effort(env: Any, seed: int) -> None:
        """
        Best-effort environment seeding across Gym/Gymnasium variants.

        Parameters
        ----------
        env : Any
            Environment instance.
        seed : int
            Seed to apply.

        Notes
        -----
        Attempts, in order:
          1) env.reset(seed=seed)  (Gymnasium/newer Gym)
          2) env.seed(seed)        (older Gym)
        """
        try:
            env.reset(seed=int(seed))
            return
        except Exception:
            pass
        try:
            env.seed(int(seed))  # older gym
        except Exception:
            pass

    def _policy_act(self, obs_t: th.Tensor, *, deterministic: bool) -> Tuple[Any, Dict[str, Any]]:
        """
        Call policy.act(...) and normalize output to (action, info_dict).

        Parameters
        ----------
        obs_t : torch.Tensor
            Observation tensor on CPU (batching handled by the policy).
        deterministic : bool
            Whether to request deterministic action selection.

        Returns
        -------
        action : Any
            Raw action output from the policy.
        info_pol : dict[str, Any]
            Policy info dict if provided; otherwise an empty dict.

        Notes
        -----
        - If `want_onpolicy_extras=True`, calls policy.act(..., return_info=True).
        - If the policy returns (action, info), we validate `info` is a dict-like.
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

        Parameters
        ----------
        action_t : Any
            Action returned by policy.act(...). Can be tensor/ndarray/list/scalar.
        action_space : Any
            Environment action space (Gym-like). Used only for inference:
              - discrete if hasattr(action_space, "n")
              - continuous shape from getattr(action_space, "shape", None)

        Returns
        -------
        action_env : Any
            Action formatted for env.step(...):
              - Discrete: int
              - Continuous/unknown: numpy.ndarray (float32)

        Discrete behavior
        -----------------
        If action_space has attribute `n`, we interpret it as a Discrete space and:
          - convert to numpy
          - flatten
          - take first element and cast to int

        Continuous / unknown behavior
        -----------------------------
        - Use action_space.shape when available to shape-check or reshape in `_to_action_np`.
        - Convert to float32 numpy.
        - If policy outputs with a leading batch dim (1, act_dim, ...), drop it.
        """
        is_discrete = bool(action_space is not None and hasattr(action_space, "n"))

        if is_discrete:
            a = _to_action_np(action_t, action_shape=None)
            a = np.asarray(a).reshape(-1)
            return int(a[0])

        action_shape = getattr(action_space, "shape", None) if action_space is not None else None
        if not isinstance(action_shape, tuple):
            action_shape = None

        a = _to_action_np(action_t, action_shape=action_shape).astype(np.float32, copy=False)
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == 1:
            a = a[0]
        return a

    def _env_step(self, action_env: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Step the environment and apply fallback TimeLimit synthesis if requested.

        Parameters
        ----------
        action_env : Any
            Action passed into env.step(action_env).

        Returns
        -------
        next_obs : Any
            Next observation.
        reward : float
            Reward as float.
        done : bool
            Episode termination flag.
        info : dict[str, Any]
            Info dict (guaranteed to be a dict, possibly empty).

        Notes
        -----
        - The raw step output is normalized by `_unpack_step(...)` to handle
          Gym/Gymnasium (terminated/truncated) variants.
        - If `max_episode_steps` is set, we *synthesize* termination when the
          step count reaches the limit **only if the env has not already ended**
          the episode. In that case, we also set:
              info["TimeLimit.truncated"] = True
          if the env didn't already provide it.
        """
        step_out = self.env.step(action_env)
        next_obs, reward, done, info_env = _unpack_step(
            step_out,
            flatten_obs=self.flatten_obs,
            obs_dtype=self.obs_dtype,
        )

        self.ep_len += 1
        info_out: Dict[str, Any] = dict(info_env) if isinstance(info_env, Mapping) else {}

        if self.max_episode_steps is not None and self.ep_len >= int(self.max_episode_steps):
            if not bool(done):
                done = True
            if "TimeLimit.truncated" not in info_out:
                info_out["TimeLimit.truncated"] = True

        return next_obs, float(reward), bool(done), info_out

    def _extract_onpolicy_extras(self, info_pol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract on-policy extras from policy info dict.

        Parameters
        ----------
        info_pol : dict[str, Any]
            Policy info dict returned from policy.act(..., return_info=True).

        Returns
        -------
        extras : dict[str, Any]
            Possibly empty dict containing:
              - "log_prob": float
              - "value": float

        Notes
        -----
        - "log_prob" is read from keys: "logp" or "log_prob".
        - "value" is read from key: "value".
        - `_to_scalar` is used to robustly convert tensors/arrays to Python scalars.
        """
        if not self.want_onpolicy_extras:
            return {}
        if not isinstance(info_pol, dict):
            return {}

        extras: Dict[str, Any] = {}

        lp = None
        if "logp" in info_pol:
            lp = _to_scalar(info_pol["logp"])
        elif "log_prob" in info_pol:
            lp = _to_scalar(info_pol["log_prob"])
        if lp is not None:
            extras["log_prob"] = float(lp)

        v = None
        if "value" in info_pol:
            v = _to_scalar(info_pol["value"])
        if v is not None:
            extras["value"] = float(v)

        return extras


# =============================================================================
# Ray actor export
# =============================================================================

# Expose a Ray actor class only when Ray is available.
if ray is not None:
    RayEnvWorker = ray.remote(RayEnvRunner)  # type: ignore[attr-defined]
else:  # pragma: no cover
    RayEnvWorker = None  # type: ignore


class RayLearner:
    """
    Learner-side orchestrator that manages RayEnvWorker actors.

    This class lives on the learner process and coordinates multiple environment
    workers to collect experience in parallel.

    Responsibilities
    ----------------
    - Create N RayEnvRunner actors (workers).
    - Broadcast policy weights to workers.
    - Optionally broadcast environment state (e.g., NormalizeWrapper stats).
    - Request rollout chunks and flatten them into a single transition list.

    Notes
    -----
    - This orchestrator is intentionally lightweight: it does not own replay buffers
      or optimization logic; it only coordinates data collection.
    - Ray must be installed; otherwise `_require_ray()` raises.
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
            Serializable policy construction spec. Broadcast to workers via ray.put.
        n_workers : int
            Number of Ray workers.
        steps_per_worker : int
            Steps collected per worker per `collect()` call (default chunk size).
        base_seed : int, default=0
            Base seed for per-worker seeding:
                worker_seed_i = base_seed + i
        max_episode_steps : int, optional
            Fallback episode-length cap in workers when env doesn't provide truncation.
        want_onpolicy_extras : bool, default=False
            If True, workers request policy extras (log_prob/value).
        flatten_obs : bool, default=False
            Forwarded to workers' `_unpack_step(...)`.
        obs_dtype : Any, default=np.float32
            Forwarded to workers' `_unpack_step(...)`.

        Raises
        ------
        RuntimeError
            If Ray is not installed or Ray actor class is unavailable.
        """
        _require_ray()
        if RayEnvWorker is None:  # pragma: no cover
            raise RuntimeError("RayEnvWorker is unavailable because Ray is not installed.")

        self.n_workers = int(n_workers)
        self.steps_per_worker = int(steps_per_worker)

        # Avoid shipping the full spec repeatedly; store once in the object store.
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
            State dict payload. It will be normalized to CPU tensors before ray.put.
        """
        sd = _to_cpu_state_dict(policy_state_dict_cpu)
        sd_ref = ray.put(sd)
        ray.get([w.set_policy_weights.remote(sd_ref) for w in self.workers])

    def broadcast_env_state(self, env_state: Mapping[str, Any]) -> None:
        """
        Broadcast env/wrapper state (e.g., normalization running stats) to all workers.

        Parameters
        ----------
        env_state : Mapping[str, Any]
            Environment state payload produced on learner side (e.g., env.state_dict()).
        """
        st = dict(env_state)
        st_ref = ray.put(st)
        ray.get([w.set_env_state.remote(st_ref) for w in self.workers])

    def collect(self, deterministic: bool = False, n_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Collect rollout chunks from all workers and flatten them into one list.

        Parameters
        ----------
        deterministic : bool, default=False
            If True, request deterministic actions from workers.
        n_steps : int, optional
            Override `steps_per_worker` for this call only.

        Returns
        -------
        transitions : list[dict[str, Any]]
            Flattened list of transitions from all workers. The transition schema
            is the same as returned by RayEnvRunner.rollout(...).
        """
        steps = self.steps_per_worker if n_steps is None else int(n_steps)
        futs = [w.rollout.remote(steps, deterministic=bool(deterministic)) for w in self.workers]
        chunks = ray.get(futs)

        out: List[Dict[str, Any]] = []
        for c in chunks:
            out.extend(c)
        return out
