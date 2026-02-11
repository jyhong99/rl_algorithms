from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..utils import MinimalWrapper, RunningMeanStd
from ..utils.common_utils import _to_action_np

# =============================================================================
# Gym/Gymnasium compatibility shim
# =============================================================================
try:  # pragma: no cover
    import gymnasium as gym  # type: ignore
    BaseGymWrapper = gym.Wrapper
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        BaseGymWrapper = gym.Wrapper
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        BaseGymWrapper = MinimalWrapper


class NormalizeWrapper(BaseGymWrapper):
    """
    Online normalization wrapper for Gym/Gymnasium environments.

    This wrapper standardizes observations and/or rewards using running
    statistics computed online, and optionally harmonizes action formatting
    and time-limit semantics across Gym and Gymnasium.

    Overview
    --------
    The wrapper provides four orthogonal features:

    1. Observation normalization (optional)
       Maintains running mean/variance via :class:`~RunningMeanStd` and applies::

           obs_norm = (obs - mean) / (std + epsilon)

       with optional clipping to ``[-clip_obs, clip_obs]``.

    2. Reward normalization (optional)
       Tracks a discounted return accumulator::

           R_t = gamma * R_{t-1} + r_t

       Updates an RMS of ``R_t`` (training only) and returns::

           r_norm = r / (std(R) + epsilon)

       with optional clipping to ``[-clip_reward, clip_reward]`` and a configurable
       reset policy for the return accumulator.

    3. Box-like action handling (optional)
       - If ``action_rescale=True``: expects actions in ``[-1, 1]`` and maps to the
         environment action bounds ``[low, high]``.
       - Else, if ``clip_action > 0``: clips actions to ``[low, high]``.
       Non-Box action spaces are passed through unchanged.

    4. Time-limit truncation harmonization (best-effort)
       - Gymnasium (5-tuple): preserves ``terminated``/``truncated`` and can optionally
         enforce ``max_episode_steps`` if provided.
       - Gym (4-tuple): synthesizes/propagates ``info["TimeLimit.truncated"]`` and can
         force ``done=True`` when the time limit is reached.

    Parameters
    ----------
    env : Any
        Environment to wrap.
    obs_shape : tuple[int, ...]
        Expected observation shape for array-like observations. Structured observations
        (dict/tuple/list) are passed through unchanged.
    norm_obs : bool, default=True
        If True, normalize array-like observations using running RMS statistics.
    norm_reward : bool, default=False
        If True, normalize scalar rewards using return RMS statistics. Use with care
        in off-policy settings.
    clip_obs : float, default=10.0
        If > 0, clip normalized observations to ``[-clip_obs, clip_obs]``.
    clip_reward : float, default=10.0
        If > 0, clip normalized rewards to ``[-clip_reward, clip_reward]``.
    gamma : float, default=0.99
        Discount factor used for the return accumulator in reward normalization.
    epsilon : float, default=1e-8
        Numerical stability constant for normalization denominators.
    training : bool, default=True
        If True, update running statistics (obs/return RMS). If False, use frozen stats.
    max_episode_steps : int, optional
        If provided, the wrapper may enforce a truncation boundary at this horizon.
        This is meant as a fallback when a TimeLimit wrapper is not present.
    action_rescale : bool, default=False
        If True and the action space is Box-like, rescale policy outputs in ``[-1, 1]``
        to the environment bounds ``[low, high]``.
    clip_action : float, default=0.0
        If > 0 and ``action_rescale=False``, clip actions to ``[low, high]`` for Box-like
        spaces. The magnitude is not used; this acts as an enable/disable switch.
    reset_return_on_done : bool, default=True
        If True, reset discounted return accumulator when an episode boundary occurs.
    reset_return_on_trunc : bool, default=True
        If True, reset discounted return accumulator when a time-limit truncation occurs.
    obs_dtype : Any, default=np.float32
        Dtype enforced for array-like observations (e.g., ``np.float32``).

    Attributes
    ----------
    obs_rms : RunningMeanStd or None
        Running statistics for observations, if ``norm_obs=True``.
    ret_rms : RunningMeanStd or None
        Running statistics for discounted returns, if ``norm_reward=True``.
    training : bool
        Whether running statistics are being updated.

    Notes
    -----
    - This wrapper is intentionally "best-effort" across Gym/Gymnasium versions.
    - Structured observations (dict/tuple/list) are not normalized by default; if you need
      structured obs normalization, implement it upstream (e.g., a flattening wrapper).
    - The wrapper tries to be dependency-light; it avoids relying on Gym space classes
      directly and instead inspects attributes (low/high/shape).
    """

    def __init__(
        self,
        env: Any,
        obs_shape: Tuple[int, ...],
        *,
        norm_obs: bool = True,
        norm_reward: bool = False,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        training: bool = True,
        max_episode_steps: Optional[int] = None,
        action_rescale: bool = False,
        clip_action: float = 0.0,
        reset_return_on_done: bool = True,
        reset_return_on_trunc: bool = True,
        obs_dtype: Any = np.float32,
    ) -> None:
        super().__init__(env)

        # ---- observation normalization ----
        self.obs_shape = tuple(obs_shape)
        self.obs_dtype = obs_dtype

        self.norm_obs = bool(norm_obs)
        self.norm_reward = bool(norm_reward)

        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)

        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.training = bool(training)

        self.obs_rms: Optional[RunningMeanStd] = RunningMeanStd(shape=self.obs_shape) if self.norm_obs else None
        self.ret_rms: Optional[RunningMeanStd] = RunningMeanStd(shape=()) if self.norm_reward else None

        # ---- reward normalization state ----
        self._running_return: float = 0.0

        # ---- time-limit tracking (optional) ----
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._ep_len: int = 0

        # ---- action handling ----
        self.action_rescale = bool(action_rescale)
        self.clip_action = float(clip_action)

        # ---- return reset policy ----
        self.reset_return_on_done = bool(reset_return_on_done)
        self.reset_return_on_trunc = bool(reset_return_on_trunc)

        # ---- cache action space metadata (dependency-free) ----
        self._action_space = getattr(self.env, "action_space", None)
        self._is_box_action = (
            self._action_space is not None
            and hasattr(self._action_space, "low")
            and hasattr(self._action_space, "high")
            and hasattr(self._action_space, "shape")
        )

        if self.action_rescale and not self._is_box_action:
            raise ValueError("action_rescale=True requires a Box-like action_space with low/high/shape.")

    # -------------------------------------------------------------------------
    # Public controls
    # -------------------------------------------------------------------------

    def set_training(self, training: bool) -> None:
        """
        Set whether running statistics are updated.

        Parameters
        ----------
        training : bool
            If True, update running statistics (obs_rms/ret_rms). If False, keep them frozen.
        """
        self.training = bool(training)

    # -------------------------------------------------------------------------
    # Action formatting
    # -------------------------------------------------------------------------

    def _format_action(self, action: Any) -> Any:
        """
        Format actions for Box-like action spaces.

        Parameters
        ----------
        action : Any
            Action produced by a policy. May be a NumPy array, Python scalar,
            torch tensor, or an array-like container.

        Returns
        -------
        action_env : Any
            Environment-compatible action.
            - For Box-like spaces: returns ``np.ndarray`` of dtype ``np.float32``.
            - Otherwise: returns the input action unchanged.

        Notes
        -----
        - If ``action_rescale=True``, the wrapper expects policy outputs in ``[-1, 1]``.
        - If ``clip_action > 0`` and rescaling is disabled, the wrapper clips to bounds.
        - For Box-like spaces, this method always returns a float32 NumPy array for
          consistent env interfacing.
        """
        if not self._is_box_action:
            return action

        a_shape = tuple(getattr(self._action_space, "shape", ()))
        a = _to_action_np(action, action_shape=a_shape).astype(np.float32, copy=False)

        low = np.asarray(getattr(self._action_space, "low"), dtype=np.float32)
        high = np.asarray(getattr(self._action_space, "high"), dtype=np.float32)

        # Broadcast low/high to action shape if needed
        if low.shape != a.shape:
            low = np.broadcast_to(low, a.shape)
        if high.shape != a.shape:
            high = np.broadcast_to(high, a.shape)

        if self.action_rescale:
            a = np.clip(a, -1.0, 1.0)
            a = low + (a + 1.0) * 0.5 * (high - low)
            return a.astype(np.float32, copy=False)

        if self.clip_action > 0.0:
            a = np.clip(a, low, high)
            return a.astype(np.float32, copy=False)

        return action

    # -------------------------------------------------------------------------
    # Observation / reward normalization
    # -------------------------------------------------------------------------

    def _normalize_obs(self, obs: Any) -> Any:
        """
        Normalize observation (best-effort).

        Parameters
        ----------
        obs : Any
            Observation produced by the environment. If it is dict/tuple/list,
            it is returned unchanged. If it is array-like, it is normalized when
            ``norm_obs=True`` and the wrapper has an ``obs_rms``.

        Returns
        -------
        obs_out : Any
            Normalized observation for array-like inputs, or unchanged structured input.

        Raises
        ------
        ValueError
            If array-like observation cannot be converted to ndarray, or if its
            shape differs from ``obs_shape``.
        """
        if isinstance(obs, (dict, tuple, list)):
            return obs

        if self.obs_rms is None:
            # dtype enforcement only (best-effort)
            try:
                return np.asarray(obs, dtype=self.obs_dtype)
            except Exception:
                return obs

        try:
            obs_np = np.asarray(obs, dtype=self.obs_dtype)
        except Exception as e:
            raise ValueError(f"Could not convert observation to ndarray: {type(obs)}") from e

        if obs_np.shape != self.obs_shape:
            raise ValueError(f"Expected obs shape {self.obs_shape}, got {obs_np.shape}")

        if self.training:
            self.obs_rms.update(obs_np[None, ...])

        clip = self.clip_obs if self.clip_obs > 0.0 else None
        obs_norm = self.obs_rms.normalize(obs_np, clip=clip, eps=self.epsilon).astype(self.obs_dtype, copy=False)
        return obs_norm

    def _normalize_reward(self, reward: Any, *, done_flag: bool, truncated_flag: bool) -> float:
        """
        Normalize reward using return RMS (if enabled).

        Parameters
        ----------
        reward : Any
            Scalar reward. Will be cast to float.
        done_flag : bool
            True if an episode boundary occurred (terminated or truncated).
        truncated_flag : bool
            True if the boundary was a time-limit truncation.

        Returns
        -------
        reward_out : float
            Normalized reward if enabled, otherwise the raw reward as float.

        Notes
        -----
        - Uses a discounted return accumulator ``R_t`` to estimate return scale.
        - RMS statistics are updated only when ``training=True``.
        - The return accumulator reset behavior is controlled by:
          ``reset_return_on_done`` and ``reset_return_on_trunc``.
        """
        if self.ret_rms is None:
            return float(reward)

        r = float(reward)

        self._running_return = self.gamma * self._running_return + r
        if self.training:
            self.ret_rms.update(np.asarray([self._running_return], dtype=np.float64))

        std = float(self.ret_rms.std(eps=self.epsilon))
        r_norm = r / (std + self.epsilon)

        if self.clip_reward > 0.0:
            r_norm = float(np.clip(r_norm, -self.clip_reward, self.clip_reward))

        if (done_flag and self.reset_return_on_done) or (truncated_flag and self.reset_return_on_trunc):
            self._running_return = 0.0

        return float(r_norm)

    # -------------------------------------------------------------------------
    # Gym/Gymnasium API
    # -------------------------------------------------------------------------

    def reset(self, **kwargs) -> Any:
        """
        Reset environment and normalize the initial observation.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to ``env.reset(**kwargs)``. This commonly includes ``seed`` and
            ``options`` depending on Gym/Gymnasium version.

        Returns
        -------
        out : Any
            - Gymnasium: ``(obs, info)``
            - Gym: ``obs``

        Notes
        -----
        - Always resets the reward return accumulator and the internal episode-length counter.
        - Normalizes the returned observation according to ``norm_obs``.
        """
        out = self.env.reset(**kwargs)
        self._running_return = 0.0
        self._ep_len = 0

        # Gymnasium: (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            obs = self._normalize_obs(obs)
            return obs, info

        # Gym: obs
        obs = out
        obs = self._normalize_obs(obs)
        return obs

    def step(self, action: Any) -> Any:
        """
        Step environment with optional action formatting and normalization.

        Parameters
        ----------
        action : Any
            Policy action passed to the wrapped environment.

        Returns
        -------
        out : Any
            - Gymnasium: ``(obs, reward, terminated, truncated, info)``
            - Gym: ``(obs, reward, done, info)``

        Notes
        -----
        - Applies action rescaling/clipping for Box-like action spaces.
        - Normalizes observations and rewards depending on configuration.
        - Harmonizes time-limit truncation best-effort:
          * For Gymnasium, may set ``truncated=True`` and annotate ``info["TimeLimit.truncated"]=True``.
          * For Gym, may force ``done=True`` and annotate ``info_out["TimeLimit.truncated"]=True``.
        """
        action_env = self._format_action(action)
        out = self.env.step(action_env)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated_b = bool(terminated)
            truncated_b = bool(truncated)

            self._ep_len += 1

            # Optional local enforcement of max_episode_steps
            if self.max_episode_steps is not None and self._ep_len >= self.max_episode_steps:
                if (not terminated_b) and (not truncated_b):
                    # Don't override an existing TimeLimit signal if present
                    info_map = dict(info)
                    if not bool(info_map.get("TimeLimit.truncated", False)):
                        truncated_b = True
                        truncated = True
                        info_map["TimeLimit.truncated"] = True
                        info = info_map

            done_flag = terminated_b or truncated_b
            truncated_flag = truncated_b or bool(dict(info).get("TimeLimit.truncated", False))

            obs = self._normalize_obs(obs)
            reward_f = self._normalize_reward(reward, done_flag=done_flag, truncated_flag=truncated_flag)

            if done_flag:
                self._ep_len = 0

            return obs, reward_f, terminated, truncated, info

        # Gym: (obs, reward, done, info)
        obs, reward, done, info = out
        done_b = bool(done)

        self._ep_len += 1
        info_out: Dict[str, Any] = dict(info) if isinstance(info, Mapping) else {"info": info}

        truncated_flag = bool(info_out.get("TimeLimit.truncated", False))

        # Optional local enforcement of max_episode_steps
        if self.max_episode_steps is not None and self._ep_len >= self.max_episode_steps and not truncated_flag:
            truncated_flag = True
            info_out["TimeLimit.truncated"] = True
            # Gym (4-tuple) has no explicit truncated; force boundary to be explicit in `done`.
            if not done_b:
                done = True
                done_b = True

        obs = self._normalize_obs(obs)
        reward_f = self._normalize_reward(reward, done_flag=done_b, truncated_flag=truncated_flag)

        if done_b:
            self._ep_len = 0

        return obs, reward_f, done, info_out

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize wrapper state (running statistics + relevant configuration).

        Returns
        -------
        state : Dict[str, Any]
            JSON-like Python dict that can be used for checkpointing. Contains:
            - scalar hyperparameters (gamma, epsilon, clipping, etc.)
            - episode-length counters and return accumulator
            - action formatting options
            - running statistics state for obs and returns when enabled

        Notes
        -----
        - ``obs_dtype`` is stored as a NumPy dtype name (e.g., ``"float32"``).
        - ``obs_shape`` is stored for sanity/debug; load_state_dict does not override it.
        """
        state: Dict[str, Any] = {
            "obs_shape": self.obs_shape,
            "norm_obs": self.norm_obs,
            "norm_reward": self.norm_reward,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "training": self.training,
            "running_return": float(self._running_return),
            "max_episode_steps": self.max_episode_steps,
            "ep_len": int(self._ep_len),
            "action_rescale": self.action_rescale,
            "clip_action": self.clip_action,
            "reset_return_on_done": self.reset_return_on_done,
            "reset_return_on_trunc": self.reset_return_on_trunc,
            "obs_dtype": np.dtype(self.obs_dtype).name,
        }

        if self.obs_rms is not None:
            state["obs_rms"] = self.obs_rms.state_dict()

        if self.ret_rms is not None:
            state["ret_rms"] = self.ret_rms.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore wrapper state from a checkpoint payload.

        Parameters
        ----------
        state : Dict[str, Any]
            State dict produced by :meth:`state_dict`.

        Notes
        -----
        - Does not override ``obs_shape`` (the wrapper is assumed to be constructed correctly).
        - Restores running stats only if the wrapper was constructed with normalization enabled.
        - Restores ``obs_dtype`` from the stored NumPy dtype name when possible.
        """
        self._running_return = float(state.get("running_return", 0.0))
        self._ep_len = int(state.get("ep_len", 0))

        # Scalar hyperparameters (best-effort)
        if "clip_obs" in state:
            self.clip_obs = float(state["clip_obs"])
        if "clip_reward" in state:
            self.clip_reward = float(state["clip_reward"])
        if "gamma" in state:
            self.gamma = float(state["gamma"])
        if "epsilon" in state:
            self.epsilon = float(state["epsilon"])
        if "training" in state:
            self.training = bool(state["training"])

        # dtype restore
        if "obs_dtype" in state:
            try:
                self.obs_dtype = np.dtype(str(state["obs_dtype"])).type
            except Exception:
                pass

        # additional configs
        if "max_episode_steps" in state:
            v = state["max_episode_steps"]
            self.max_episode_steps = None if v is None else int(v)

        if "action_rescale" in state:
            self.action_rescale = bool(state["action_rescale"])
        if "clip_action" in state:
            self.clip_action = float(state["clip_action"])
        if "reset_return_on_done" in state:
            self.reset_return_on_done = bool(state["reset_return_on_done"])
        if "reset_return_on_trunc" in state:
            self.reset_return_on_trunc = bool(state["reset_return_on_trunc"])

        # running statistics (only if enabled at construction time)
        if self.obs_rms is not None and "obs_rms" in state:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if self.ret_rms is not None and "ret_rms" in state:
            self.ret_rms.load_state_dict(state["ret_rms"])
