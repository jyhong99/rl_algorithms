from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Mapping

import numpy as np

from ..utils import MinimalWrapper, RunningMeanStd
from ..utils.common_utils import to_action_np


# =============================================================================
# Gym/Gymnasium compatibility shim
# =============================================================================

try:  # pragma: no cover
    import gymnasium as gym  # type: ignore
    BaseGymWrapper = gym.Wrapper
    _HAS_GYMNASIUM = True
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        BaseGymWrapper = gym.Wrapper  # gym.core.Wrapper also works; gym.Wrapper is standard
        _HAS_GYMNASIUM = False
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        BaseGymWrapper = MinimalWrapper
        _HAS_GYMNASIUM = False


class NormalizeWrapper(BaseGymWrapper):
    """
    Online normalization wrapper for Gym/Gymnasium environments.

    Features
    --------
    1) Observation normalization (optional)
       - Maintains running mean/variance (RunningMeanStd) and applies:
         (obs - mean) / (std + eps), with optional clipping.

    2) Reward normalization (optional)
       - Maintains a discounted-return accumulator:
           R_t = gamma * R_{t-1} + r_t
       - Updates return RMS from R_t (training only) and returns:
           r_norm = r / (std(R) + eps)
       - Optional clipping and configurable reset policy.

    3) Action handling for Box-like action spaces (optional)
       - action_rescale=True: expects actions in [-1, 1], maps to [low, high]
       - else: clip_action > 0 clips to [low, high]

    4) Time-limit truncation harmonization (best-effort)
       - Gymnasium (5-tuple): uses terminated/truncated as returned, and can
         optionally enforce max_episode_steps if provided.
       - Gym (4-tuple): synthesizes/propagates info["TimeLimit.truncated"] and
         can force done=True when the time limit is reached.

    Parameters
    ----------
    env : Any
        Environment to wrap.
    obs_shape : Tuple[int, ...]
        Expected observation shape for array-like observations.
        (Structured obs: dict/tuple/list are passed through unchanged.)
    norm_obs : bool, optional
        Enable observation normalization.
    norm_reward : bool, optional
        Enable reward normalization (use with care for off-policy).
    clip_obs : float, optional
        If > 0, clip normalized observations to [-clip_obs, clip_obs].
    clip_reward : float, optional
        If > 0, clip normalized rewards to [-clip_reward, clip_reward].
    gamma : float, optional
        Discount factor used in return accumulator for reward normalization.
    epsilon : float, optional
        Numerical stability constant.
    training : bool, optional
        If True, updates running stats (obs_rms / ret_rms).
    max_episode_steps : Optional[int], optional
        If provided, wrapper may enforce truncation at this horizon.
    action_rescale : bool, optional
        If True, rescale actions from [-1,1] to [low,high] for Box-like spaces.
    clip_action : float, optional
        If > 0 and action_rescale is False, clip actions to [low,high].
        (Magnitude is not used; it acts as an on/off switch.)
    reset_return_on_done : bool, optional
        Reset discounted return accumulator when done boundary occurs.
    reset_return_on_trunc : bool, optional
        Reset discounted return accumulator when truncation boundary occurs.
    obs_dtype : Any, optional
        Dtype enforced for array-like observations (e.g., np.float32).
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

        self.obs_shape = tuple(obs_shape)
        self.obs_dtype = obs_dtype

        self.norm_obs = bool(norm_obs)
        self.norm_reward = bool(norm_reward)

        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)

        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.training = bool(training)

        # Running statistics
        self.obs_rms: Optional[RunningMeanStd] = RunningMeanStd(shape=self.obs_shape) if self.norm_obs else None
        self.ret_rms: Optional[RunningMeanStd] = RunningMeanStd(shape=()) if self.norm_reward else None

        # Discounted return accumulator for reward normalization
        self._running_return: float = 0.0

        # Episode length tracking (for optional time-limit enforcement)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._ep_len: int = 0

        # Action handling configuration
        self.action_rescale = bool(action_rescale)
        self.clip_action = float(clip_action)

        # Reward-return reset policy
        self.reset_return_on_done = bool(reset_return_on_done)
        self.reset_return_on_trunc = bool(reset_return_on_trunc)

        # Cache action space metadata (best-effort; dependency-free)
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
        """Enable/disable running-stat updates."""
        self.training = bool(training)

    # -------------------------------------------------------------------------
    # Action formatting
    # -------------------------------------------------------------------------

    def _format_action(self, action: Any) -> Any:
        """
        Apply action rescaling and/or clipping for Box-like action spaces.

        Policy
        ------
        - If action_rescale=True:
            * expects action in [-1, 1]
            * maps to [low, high]
        - Else if clip_action > 0:
            * clips to [low, high]
        - Non-Box actions: returned unchanged.

        Notes
        -----
        - Uses to_action_np(...) to tolerate torch tensors / scalars.
        - Always returns np.float32 for Box actions for consistent env interfacing.
        """
        if not self._is_box_action:
            return action

        a_shape = tuple(getattr(self._action_space, "shape", ()))
        a = to_action_np(action, action_shape=a_shape).astype(np.float32, copy=False)

        low = np.asarray(self._action_space.low, dtype=np.float32)
        high = np.asarray(self._action_space.high, dtype=np.float32)

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
        Normalize observations (best-effort).

        Policy
        ------
        - dict/tuple/list observations are passed through unchanged.
        - array-like observations are converted to obs_dtype.
        - if norm_obs=True and training=True, updates obs_rms with batch dim.
        - applies RMS normalization and optional clipping.

        Raises
        ------
        ValueError
            If array-like obs cannot be converted, or shape mismatches obs_shape.
        """
        if isinstance(obs, (dict, tuple, list)):
            return obs

        # No RMS: dtype enforcement only (best-effort)
        if self.obs_rms is None:
            try:
                return np.asarray(obs, dtype=self.obs_dtype)
            except Exception:
                return obs

        # RMS normalization path (array-like)
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
            Scalar reward (will be cast to float).
        done_flag : bool
            True if episode boundary occurred (terminated or truncated).
        truncated_flag : bool
            True if boundary was a time-limit truncation.

        Returns
        -------
        float
            Normalized reward (or raw reward if norm_reward=False).
        """
        if self.ret_rms is None:
            return float(reward)

        r = float(reward)

        # discounted return accumulator
        self._running_return = self.gamma * self._running_return + r

        if self.training:
            self.ret_rms.update(np.asarray([self._running_return], dtype=np.float64))

        std = float(self.ret_rms.std(eps=self.epsilon))  # scalar
        r_norm = r / (std + self.epsilon)

        if self.clip_reward > 0.0:
            r_norm = float(np.clip(r_norm, -self.clip_reward, self.clip_reward))

        # reset policy for return accumulator
        if (done_flag and self.reset_return_on_done) or (truncated_flag and self.reset_return_on_trunc):
            self._running_return = 0.0

        return float(r_norm)

    # -------------------------------------------------------------------------
    # Gym/Gymnasium API
    # -------------------------------------------------------------------------

    def reset(self, **kwargs) -> Any:
        """
        Reset env and normalize initial observation.

        Returns
        -------
        - Gymnasium: (obs, info)
        - Gym: obs
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
        Step env with optional action formatting and obs/reward normalization.

        Returns
        -------
        - Gymnasium: (obs, reward, terminated, truncated, info)
        - Gym: (obs, reward, done, info)
        """
        action = self._format_action(action)
        out = self.env.step(action)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated_b = bool(terminated)
            truncated_b = bool(truncated)

            self._ep_len += 1

            # Optional local enforcement of max_episode_steps (in case TimeLimit wrapper is absent)
            if (
                self.max_episode_steps is not None
                and self._ep_len >= self.max_episode_steps
                and (not terminated_b)
                and (not truncated_b)
                and (not bool(dict(info).get("TimeLimit.truncated", False)))
            ):
                truncated_b = True
                truncated = True
                info = dict(info)
                info["TimeLimit.truncated"] = True

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

        # Synthesize/normalize TimeLimit.truncated when max_episode_steps is provided.
        truncated_flag = bool(info_out.get("TimeLimit.truncated", False))
        if self.max_episode_steps is not None and self._ep_len >= self.max_episode_steps and not truncated_flag:
            truncated_flag = True
            info_out["TimeLimit.truncated"] = True

            # Force episode boundary for Gym 4-tuple to keep semantics consistent
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
        Serialize wrapper state (stats + relevant configuration).

        Notes
        -----
        - `obs_dtype` is stored as a NumPy dtype string (e.g., 'float32').
        - `obs_shape` is stored for sanity/debug; load_state_dict does not override it.
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
        Restore wrapper state.

        Policy
        ------
        - Does not override obs_shape.
        - Restores scalar hyperparameters and normalization stats when enabled.
        - Restores obs_dtype from a NumPy dtype name when possible.
        """
        self._running_return = float(state.get("running_return", 0.0))
        self._ep_len = int(state.get("ep_len", 0))

        # Restore scalar hyperparameters (best-effort; keep current if missing)
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

        # Restore dtype
        if "obs_dtype" in state:
            try:
                self.obs_dtype = np.dtype(str(state["obs_dtype"])).type
            except Exception:
                pass

        # Restore added configs
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

        # Restore running stats (only if the wrapper was constructed with them enabled)
        if self.obs_rms is not None and "obs_rms" in state:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if self.ret_rms is not None and "ret_rms" in state:
            self.ret_rms.load_state_dict(state["ret_rms"])