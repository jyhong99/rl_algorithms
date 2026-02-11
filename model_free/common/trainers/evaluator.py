from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from ..utils.train_utils import _env_reset, _unpack_step, _to_action_np


class Evaluator:
    """
    RL policy evaluator via episodic rollouts.

    This utility runs episode rollouts in an environment using a given agent/policy
    and reports aggregate episode-return statistics.

    The design goal is robustness: missing methods/attributes are tolerated.
    Training/eval mode switches are performed in a best-effort manner.

    Expected interfaces (duck-typed)
    --------------------------------
    env:
        - reset(**kwargs) -> obs
        - step(action) -> step_output
        - optional: action_space (Gym-like)
        - optional: set_training(bool) to freeze running stats (e.g., normalization)

    agent:
        - act(obs, deterministic: bool) -> action
        - optional: set_training(bool)
        - optional: attribute `training` (bool-like) to snapshot current mode

    Notes
    -----
    - Step output formatting is delegated to `_unpack_step(...)`.
    - Action formatting uses `_to_action_np(...)` to normalize agent outputs.
    - If we cannot snapshot a previous training flag, we keep the object in eval mode
      after evaluation (conservative choice).
    """

    def __init__(
        self,
        env: Any,
        *,
        episodes: int = 10,
        deterministic: bool = True,
        show_progress: bool = True,
        max_episode_steps: Optional[int] = None,
        base_seed: Optional[int] = None,
        seed_increment: int = 1,
        # Step unpacking / observation formatting
        flatten_obs: bool = False,
        obs_dtype: Any = np.float32,
    ) -> None:
        """
        Parameters
        ----------
        env : Any
            Environment instance supporting `reset` and `step`.
        episodes : int, default=10
            Number of evaluation episodes to run.
        deterministic : bool, default=True
            If True, run the agent in deterministic mode (e.g., mean action).
        show_progress : bool, default=True
            If True and tqdm is available, display a progress bar.
        max_episode_steps : int, optional
            If set, hard-truncate each episode after this many environment steps.
        base_seed : int, optional
            If provided, each episode reset will use:
                seed = base_seed + ep * seed_increment
        seed_increment : int, default=1
            Episode-to-episode seed increment when `base_seed` is provided.
            Must be positive when `base_seed` is set.
        flatten_obs : bool, default=False
            Forwarded to `_unpack_step(...)`. If True, flattens observations.
        obs_dtype : Any, default=np.float32
            Forwarded to `_unpack_step(...)`. Cast/convert observation dtype.

        Raises
        ------
        ValueError
            If `episodes <= 0`, or if `max_episode_steps` is non-positive,
            or if `base_seed` is set while `seed_increment <= 0`.
        """
        self.env = env

        self.episodes = int(episodes)
        self.deterministic = bool(deterministic)
        self.show_progress = bool(show_progress)

        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.base_seed = None if base_seed is None else int(base_seed)
        self.seed_increment = int(seed_increment)

        self.flatten_obs = bool(flatten_obs)
        self.obs_dtype = obs_dtype

        self._validate()

    def _validate(self) -> None:
        """
        Validate constructor arguments.

        Raises
        ------
        ValueError
            If arguments are inconsistent or invalid.
        """
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        if self.base_seed is not None and self.seed_increment <= 0:
            raise ValueError(
                f"seed_increment must be positive when base_seed is set, got {self.seed_increment}"
            )

    def evaluate(self, agent: Any) -> Dict[str, float]:
        """
        Evaluate `agent` for a fixed number of episodes.

        Parameters
        ----------
        agent : Any
            Agent/policy object implementing:
                act(obs, deterministic: bool) -> action

        Returns
        -------
        metrics : dict[str, float]
            Dictionary with:
            - "eval/return_mean": mean episode return across episodes
            - "eval/return_std" : standard deviation of episode return

        Notes
        -----
        - This method attempts to set both `env` and `agent` into evaluation mode
          via `set_training(False)` if available.
        - Previous `training` flags are best-effort snapshotted and restored.
        """
        prev_agent_training = self._snapshot_training_flag(agent)
        prev_env_training = self._snapshot_training_flag(self.env)

        # Freeze env running stats (e.g., observation normalization) during evaluation.
        self._set_training(self.env, False)

        returns = np.empty(self.episodes, dtype=np.float64)

        # Progress bar (optional)
        ep_iter = range(self.episodes)
        pbar = None
        if self.show_progress and (tqdm is not None):
            pbar = tqdm(ep_iter, desc="Eval", unit="ep", leave=False, dynamic_ncols=True)
            ep_iter = pbar

        # Cache action space interpretation once (Gym-style envs typically fixed).
        is_discrete, action_shape = self._infer_action_space(self.env)

        try:
            self._set_training(agent, False)

            for ep in ep_iter:
                ep = int(ep)
                obs = _env_reset(self.env, **self._reset_kwargs_for_episode(ep))
                ep_return = self._run_episode(
                    agent=agent,
                    obs0=obs,
                    is_discrete=is_discrete,
                    action_shape=action_shape,
                )
                returns[ep] = ep_return

                if pbar is not None:
                    pbar.set_postfix({"ret": f"{ep_return:.2f}"}, refresh=False)

        finally:
            # Restore training flags (best-effort).
            self._restore_training_flag(agent, prev_agent_training)
            self._restore_training_flag(self.env, prev_env_training)

            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass

        return {
            "eval/return_mean": float(np.mean(returns)),
            "eval/return_std": float(np.std(returns)),
        }

    # ------------------------------------------------------------------
    # Rollout primitives
    # ------------------------------------------------------------------
    def _run_episode(
        self,
        *,
        agent: Any,
        obs0: Any,
        is_discrete: bool,
        action_shape: Optional[Tuple[int, ...]],
    ) -> float:
        """
        Run a single episode rollout starting from an initial observation.

        Parameters
        ----------
        agent : Any
            Policy object with `act(...)`.
        obs0 : Any
            Initial observation returned by env.reset(...).
        is_discrete : bool
            If True, env expects an integer action.
        action_shape : tuple[int, ...], optional
            Expected shape for continuous action vectors.

        Returns
        -------
        ep_return : float
            Sum of rewards over the episode (optionally truncated).
        """
        obs = obs0
        ep_return = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.act(obs, deterministic=self.deterministic)
            action_env = self._format_action_for_env(
                action=action,
                is_discrete=is_discrete,
                action_shape=action_shape,
            )

            step_out = self.env.step(action_env)
            obs, reward, done, _info = _unpack_step(
                step_out,
                flatten_obs=self.flatten_obs,
                obs_dtype=self.obs_dtype,
            )

            ep_return += float(reward)
            steps += 1

            if self.max_episode_steps is not None and steps >= self.max_episode_steps:
                done = True

        return ep_return

    # ------------------------------------------------------------------
    # Training-flag handling (best-effort)
    # ------------------------------------------------------------------
    @staticmethod
    def _snapshot_training_flag(obj: Any) -> Optional[bool]:
        """
        Snapshot `obj.training` if present and readable.

        Parameters
        ----------
        obj : Any
            Object that may expose a `training` attribute.

        Returns
        -------
        flag : bool or None
            - bool: previous training state if attribute exists and is readable
            - None: if attribute doesn't exist or cannot be read safely
        """
        try:
            val = getattr(obj, "training")
            return None if val is None else bool(val)
        except Exception:
            return None

    @staticmethod
    def _set_training(obj: Any, training: bool) -> None:
        """
        Best-effort: call `obj.set_training(training)` if available.

        Parameters
        ----------
        obj : Any
            Object that may define `set_training(bool)`.
        training : bool
            Desired training flag (True for training mode, False for eval mode).

        Notes
        -----
        Failures are intentionally silenced to keep evaluation robust.
        """
        fn = getattr(obj, "set_training", None)
        if callable(fn):
            try:
                fn(bool(training))
            except Exception:
                pass

    @classmethod
    def _restore_training_flag(cls, obj: Any, prev_flag: Optional[bool]) -> None:
        """
        Restore training state if it was previously snapshotted.

        Parameters
        ----------
        obj : Any
            Target object (env or agent).
        prev_flag : bool or None
            The snapshotted training flag.

        Notes
        -----
        If `prev_flag` is None (unknown), we keep the object in eval mode
        (training=False) to avoid accidentally enabling training-time behavior.
        """
        if prev_flag is None:
            cls._set_training(obj, False)
        else:
            cls._set_training(obj, bool(prev_flag))

    # ------------------------------------------------------------------
    # Reset / action-space / action formatting
    # ------------------------------------------------------------------
    def _reset_kwargs_for_episode(self, ep: int) -> Dict[str, Any]:
        """
        Build reset keyword arguments for a given episode index.

        Parameters
        ----------
        ep : int
            Episode index in [0, episodes).

        Returns
        -------
        kwargs : dict[str, Any]
            - {} if `base_seed` is None
            - {"seed": base_seed + ep * seed_increment} otherwise
        """
        if self.base_seed is None:
            return {}
        seed = int(self.base_seed) + int(ep) * int(self.seed_increment)
        return {"seed": seed}

    @staticmethod
    def _infer_action_space(env: Any) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        """
        Infer action-space type and expected continuous action shape.

        Parameters
        ----------
        env : Any
            Environment that may expose `action_space` in a Gym-like manner.

        Returns
        -------
        is_discrete : bool
            True if `env.action_space` exists and has attribute `n` (Discrete space).
        action_shape : tuple[int, ...] or None
            If continuous, returns `env.action_space.shape` when available.

        Notes
        -----
        - This is heuristic (duck-typed). It does not require Gym imports.
        - If we cannot infer a shape for continuous actions, `action_shape` is None,
          and `_to_action_np` is expected to handle it reasonably.
        """
        action_space = getattr(env, "action_space", None)
        is_discrete = bool(action_space is not None and hasattr(action_space, "n"))
        if is_discrete:
            return True, None

        try:
            shp = getattr(action_space, "shape", None)
            if isinstance(shp, tuple):
                return False, tuple(int(x) for x in shp)
        except Exception:
            pass

        return False, None

    @staticmethod
    def _format_action_for_env(
        *,
        action: Any,
        is_discrete: bool,
        action_shape: Optional[Tuple[int, ...]],
    ) -> Any:
        """
        Convert agent output into an env-compatible action.

        Parameters
        ----------
        action : Any
            Raw output from `agent.act(...)`. Can be scalar, list, np.ndarray,
            torch tensor, etc.
        is_discrete : bool
            Whether env expects an integer action.
        action_shape : tuple[int, ...], optional
            Expected shape for continuous action vectors.

        Returns
        -------
        action_env : Any
            Action formatted for `env.step(...)`:
            - Discrete: int
            - Continuous: numpy array (shape ~ action_shape if provided)

        Notes
        -----
        Discrete conversion rule:
            - Convert to numpy, flatten, and take the first element as int.
            - This matches common agent outputs that may return [a] or np.array([a]).
        """
        if is_discrete:
            a = _to_action_np(action, action_shape=None)
            a = np.asarray(a).reshape(-1)
            return int(a[0])

        return _to_action_np(action, action_shape=action_shape)
