from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# tqdm is optional: keep Evaluator importable in minimal environments.
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from ..utils.train_utils import env_reset, unpack_step, to_action_np


class Evaluator:
    """
    RL policy evaluator (episode rollouts).

    This class runs episode rollouts in `env` using `agent.act(...)` and returns
    aggregate statistics of episode returns.

    Expected interfaces (duck-typed)
    --------------------------------
    env:
      - reset(...) and step(action)
      - (optional) action_space compatible with Gym-like spaces
      - (optional) set_training(bool) to freeze running stats (e.g., NormalizeWrapper)

    agent:
      - act(obs, deterministic: bool) -> action
      - (optional) set_training(bool)
      - (optional) training attribute (bool) for snapshotting

    Notes
    -----
    - Best-effort: missing methods/attributes are tolerated.
    - Observation formatting is delegated to `unpack_step(...)`.
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
        # step unpacking
        flatten_obs: bool = False,
        obs_dtype: Any = np.float32,
    ) -> None:
        """
        Parameters
        ----------
        env : Any
            Environment instance.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool
            If True, run agent in deterministic mode (e.g., mean action).
        show_progress : bool
            If True and tqdm is available, show a progress bar.
        max_episode_steps : int, optional
            If set, hard-truncate each episode after this many steps.
        base_seed : int, optional
            If set, each episode reset uses:
            seed = base_seed + ep * seed_increment
        seed_increment : int
            Episode-to-episode seed increment when base_seed is set.
        flatten_obs : bool
            Forwarded to unpack_step(...). If True, flattens observations.
        obs_dtype : Any
            Forwarded to unpack_step(...). Cast/convert observation dtype.
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
        """Validate constructor arguments."""
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        if self.base_seed is not None and self.seed_increment <= 0:
            raise ValueError(f"seed_increment must be positive when base_seed is set, got {self.seed_increment}")

    def evaluate(self, agent: Any) -> Dict[str, float]:
        """
        Evaluate `agent` for `episodes` episodes.

        Parameters
        ----------
        agent : Any
            Agent/policy object with `act(obs, deterministic=...)`.

        Returns
        -------
        metrics : Dict[str, float]
            - eval/return_mean: mean episode return
            - eval/return_std : std episode return
        """
        prev_agent_training = self._snapshot_training_flag(agent)
        prev_env_training = self._snapshot_training_flag(self.env)

        # Freeze env stats (e.g., normalization) during evaluation.
        self._set_training(self.env, False)

        returns = np.empty(self.episodes, dtype=np.float64)

        ep_iter = range(self.episodes)
        pbar = None
        if self.show_progress and (tqdm is not None):
            pbar = tqdm(ep_iter, desc="Eval", unit="ep", leave=False, dynamic_ncols=True)
            ep_iter = pbar

        is_discrete, action_shape = self._infer_action_space(self.env)

        try:
            self._set_training(agent, False)

            for ep in ep_iter:
                obs = env_reset(self.env, **self._reset_kwargs_for_episode(int(ep)))

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
                    obs, reward, done, _info = unpack_step(
                        step_out,
                        flatten_obs=self.flatten_obs,
                        obs_dtype=self.obs_dtype,
                    )
                    ep_return += float(reward)

                    steps += 1
                    if self.max_episode_steps is not None and steps >= self.max_episode_steps:
                        done = True

                returns[int(ep)] = ep_return
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
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _snapshot_training_flag(obj: Any) -> Optional[bool]:
        """
        Best-effort snapshot of `obj.training` if present.

        Returns
        -------
        flag : Optional[bool]
            training state if readable, else None.
        """
        try:
            val = getattr(obj, "training")
            return None if val is None else bool(val)
        except Exception:
            return None

    @staticmethod
    def _set_training(obj: Any, training: bool) -> None:
        """
        Best-effort call to `obj.set_training(training)` if present.
        Silently ignores failures to keep evaluation robust.
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
        Restore `obj` training state if we previously captured it.

        If we could not snapshot the previous flag (prev_flag is None),
        we leave the object in eval mode (training=False) to avoid
        accidentally enabling training-time behavior.
        """
        if prev_flag is None:
            cls._set_training(obj, False)
        else:
            cls._set_training(obj, bool(prev_flag))

    def _reset_kwargs_for_episode(self, ep: int) -> Dict[str, Any]:
        """
        Build reset kwargs for a given episode index.

        Returns
        -------
        kwargs : Dict[str, Any]
            Empty dict if base_seed is None; else {"seed": computed_seed}.
        """
        if self.base_seed is None:
            return {}
        seed = int(self.base_seed) + int(ep) * int(self.seed_increment)
        return {"seed": seed}

    @staticmethod
    def _infer_action_space(env: Any) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        """
        Infer whether env uses a discrete action space and the continuous action shape.

        Returns
        -------
        is_discrete : bool
            True if env.action_space has attribute `n`.
        action_shape : tuple[int, ...] or None
            Shape of continuous action if available; otherwise None.
        """
        action_space = getattr(env, "action_space", None)
        is_discrete = bool(action_space is not None and hasattr(action_space, "n"))

        if is_discrete:
            return True, None

        # Continuous: attempt to read `.shape`.
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
            Raw output from agent.act(...).
        is_discrete : bool
            Whether env expects an integer action.
        action_shape : tuple[int, ...], optional
            Expected shape for continuous actions.

        Returns
        -------
        action_env : Any
            Action in a format accepted by env.step(...).

        Notes
        -----
        - Discrete: coerces to int using the first element.
        - Continuous: uses `to_action_np(action, action_shape=...)`.
        """
        if is_discrete:
            a = to_action_np(action, action_shape=None)
            a = np.asarray(a).reshape(-1)
            return int(a[0])

        return to_action_np(action, action_shape=action_shape)