from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import math

from ..utils.train_utils import _env_reset, _unpack_step, _maybe_call, _format_env_action


def train_single_env(trainer: Any, pbar: Any, msg_pbar: Any) -> None:
    """
    Training loop for a single (non-vectorized) environment.

    This loop performs online interaction with `trainer.train_env` and hands
    transitions to `trainer.algo` step-by-step.

    Responsibilities
    ----------------
    1) Rollout:
         - action selection via `algo.act(obs, deterministic=False)`
         - environment stepping and robust step unpacking via `_unpack_step`
    2) Transition handoff:
         - `algo.on_env_step(transition)`
    3) Callbacks (best-effort):
         - `callbacks.on_step(trainer, transition=...)`
         - `callbacks.on_update(trainer, metrics=...)`
         - if a callback returns False, training exits early
    4) Updates:
         - when `algo.ready_to_update()` is True
         - supports Update-To-Data (UTD): multiple `algo.update()` calls per env step
    5) Logging / progress UI:
         - periodic system-level logging (`sys/` counters)
         - progress bar postfix updates

    Parameters
    ----------
    trainer : Any
        Trainer-like object (duck-typed) providing:
          - train_env : environment with `reset()` and `step(action)`
          - algo : object with:
              act(obs, deterministic: bool) -> action
              on_env_step(transition: Mapping[str, Any]) -> None
              ready_to_update() -> bool
              update() -> Mapping[str, Any] or {}
          - total_env_steps : int
          - global_env_step : int
          - global_update_step : int
          - episode_idx : int
          - utd : float (>= 1 implies multiple updates per env step; capped)
          - callbacks : optional object (invoked via `_maybe_call`)
              - on_step(trainer, transition=...)
              - on_update(trainer, metrics=...)
          - logger : optional object with `log(metrics, step, prefix, pbar=...)`
          - log_every_steps : int (system metrics cadence; 0 disables)
          - flatten_obs : bool (forwarded to `_unpack_step`)
          - step_obs_dtype : dtype (forwarded to `_unpack_step`)
          - max_episode_steps : Optional[int] (trainer-side TimeLimit fallback)
          - _normalize_enabled : bool
              If True, trainer-side TimeLimit is NOT applied here (to avoid
              conflicting with normalization wrappers / TimeLimit wrappers).
          - _episode_stats_enabled : bool
              If True, episode indexing/logging is owned elsewhere.
          - internal counters (created/updated here):
              _ep_return : float
              _ep_len : int
          - optional stop flag:
              _stop_training : bool (if set True externally, loop breaks)

    pbar : Any
        Progress-bar like object with:
          - update(int)
          - set_postfix(dict, refresh=...)
    msg_pbar : Any
        A "message" progress bar passed into logger calls (optional). Typically a
        tqdm instance used by the learner for printing metrics.

    Returns
    -------
    None

    Notes
    -----
    Trainer-side TimeLimit fallback:
      - Applied only when: (not trainer._normalize_enabled) AND trainer.max_episode_steps is set.
      - This mirrors your existing coupling. If you want a cleaner contract,
        replace the condition with an explicit flag like trainer._time_limit_enabled.

    UTD (Update-To-Data):
      - For utd <= 1.0: performs 1 update when ready_to_update() is True.
      - For utd > 1.0 : performs ceil(utd) updates (capped at 1000), stopping early
        if ready_to_update() becomes False.
    """
    env = getattr(trainer, "train_env", None)
    algo = getattr(trainer, "algo", None)

    obs = _env_reset(env)

    trainer._ep_return = 0.0
    trainer._ep_len = 0

    last_sys_log_step = int(getattr(trainer, "global_env_step", 0))
    total_env_steps = int(getattr(trainer, "total_env_steps", 0))

    while int(getattr(trainer, "global_env_step", 0)) < total_env_steps:
        # ------------------------------------------------------------------
        # 1) Act + env.step
        # ------------------------------------------------------------------
        action = algo.act(obs, deterministic=False)

        action_space = getattr(env, "action_space", None)
        action_env = _format_env_action(action, action_space)

        step_out = env.step(action_env)
        next_obs, reward, done, info = _unpack_step(
            step_out,
            flatten_obs=bool(getattr(trainer, "flatten_obs", False)),
            obs_dtype=getattr(trainer, "step_obs_dtype", None),
        )

        # Update global env counter and UI
        trainer.global_env_step = int(getattr(trainer, "global_env_step", 0)) + 1
        _pbar_update_best_effort(pbar, 1)

        # Update episode counters
        trainer._ep_return += float(reward)
        trainer._ep_len += 1

        # ------------------------------------------------------------------
        # 2) Optional trainer-side time limit (fallback)
        # ------------------------------------------------------------------
        done, info_dict = _maybe_apply_trainer_time_limit(trainer, bool(done), info)

        # ------------------------------------------------------------------
        # 3) Transition handoff to algo + callbacks
        # ------------------------------------------------------------------
        transition = _build_transition(
            obs=obs,
            action_env=action_env,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info_dict,
        )
        algo.on_env_step(transition)

        if getattr(trainer, "callbacks", None) is not None:
            cont = _maybe_call(trainer.callbacks, "on_step", trainer, transition=transition)
            if cont is False:
                return

        obs = next_obs

        # ------------------------------------------------------------------
        # 4) Episode boundary
        # ------------------------------------------------------------------
        if done:
            _on_episode_end(trainer)
            obs = _env_reset(env)

        # ------------------------------------------------------------------
        # 5) Updates (UTD)
        # ------------------------------------------------------------------
        if algo.ready_to_update():
            _run_updates(trainer, pbar=msg_pbar)

        # ------------------------------------------------------------------
        # 6) Periodic system logging + UI postfix
        # ------------------------------------------------------------------
        last_sys_log_step = _maybe_log_sys(trainer, last_sys_log_step, pbar=msg_pbar)
        _maybe_set_pbar_postfix(trainer, pbar)

        if getattr(trainer, "_stop_training", False):
            break


# =============================================================================
# Internal helpers
# =============================================================================
def _pbar_update_best_effort(pbar: Any, n: int) -> None:
    """Best-effort progress bar update."""
    try:
        pbar.update(int(n))
    except Exception:
        pass


def _build_transition(
    *,
    obs: Any,
    action_env: Any,
    reward: Any,
    next_obs: Any,
    done: bool,
    info: Any,
) -> Dict[str, Any]:
    """
    Build a standardized transition dict for algo.on_env_step(...).

    Parameters
    ----------
    obs : Any
        Observation before the environment step.
    action_env : Any
        Action passed to env.step(...), already formatted for the environment.
    reward : Any
        Reward returned by the environment step (will be cast to float).
    next_obs : Any
        Next observation returned by the environment.
    done : bool
        Episode termination flag (may be synthesized by trainer time limit).
    info : Any
        Step info payload. If mapping-like, it is copied into a dict; otherwise {}.

    Returns
    -------
    transition : Dict[str, Any]
        Transition dictionary with keys:
          - obs, action, reward, next_obs, done, info
    """
    info_dict = dict(info) if isinstance(info, Mapping) else {}
    return {
        "obs": obs,
        "action": action_env,
        "reward": float(reward),
        "next_obs": next_obs,
        "done": bool(done),
        "info": info_dict,
    }


def _maybe_apply_trainer_time_limit(trainer: Any, done: bool, info: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Apply trainer-side max_episode_steps termination if enabled.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with:
          - _normalize_enabled : bool
          - max_episode_steps : Optional[int]
          - _ep_len : int
    done : bool
        Current done flag from environment.
    info : Any
        Environment info payload.

    Returns
    -------
    done_out : bool
        Possibly overridden done flag if trainer-side time limit triggers.
    info_out : Dict[str, Any]
        Info dict guaranteed to be a dict.

    Notes
    -----
    This is a fallback time-limit synthesis. It only triggers when:
      - trainer._normalize_enabled is False
      - trainer.max_episode_steps is not None
      - trainer._ep_len >= max_episode_steps
      - done is currently False

    When triggered, sets:
      - done_out = True
      - info_out["TimeLimit.truncated"] = True
    """
    info_dict: Dict[str, Any] = dict(info) if isinstance(info, Mapping) else {}

    normalize_enabled = bool(getattr(trainer, "_normalize_enabled", False))
    max_episode_steps = getattr(trainer, "max_episode_steps", None)

    if (not normalize_enabled) and (max_episode_steps is not None):
        if int(getattr(trainer, "_ep_len", 0)) >= int(max_episode_steps) and (not bool(done)):
            done = True
            info_dict["TimeLimit.truncated"] = True

    return bool(done), info_dict


def _on_episode_end(trainer: Any) -> None:
    """
    Handle end-of-episode bookkeeping.

    Parameters
    ----------
    trainer : Any
        Trainer-like object. Updates:
          - episode_idx (unless episode stats are owned externally)
          - resets trainer._ep_return and trainer._ep_len

    Notes
    -----
    If `trainer._episode_stats_enabled` is True, episode indexing/logging is assumed
    to be handled elsewhere (e.g., an EpisodeStatsCallback).
    """
    if not bool(getattr(trainer, "_episode_stats_enabled", False)):
        trainer.episode_idx = int(getattr(trainer, "episode_idx", 0)) + 1

    trainer._ep_return = 0.0
    trainer._ep_len = 0


def _run_updates(trainer: Any, pbar: Any) -> None:
    """
    Run one or more learner updates according to UTD budget.

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing:
          - algo.ready_to_update() -> bool
          - algo.update() -> Mapping[str, Any] or {}
          - utd : float
          - callbacks.on_update(...) (optional)
          - logger.log(...) (optional)
          - global_update_step : int (updated here)
          - global_env_step : int (used as logger step)
    pbar : Any
        Progress bar-like object passed to logger (optional).

    Notes
    -----
    Update budget:
      - utd <= 1.0  -> budget = 1
      - utd > 1.0   -> budget = ceil(utd)
      - budget capped to [1, 1000]

    global_update_step increment:
      - Prefer metrics["sys/num_updates"] if present
      - Fallback to metrics["offpolicy/updates_ran"] for backward compatibility
      - Else increment by 1
    """
    algo = getattr(trainer, "algo", None)

    utd = float(getattr(trainer, "utd", 1.0))
    budget = 1 if utd <= 1.0 else int(math.ceil(utd))
    budget = max(1, min(budget, 1000))  # safety cap

    for _ in range(budget):
        if not algo.ready_to_update():
            break

        m = algo.update()
        metrics = dict(m) if isinstance(m, Mapping) else None

        inc = _infer_num_updates(metrics)
        trainer.global_update_step = int(getattr(trainer, "global_update_step", 0)) + inc

        if getattr(trainer, "callbacks", None) is not None:
            cont = _maybe_call(trainer.callbacks, "on_update", trainer, metrics=metrics)
            if cont is False:
                return

        logger = getattr(trainer, "logger", None)
        if logger is not None and metrics:
            try:
                logger.log(
                    metrics,
                    step=int(getattr(trainer, "global_env_step", 0)),
                    pbar=pbar,
                    prefix="train/",
                )
            except Exception:
                pass


def _infer_num_updates(metrics: Optional[Mapping[str, Any]]) -> int:
    """
    Infer how many learner updates were actually executed from metrics.

    Parameters
    ----------
    metrics : Mapping[str, Any] or None
        Update metrics returned by algo.update().

    Returns
    -------
    inc : int
        A positive integer update increment (>= 1).
    """
    if not metrics:
        return 1

    v = metrics.get("sys/num_updates", None)
    if v is None:
        v = metrics.get("offpolicy/updates_ran", None)

    try:
        inc = int(float(v)) if v is not None else 1
    except Exception:
        inc = 1

    return max(1, inc)


def _maybe_log_sys(trainer: Any, last_sys_log_step: int, pbar: Any) -> int:
    """
    Periodically log system-level counters.

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing:
          - logger.log(...) (optional)
          - log_every_steps : int
          - global_env_step / global_update_step
          - episode_idx (if episode stats not externally owned)
    last_sys_log_step : int
        The last env step at which we emitted a sys log.
    pbar : Any
        Progress bar-like object passed to logger (optional).

    Returns
    -------
    new_last_sys_log_step : int
        Updated last log step (unchanged if no log emitted).

    Notes
    -----
    - If log_every_steps <= 0, sys logging is disabled.
    - Logged payload keys are emitted under prefix "sys/":
        - env_step, update_step, and optionally episode
    """
    logger = getattr(trainer, "logger", None)
    log_every = int(getattr(trainer, "log_every_steps", 0))
    if logger is None or log_every <= 0:
        return last_sys_log_step

    now = int(getattr(trainer, "global_env_step", 0))
    if (now - int(last_sys_log_step)) < log_every:
        return last_sys_log_step

    payload: Dict[str, Any] = {
        "env_step": int(getattr(trainer, "global_env_step", 0)),
        "update_step": int(getattr(trainer, "global_update_step", 0)),
    }
    if not bool(getattr(trainer, "_episode_stats_enabled", False)):
        payload["episode"] = int(getattr(trainer, "episode_idx", 0))

    try:
        logger.log(payload, step=now, pbar=pbar, prefix="sys/")
    except Exception:
        pass

    return now


def _maybe_set_pbar_postfix(trainer: Any, pbar: Any) -> None:
    """
    Update progress bar postfix with common training counters (best-effort).

    Postfix keys
    ------------
    ep : int
        Current episode index (trainer-owned unless episode stats external).
    ret : str
        Current episode return formatted to 2 decimals.
    updates : int
        Global update step counter.
    """
    try:
        pbar.set_postfix(
            {
                "ep": int(getattr(trainer, "episode_idx", 0)),
                "ret": f"{float(getattr(trainer, '_ep_return', 0.0)):.2f}",
                "updates": int(getattr(trainer, "global_update_step", 0)),
            },
            refresh=False,
        )
    except Exception:
        pass
