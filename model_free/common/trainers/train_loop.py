from __future__ import annotations

from typing import Any, Dict, Mapping

import math

from ..utils.train_utils import env_reset, unpack_step, maybe_call, format_env_action


def train_single_env(trainer: Any, pbar: Any, msg_pbar: Any) -> None:
    """
    Train loop for a single (non-vectorized) environment.

    This function performs:
      - env rollout (step-by-step)
      - transition handoff to algo via `algo.on_env_step(transition)`
      - optional callbacks on each step and update
      - update(s) when `algo.ready_to_update()` is True (supports UTD)
      - periodic system logging

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing (duck-typed):
          - train_env : env with step/reset
          - algo : object with:
              act(obs, deterministic: bool) -> action
              on_env_step(transition: Mapping) -> None
              ready_to_update() -> bool
              update() -> Mapping[str, Any] or {}
          - total_env_steps : int
          - global_env_step : int
          - global_update_step : int
          - episode_idx : int
          - utd : float (>=1 implies multiple updates per env step, capped)
          - callbacks : optional, with on_step/on_update hooks (invoked via maybe_call)
          - logger : optional, with log(metrics, step, prefix)
          - log_every_steps : int (system metrics cadence; 0 disables)
          - flatten_obs : bool (forwarded to unpack_step)
          - step_obs_dtype : dtype (forwarded to unpack_step)
          - max_episode_steps : Optional[int] (trainer-side time limit)
          - _normalize_enabled : bool (if True, do not apply trainer-side time limit here)
          - _episode_stats_enabled : bool (if True, episode counting is owned elsewhere)
          - internal counters: _ep_return, _ep_len

    pbar : Any
        Progress bar-like object providing update(int) and set_postfix(dict, ...).

    Returns
    -------
    None

    Notes
    -----
    - Trainer-side time limit is applied only when:
        (not _normalize_enabled) and max_episode_steps is set.
      This preserves previous behavior but avoids overriding wrappers that may
      already manage TimeLimit. (If you want a cleaner rule, key it on an explicit
      trainer flag like `_time_limit_enabled` instead of `_normalize_enabled`.)
    - Update-To-Data (UTD): for utd > 1.0, uses ceil(utd) updates, capped to 1000,
      and stops early if ready_to_update() becomes False.
    """
    obs = env_reset(getattr(trainer, "train_env", None))

    trainer._ep_return = 0.0
    trainer._ep_len = 0
    last_sys_log_step = int(getattr(trainer, "global_env_step", 0))

    total_env_steps = int(getattr(trainer, "total_env_steps", 0))

    while int(getattr(trainer, "global_env_step", 0)) < total_env_steps:
        env = getattr(trainer, "train_env", None)
        algo = getattr(trainer, "algo", None)

        # ------------------------------------------------------------------
        # 1) Act and step environment
        # ------------------------------------------------------------------
        action = algo.act(obs, deterministic=False)
        action_space = getattr(env, "action_space", None)
        action_env = format_env_action(action, action_space)

        step_out = env.step(action_env)
        next_obs, reward, done, info = unpack_step(
            step_out,
            flatten_obs=bool(getattr(trainer, "flatten_obs", False)),
            obs_dtype=getattr(trainer, "step_obs_dtype", None),
        )

        trainer.global_env_step = int(getattr(trainer, "global_env_step", 0)) + 1
        try:
            pbar.update(1)
        except Exception:
            pass

        trainer._ep_return += float(reward)
        trainer._ep_len += 1

        # ------------------------------------------------------------------
        # 2) Optional trainer-side time limit
        # ------------------------------------------------------------------
        done, info = _maybe_apply_trainer_time_limit(trainer, done, info)

        # ------------------------------------------------------------------
        # 3) Hand off transition to algo
        # ------------------------------------------------------------------
        transition = {
            "obs": obs,
            "action": action_env,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": bool(done),
            "info": dict(info) if isinstance(info, Mapping) else {},
        }
        algo.on_env_step(transition)

        # Callbacks: on_step
        if getattr(trainer, "callbacks", None) is not None:
            cont = maybe_call(trainer.callbacks, "on_step", trainer, transition=transition)
            if cont is False:
                return

        obs = next_obs

        # ------------------------------------------------------------------
        # 4) Episode boundary handling
        # ------------------------------------------------------------------
        if done:
            _on_episode_end(trainer)
            obs = env_reset(env)

        # ------------------------------------------------------------------
        # 5) Updates (UTD)
        # ------------------------------------------------------------------
        if algo.ready_to_update():
            _run_updates(trainer, pbar=msg_pbar)

        # ------------------------------------------------------------------
        # 6) Periodic system logging and progress postfix
        # ------------------------------------------------------------------
        last_sys_log_step = _maybe_log_sys(trainer, last_sys_log_step, pbar=msg_pbar)
        _maybe_set_pbar_postfix(trainer, pbar)
        
        if getattr(trainer, "_stop_training", False):
            break



# =============================================================================
# Internal helpers
# =============================================================================
def _maybe_apply_trainer_time_limit(trainer: Any, done: bool, info: Any) -> tuple[bool, Dict[str, Any]]:
    """
    Apply trainer-side max_episode_steps if enabled.

    Returns
    -------
    done : bool
        Possibly overridden done flag.
    info : Dict[str, Any]
        Info dict (ensured to be a dict).
    """
    info_dict: Dict[str, Any] = dict(info) if isinstance(info, Mapping) else {}

    # Preserve original behavior: only do this if "not _normalize_enabled".
    # If that coupling is accidental, consider replacing with a dedicated flag.
    normalize_enabled = bool(getattr(trainer, "_normalize_enabled", False))
    max_episode_steps = getattr(trainer, "max_episode_steps", None)

    if (not normalize_enabled) and (max_episode_steps is not None):
        if int(getattr(trainer, "_ep_len", 0)) >= int(max_episode_steps) and (not bool(done)):
            done = True
            info_dict["TimeLimit.truncated"] = True

    return bool(done), info_dict


def _on_episode_end(trainer: Any) -> None:
    """
    Handle end-of-episode bookkeeping on trainer.

    Notes
    -----
    - If EpisodeStatsCallback is enabled, it owns episode indexing/logging.
    - Always resets trainer-side episode counters.
    """
    if not bool(getattr(trainer, "_episode_stats_enabled", False)):
        trainer.episode_idx = int(getattr(trainer, "episode_idx", 0)) + 1

    trainer._ep_return = 0.0
    trainer._ep_len = 0


def _run_updates(trainer: Any, pbar: Any) -> None:
    algo = getattr(trainer, "algo", None)

    utd = float(getattr(trainer, "utd", 1.0))
    n_updates_budget = 1 if utd <= 1.0 else int(math.ceil(utd))
    n_updates_budget = max(1, min(n_updates_budget, 1000))  # safety cap

    for _ in range(n_updates_budget):
        if not algo.ready_to_update():
            break

        m = algo.update()
        metrics = dict(m) if isinstance(m, Mapping) else None

        # -----------------------------
        # NEW: real learner update count
        # -----------------------------
        inc = 1
        if metrics is not None:
            v = metrics.get("sys/num_updates", None)
            if v is None:
                # backward-compat with your OffPolicyAlgorithm metric name
                v = metrics.get("offpolicy/updates_ran", None)
            try:
                inc = int(float(v)) if v is not None else 1
            except Exception:
                inc = 1
        trainer.global_update_step = int(getattr(trainer, "global_update_step", 0)) + max(1, inc)

        if getattr(trainer, "callbacks", None) is not None:
            cont = maybe_call(trainer.callbacks, "on_update", trainer, metrics=metrics)
            if cont is False:
                return

        logger = getattr(trainer, "logger", None)
        if logger is not None and metrics:
            try:
                logger.log(metrics, step=int(getattr(trainer, "global_env_step", 0)), pbar=pbar, prefix="train/")
            except Exception:
                pass



def _maybe_log_sys(trainer: Any, last_sys_log_step: int, pbar: Any) -> int:
    """
    Periodically log system-level counters.

    Returns
    -------
    last_sys_log_step : int
        Updated last log step.
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
    """Update progress bar postfix (best-effort)."""
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