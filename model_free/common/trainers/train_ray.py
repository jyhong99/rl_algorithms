from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import math
import numpy as np

from ..utils.train_utils import maybe_call, infer_env_id, wrap_make_env_with_normalize


def train_ray(trainer: Any, pbar: Any, msg_pbar: Any) -> None:
    """
    Ray multi-worker rollout loop.

    This loop collects rollouts from Ray workers and feeds them into `trainer.algo`,
    then performs updates and periodically synchronizes policy weights (and optional
    environment normalization state) back to workers.

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing (duck-typed):
          - train_env, eval_env, algo, callbacks, logger
          - n_envs : int
          - rollout_steps_per_env : int
          - seed : int
          - utd : float
          - total_env_steps : int
          - max_episode_steps : Optional[int]
          - flatten_obs : bool
          - step_obs_dtype : dtype
          - global_env_step, global_update_step, episode_idx : int
          - _normalize_enabled : bool
          - ray_env_make_fn : Optional[Callable[[], env]]
          - ray_policy_spec : Optional[Any]
          - ray_want_onpolicy_extras : Optional[bool]
          - sync_weights_every_updates : int
          - _warn(msg) optional

    pbar : Any
        Progress bar-like object providing update(int) and set_postfix(dict, ...).

    Returns
    -------
    None

    Notes
    -----
    - On-policy: expects `algo.remaining_rollout_steps()` to exist, and performs exactly one
      update when ready (mode="onpolicy").
    - Off-policy: performs up to `max_update_calls_per_iter` update calls per iteration
      (mode="offpolicy") while `algo.ready_to_update()` stays True.
    - Ray is initialized best-effort if not already initialized.
    """
    _ensure_ray_env_make_fn(trainer)
    _ensure_ray_policy_spec(trainer)

    is_off_policy = bool(getattr(trainer.algo, "is_off_policy", False))
    want_onpolicy_extras = _infer_want_onpolicy_extras(trainer, is_off_policy=is_off_policy)

    runner, get_policy_state_dict_cpu_fn = _build_ray_runner(
        trainer,
        want_onpolicy_extras=want_onpolicy_extras,
    )

    _initial_broadcast(trainer, runner, get_policy_state_dict_cpu_fn)

    last_sys_log_step = int(getattr(trainer, "global_env_step", 0))
    max_update_calls_per_iter = _infer_max_update_calls_per_iter(trainer)

    while int(getattr(trainer, "global_env_step", 0)) < int(getattr(trainer, "total_env_steps", 0)):
        if is_off_policy:
            algo = getattr(trainer, "algo", None)
            warmup = int(getattr(algo, "warmup_steps", 0))
            update_after = int(getattr(algo, "update_after", 0))
            gate = max(warmup, update_after)

            # learner 관점에서 env_steps 기준이 가장 정확합니다.
            env_steps = int(getattr(algo, "_env_steps", 0))  # OffPolicyAlgorithm이 on_env_step에서 증가시킴
            pause_after = bool(getattr(trainer, "pause_rollout_after_warmup", True))

            # gate를 넘었고, 업데이트가 가능한 동안은 collect를 쉬고 update만 수행
            if pause_after and env_steps >= gate and algo.ready_to_update():
                if not run_updates(trainer, mode="offpolicy", max_calls=int(max_update_calls_per_iter), pbar=msg_pbar):
                    return

                maybe_sync_worker_weights_and_env(trainer, runner, get_policy_state_dict_cpu_fn)
                last_sys_log_step = _maybe_log_sys_ray(trainer, last_sys_log_step, pbar=pbar)

                continue

        # -------------------------
        # 기존: collect -> ingest ...
        # -------------------------
        per_worker, target_new = _decide_collection_size(trainer, is_off_policy=is_off_policy)
        rollout = runner.collect(deterministic=False, n_steps=int(per_worker))
        n_new_total = int(len(rollout))
        if n_new_total <= 0:
            continue

        rollout, n_new = _trim_onpolicy_rollout(rollout, target_new=target_new)

        # Bookkeeping
        trainer.global_env_step = int(getattr(trainer, "global_env_step", 0)) + int(n_new)
        try:
            pbar.update(int(n_new))
        except Exception:
            pass

        # Ingest
        _ingest_rollout(trainer, rollout)

        # Callbacks per transition (same behavior as original)
        if getattr(trainer, "callbacks", None) is not None:
            for tr in rollout:
                cont = maybe_call(trainer.callbacks, "on_step", trainer, transition=tr)
                if cont is False:
                    return

        # Updates
        if is_off_policy:
            if not run_updates(trainer, mode="offpolicy", max_calls=int(max_update_calls_per_iter), pbar=msg_pbar):
                return
        else:
            # On-policy: if rollout collection finished the batch, algo may become ready.
            if not run_updates(trainer, mode="onpolicy", max_calls=1, pbar=msg_pbar):
                return

        # Sync weights / env stats
        maybe_sync_worker_weights_and_env(trainer, runner, get_policy_state_dict_cpu_fn)

        # Sys log
        last_sys_log_step = _maybe_log_sys_ray(trainer, last_sys_log_step, pbar=pbar)

        try:
            pbar.set_postfix({"updates": int(getattr(trainer, "global_update_step", 0))}, refresh=False)
        except Exception:
            pass
        
        if getattr(trainer, "_stop_training", False):
            break

def _infer_num_updates_from_metrics(metrics: Any) -> int:
    """
    Infer 'how many learner updates were performed' from metrics.

    Convention:
      - Prefer sys/num_updates or num_updates or n_updates, if present.
      - Else default to 1 (one update() call == one learner update).
    """
    if not isinstance(metrics, Mapping):
        return 1

    for key in ("sys/num_updates", "num_updates", "n_updates", "update_steps", "train/num_updates"):
        v = metrics.get(key, None)
        if v is None:
            continue
        try:
            k = int(v)
            return max(1, k)
        except Exception:
            pass
    return 1


def run_updates(trainer: Any, *, mode: str, max_calls: int, pbar: Any) -> bool:
    if mode not in ("onpolicy", "offpolicy"):
        raise ValueError(f"Unknown update mode: {mode}")

    algo = getattr(trainer, "algo", None)

    if mode == "onpolicy":
        if not algo.ready_to_update():
            return True

        metrics = algo.update()
        k = _infer_num_updates_from_metrics(metrics)
        trainer.global_update_step += k

        return _handle_update_side_effects(trainer, metrics, pbar=pbar)

    # offpolicy
    n_calls = 0
    while n_calls < int(max_calls):
        if not algo.ready_to_update():
            break

        metrics = algo.update()
        k = _infer_num_updates_from_metrics(metrics)
        trainer.global_update_step += k

        n_calls += 1  # ✅ 이거 없으면 루프 제어가 깨짐

        if not _handle_update_side_effects(trainer, metrics, pbar=pbar):
            return False

    return True



def maybe_sync_worker_weights_and_env(
    trainer: Any,
    runner: Any,
    get_policy_state_dict_cpu_fn: Callable[[Any], Dict[str, Any]],
) -> bool:
    """
    Broadcast latest policy weights (and optionally NormalizeWrapper state) to Ray workers.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with:
          - sync_weights_every_updates : int
          - global_update_step : int
          - algo : policy holder
          - train_env : optional, may implement state_dict
          - _normalize_enabled : bool
    runner : Any
        Ray runner providing broadcast_policy(...) and broadcast_env_state(...).
    get_policy_state_dict_cpu_fn : Callable
        Function that returns a CPU state_dict for broadcasting.

    Returns
    -------
    did_sync : bool
        True if sync was performed, else False.
    """
    every = int(getattr(trainer, "sync_weights_every_updates", 0))
    if every <= 0:
        return False

    if (int(getattr(trainer, "global_update_step", 0)) % every) != 0:
        return False

    runner.broadcast_policy(get_policy_state_dict_cpu_fn(getattr(trainer, "algo", None)))

    if bool(getattr(trainer, "_normalize_enabled", False)) and callable(getattr(trainer.train_env, "state_dict", None)):
        try:
            runner.broadcast_env_state(trainer.train_env.state_dict())
        except Exception:
            pass

    return True


# =============================================================================
# Internal helpers
# =============================================================================
def _ensure_ray_env_make_fn(trainer: Any) -> None:
    """
    Ensure trainer.ray_env_make_fn is populated.

    If missing, attempts to infer env_id from train_env and creates a default gym.make maker.
    If normalization is enabled, wraps maker with NormalizeWrapper parameters recovered from train_env.
    """
    if getattr(trainer, "ray_env_make_fn", None) is not None:
        return

    env_id = infer_env_id(getattr(trainer, "train_env", None))
    if env_id is None:
        raise ValueError("n_envs > 1 requires ray_env_make_fn OR a resolvable train_env.spec.id to auto-make envs.")

    def _default_make_env() -> Any:
        try:
            import gymnasium as gym
        except Exception:
            import gym  # type: ignore
        return gym.make(env_id)

    make_fn: Callable[[], Any] = _default_make_env

    # If normalization enabled, auto wrap the default maker too.
    if bool(getattr(trainer, "_normalize_enabled", False)):
        obs_shape = getattr(getattr(trainer, "train_env", None), "obs_shape", None)
        if obs_shape is None:
            raise RuntimeError("normalize=True but could not recover obs_shape from train_env.")

        te = getattr(trainer, "train_env", None)
        make_fn = wrap_make_env_with_normalize(
            make_fn,
            obs_shape=tuple(obs_shape),
            norm_obs=bool(getattr(te, "norm_obs", True)),
            norm_reward=bool(getattr(te, "norm_reward", False)),
            clip_obs=float(getattr(te, "clip_obs", 10.0)),
            clip_reward=float(getattr(te, "clip_reward", 10.0)),
            gamma=float(getattr(te, "gamma", 0.99)),
            epsilon=float(getattr(te, "epsilon", 1e-8)),
            training=True,
            max_episode_steps=getattr(trainer, "max_episode_steps", None),
            action_rescale=bool(getattr(te, "action_rescale", False)),
            clip_action=float(getattr(te, "clip_action", 0.0)),
            reset_return_on_done=bool(getattr(te, "reset_return_on_done", True)),
            reset_return_on_trunc=bool(getattr(te, "reset_return_on_trunc", True)),
            obs_dtype=getattr(te, "obs_dtype", np.float32),
        )

    trainer.ray_env_make_fn = make_fn


def _ensure_ray_policy_spec(trainer: Any) -> None:
    """
    Ensure trainer.ray_policy_spec is populated.

    If missing, uses algo.get_ray_policy_factory_spec().
    """
    if getattr(trainer, "ray_policy_spec", None) is not None:
        return

    fn = getattr(getattr(trainer, "algo", None), "get_ray_policy_factory_spec", None)
    if not callable(fn):
        raise ValueError("n_envs > 1 requires ray_policy_spec OR algo.get_ray_policy_factory_spec().")
    trainer.ray_policy_spec = fn()


def _infer_want_onpolicy_extras(trainer: Any, *, is_off_policy: bool) -> bool:
    """
    Infer whether Ray workers should return on-policy extras (logp, value, etc.).

    Precedence
    ----------
    - trainer.ray_want_onpolicy_extras if not None
    - else: True for on-policy algorithms, False for off-policy algorithms
    """
    if getattr(trainer, "ray_want_onpolicy_extras", None) is not None:
        return bool(trainer.ray_want_onpolicy_extras)
    return (not bool(is_off_policy))


def _build_ray_runner(trainer: Any, *, want_onpolicy_extras: bool) -> tuple[Any, Callable[[Any], Dict[str, Any]]]:
    """
    Construct RayLearner runner and return it along with get_policy_state_dict_cpu function.
    """
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # NOTE: Keep these imports local to avoid importing ray-related modules when unused.
    from model_free.common.trainers.ray_workers import RayLearner
    from ..utils.ray_utils import get_policy_state_dict_cpu

    runner = RayLearner(
        env_make_fn=getattr(trainer, "ray_env_make_fn", None),
        policy_spec=getattr(trainer, "ray_policy_spec", None),
        n_workers=int(getattr(trainer, "n_envs", 1)),
        steps_per_worker=int(getattr(trainer, "rollout_steps_per_env", 1)),
        base_seed=int(getattr(trainer, "seed", 0)),
        max_episode_steps=getattr(trainer, "max_episode_steps", None),
        want_onpolicy_extras=bool(want_onpolicy_extras),
        flatten_obs=bool(getattr(trainer, "flatten_obs", False)),
        obs_dtype=getattr(trainer, "step_obs_dtype", np.float32),
    )
    return runner, get_policy_state_dict_cpu


def _initial_broadcast(trainer: Any, runner: Any, get_policy_state_dict_cpu_fn: Callable[[Any], Dict[str, Any]]) -> None:
    """
    Initial broadcast of policy weights and (optional) env normalization state to workers.
    """
    runner.broadcast_policy(get_policy_state_dict_cpu_fn(getattr(trainer, "algo", None)))

    if bool(getattr(trainer, "_normalize_enabled", False)) and callable(getattr(trainer.train_env, "state_dict", None)):
        try:
            runner.broadcast_env_state(trainer.train_env.state_dict())
        except Exception:
            pass


def _infer_max_update_calls_per_iter(trainer: Any) -> int:
    """
    Infer update call budget per iteration for off-policy mode.

    Uses a base budget and scales by UTD when utd > 0.
    """
    max_calls = 10_000
    utd = float(getattr(trainer, "utd", 1.0))
    if utd > 0.0:
        max_calls = int(max_calls * utd)
    return max(1, int(max_calls))


def _decide_collection_size(trainer: Any, *, is_off_policy: bool) -> tuple[int, Optional[int]]:
    """
    Decide per-worker collection steps and optional on-policy target trim.

    Returns
    -------
    per_worker : int
        Steps each worker should collect this iteration.
    target_new : Optional[int]
        Total desired number of transitions (on-policy only). If set, rollout may be trimmed.
    """
    n_envs = int(getattr(trainer, "n_envs", 1))
    steps_per_env = int(getattr(trainer, "rollout_steps_per_env", 1))

    if is_off_policy:
        return steps_per_env, None

    # On-policy path: use algo.remaining_rollout_steps()
    fn_rem = getattr(getattr(trainer, "algo", None), "remaining_rollout_steps", None)
    if not callable(fn_rem):
        raise ValueError(
            "Ray on-policy path requires algo.remaining_rollout_steps(). Implement it in your on-policy driver."
        )

    remaining = int(fn_rem())
    if remaining <= 0:
        # If algo is already ready, let the outer loop call updates by returning a tiny collection.
        # The caller handles the 'remaining<=0' case by running updates first in the original code.
        return 1, 0

    per_worker = int(math.ceil(float(remaining) / float(n_envs)))
    per_worker = max(1, min(per_worker, steps_per_env))
    return per_worker, int(remaining)


def _trim_onpolicy_rollout(rollout: Any, *, target_new: Optional[int]) -> tuple[Any, int]:
    """
    Trim rollout to exactly target_new transitions (on-policy only).

    Returns
    -------
    rollout : same type as input
    n_new : int
        Number of transitions after trimming.
    """
    n_total = int(len(rollout))
    if target_new is None:
        return rollout, n_total

    if int(target_new) <= 0:
        return rollout[:0], 0

    if n_total > int(target_new):
        return rollout[: int(target_new)], int(target_new)

    return rollout, n_total


def _ingest_rollout(trainer: Any, rollout: Any) -> None:
    """
    Feed collected rollout into algo.

    Prefers algo.on_rollout(rollout) if available, else calls algo.on_env_step per transition.
    """
    algo = getattr(trainer, "algo", None)

    on_rollout = getattr(algo, "on_rollout", None)
    if callable(on_rollout):
        on_rollout(rollout)
        return

    for tr in rollout:
        algo.on_env_step(tr)


def _handle_update_side_effects(trainer: Any, metrics: Any, pbar: Any) -> bool:
    """
    Apply common side effects after a single update call:
      - callbacks.on_update
      - logger.log(train/...)

    Returns
    -------
    ok : bool
        False if callbacks request stop; True otherwise.
    """
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else None

    if getattr(trainer, "callbacks", None) is not None:
        cont = maybe_call(trainer.callbacks, "on_update", trainer, metrics=metrics_map)
        if cont is False:
            return False

    if getattr(trainer, "logger", None) is not None and metrics_map:
        try:
            trainer.logger.log(metrics_map, step=int(getattr(trainer, "global_env_step", 0)), pbar=pbar, prefix="train/")
        except Exception:
            pass

    return True


def _maybe_log_sys_ray(trainer: Any, last_sys_log_step: int, pbar: Any) -> int:
    """
    Periodically log system-level counters from the Ray loop.
    """
    logger = getattr(trainer, "logger", None)
    log_every = int(getattr(trainer, "log_every_steps", 0))
    if logger is None or log_every <= 0:
        return last_sys_log_step

    now = int(getattr(trainer, "global_env_step", 0))
    if (now - int(last_sys_log_step)) < log_every:
        return last_sys_log_step

    payload = {
        "env_step": int(getattr(trainer, "global_env_step", 0)),
        "update_step": int(getattr(trainer, "global_update_step", 0)),
        "episode": int(getattr(trainer, "episode_idx", 0)),
    }
    try:
        logger.log(payload, step=now, pbar=pbar, prefix="sys/")
    except Exception:
        pass

    return now