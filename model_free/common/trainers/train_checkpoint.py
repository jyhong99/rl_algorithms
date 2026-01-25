from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import time

import torch as th


def save_checkpoint(trainer: Any, path: Optional[str] = None) -> Optional[str]:
    """
    Save a trainer checkpoint (trainer counters + optional env states + algo artifact).

    The checkpoint is split into:
      1) a Torch file at `path` that contains metadata + relative reference to algo file
      2) an algorithm artifact saved by `trainer.algo.save(...)` at `<root>_algo<ext>`

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing (duck-typed):
          - ckpt_dir : str
          - run_dir : str
          - checkpoint_prefix : str
          - global_env_step : int
          - global_update_step : int
          - episode_idx : int
          - algo : object with save(path: str) method
          - train_env / eval_env (optional): may implement state_dict()
          - _warn(msg: str) method (best-effort)
          - strict_checkpoint : bool (optional; if True, re-raise algo save error)
    path : str, optional
        Output checkpoint path. If relative, it is resolved under `trainer.run_dir`.
        If omitted, it is created under `trainer.ckpt_dir` using checkpoint_prefix.

    Returns
    -------
    saved_path : Optional[str]
        Absolute/normalized path of the written torch checkpoint file, or None on failure.

    Notes
    -----
    - This function is robust by default: failures to save algo/env state will not abort
      unless `trainer.strict_checkpoint` is True for algo save.
    - A single warning is emitted (best-effort) per trainer instance by using
      `trainer._warned_checkpoint` guard.
    """
    try:
        ckpt_path = _resolve_checkpoint_path(trainer, path)
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))

        root, ext = os.path.splitext(ckpt_path)
        algo_abs_path = f"{root}_algo{ext}"

        algo_rel_path, algo_ok, algo_err = _try_save_algo(trainer, algo_abs_path, ckpt_dir)

        ckpt: Dict[str, Any] = {
            "trainer": {
                "global_env_step": int(getattr(trainer, "global_env_step", 0)),
                "global_update_step": int(getattr(trainer, "global_update_step", 0)),
                "episode_idx": int(getattr(trainer, "episode_idx", 0)),
                "timestamp": float(time.time()),
            },
            "algo_path": algo_rel_path,          # relative to ckpt_dir when possible
            "algo_save_ok": bool(algo_ok),
            "algo_save_error": algo_err,
        }

        # Optional environment state snapshots.
        train_env_state = _maybe_env_state_dict(getattr(trainer, "train_env", None))
        if train_env_state is not None:
            ckpt["train_env_state"] = train_env_state

        eval_env_state = _maybe_env_state_dict(getattr(trainer, "eval_env", None))
        if eval_env_state is not None:
            ckpt["eval_env_state"] = eval_env_state

        # Ensure directory exists (quietly).
        os.makedirs(ckpt_dir, exist_ok=True)

        th.save(ckpt, ckpt_path)
        return ckpt_path

    except Exception as e:
        _warn_once(trainer, f"Trainer checkpoint save failed: {type(e).__name__}: {e}")
        return None


def load_checkpoint(trainer: Any, path: str) -> None:
    """
    Load a trainer checkpoint (trainer counters + optional env states + algo artifact).

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing (duck-typed):
          - algo : object with load(path: str) method
          - train_env / eval_env (optional): may implement load_state_dict(...)
          - _warn(msg: str) method (best-effort)
    path : str
        Path to a torch checkpoint file created by `save_checkpoint`.

    Notes
    -----
    - Loads trainer counters from the "trainer" dict.
    - Loads algo artifact from "algo_path" if present:
      relative paths are resolved relative to the checkpoint directory.
    - Env states are loaded only if corresponding keys exist and env supports load_state_dict.
    """
    sd = th.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"Invalid checkpoint format (expected dict), got {type(sd).__name__}")

    # ✅ 방어: trainer 키가 없으면, 거의 확실히 algo artifact를 잘못 집은 것
    if "trainer" not in sd:
        raise ValueError(
            f"Checkpoint file does not contain 'trainer' state. "
            f"Did you pass an algo artifact (e.g., *_algo.pt)? path={path}"
        )

    _restore_trainer_counters(trainer, sd.get("trainer", {}))
    _try_load_algo(trainer, sd.get("algo_path", None), ckpt_path=path)
    _maybe_env_load_state(getattr(trainer, "train_env", None), sd.get("train_env_state", None))
    _maybe_env_load_state(getattr(trainer, "eval_env", None), sd.get("eval_env_state", None))


# =============================================================================
# Internal helpers
# =============================================================================
def _resolve_checkpoint_path(trainer: Any, path: Optional[str]) -> str:
    """
    Resolve checkpoint file path.

    Rules
    -----
    - If path is None: <trainer.ckpt_dir>/<prefix>_<global_env_step>.pt
    - If path is relative: resolve under trainer.run_dir
    - Ensure extension exists; default ".pt"
    """
    if path is None:
        prefix = str(getattr(trainer, "checkpoint_prefix", "ckpt"))
        step = int(getattr(trainer, "global_env_step", 0))
        fname = f"{prefix}_{step:012d}.pt"
        base_dir = str(getattr(trainer, "ckpt_dir", "."))
        ckpt_path = os.path.join(base_dir, fname)
    else:
        ckpt_path = path
        if not os.path.isabs(ckpt_path):
            run_dir = str(getattr(trainer, "run_dir", "."))
            ckpt_path = os.path.join(run_dir, ckpt_path)

    root, ext = os.path.splitext(ckpt_path)
    if not ext:
        ckpt_path = root + ".pt"
    return ckpt_path


def _try_save_algo(trainer: Any, algo_abs_path: str, ckpt_dir: str) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Save algorithm artifact via `trainer.algo.save(...)`.

    Returns
    -------
    algo_path_saved : Optional[str]
        Relative path to ckpt_dir if possible, else absolute path; None if failed.
    ok : bool
        Whether algo save succeeded.
    err : Optional[str]
        Error string if failed.
    """
    try:
        trainer.algo.save(algo_abs_path)  # may raise
        try:
            rel = os.path.relpath(algo_abs_path, start=ckpt_dir)
            return rel, True, None
        except Exception:
            return algo_abs_path, True, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _warn_once(trainer, f"Algo checkpoint save failed: {err}")
        if bool(getattr(trainer, "strict_checkpoint", False)):
            raise
        return None, False, err


def _try_load_algo(trainer: Any, algo_path: Any, *, ckpt_path: str) -> None:
    """
    Load algorithm artifact via `trainer.algo.load(...)` (best-effort).

    Parameters
    ----------
    algo_path : Any
        Typically a string or None. If relative, resolve against checkpoint directory.
    ckpt_path : str
        Torch checkpoint file path (used for resolving relative algo_path).
    """
    if not isinstance(algo_path, str) or not algo_path:
        return

    load_path = algo_path
    if not os.path.isabs(load_path):
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        load_path = os.path.join(ckpt_dir, load_path)

    try:
        trainer.algo.load(load_path)
    except Exception as e:
        _warn_once(trainer, f"Algo checkpoint load failed: {type(e).__name__}: {e}")


def _restore_trainer_counters(trainer: Any, t: Any) -> None:
    """
    Restore trainer counters from checkpoint content.

    Missing keys default to 0.
    """
    if not isinstance(t, dict):
        t = {}
    trainer.global_env_step = int(t.get("global_env_step", 0))
    trainer.global_update_step = int(t.get("global_update_step", 0))
    trainer.episode_idx = int(t.get("episode_idx", 0))


def _maybe_env_state_dict(env: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort call to env.state_dict().

    Returns
    -------
    state : dict or None
        None if env is None, missing method, or state_dict failed.
    """
    if env is None:
        return None
    fn = getattr(env, "state_dict", None)
    if not callable(fn):
        return None
    try:
        out = fn()
        return out if isinstance(out, dict) else out  # allow non-dict if user wants
    except Exception:
        return None


def _maybe_env_load_state(env: Any, state: Any) -> None:
    """
    Best-effort call to env.load_state_dict(state) if both are available.
    """
    if env is None or state is None:
        return
    fn = getattr(env, "load_state_dict", None)
    if not callable(fn):
        return
    try:
        fn(state)
    except Exception:
        pass


def _warn_once(trainer: Any, msg: str) -> None:
    """
    Emit a warning once per trainer instance (best-effort).

    Uses `trainer._warned_checkpoint` guard and `trainer._warn(msg)` method.
    Silently ignores failures to avoid crashing checkpoint IO.
    """
    try:
        if bool(getattr(trainer, "_warned_checkpoint", False)):
            return
        setattr(trainer, "_warned_checkpoint", True)
        warn_fn = getattr(trainer, "_warn", None)
        if callable(warn_fn):
            warn_fn(str(msg))
    except Exception:
        pass