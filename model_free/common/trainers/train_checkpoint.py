from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import time

import torch as th


def save_checkpoint(trainer: Any, path: Optional[str] = None) -> Optional[str]:
    """
    Save a trainer checkpoint (trainer counters + optional env states + algo artifact).

    This routine writes *two* artifacts:

    1) A Torch checkpoint file at `ckpt_path` that contains:
       - trainer counters (global steps, episode index, timestamp)
       - optional environment state snapshots (train/eval)
       - a reference to the algorithm artifact (relative when possible)

    2) An algorithm artifact saved by `trainer.algo.save(...)` at:
           <root>_algo<ext>
       where (root, ext) come from the main checkpoint file path.

    Parameters
    ----------
    trainer : Any
        Trainer-like object (duck-typed) providing:
          - ckpt_dir : str
          - run_dir : str
          - checkpoint_prefix : str
          - global_env_step : int
          - global_update_step : int
          - episode_idx : int
          - algo : object with `save(path: str) -> None`
          - train_env / eval_env (optional): may implement `state_dict()`
          - _warn(msg: str) method (optional, best-effort)
          - strict_checkpoint : bool (optional)
              If True, algorithm artifact save errors are re-raised.
              If False/absent, algorithm save errors are recorded but do not abort.
    path : str, optional
        Output checkpoint path.
        - If None: uses `<trainer.ckpt_dir>/<prefix>_<global_env_step>.pt`
        - If relative: resolved under `trainer.run_dir`
        - If missing extension: default ".pt"

    Returns
    -------
    saved_path : str or None
        Absolute/normalized path of the written Torch checkpoint file, or None if:
          - path resolution fails
          - directory creation fails
          - torch save fails
        Algorithm artifact failure alone does *not* cause None unless
        `trainer.strict_checkpoint` is True.

    Notes
    -----
    Robustness policy:
      - Env state snapshot failures are ignored.
      - Algo save failure is recorded in the main checkpoint dict and emits a warning
        once, unless `strict_checkpoint=True`.
      - Any top-level failure results in returning None (and warning once).
    """
    try:
        ckpt_path = _resolve_checkpoint_path(trainer, path)
        ckpt_abs = os.path.abspath(ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_abs)

        root, ext = os.path.splitext(ckpt_abs)
        algo_abs_path = f"{root}_algo{ext}"

        # Ensure directory exists (quietly).
        os.makedirs(ckpt_dir, exist_ok=True)

        algo_rel_path, algo_ok, algo_err = _try_save_algo(trainer, algo_abs_path, ckpt_dir)

        ckpt: Dict[str, Any] = {
            "trainer": {
                "global_env_step": int(getattr(trainer, "global_env_step", 0)),
                "global_update_step": int(getattr(trainer, "global_update_step", 0)),
                "episode_idx": int(getattr(trainer, "episode_idx", 0)),
                "timestamp": float(time.time()),
            },
            # relative to ckpt_dir when possible; otherwise absolute; None on failure
            "algo_path": algo_rel_path,
            "algo_save_ok": bool(algo_ok),
            "algo_save_error": algo_err,
        }

        # Optional environment state snapshots (best-effort).
        train_env_state = _maybe_env_state_dict(getattr(trainer, "train_env", None))
        if train_env_state is not None:
            ckpt["train_env_state"] = train_env_state

        eval_env_state = _maybe_env_state_dict(getattr(trainer, "eval_env", None))
        if eval_env_state is not None:
            ckpt["eval_env_state"] = eval_env_state

        th.save(ckpt, ckpt_abs)
        return ckpt_abs

    except Exception as e:
        _warn_once(trainer, f"Trainer checkpoint save failed: {type(e).__name__}: {e}")
        return None


def load_checkpoint(trainer: Any, path: str) -> None:
    """
    Load a trainer checkpoint (trainer counters + optional env states + algo artifact).

    Parameters
    ----------
    trainer : Any
        Trainer-like object (duck-typed) providing:
          - algo : object with `load(path: str) -> None`
          - train_env / eval_env (optional): may implement `load_state_dict(state)`
          - _warn(msg: str) method (optional, best-effort)
          - attributes to be restored (best-effort assignment):
              global_env_step, global_update_step, episode_idx
    path : str
        Path to a Torch checkpoint file created by `save_checkpoint(...)`.

    Raises
    ------
    ValueError
        If the file does not contain a dict-like checkpoint or is missing the
        required "trainer" key (common when user accidentally passes *_algo.pt).

    Notes
    -----
    Loading behavior:
      - Trainer counters are restored from the "trainer" dict.
      - Algo artifact is loaded from "algo_path" if present; relative paths are
        resolved relative to the checkpoint directory.
      - Env states are loaded only if:
          (a) corresponding keys exist in the checkpoint, and
          (b) env implements load_state_dict(...)
        Failures are ignored (warning once).
    """
    sd = th.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"Invalid checkpoint format (expected dict), got {type(sd).__name__}")

    # Defensive: if missing "trainer", user probably pointed to the algo artifact.
    if "trainer" not in sd:
        raise ValueError(
            "Checkpoint file does not contain 'trainer' state. "
            "Did you pass an algo artifact (e.g., *_algo.pt)? "
            f"path={path}"
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
    Resolve the main Torch checkpoint file path.

    Parameters
    ----------
    trainer : Any
        Trainer-like object. Used for default directories/names.
    path : str, optional
        Desired output path or None.

    Returns
    -------
    ckpt_path : str
        A filesystem path to the main Torch checkpoint file.

    Rules
    -----
    - If `path is None`:
        <trainer.ckpt_dir>/<prefix>_<global_env_step>.pt
      where:
        prefix = trainer.checkpoint_prefix (default "ckpt")
        global_env_step = trainer.global_env_step (default 0)
    - If `path` is relative:
        resolve it under `trainer.run_dir` (default ".")
    - If no extension:
        append ".pt"

    Notes
    -----
    This function does *not* create directories.
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


def _try_save_algo(
    trainer: Any,
    algo_abs_path: str,
    ckpt_dir: str,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Save the algorithm artifact via `trainer.algo.save(...)` (best-effort).

    Parameters
    ----------
    trainer : Any
        Trainer-like object with `algo.save(...)` and optional strict policy.
    algo_abs_path : str
        Absolute path where the algo artifact should be written.
    ckpt_dir : str
        Directory of the main checkpoint file (used for relative path conversion).

    Returns
    -------
    algo_path_saved : str or None
        - If save succeeds: relative path to `ckpt_dir` when possible, else absolute.
        - If save fails: None.
    ok : bool
        True if save succeeded, else False.
    err : str or None
        Error string if failed, else None.

    Notes
    -----
    - If save fails, a warning is emitted once.
    - If `trainer.strict_checkpoint` is True, the exception is re-raised.
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
    trainer : Any
        Trainer-like object with `algo.load(path)`.
    algo_path : Any
        Usually a string or None. If relative, it is resolved against the directory
        containing `ckpt_path`.
    ckpt_path : str
        Path to the main Torch checkpoint file (used to resolve relative algo_path).

    Notes
    -----
    - If `algo_path` is missing or not a non-empty string, no action is taken.
    - Failures emit a warning once and are otherwise ignored.
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

    Parameters
    ----------
    trainer : Any
        Object to receive restored counters via attribute assignment.
    t : Any
        Expected to be a dict-like object with keys:
          - global_env_step
          - global_update_step
          - episode_idx

    Notes
    -----
    Missing or malformed input defaults counters to 0.
    """
    if not isinstance(t, dict):
        t = {}
    trainer.global_env_step = int(t.get("global_env_step", 0))
    trainer.global_update_step = int(t.get("global_update_step", 0))
    trainer.episode_idx = int(t.get("episode_idx", 0))


def _maybe_env_state_dict(env: Any) -> Optional[Any]:
    """
    Snapshot environment state via `env.state_dict()` (best-effort).

    Parameters
    ----------
    env : Any
        Environment/wrapper instance.

    Returns
    -------
    state : Any or None
        - Returns the output of env.state_dict() if available and successful.
        - Returns None if env is None, state_dict is missing, or an exception occurs.

    Notes
    -----
    - The return type is intentionally `Any` (not forced to dict) to support
      user-defined wrappers that serialize custom structures.
    - The returned object must be Torch-serializable for `th.save` to work.
    """
    if env is None:
        return None

    fn = getattr(env, "state_dict", None)
    if not callable(fn):
        return None

    try:
        return fn()
    except Exception:
        return None


def _maybe_env_load_state(env: Any, state: Any) -> None:
    """
    Restore environment state via `env.load_state_dict(state)` (best-effort).

    Parameters
    ----------
    env : Any
        Environment/wrapper instance.
    state : Any
        State payload previously produced by `state_dict()`.

    Notes
    -----
    - No-op if env/state is missing or load_state_dict is unavailable.
    - Exceptions are swallowed to avoid breaking checkpoint loading flows.
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

    Parameters
    ----------
    trainer : Any
        Trainer-like object. Uses:
          - trainer._warned_checkpoint : bool guard flag (set on first warning)
          - trainer._warn(msg) : callable warning hook (optional)
    msg : str
        Warning message.

    Notes
    -----
    - This function must never raise. It silently ignores all failures.
    - The guard prevents repeated spam from repeated checkpoint failures.
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
