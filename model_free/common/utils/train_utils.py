from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import os
import random

import numpy as np
import torch as th

# NOTE: this module is "train_utils"-ish. Keep it light: utilities + builder only.
from .common_utils import to_action_np, to_flat_np, to_scalar
from ..wrappers.normalize_wrapper import NormalizeWrapper


# =============================================================================
# Progress bar (tqdm optional)
# =============================================================================
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


class DummyTqdm:
    """
    Minimal tqdm-compatible progress bar.

    Used when `tqdm` is not installed to keep training code importable and runnable.
    """

    def __init__(self, total: int = 0, initial: int = 0, **kwargs: Any) -> None:
        self.total = int(total)
        self.n = int(initial)

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return


def make_pbar(**kwargs: Any) -> Any:
    """
    Create a progress bar instance.

    Returns
    -------
    pbar : Any
        `tqdm(...)` instance if available, else DummyTqdm.
    """
    if tqdm is None:
        return DummyTqdm(**kwargs)
    return tqdm(**kwargs)


# =============================================================================
# Small utilities (keep minimal and predictable)
# =============================================================================
def maybe_call(obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """
    Call `getattr(obj, method)(*args, **kwargs)` if it exists and is callable.

    Returns
    -------
    out : Any
        Return value of method, or None if missing/not callable or if obj is None.
    """
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def format_env_action(action: Any, action_space: Any) -> Any:
    """
    Convert policy output into an environment-compatible action (best-effort).

    Parameters
    ----------
    action : Any
        Policy output (torch tensor / numpy array / scalar / etc.).
    action_space : Any
        Environment action_space (Gym/Gymnasium space). If None, returns `action` as-is.

    Returns
    -------
    action_env : Any
        Discrete -> int
        MultiDiscrete -> int64 ndarray
        MultiBinary -> int8 ndarray (thresholded)
        Box -> float32 ndarray with declared shape (squeezes leading batch dim if present)
        Unknown -> best-effort float32 ndarray or raw action

    Notes
    -----
    This is intentionally permissive: different policies emit different formats.
    """
    if action_space is None:
        return action

    # Prefer explicit type checks when gym/gymnasium is available.
    try:  # pragma: no cover
        import gymnasium as _gym  # type: ignore

        _spaces = _gym.spaces
    except Exception:  # pragma: no cover
        try:
            import gym as _gym  # type: ignore

            _spaces = _gym.spaces  # type: ignore
        except Exception:  # pragma: no cover
            _spaces = None

    if _spaces is not None:
        if isinstance(action_space, _spaces.Discrete):
            a = np.asarray(to_action_np(action, action_shape=None)).reshape(-1)
            return int(a[0])

        if isinstance(action_space, _spaces.MultiDiscrete):
            shp = tuple(action_space.shape)
            a = np.asarray(to_action_np(action, action_shape=shp)).reshape(shp)
            return a.astype(np.int64, copy=False)

        if isinstance(action_space, _spaces.MultiBinary):
            shp = tuple(action_space.shape)
            a = np.asarray(to_action_np(action, action_shape=shp)).reshape(shp)
            return (a > 0.5).astype(np.int8, copy=False)

        if isinstance(action_space, _spaces.Box):
            shp = tuple(action_space.shape)
            a = np.asarray(to_action_np(action, action_shape=shp), dtype=np.float32).reshape(shp)
            if a.ndim >= 2 and a.shape[0] == 1:  # policy emits (1,A), env expects (A,)
                a = a[0]
            return a

    # Fallback: legacy heuristic
    if hasattr(action_space, "n"):
        a = np.asarray(to_action_np(action, action_shape=None)).reshape(-1)
        return int(a[0])

    shp = getattr(action_space, "shape", None)
    action_shape: Optional[Tuple[int, ...]] = None
    if isinstance(shp, tuple):
        try:
            action_shape = tuple(int(x) for x in shp)
        except Exception:
            action_shape = None

    a = np.asarray(to_action_np(action, action_shape=action_shape), dtype=np.float32)
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == 1:
        a = a[0]
    return a


def sync_normalize_state(train_env: Any, eval_env: Any) -> bool:
    """
    Copy NormalizeWrapper running stats from train_env -> eval_env (best-effort).

    Parameters
    ----------
    train_env : Any
        Training environment, ideally NormalizeWrapper-enabled.
    eval_env : Any
        Evaluation environment, ideally NormalizeWrapper-enabled.

    Returns
    -------
    ok : bool
        True if state sync succeeded, else False.

    Notes
    -----
    Freezes eval stats via eval_env.set_training(False) when supported.
    """
    ok = False
    if callable(getattr(train_env, "state_dict", None)) and callable(getattr(eval_env, "load_state_dict", None)):
        try:
            eval_env.load_state_dict(train_env.state_dict())
            ok = True
        except Exception:
            ok = False

    if callable(getattr(eval_env, "set_training", None)):
        try:
            eval_env.set_training(False)
        except Exception:
            pass

    return ok


def infer_env_id(env: Any) -> Optional[str]:
    """
    Best-effort inference of env_id from Gym/Gymnasium env.spec.id.

    Returns
    -------
    env_id : Optional[str]
        `env.spec.id` if available, else None.
    """
    try:
        spec = getattr(env, "spec", None)
        if spec is None and hasattr(env, "unwrapped"):
            spec = getattr(env.unwrapped, "spec", None)
        if spec is None:
            return None
        return getattr(spec, "id", None)
    except Exception:
        return None


def wrap_make_env_with_normalize(
    make_env: Callable[[], Any],
    *,
    obs_shape: Tuple[int, ...],
    norm_obs: bool,
    norm_reward: bool,
    clip_obs: float,
    clip_reward: float,
    gamma: float,
    epsilon: float,
    training: bool,
    max_episode_steps: Optional[int] = None,
    action_rescale: bool = False,
    clip_action: float = 0.0,
    reset_return_on_done: bool = True,
    reset_return_on_trunc: bool = True,
    obs_dtype: Any = np.float32,
) -> Callable[[], Any]:
    """
    Wrap an env factory with NormalizeWrapper.

    Returns
    -------
    wrapped_make_env : Callable[[], Any]
        Factory that builds env then wraps it with NormalizeWrapper.
    """

    def _fn() -> Any:
        env = make_env()
        return NormalizeWrapper(
            env,
            obs_shape=obs_shape,
            norm_obs=bool(norm_obs),
            norm_reward=bool(norm_reward),
            clip_obs=float(clip_obs),
            clip_reward=float(clip_reward),
            gamma=float(gamma),
            epsilon=float(epsilon),
            training=bool(training),
            max_episode_steps=max_episode_steps,
            action_rescale=bool(action_rescale),
            clip_action=float(clip_action),
            reset_return_on_done=bool(reset_return_on_done),
            reset_return_on_trunc=bool(reset_return_on_trunc),
            obs_dtype=obs_dtype,
        )

    return _fn


# =============================================================================
# RNG seeding
# =============================================================================
def set_random_seed(
    seed: int,
    *,
    deterministic: bool = True,
    verbose: bool = False,
    set_torch_threads_to_one: bool = False,
) -> None:
    """
    Seed Python/NumPy/PyTorch RNGs for reproducibility (best-effort).

    Parameters
    ----------
    seed : int
        Base seed.
    deterministic : bool
        If True, configures PyTorch for deterministic behavior where possible.
    verbose : bool
        If True, prints a short summary.
    set_torch_threads_to_one : bool
        If True, limits intra-/interop threads (useful for some Ray worker setups).
    """
    seed = int(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    if deterministic:
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
        try:
            th.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if set_torch_threads_to_one:
        try:
            th.set_num_threads(1)
            th.set_num_interop_threads(1)
        except Exception:
            pass

    if verbose:
        print(f"[set_random_seed] seed={seed}, deterministic={deterministic}, cuda={th.cuda.is_available()}")


# =============================================================================
# Gym/Gymnasium compat helpers
# =============================================================================
def env_reset(env: Any, *, return_info: bool = False, **kwargs: Any):
    """
    Gym/Gymnasium compatible reset.

    Behavior
    --------
    - Gymnasium: env.reset(...) -> (obs, info)
    - Gym:       env.reset(...) -> obs

    Parameters
    ----------
    env : Any
        Environment with reset(**kwargs).
    return_info : bool, optional
        If True, always return (obs, info_dict). If False, return obs only.

    Returns
    -------
    obs : Any
        Observation.
    info : Dict[str, Any]
        Only when return_info=True. Gym: {} ; Gymnasium: returned info (coerced to dict).
    """
    out = env.reset(**kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        info_dict = _to_info_dict(info)
    else:
        obs, info_dict = out, {}

    if return_info:
        return obs, info_dict
    return obs


def _to_info_dict(info: Any) -> Dict[str, Any]:
    """Coerce info to a plain dict."""
    if info is None:
        return {}
    if isinstance(info, dict):
        return dict(info)
    if isinstance(info, Mapping):
        return dict(info)
    return {"_info": info}


def _process_obs(
    obs: Any,
    *,
    flatten: bool = False,
    obs_dtype: Any = np.float32,
) -> Any:
    """
    Cast/flatten array-like obs without changing container structure.

    - dict/tuple/list: return as-is
    - else: np.asarray(dtype) or to_flat_np(dtype)
    """
    if isinstance(obs, dict) or isinstance(obs, (tuple, list)):
        return obs
    try:
        return to_flat_np(obs, dtype=obs_dtype) if flatten else np.asarray(obs, dtype=obs_dtype)
    except Exception:
        return obs


def unpack_step(
    step_out: Any,
    *,
    flatten_obs: bool = False,
    obs_dtype: Any = np.float32,
) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Normalize env.step() outputs into (next_obs, reward, done, info).

    Supports:
      - Gym:       (obs, reward, done, info)
      - Gymnasium: (obs, reward, terminated, truncated, info)
    """
    if not isinstance(step_out, tuple):
        raise ValueError(f"env.step(...) must return tuple, got: {type(step_out)}")

    n = len(step_out)
    if n == 4:
        next_obs, reward, done, info = step_out
        terminated, truncated = done, False
    elif n == 5:
        next_obs, reward, terminated, truncated, info = step_out
    else:
        raise ValueError(f"Unsupported step() return signature (len={n}).")

    next_obs = _process_obs(next_obs, flatten=flatten_obs, obs_dtype=obs_dtype)
    info_d = _to_info_dict(info)

    r = to_scalar(reward)
    t = to_scalar(terminated)
    tr = to_scalar(truncated)
    if r is None or t is None or tr is None:
        raise ValueError("reward/terminated/truncated must be scalar-like for unpack_step().")

    done_flag = bool(t) or bool(tr)
    if bool(tr) and "TimeLimit.truncated" not in info_d:
        info_d["TimeLimit.truncated"] = True

    return next_obs, float(r), done_flag, info_d