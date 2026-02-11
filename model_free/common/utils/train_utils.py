from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import os
import random

import numpy as np
import torch as th

from .common_utils import _to_action_np, _to_flat_np, _to_scalar
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

    This class is a small compatibility shim used when `tqdm` is not available.
    It implements only the methods typically used by training loops:
    - `update(n)`
    - `close()`

    Parameters
    ----------
    total : int, default=0
        Total expected steps for the progress bar (informational only here).
    initial : int, default=0
        Initial counter value.
    **kwargs : Any
        Ignored. Included to be signature-compatible with `tqdm(...)`.

    Notes
    -----
    - This is intentionally tiny: it does not render anything to the console.
    - Keeping the same interface allows training code to remain importable and runnable
      without optional dependencies.
    """

    def __init__(self, total: int = 0, initial: int = 0, **kwargs: Any) -> None:
        self.total = int(total)
        self.n = int(initial)

    def update(self, n: int = 1) -> None:
        """
        Increment the internal counter.

        Parameters
        ----------
        n : int, default=1
            Amount to increment.
        """
        self.n += int(n)

    def close(self) -> None:
        """
        Close the progress bar.

        Notes
        -----
        No-op for DummyTqdm.
        """
        return


def _make_pbar(**kwargs: Any) -> Any:
    """
    Create a progress bar instance.

    Parameters
    ----------
    **kwargs : Any
        Passed through to `tqdm(...)` or `DummyTqdm(...)`.

    Returns
    -------
    pbar : Any
        `tqdm(...)` instance if available, else `DummyTqdm`.

    Notes
    -----
    This function centralizes the dependency on tqdm so the rest of the codebase
    can remain clean and optional-dependency friendly.
    """
    if tqdm is None:
        return DummyTqdm(**kwargs)
    return tqdm(**kwargs)


# =============================================================================
# Small utilities (keep minimal and predictable)
# =============================================================================
def _maybe_call(obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """
    Call a method on `obj` if it exists and is callable.

    This is a permissive helper used for optional hooks like:
    - env.set_training(...)
    - env.close()
    - callback.on_step(...)

    Parameters
    ----------
    obj : Any
        Target object. If None, this returns None.
    method : str
        Method name to look up via `getattr(obj, method, None)`.
    *args : Any
        Positional arguments forwarded to the method call.
    **kwargs : Any
        Keyword arguments forwarded to the method call.

    Returns
    -------
    out : Any
        Return value of the called method, or None if:
        - `obj` has no such attribute
        - attribute is not callable
        - `obj` is None

    Notes
    -----
    This helper intentionally does NOT catch exceptions raised by the method call.
    If you want a "never raise" semantics, use a different helper (e.g., `_safe_call`).
    """
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _format_env_action(action: Any, action_space: Any) -> Any:
    """
    Convert a policy output into an environment-compatible action (best-effort).

    This function tries to standardize policy outputs (torch tensors, numpy arrays,
    scalars, batched shapes) into the format expected by Gym/Gymnasium action spaces.

    Parameters
    ----------
    action : Any
        Policy output. Common cases:
        - torch.Tensor
        - numpy.ndarray
        - python scalar
        - list/tuple
        - batched action like shape (1, A)
    action_space : Any
        Environment action space (Gym/Gymnasium). If None, returns `action` as-is.

    Returns
    -------
    action_env : Any
        Action formatted for env.step(...). Typical mappings:
        - Discrete      -> int
        - MultiDiscrete -> np.ndarray[int64] with `action_space.shape`
        - MultiBinary   -> np.ndarray[int8] with `action_space.shape` (thresholded at 0.5)
        - Box           -> np.ndarray[float32] with `action_space.shape`
        - Fallback      -> best-effort np.ndarray[float32] or int for `space.n`

    Notes
    -----
    - This is intentionally permissive because different policies emit different
      action types and shapes.
    - For Box spaces, if the policy emits (1, A) and env expects (A,), it squeezes
      the leading batch dimension.
    """
    if action_space is None:
        return action

    # Prefer explicit space type checks when gym/gymnasium is importable.
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
            a = np.asarray(_to_action_np(action, action_shape=None)).reshape(-1)
            return int(a[0])

        if isinstance(action_space, _spaces.MultiDiscrete):
            shp = tuple(action_space.shape)
            a = np.asarray(_to_action_np(action, action_shape=shp)).reshape(shp)
            return a.astype(np.int64, copy=False)

        if isinstance(action_space, _spaces.MultiBinary):
            shp = tuple(action_space.shape)
            a = np.asarray(_to_action_np(action, action_shape=shp)).reshape(shp)
            return (a > 0.5).astype(np.int8, copy=False)

        if isinstance(action_space, _spaces.Box):
            shp = tuple(action_space.shape)
            a = np.asarray(_to_action_np(action, action_shape=shp), dtype=np.float32).reshape(shp)
            if a.ndim >= 2 and a.shape[0] == 1:  # policy emits (1,A), env expects (A,)
                a = a[0]
            return a

    # Fallback: legacy heuristics (no space classes available)
    if hasattr(action_space, "n"):
        a = np.asarray(_to_action_np(action, action_shape=None)).reshape(-1)
        return int(a[0])

    shp = getattr(action_space, "shape", None)
    action_shape: Optional[Tuple[int, ...]] = None
    if isinstance(shp, tuple):
        try:
            action_shape = tuple(int(x) for x in shp)
        except Exception:
            action_shape = None

    a = np.asarray(_to_action_np(action, action_shape=action_shape), dtype=np.float32)
    if a.ndim >= 2 and a.shape[0] == 1:
        a = a[0]
    return a


def _sync_normalize_state(train_env: Any, eval_env: Any) -> bool:
    """
    Synchronize NormalizeWrapper running statistics from train_env to eval_env.

    Parameters
    ----------
    train_env : Any
        Training environment. Typically an instance of `NormalizeWrapper`, or a wrapper
        stack that exposes `state_dict()`.
    eval_env : Any
        Evaluation environment. Typically also `NormalizeWrapper`, or wrapper stack
        exposing `load_state_dict()`.

    Returns
    -------
    ok : bool
        True if state sync succeeded, else False.

    Notes
    -----
    - This is best-effort and will not raise on sync failure.
    - If `eval_env.set_training(False)` is available, it is called to freeze
      normalization statistics during evaluation.
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


def _infer_env_id(env: Any) -> Optional[str]:
    """
    Best-effort inference of `env_id` from Gym/Gymnasium `env.spec.id`.

    Parameters
    ----------
    env : Any
        Environment instance. `env.spec.id` is checked first; if missing, also tries
        `env.unwrapped.spec.id` when available.

    Returns
    -------
    env_id : Optional[str]
        Environment id string if available (e.g., "CartPole-v1"), else None.
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


def _wrap_make_env_with_normalize(
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
    Wrap an env factory with `NormalizeWrapper`.

    Parameters
    ----------
    make_env : Callable[[], Any]
        Factory function that returns a raw environment instance.
    obs_shape : Tuple[int, ...]
        Observation shape expected by the normalization wrapper.
    norm_obs : bool
        Whether to normalize observations.
    norm_reward : bool
        Whether to normalize rewards/returns.
    clip_obs : float
        Clip threshold for normalized observations.
    clip_reward : float
        Clip threshold for normalized rewards/returns.
    gamma : float
        Discount factor used for return normalization.
    epsilon : float
        Small constant for numerical stability in running stats.
    training : bool
        If True, wrapper updates running stats; if False, stats are frozen.
    max_episode_steps : Optional[int], default=None
        Optional time limit used by the wrapper to manage returns.
    action_rescale : bool, default=False
        Whether to rescale actions to env bounds internally (wrapper-dependent).
    clip_action : float, default=0.0
        If > 0, clip actions to [-clip_action, +clip_action] before env.step.
    reset_return_on_done : bool, default=True
        Whether to reset running return accumulator on terminal transitions.
    reset_return_on_trunc : bool, default=True
        Whether to reset running return accumulator on truncations (time limits).
    obs_dtype : Any, default=np.float32
        Observation dtype used when the wrapper stores/updates stats.

    Returns
    -------
    wrapped_make_env : Callable[[], Any]
        New factory that creates the env and wraps it with `NormalizeWrapper`.

    Notes
    -----
    This pattern keeps your outer training code clean: you can pass around a single
    `make_env` callable and decide wrapping behavior centrally.
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
def _set_random_seed(
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
    deterministic : bool, default=True
        If True, configures PyTorch for deterministic behavior where possible.
        This can reduce performance and may raise errors for unsupported ops on
        some platforms.
    verbose : bool, default=False
        If True, prints a short summary.
    set_torch_threads_to_one : bool, default=False
        If True, limits PyTorch intra-/interop threads. This can be useful for:
        - reducing CPU oversubscription in multi-worker setups (Ray, multiprocessing)
        - improving determinism in some environments

    Returns
    -------
    None

    Notes
    -----
    This is a *best-effort* determinism helper:
      - Full determinism is not guaranteed across different GPU drivers, CUDA versions,
        and non-deterministic ops.
      - `CUBLAS_WORKSPACE_CONFIG` is set to a safe default when deterministic=True.
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
def _to_info_dict(info: Any) -> Dict[str, Any]:
    """
    Coerce an environment `info` object to a plain Python dict.

    Parameters
    ----------
    info : Any
        Info object returned by env.reset/step.

    Returns
    -------
    info_dict : Dict[str, Any]
        - If `info` is dict/Mapping -> copied to a plain dict
        - If `info` is None -> {}
        - Else -> {"_info": info}

    Notes
    -----
    This helper ensures downstream logging code can safely assume a `dict`.
    """
    if info is None:
        return {}
    if isinstance(info, dict):
        return dict(info)
    if isinstance(info, Mapping):
        return dict(info)
    return {"_info": info}


def _env_reset(env: Any, *, return_info: bool = False, **kwargs: Any):
    """
    Reset an environment with Gym/Gymnasium compatibility.

    Behavior
    --------
    - Gymnasium: `env.reset(...) -> (obs, info)`
    - Gym:       `env.reset(...) -> obs`

    Parameters
    ----------
    env : Any
        Environment instance exposing `reset(**kwargs)`.
    return_info : bool, default=False
        If True, always returns `(obs, info_dict)`. If False, returns `obs`.
    **kwargs : Any
        Forwarded to `env.reset(...)` (e.g., seed=..., options=...).

    Returns
    -------
    obs : Any
        Observation.
    info : Dict[str, Any]
        Only returned when `return_info=True`.
        - Gym: {}
        - Gymnasium: returned info coerced to dict

    Notes
    -----
    This function intentionally does not flatten/cast observations; use `_process_obs`
    if you need dtype normalization.
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


def _process_obs(
    obs: Any,
    *,
    flatten: bool = False,
    obs_dtype: Any = np.float32,
) -> Any:
    """
    Optionally cast/flatten array-like observations without changing containers.

    Parameters
    ----------
    obs : Any
        Observation returned by env.reset/step. Can be:
        - scalar / array-like
        - dict/tuple/list (returned unchanged)
    flatten : bool, default=False
        If True and `obs` is array-like, flatten to 1D via `_to_flat_np`.
        If False, convert via `np.asarray`.
    obs_dtype : Any, default=np.float32
        Target dtype used for numpy conversion.

    Returns
    -------
    out : Any
        Processed observation. Container structures are preserved:
        - dict/tuple/list -> returned as-is
        - array-like -> numpy array (flattened if requested)
        - on conversion failure -> original `obs`

    Notes
    -----
    This helper is intentionally conservative: it does not attempt to recursively
    process dict observations. If you use dict observations, handle them upstream
    with a dedicated preprocessor.
    """
    if isinstance(obs, dict) or isinstance(obs, (tuple, list)):
        return obs

    try:
        if flatten:
            return _to_flat_np(obs, dtype=obs_dtype)
        return np.asarray(obs, dtype=obs_dtype)
    except Exception:
        return obs


def _unpack_step(
    step_out: Any,
    *,
    flatten_obs: bool = False,
    obs_dtype: Any = np.float32,
) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Normalize env.step(...) outputs into a stable 4-tuple:

        (next_obs, reward, done, info_dict)

    Supported signatures
    --------------------
    - Gym:       (obs, reward, done, info)
    - Gymnasium: (obs, reward, terminated, truncated, info)

    Parameters
    ----------
    step_out : Any
        Return value from `env.step(action)`.
    flatten_obs : bool, default=False
        If True, flattens array-like observations into 1D vectors.
    obs_dtype : Any, default=np.float32
        Dtype used when converting array-like observations.

    Returns
    -------
    next_obs : Any
        Next observation, optionally processed by `_process_obs`.
    reward : float
        Reward as a Python float.
    done : bool
        True iff terminated or truncated.
    info : Dict[str, Any]
        Info coerced to a plain dict. If truncated=True, injects:
            info["TimeLimit.truncated"] = True
        if not already present.

    Raises
    ------
    ValueError
        If:
        - step_out is not a tuple,
        - step_out length is not 4 or 5,
        - reward/terminated/truncated are not scalar-like.

    Notes
    -----
    - Gymnasium distinguishes termination vs truncation; this function collapses them
      into a single `done` flag for compatibility with older pipelines.
    - If you need to preserve the distinction, return both flags instead.
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

    r = _to_scalar(reward)
    t = _to_scalar(terminated)
    tr = _to_scalar(truncated)
    if r is None or t is None or tr is None:
        raise ValueError("reward/terminated/truncated must be scalar-like for unpack_step().")

    done_flag = bool(t) or bool(tr)
    if bool(tr) and "TimeLimit.truncated" not in info_d:
        info_d["TimeLimit.truncated"] = True

    return next_obs, float(r), done_flag, info_d
