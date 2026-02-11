from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch as th

from ..utils.common_utils import _to_scalar
from ..utils.logger_utils import _make_run_dir


class Logger:
    """
    Scalar-first experiment logger (frontend).

    The `Logger` is responsible for **frontend concerns**:
    - Resolving and owning a per-run directory (`run_dir`)
    - Inferring global step (via bound trainer or custom callable)
    - Normalizing metric keys (prefixing and path normalization)
    - Per-key throttling (log a key every N steps)
    - Optional dropping of non-finite values (NaN/Inf)
    - In-memory aggregation buffer (`record` -> `dump` / `dump_stats`)
    - Console printing at a configured cadence
    - Best-effort dumps of runtime metadata and experiment configs

    Writer backends are responsible for **I/O concerns**:
    - file/network I/O
    - serialization formats
    - buffering, flush, close semantics

    Parameters
    ----------
    log_dir : str, default="./runs"
        Root directory for experiment runs.
    exp_name : str, default="exp"
        Experiment name used as a subdirectory under `log_dir`.
    run_id : str, optional
        Explicit run identifier with highest priority. When provided, it is used
        to form a deterministic run directory name.
    run_name : str, optional
        Human-friendly run identifier used when `run_id` is not provided.
        Exact usage depends on `_make_run_dir`.
    overwrite : bool, default=False
        If True, reuse an existing run directory if it already exists.
        (Semantics depend on `_make_run_dir`.)
    resume : bool, default=False
        If True, treat the resolved run directory as an existing run and append to it.
        The directory is still created with `exist_ok=True` for robustness.
    writers : Iterable, optional
        Writer backend instances (e.g., CSV/JSONL/TensorBoard/W&B). If None,
        no writers are attached at initialization.
    console_every : int, default=1
        Print to stdout every N calls to `log()` (call-count based, not step-based).
        Set <= 0 to disable console output.
    flush_every : int, default=200
        Flush writers every N calls to `log()`. Set <= 0 to disable periodic flushing.
    drop_non_finite : bool, default=False
        If True, discard NaN/Inf scalars rather than writing them.
    strict : bool, default=False
        If True, re-raise exceptions encountered during writer operations or metadata dumps.
        If False, errors are recorded in `_errors` and execution continues (best-effort).

    Attributes
    ----------
    run_dir : str
        Resolved run directory path owned by this logger.
    strict : bool
        Strict mode flag controlling exception propagation.
    console_every : int
        Console printing cadence (calls to `log()`).
    flush_every : int
        Flush cadence (calls to `log()`).
    drop_non_finite : bool
        Whether to drop NaN/Inf scalars.
    _errors : list of str
        Collected error messages from best-effort operations.
    _writers : list
        Attached writer backends.

    Notes
    -----
    - Scalar conversion is delegated to `_to_scalar(x)`, which is expected to return
      a float-like scalar or None when conversion is not possible.
    - Step inference uses (in priority order):
        1) explicit `step` argument
        2) custom callable set via `set_step_fn`
        3) bound trainer attribute `global_env_step`
        4) fallback to 0
    """

    def __init__(
        self,
        *,
        log_dir: str = "./runs",
        exp_name: str = "exp",
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        overwrite: bool = False,
        resume: bool = False,
        writers: Optional[Iterable[Any]] = None,
        console_every: int = 1,
        flush_every: int = 200,
        drop_non_finite: bool = False,
        strict: bool = False,
    ) -> None:
        self.strict = bool(strict)
        self._errors: List[str] = []

        # Resolve and create the run directory.
        self.run_dir = _make_run_dir(
            log_dir=log_dir,
            exp_name=exp_name,
            run_id=run_id,
            run_name=run_name,
            overwrite=bool(overwrite),
            resume=bool(resume),
        )
        os.makedirs(self.run_dir, exist_ok=True)

        # Console/flush cadence is based on number of `log()` calls.
        self.console_every = int(console_every)
        self.flush_every = int(flush_every)
        self.drop_non_finite = bool(drop_non_finite)

        # Timing counters.
        self._start_time = time.time()
        self._log_calls = 0

        # Step inference hooks.
        self._step_fn: Optional[Callable[[], int]] = None
        self._trainer_ref: Optional[Any] = None

        # Per-key throttling: full_key -> every_n_steps.
        self._key_every: Dict[str, int] = {}

        # In-memory aggregation buffer: full_key -> list of floats.
        self._buffer: Dict[str, List[float]] = defaultdict(list)

        # Attached writer backends.
        self._writers: List[Any] = list(writers) if writers is not None else []

        # Best-effort metadata dump at creation time.
        try:
            self.dump_metadata(filename="metadata.json")
        except Exception as e:
            self._handle_exception(e, "dump_metadata")

    # ---------------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------------
    def __enter__(self) -> "Logger":
        """
        Enter context manager.

        Returns
        -------
        Logger
            The logger itself.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit context manager and close all writers (best-effort).

        Notes
        -----
        - Writer failures during close obey the `strict` policy.
        """
        self.close()

    # ---------------------------------------------------------------------
    # Binding / step inference
    # ---------------------------------------------------------------------
    def set_step_fn(self, fn: Optional[Callable[[], int]]) -> None:
        """
        Set a callable used to infer the current global step.

        Parameters
        ----------
        fn : callable or None
            Callable that returns the current step as an integer.
            Set to None to disable custom inference.

        Notes
        -----
        - If provided, this callable takes precedence over a bound trainer.
        - Exceptions inside the callable are swallowed and step falls back to 0.
        """
        self._step_fn = fn

    def bind_trainer(self, trainer: Any) -> None:
        """
        Bind a trainer object for default step inference.

        The default extractor attempts to read:
        ``trainer.global_env_step``

        Parameters
        ----------
        trainer : Any
            Trainer-like object that (optionally) exposes `global_env_step`.

        Notes
        -----
        - This method sets an internal default step function which is used by
          `_infer_step()` when no explicit `step` is provided.
        - On any access/cast failure, the inferred step defaults to 0.
        """
        self._trainer_ref = trainer

        def _default_step() -> int:
            try:
                return int(getattr(trainer, "global_env_step", 0))
            except Exception:
                return 0

        self._step_fn = _default_step

    def _infer_step(self, step: Optional[int]) -> int:
        """
        Infer the step according to the logger's step policy.

        Parameters
        ----------
        step : int, optional
            Explicit step override.

        Returns
        -------
        int
            Inferred step (>=0 by convention; not strictly enforced).
        """
        if step is not None:
            return int(step)
        if self._step_fn is not None:
            try:
                return int(self._step_fn())
            except Exception:
                return 0
        if self._trainer_ref is not None:
            try:
                return int(getattr(self._trainer_ref, "global_env_step", 0))
            except Exception:
                return 0
        return 0

    # ---------------------------------------------------------------------
    # Error handling
    # ---------------------------------------------------------------------
    def _handle_exception(self, err: BaseException, context: str) -> None:
        """
        Handle an internal exception according to the `strict` policy.

        Parameters
        ----------
        err : BaseException
            The exception object.
        context : str
            Human-readable context string indicating where the error occurred.

        Notes
        -----
        - In non-strict mode, errors are recorded in `_errors` and execution continues.
        - In strict mode, the exception is re-raised.
        """
        msg = f"[{self.__class__.__name__}] {context}: {type(err).__name__}: {err}"
        self._errors.append(msg)
        if self.strict:
            raise err

    # ---------------------------------------------------------------------
    # Key normalization and throttling
    # ---------------------------------------------------------------------
    @staticmethod
    def _norm_prefix(prefix: str) -> str:
        """
        Normalize a prefix into a canonical path-like form.

        Parameters
        ----------
        prefix : str
            User-provided prefix, e.g., "train", "eval", "rollout/env0".

        Returns
        -------
        str
            Normalized prefix ending with '/' when non-empty, otherwise ''.

        Notes
        -----
        - Backslashes are converted to forward slashes.
        - Leading/trailing slashes are stripped.
        """
        p = str(prefix).strip()
        if not p:
            return ""
        p = p.replace("\\", "/").strip("/")
        return p + "/"

    @staticmethod
    def _norm_key(key: Any) -> str:
        """
        Normalize a metric key.

        Parameters
        ----------
        key : Any
            Metric key, typically str-like.

        Returns
        -------
        str
            Normalized key without a leading '/'.

        Notes
        -----
        - Converts backslashes to forward slashes.
        - Strips whitespace and leading slashes to avoid accidental absolute paths.
        """
        k = str(key).strip().replace("\\", "/")
        return k.lstrip("/")

    def _join_name(self, prefix: str, key: Any) -> str:
        """
        Join prefix and key into a full metric name.

        Parameters
        ----------
        prefix : str
            Prefix (e.g., "train").
        key : Any
            Metric key (e.g., "loss").

        Returns
        -------
        str
            Full key name (e.g., "train/loss").
        """
        p = self._norm_prefix(prefix)
        k = self._norm_key(key)
        return f"{p}{k}" if p else k

    def set_key_every(self, mapping: Mapping[str, int]) -> None:
        """
        Set per-key throttling: log a key only every N steps.

        Parameters
        ----------
        mapping : Mapping[str, int]
            Mapping from full metric key to `every_n_steps`.

            - If `every_n_steps <= 0`, throttling is removed for that key.
            - Otherwise, the key is logged only when `step % every_n_steps == 0`.

        Notes
        -----
        Keys are normalized via `_norm_key` before storing.
        """
        for k, v in mapping.items():
            kk = self._norm_key(k)
            vv = int(v)
            if vv <= 0:
                self._key_every.pop(kk, None)
            else:
                self._key_every[kk] = vv

    def _should_log_key(self, full_key: str, step: int) -> bool:
        """
        Decide whether a given key should be logged at the current step.

        Parameters
        ----------
        full_key : str
            Fully-qualified metric key (after prefix normalization).
        step : int
            Current step.

        Returns
        -------
        bool
            True if the key should be logged; False if throttled.
        """
        every = self._key_every.get(full_key, None)
        if every is None or every <= 0:
            return True
        return (int(step) % int(every)) == 0

    # ---------------------------------------------------------------------
    # Public logging APIs
    # ---------------------------------------------------------------------
    def log(
        self,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
        *,
        pbar: Optional[Any] = None,
        prefix: str = "",
    ) -> None:
        """
        Immediately write metrics to writer backends (and optionally console).

        Parameters
        ----------
        metrics : Mapping[str, Any]
            Metric mapping. Values are converted to floats using `_to_scalar`.
            Non-convertible values are skipped.
        step : int, optional
            Explicit step override. If omitted, step is inferred.
        pbar : Any, optional
            Optional progress-bar-like object (e.g., tqdm) supporting
            `set_description_str`. If provided, console output is routed through it.
        prefix : str, default=""
            Optional prefix applied to all keys (e.g., "train", "eval").

        Notes
        -----
        - Throttling is applied after key normalization (`set_key_every`).
        - Metadata keys are injected into every emitted row:
          ``step``, ``wall_time``, ``timestamp``.
        - Console printing and periodic flushing are based on number of calls to `log()`.
        """
        s = self._infer_step(step)
        self._log_calls += 1

        row: Dict[str, float] = {}

        # Convert metrics to float scalars and apply throttling.
        for k, v in metrics.items():
            name = self._join_name(prefix, k)

            val = _to_scalar(v)
            if val is None:
                continue

            try:
                fval = float(val)
            except Exception:
                continue

            if self.drop_non_finite and (not np.isfinite(fval)):
                continue

            if not self._should_log_key(name, s):
                continue

            row[name] = fval

        # Inject metadata fields (always present).
        now = time.time()
        row["step"] = float(int(s))
        row["wall_time"] = float(now - self._start_time)
        row["timestamp"] = float(now)

        # Dispatch to writers.
        for w in self._writers:
            try:
                w.write(row)
            except Exception as e:
                self._handle_exception(e, f"writer.write({w.__class__.__name__})")

        # Console printing.
        if self.console_every > 0 and (self._log_calls % self.console_every == 0):
            self._print_console(row, pbar=pbar)

        # Periodic flush.
        if self.flush_every > 0 and (self._log_calls % self.flush_every == 0):
            self.flush()

    def record(self, metrics: Mapping[str, Any], *, prefix: str = "") -> None:
        """
        Record metrics in an in-memory buffer for later aggregation.

        Parameters
        ----------
        metrics : Mapping[str, Any]
            Metric mapping. Values are converted to floats using `_to_scalar`.
            Non-convertible values are skipped.
        prefix : str, default=""
            Optional prefix applied to all keys recorded into the buffer.

        Notes
        -----
        - This method does not call writers.
        - Use `dump()` or `dump_stats()` to aggregate buffered values and emit scalars.
        """
        for k, v in metrics.items():
            name = self._join_name(prefix, k)

            val = _to_scalar(v)
            if val is None:
                continue

            try:
                fval = float(val)
            except Exception:
                continue

            if self.drop_non_finite and (not np.isfinite(fval)):
                continue

            self._buffer[name].append(fval)

    def dump(
        self,
        step: Optional[int] = None,
        *,
        prefix: str = "",
        agg: str = "mean",
        clear: bool = True,
    ) -> None:
        """
        Aggregate buffered scalars and emit them via `log()`.

        Parameters
        ----------
        step : int, optional
            Explicit step override.
        prefix : str, default=""
            Optional prefix applied to output keys at dump time.
            This is applied *after* aggregation.
        agg : {"mean", "min", "max", "std"}, default="mean"
            Aggregation operator.
        clear : bool, default=True
            If True, clear the buffer after dumping.

        Raises
        ------
        ValueError
            If `agg` is not one of {"mean", "min", "max", "std"}.

        Notes
        -----
        - If the buffer is empty, this is a no-op.
        - Output keys preserve original recording keys unless `prefix` is given.
        """
        op = str(agg).lower().strip()
        if op not in ("mean", "min", "max", "std"):
            raise ValueError(f"Unknown agg={agg!r}. Use mean|min|max|std.")

        out: Dict[str, float] = {}
        for k, vals in self._buffer.items():
            if not vals:
                continue
            a = np.asarray(vals, dtype=np.float64)
            if op == "mean":
                out[k] = float(np.mean(a))
            elif op == "min":
                out[k] = float(np.min(a))
            elif op == "max":
                out[k] = float(np.max(a))
            else:
                out[k] = float(np.std(a))

        if clear:
            self._buffer.clear()

        if not out:
            return

        if prefix:
            out = {self._join_name(prefix, k): v for k, v in out.items()}

        # Prefix is already applied above (if requested), so pass prefix="".
        self.log(out, step=step, prefix="")

    def dump_stats(
        self,
        step: Optional[int] = None,
        *,
        prefix: str = "",
        clear: bool = True,
        suffixes: Tuple[str, ...] = ("mean", "min", "max", "std"),
    ) -> None:
        """
        Emit multiple statistics per buffered key (mean/min/max/std).

        Parameters
        ----------
        step : int, optional
            Explicit step override.
        prefix : str, default=""
            Optional prefix applied to output keys at dump time.
        clear : bool, default=True
            If True, clear the buffer after dumping.
        suffixes : tuple of str, default=("mean","min","max","std")
            Statistics to compute. Each element must be in {"mean","min","max","std"}.

        Raises
        ------
        ValueError
            If any suffix is not supported.

        Notes
        -----
        - For each recorded key `k`, output keys are of the form:
          `k_mean`, `k_min`, `k_max`, `k_std` (subset depending on `suffixes`).
        """
        use = tuple(str(s).lower().strip() for s in suffixes)
        for sfx in use:
            if sfx not in ("mean", "min", "max", "std"):
                raise ValueError(f"Unsupported suffix stat: {sfx!r}")

        out: Dict[str, float] = {}
        for k, vals in self._buffer.items():
            if not vals:
                continue
            a = np.asarray(vals, dtype=np.float64)
            if "mean" in use:
                out[f"{k}_mean"] = float(np.mean(a))
            if "min" in use:
                out[f"{k}_min"] = float(np.min(a))
            if "max" in use:
                out[f"{k}_max"] = float(np.max(a))
            if "std" in use:
                out[f"{k}_std"] = float(np.std(a))

        if clear:
            self._buffer.clear()

        if not out:
            return

        if prefix:
            out = {self._join_name(prefix, k): v for k, v in out.items()}

        self.log(out, step=step, prefix="")

    # ---------------------------------------------------------------------
    # Config / metadata
    # ---------------------------------------------------------------------
    def dump_config(self, config: Mapping[str, Any], filename: str = "config.json") -> None:
        """
        Dump experiment configuration as JSON into `run_dir`.

        Parameters
        ----------
        config : Mapping[str, Any]
            Configuration mapping. Must be JSON-serializable or convertible via `default=str`.
        filename : str, default="config.json"
            Output filename under `run_dir`.

        Notes
        -----
        - Performs a shallow copy to avoid mutating caller objects.
        - Falls back to stringifying keys if `dict(config)` fails.
        """
        path = os.path.join(self.run_dir, filename)
        try:
            payload = dict(config)
        except Exception:
            payload = {str(k): v for k, v in config.items()}  # type: ignore[attr-defined]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def dump_metadata(self, filename: str = "metadata.json") -> None:
        """
        Dump runtime/environment metadata as JSON into `run_dir`.

        Parameters
        ----------
        filename : str, default="metadata.json"
            Output filename under `run_dir`.

        Notes
        -----
        Captures (best-effort):
        - Run directory and start timestamp
        - Hostname/FQDN/PID and Python/platform info
        - Torch/CUDA information (if available)
        - Git information (commit/branch/dirty) when inside a git repo
        """
        meta: Dict[str, Any] = {
            "run_dir": self.run_dir,
            "start_time_unix": float(self._start_time),
            "start_time_iso": datetime.fromtimestamp(self._start_time).isoformat(),
            "host": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "pid": os.getpid(),
            "python": sys.version.replace("\n", " "),
            "platform": sys.platform,
        }

        # Torch/CUDA info (best-effort).
        try:
            meta["torch"] = str(getattr(th, "__version__", "unknown"))
            meta["cuda_available"] = bool(th.cuda.is_available())
            if th.cuda.is_available():
                meta["cuda_device_count"] = int(th.cuda.device_count())
                meta["cuda_device_name0"] = str(th.cuda.get_device_name(0))
        except Exception:
            pass

        meta["git"] = self._get_git_info()

        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _get_git_info() -> Dict[str, Any]:
        """
        Collect minimal git information (best-effort).

        Returns
        -------
        Dict[str, Any]
            Dictionary with optional keys:
            - "commit": str
            - "branch": str
            - "dirty": bool

            Returns an empty dict if not inside a git repo or git is unavailable.
        """

        def _run(args: List[str]) -> Optional[str]:
            try:
                out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
                return out.decode("utf-8", errors="ignore").strip()
            except Exception:
                return None

        info: Dict[str, Any] = {}
        inside = _run(["git", "rev-parse", "--is-inside-work-tree"])
        if inside not in ("true", "True"):
            return info

        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain"])

        if commit is not None:
            info["commit"] = commit
        if branch is not None:
            info["branch"] = branch
        if status is not None:
            info["dirty"] = bool(status.strip() != "")

        return info

    # ---------------------------------------------------------------------
    # Writer lifecycle
    # ---------------------------------------------------------------------
    def add_writer(self, writer: Any) -> None:
        """
        Attach a single writer backend.

        Parameters
        ----------
        writer : Any
            Writer-like object exposing `write()`, `flush()`, and `close()`.

        Notes
        -----
        - This method does not validate the interface strictly to keep the logger lightweight.
        """
        self._writers.append(writer)

    def add_writers(self, writers: Iterable[Any]) -> None:
        """
        Attach multiple writer backends.

        Parameters
        ----------
        writers : Iterable[Any]
            Iterable of writer-like objects.
        """
        for w in writers:
            self.add_writer(w)

    def flush(self) -> None:
        """
        Flush all writers (best-effort unless `strict=True`).

        Notes
        -----
        - Errors are recorded and optionally raised according to the `strict` policy.
        """
        for w in self._writers:
            try:
                w.flush()
            except Exception as e:
                self._handle_exception(e, f"writer.flush({w.__class__.__name__})")

    def close(self) -> None:
        """
        Flush and close all writers (best-effort unless `strict=True`).

        Notes
        -----
        - Calls `flush()` first.
        - Closes each writer even if flushing fails.
        """
        try:
            self.flush()
        finally:
            for w in self._writers:
                try:
                    w.close()
                except Exception as e:
                    self._handle_exception(e, f"writer.close({w.__class__.__name__})")

    # ---------------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------------
    @staticmethod
    def _print_console(row: Mapping[str, float], *, pbar: Optional[Any] = None) -> None:
        """
        Render a compact console line for the current logging row.

        Parameters
        ----------
        row : Mapping[str, float]
            Logging row (includes meta keys `step`, `wall_time`, `timestamp`).
        pbar : Any, optional
            Progress-bar-like object supporting `set_description_str`. If provided,
            the message is routed to the progress bar instead of printing a new line.

        Notes
        -----
        - Prefers a curated set of common RL metrics; falls back to the first few keys.
        - Avoids printing meta keys except `step` and elapsed wall time.
        """
        step = int(row.get("step", 0.0))
        wall = float(row.get("wall_time", 0.0))

        preferred = (
            "train/loss",
            "train/actor_loss",
            "train/critic_loss",
            "train/entropy",
            "train/lr",
            "rollout/ep_return_mean",
            "eval/return_mean",
            "eval/len_mean",
        )

        shown: List[str] = []
        for k in preferred:
            if k in row:
                try:
                    shown.append(f"{k}={float(row[k]):.4g}")
                except Exception:
                    pass

        if not shown:
            for k, v in row.items():
                if k in ("step", "wall_time", "timestamp"):
                    continue
                try:
                    shown.append(f"{k}={float(v):.4g}")
                except Exception:
                    continue
                if len(shown) >= 6:
                    break

        msg = f"[step={step} | t={wall:.1f}s] " + " ".join(shown)

        # Prefer progress bar update if provided.
        if pbar is not None:
            try:
                pbar.set_description_str(msg, refresh=True)
                return
            except Exception:
                pass

        print(msg)
