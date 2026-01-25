from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch as th
from ..utils.common_utils import to_scalar
from ..utils.log_utils import make_run_dir



# =============================================================================
# Logger
# =============================================================================
class Logger:
    """
    Scalar-first experiment logger.

    This class provides a lightweight frontend for experiment logging.
    It owns:
      - step inference
      - prefix/key normalization
      - key throttling (log every N steps per key)
      - in-memory aggregation buffer (record/dump/dump_stats)
      - console printing
      - run metadata/config dumps

    Writers (backends) own:
      - file/network IO
      - serialization format
      - flush/close semantics

    Parameters
    ----------
    log_dir : str
        Root directory for runs (e.g., "./runs").
    exp_name : str
        Experiment name subdirectory under log_dir.
    run_id : Optional[str]
        Explicit run identifier (filesystem-level). Highest priority.
    run_name : Optional[str]
        Alternative identifier used if run_id is None.
    overwrite : bool
        If True, reuse the same run directory even if it exists.
    resume : bool
        If True, use the computed run directory directly (assumed to exist).
        The directory is still created with exist_ok=True for robustness.
    writers : Optional[Iterable[Writer]]
        Writer instances (TensorBoard/CSV/JSONL/etc). May be empty/None.
    console_every : int
        Print to stdout every N `log()` calls. (Call-count based, not step based.)
        Set <= 0 to disable.
    flush_every : int
        Flush writers every N `log()` calls. Set <= 0 to disable.
    drop_non_finite : bool
        If True, drop NaN/Inf scalars instead of writing them.
    strict : bool
        If True, re-raise exceptions from writers/metadata dumps.

    Notes
    -----
    - This logger assumes `to_scalar(x)` returns either:
      - float-compatible scalar, or
      - None if conversion is not possible.
    - For step inference, bind a trainer via `bind_trainer()` or set a custom
      callable via `set_step_fn()`.
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
        writers: Optional[Iterable] = None,
        console_every: int = 1,
        flush_every: int = 200,
        drop_non_finite: bool = False,
        strict: bool = False,
    ) -> None:
        self.strict = bool(strict)
        self._errors: List[str] = []

        self.run_dir = make_run_dir(
            log_dir=log_dir,
            exp_name=exp_name,
            run_id=run_id,
            run_name=run_name,
            overwrite=bool(overwrite),
            resume=bool(resume),
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.console_every = int(console_every)
        self.flush_every = int(flush_every)
        self.drop_non_finite = bool(drop_non_finite)

        self._start_time = time.time()
        self._log_calls = 0

        # Step inference hooks
        self._step_fn: Optional[Callable[[], int]] = None
        self._trainer_ref: Optional[Any] = None

        # Key throttling: full_key -> every_n_steps
        self._key_every: Dict[str, int] = {}

        # Aggregation buffer: full_key -> list of floats
        self._buffer: Dict[str, List[float]] = defaultdict(list)

        # Writers
        self._writers: List = list(writers) if writers is not None else []

        # Best-effort metadata dump
        try:
            self.dump_metadata(filename="metadata.json")
        except Exception as e:
            self._handle_exception(e, "dump_metadata")

    # ---------------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------------
    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------------------------------------------------
    # Binding / step inference
    # ---------------------------------------------------------------------
    def set_step_fn(self, fn: Optional[Callable[[], int]]) -> None:
        """
        Set a custom callable that returns the current global step.

        Parameters
        ----------
        fn : Optional[Callable[[], int]]
            Callable that returns a step integer. Set None to disable.
        """
        self._step_fn = fn

    def bind_trainer(self, trainer: Any) -> None:
        """
        Bind a trainer object for default step inference.

        The default step extractor tries:
        trainer.global_env_step (int), falling back to 0 on error.
        """
        self._trainer_ref = trainer

        def _default_step() -> int:
            try:
                return int(getattr(trainer, "global_env_step", 0))
            except Exception:
                return 0

        self._step_fn = _default_step

    def _infer_step(self, step: Optional[int]) -> int:
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
        msg = f"[{self.__class__.__name__}] {context}: {type(err).__name__}: {err}"
        self._errors.append(msg)
        if self.strict:
            raise err

    # ---------------------------------------------------------------------
    # Key normalization and throttling
    # ---------------------------------------------------------------------
    @staticmethod
    def _norm_prefix(prefix: str) -> str:
        p = str(prefix).strip()
        if not p:
            return ""
        p = p.replace("\\", "/").strip("/")
        return p + "/"

    @staticmethod
    def _norm_key(key: Any) -> str:
        k = str(key).strip().replace("\\", "/")
        return k.lstrip("/")

    def _join_name(self, prefix: str, key: Any) -> str:
        p = self._norm_prefix(prefix)
        k = self._norm_key(key)
        return f"{p}{k}" if p else k

    def set_key_every(self, mapping: Mapping[str, int]) -> None:
        """
        Set per-key throttling: log a key only every N steps.

        Parameters
        ----------
        mapping : Mapping[str, int]
            full_key -> every_n_steps.
            If value <= 0, throttling for that key is removed.
        """
        for k, v in mapping.items():
            kk = self._norm_key(k)
            vv = int(v)
            if vv <= 0:
                self._key_every.pop(kk, None)
            else:
                self._key_every[kk] = vv

    def _should_log_key(self, full_key: str, step: int) -> bool:
        every = self._key_every.get(full_key, None)
        if every is None or every <= 0:
            return True
        return (int(step) % int(every)) == 0

    # ---------------------------------------------------------------------
    # Public logging APIs
    # ---------------------------------------------------------------------
    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None, *, pbar: Optional[Any] = None,  prefix: str = "") -> None:
        """
        Immediately write metrics to writers (and optionally console).

        Parameters
        ----------
        metrics : Mapping[str, Any]
            Key-value mapping. Values are converted via `to_scalar`.
        step : Optional[int]
            If provided, overrides inferred step.
        prefix : str
            Optional prefix for all keys (e.g., "train", "eval").
        """
        s = self._infer_step(step)
        self._log_calls += 1

        row: Dict[str, float] = {}

        for k, v in metrics.items():
            name = self._join_name(prefix, k)
            val = to_scalar(v)
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

        # meta
        row["step"] = float(int(s))
        row["wall_time"] = float(time.time() - self._start_time)
        row["timestamp"] = float(time.time())

        # writers
        for w in self._writers:
            try:
                w.write(row)
            except Exception as e:
                self._handle_exception(e, f"writer.write({w.__class__.__name__})")

        # console
        if self.console_every > 0 and (self._log_calls % self.console_every == 0):
            self._print_console(row, pbar=pbar)

        # periodic flush
        if self.flush_every > 0 and (self._log_calls % self.flush_every == 0):
            self.flush()

    def record(self, metrics: Mapping[str, Any], *, prefix: str = "") -> None:
        """
        Record metrics into an in-memory buffer for later aggregation.

        Use `dump()` or `dump_stats()` to write aggregated scalars.
        """
        for k, v in metrics.items():
            name = self._join_name(prefix, k)
            val = to_scalar(v)
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
        Aggregate buffered scalars and write them via `log()`.

        Parameters
        ----------
        step : Optional[int]
            Step override.
        prefix : str
            Optional prefix applied to ALL aggregated keys at output time.
        agg : str
            Aggregation operator in {"mean","min","max","std"}.
        clear : bool
            If True, clears the buffer after dumping.
        """
        agg = str(agg).lower().strip()
        if agg not in ("mean", "min", "max", "std"):
            raise ValueError(f"Unknown agg={agg!r}. Use mean|min|max|std.")

        out: Dict[str, float] = {}
        for k, vals in self._buffer.items():
            if not vals:
                continue
            a = np.asarray(vals, dtype=np.float64)
            if agg == "mean":
                out[k] = float(np.mean(a))
            elif agg == "min":
                out[k] = float(np.min(a))
            elif agg == "max":
                out[k] = float(np.max(a))
            else:
                out[k] = float(np.std(a))

        if clear:
            self._buffer.clear()

        if not out:
            return

        # Apply output prefix in a predictable way
        if prefix:
            out = {self._join_name(prefix, k): v for k, v in out.items()}

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
        Dump multiple statistics per buffered key (e.g., mean/min/max/std).

        Parameters
        ----------
        step : Optional[int]
            Step override.
        prefix : str
            Optional prefix applied to output keys.
        clear : bool
            If True, clears buffer after dumping.
        suffixes : Tuple[str, ...]
            Subset of {"mean","min","max","std"}.
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
        Dump experiment config as JSON into run_dir.

        Parameters
        ----------
        config : Mapping[str, Any]
            Configuration mapping.
        filename : str
            Output filename under run_dir.
        """
        path = os.path.join(self.run_dir, filename)
        try:
            payload = dict(config)  # shallow copy
        except Exception:
            payload = {str(k): v for k, v in config.items()}  # type: ignore[attr-defined]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def dump_metadata(self, filename: str = "metadata.json") -> None:
        """
        Dump runtime/environment metadata into run_dir.

        Includes:
        - host/pid/python/platform
        - torch/cuda info (best-effort)
        - git info (best-effort)
        """
        meta: Dict[str, Any] = {}
        meta["run_dir"] = self.run_dir
        meta["start_time_unix"] = float(self._start_time)
        meta["start_time_iso"] = datetime.fromtimestamp(self._start_time).isoformat()

        meta["host"] = socket.gethostname()
        meta["fqdn"] = socket.getfqdn()
        meta["pid"] = os.getpid()
        meta["python"] = sys.version.replace("\n", " ")
        meta["platform"] = sys.platform

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
            Keys may include: commit, branch, dirty.
            Returns empty dict when not in a git repo or git not available.
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
    # Flush / close
    # ---------------------------------------------------------------------
    def flush(self) -> None:
        """Flush all writers (best-effort unless strict=True)."""
        for w in self._writers:
            try:
                w.flush()
            except Exception as e:
                self._handle_exception(e, f"writer.flush({w.__class__.__name__})")

    def close(self) -> None:
        """Flush and close all writers (best-effort unless strict=True)."""
        try:
            self.flush()
        finally:
            for w in self._writers:
                try:
                    w.close()
                except Exception as e:
                    self._handle_exception(e, f"writer.close({w.__class__.__name__})")

    # ---------------------------------------------------------------------
    # Console
    # ---------------------------------------------------------------------
    @staticmethod
    def _print_console(row: Mapping[str, float], *, pbar: Optional[Any] = None) -> None:
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

        # ---- Update the fixed message line (no new lines) ----
        if pbar is not None:
            try:
                pbar.set_description_str(msg, refresh=True)
                return
            except Exception:
                pass

        # fallback
        print(msg)