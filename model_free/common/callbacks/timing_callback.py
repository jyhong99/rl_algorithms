from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import time

from .base_callback import BaseCallback
from ..utils.callback_utils import _safe_env_step, _safe_update_step


# =============================================================================
# Small utility components
# =============================================================================
@dataclass
class GapAverager:
    """
    Estimate the mean wall-time gap between successive invocations (milliseconds).

    This is a coarse proxy for "how frequently a hook is called" and is useful for
    spotting regressions such as:
      - slower environment stepping
      - slower update loop execution
      - heavier callback/logging overhead

    The averager is *best-effort* and intentionally robust against long pauses
    (debugger stops, machine sleep, queue stalls, etc.) via outlier filtering.

    Parameters
    ----------
    max_gap_sec : float, default=600.0
        Maximum wall-time gap (seconds) that is considered a "valid" sample.
        Gaps larger than this are ignored to avoid skewing the mean due to pauses.

    Notes
    -----
    - Gaps < 0 are ignored defensively (should not happen in normal operation).
    - This measures wall-time between callback invocations, not the runtime of
      env.step() or optimizer.step() directly.
    """

    max_gap_sec: float = 600.0
    _last_t: float = 0.0
    _acc_ms: float = 0.0
    _n: int = 0

    def reset(self, now: Optional[float] = None) -> None:
        """
        Reset internal state and start a new measurement window.

        Parameters
        ----------
        now : Optional[float], default=None
            Wall time (seconds) to use as the baseline. If None, uses ``time.time()``.
        """
        t = time.time() if now is None else float(now)
        self._last_t = t
        self._acc_ms = 0.0
        self._n = 0

    def tick(self, now: Optional[float] = None) -> None:
        """
        Record one invocation and accumulate a filtered wall-time gap.

        Parameters
        ----------
        now : Optional[float], default=None
            Current wall time (seconds). If None, uses ``time.time()``.
        """
        t = time.time() if now is None else float(now)
        gap = t - self._last_t
        self._last_t = t

        # Accumulate only "reasonable" gaps to avoid skew from long pauses.
        if 0.0 <= gap <= self.max_gap_sec:
            self._acc_ms += gap * 1000.0
            self._n += 1

    def mean_ms(self) -> Optional[float]:
        """
        Return the mean invocation gap in milliseconds for the current window.

        Returns
        -------
        Optional[float]
            Mean gap in milliseconds, or None if no valid samples exist yet.
        """
        if self._n <= 0:
            return None
        return float(self._acc_ms / float(self._n))


@dataclass
class ThroughputMeter:
    """
    Track average and delta throughput of a monotonically increasing counter.

    This meter is used for counters such as:
      - environment steps (global env step)
      - update steps (global update step)

    Given a counter value ``count`` and wall time ``now``:

    - Average throughput since last reset:
        avg_per_sec = count / (now - t0)

    - Delta throughput since last commit:
        delta_per_sec = (count - last_count) / (now - last_t)

    Notes
    -----
    - Assumes ``count`` is monotonically non-decreasing.
    - ``delta_per_sec`` returns None if there was no progress (d_count <= 0) or
      the time delta is too small.
    """

    _t0: float = 0.0
    _last_count: int = 0
    _last_t: float = 0.0

    def reset(self, *, count: int, now: Optional[float] = None) -> None:
        """
        Reset the baseline for average and delta throughput.

        Parameters
        ----------
        count : int
            Current counter value at reset time.
        now : Optional[float], default=None
            Current wall time (seconds). If None, uses ``time.time()``.
        """
        t = time.time() if now is None else float(now)
        self._t0 = t
        self._last_count = int(count)
        self._last_t = t

    def avg_per_sec(self, *, count: int, now: Optional[float] = None) -> float:
        """
        Compute average throughput since last reset.

        Parameters
        ----------
        count : int
            Current counter value.
        now : Optional[float], default=None
            Current wall time (seconds). If None, uses ``time.time()``.

        Returns
        -------
        float
            Average throughput in counts/sec. Elapsed time is clamped to avoid
            division by zero.
        """
        t = time.time() if now is None else float(now)
        wall = max(1e-9, t - self._t0)
        return float(count) / wall

    def delta_per_sec(self, *, count: int, now: Optional[float] = None) -> Optional[float]:
        """
        Compute delta throughput since last commit.

        Parameters
        ----------
        count : int
            Current counter value.
        now : Optional[float], default=None
            Current wall time (seconds). If None, uses ``time.time()``.

        Returns
        -------
        Optional[float]
            Delta throughput in counts/sec if progress is positive and dt is valid;
            otherwise None.
        """
        t = time.time() if now is None else float(now)
        d_count = int(count) - int(self._last_count)
        d_t = float(t - self._last_t)

        if d_count > 0 and d_t > 1e-9:
            return float(d_count) / d_t
        return None

    def commit(self, *, count: int, now: Optional[float] = None) -> None:
        """
        Commit the current counter/time as the new delta baseline.

        Parameters
        ----------
        count : int
            Counter value to commit.
        now : Optional[float], default=None
            Current wall time (seconds). If None, uses ``time.time()``.
        """
        t = time.time() if now is None else float(now)
        self._last_count = int(count)
        self._last_t = t


@dataclass
class DeltaGate:
    """
    Delta-based trigger gate: fires when (current - last_committed) >= every.

    This is robust to:
      - counter jumps (e.g., env steps updated in batches)
      - irregular callback invocation frequencies

    Parameters
    ----------
    every : int
        Trigger threshold in counter units. If <= 0, the gate never triggers.

    Notes
    -----
    Usage pattern:
      - reset(current=...)
      - if ready(current=...): do work; commit(current=...)
    """

    every: int
    _last: int = 0
    _inited: bool = False

    def reset(self, *, current: int) -> None:
        """
        Initialize the baseline at the given counter value.

        Parameters
        ----------
        current : int
            Counter value used as the initial baseline.
        """
        self._last = int(current)
        self._inited = True

    def ready(self, *, current: int) -> bool:
        """
        Check whether the gate condition is satisfied.

        Parameters
        ----------
        current : int
            Current counter value.

        Returns
        -------
        bool
            True if (current - last_committed) >= every, else False.

        Notes
        -----
        - On the first call (not initialized), this initializes the gate and returns False.
        - If ``every <= 0``, it never triggers.
        """
        if self.every <= 0:
            return False

        cur = int(current)
        if not self._inited:
            self.reset(current=cur)
            return False

        return (cur - self._last) >= self.every

    def commit(self, *, current: int) -> None:
        """
        Commit the current counter value as the new baseline.

        Parameters
        ----------
        current : int
            Counter value to store as the baseline.
        """
        self._last = int(current)
        self._inited = True


# =============================================================================
# Callback: Timing / Throughput logging
# =============================================================================
class TimingCallback(BaseCallback):
    """
    Log coarse throughput and timing signals for regression/bottleneck detection.

    This callback does not precisely time ``env.step()`` or optimizer execution.
    Instead, it logs *coarse* wall-clock and throughput signals that are usually
    sufficient to detect:
      - performance regressions (slower steps/updates)
      - sudden slowdowns due to I/O or logging
      - mismatched collection/update ratios

    Reported metrics (best-effort)
    ------------------------------
    Step-side (on_step, logged periodically by env-step progress):
      - env_steps_per_sec_avg
          Average environment-step throughput since last reset.
      - env_steps_per_sec_delta
          Environment-step throughput since last commit (may be absent if no progress).
      - updates_per_sec_avg
          Average update throughput (cross-signal) if update counter is available.
      - step_time_ms_mean
          Mean wall-time gap between on_step invocations (coarse, filtered).

    Update-side (on_update, logged periodically by update-step progress):
      - updates_per_sec_avg
          Average update throughput since last reset.
      - updates_per_sec_delta
          Update throughput since last commit (may be absent if no progress).
      - env_steps_per_sec_avg
          Average env-step throughput (cross-signal) if step counter is available.
      - update_time_ms_mean
          Mean wall-time gap between on_update invocations (coarse, filtered).

    Parameters
    ----------
    log_every_steps : int, default=5_000
        Log step-side metrics after this many *environment steps* have progressed
        since the previous step-side log. If <= 0, step-side logging never triggers.
    log_every_updates : int, default=200
        Log update-side metrics after this many *update steps* have progressed
        since the previous update-side log. If <= 0, update-side logging never triggers.
    log_prefix : str, default="perf/"
        Prefix used for all emitted keys via ``BaseCallback.log(..., prefix=...)``.
    max_gap_sec : float, default=600.0
        Maximum gap (seconds) considered valid when computing mean wall gaps.

    Trainer contract (duck-typed)
    -----------------------------
    Expected step counters are read using:
      - ``safe_env_step(trainer)``   -> env step counter (int-like)
      - ``safe_update_step(trainer)``-> update step counter (int-like)

    Notes
    -----
    - This callback uses delta gating (DeltaGate) instead of mod gating to be robust
      to counter jumps and irregular invocation frequency.
    - Logging steps:
        * on_step logs with step=env_step
        * on_update logs with step=update_step
      This makes time series alignment explicit in dashboards.
    """

    def __init__(
        self,
        *,
        log_every_steps: int = 5_000,
        log_every_updates: int = 200,
        log_prefix: str = "perf/",
        max_gap_sec: float = 600.0,
    ) -> None:
        self.log_prefix = str(log_prefix)

        # Delta gates trigger when enough *progress* has accumulated.
        self._step_gate = DeltaGate(every=int(log_every_steps))
        self._upd_gate = DeltaGate(every=int(log_every_updates))

        # Coarse wall-time gap trackers for hook invocation frequency.
        self._step_gap = GapAverager(max_gap_sec=float(max_gap_sec))
        self._upd_gap = GapAverager(max_gap_sec=float(max_gap_sec))

        # Throughput meters for step/update counters.
        self._step_tp = ThroughputMeter()
        self._upd_tp = ThroughputMeter()

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def on_train_start(self, trainer: Any) -> bool:
        """
        Initialize gates/meters using the trainer's current counters.

        This supports resuming runs:
        - If the trainer resumes from a checkpoint with non-zero counters,
          baselines will align to the current counter values and current wall time.
        """
        now = time.time()

        step = max(0, int(_safe_env_step(trainer)))
        upd = max(0, int(_safe_update_step(trainer)))

        # Gate baselines: next trigger occurs after `every` additional progress.
        self._step_gate.reset(current=step)
        self._upd_gate.reset(current=upd)

        # Throughput baselines.
        self._step_tp.reset(count=step, now=now)
        self._upd_tp.reset(count=upd, now=now)

        # Gap baselines.
        self._step_gap.reset(now=now)
        self._upd_gap.reset(now=now)
        return True

    # =========================================================================
    # Hooks
    # =========================================================================
    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Step hook: periodically log env-step throughput and coarse step timing.

        Notes
        -----
        - Each invocation ticks the wall-gap tracker.
        - Logging is triggered by env-step progress via DeltaGate (robust to step jumps).
        """
        now = time.time()
        self._step_gap.tick(now=now)

        step = int(_safe_env_step(trainer))
        if step <= 0:
            return True

        if not self._step_gate.ready(current=step):
            return True

        payload: Dict[str, Any] = {
            "env_steps_per_sec_avg": self._step_tp.avg_per_sec(count=step, now=now),
        }

        d = self._step_tp.delta_per_sec(count=step, now=now)
        if d is not None:
            payload["env_steps_per_sec_delta"] = d

        # Cross-signal: update throughput average (if update counter is available).
        upd = int(_safe_update_step(trainer))
        if upd > 0:
            payload["updates_per_sec_avg"] = self._upd_tp.avg_per_sec(count=upd, now=now)

        m = self._step_gap.mean_ms()
        if m is not None:
            payload["step_time_ms_mean"] = m

        # Log using env-step as x-axis for step-side metrics.
        self.log(trainer, payload, step=step, prefix=self.log_prefix)

        # Commit baselines for next delta window.
        self._step_gate.commit(current=step)
        self._step_tp.commit(count=step, now=now)
        return True

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update hook: periodically log update throughput and coarse update timing.

        Notes
        -----
        - Each invocation ticks the wall-gap tracker.
        - Logging is triggered by update-step progress via DeltaGate (robust to jumps).
        """
        now = time.time()
        self._upd_gap.tick(now=now)

        upd = int(_safe_update_step(trainer))
        if upd <= 0:
            return True

        if not self._upd_gate.ready(current=upd):
            return True

        payload: Dict[str, Any] = {
            "updates_per_sec_avg": self._upd_tp.avg_per_sec(count=upd, now=now),
        }

        d = self._upd_tp.delta_per_sec(count=upd, now=now)
        if d is not None:
            payload["updates_per_sec_delta"] = d

        # Cross-signal: env-step throughput average (if available).
        step = int(_safe_env_step(trainer))
        if step > 0:
            payload["env_steps_per_sec_avg"] = self._step_tp.avg_per_sec(count=step, now=now)

        m = self._upd_gap.mean_ms()
        if m is not None:
            payload["update_time_ms_mean"] = m

        # Log using update-step as x-axis for update-side metrics.
        self.log(trainer, payload, step=upd, prefix=self.log_prefix)

        # Commit baselines for next delta window.
        self._upd_gate.commit(current=upd)
        self._upd_tp.commit(count=upd, now=now)
        return True
