from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import time

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, infer_step
from ..utils.log_utils import log


@dataclass
class GapAverager:
    """
    Estimate the mean wall-time gap between successive callback invocations (in ms).

    This is a coarse proxy for how often a callback hook is being called and can help
    detect performance regressions (e.g., slower env stepping or slower updates).

    Outlier filtering
    -----------------
    - Gaps < 0 are ignored (should not happen, but defensive).
    - Gaps > max_gap_sec are ignored (likely due to pauses, debugging, suspension).
    """
    max_gap_sec: float = 600.0
    _last_t: float = 0.0
    _acc_ms: float = 0.0
    _n: int = 0

    def reset(self, now: Optional[float] = None) -> None:
        """
        Initialize internal state.

        Parameters
        ----------
        now : Optional[float]
            If provided, use this as current wall time; otherwise use time.time().
        """
        t = time.time() if now is None else float(now)
        self._last_t = t
        self._acc_ms = 0.0
        self._n = 0

    def tick(self, now: Optional[float] = None) -> None:
        """
        Consume one "tick" (one callback invocation) and accumulate a filtered gap.

        Parameters
        ----------
        now : Optional[float]
            If provided, use this as current wall time; otherwise use time.time().
        """
        t = time.time() if now is None else float(now)
        gap = t - self._last_t
        self._last_t = t

        # Only accumulate reasonable gaps to avoid skew from long pauses.
        if 0.0 <= gap <= self.max_gap_sec:
            self._acc_ms += gap * 1000.0
            self._n += 1

    def mean_ms(self) -> Optional[float]:
        """
        Return the mean gap in milliseconds over accumulated samples.

        Returns
        -------
        Optional[float]
            None if no valid samples have been accumulated yet.
        """
        if self._n <= 0:
            return None
        return float(self._acc_ms / float(self._n))


@dataclass
class ThroughputMeter:
    """
    Track average and delta throughput of a monotonically increasing counter.

    Typical counters
    ----------------
    - env steps (global_env_step)
    - update steps (global_update_step)

    Definitions
    -----------
    - avg_per_sec: count / (now - t0) since last reset
    - delta_per_sec: (count - last_count) / (now - last_t) since last commit

    Notes
    -----
    - This assumes `count` is monotonically non-decreasing.
    - delta_per_sec returns None if there was no progress or time delta is too small.
    """
    _t0: float = 0.0
    _last_count: int = 0
    _last_t: float = 0.0

    def reset(self, *, count: int, now: Optional[float] = None) -> None:
        """
        Reset the meter baseline.

        Parameters
        ----------
        count : int
            Current counter value at reset time.
        now : Optional[float]
            Current wall time; defaults to time.time() if omitted.
        """
        t = time.time() if now is None else float(now)
        self._t0 = t
        self._last_count = int(count)
        self._last_t = t

    def avg_per_sec(self, *, count: int, now: Optional[float] = None) -> float:
        """
        Average throughput since last reset.

        Returns
        -------
        float
            count / elapsed_seconds (elapsed is clamped to avoid divide-by-zero).
        """
        t = time.time() if now is None else float(now)
        wall = max(1e-9, t - self._t0)
        return float(count) / wall

    def delta_per_sec(self, *, count: int, now: Optional[float] = None) -> Optional[float]:
        """
        Delta throughput since last commit.

        Returns
        -------
        Optional[float]
            (delta_count / delta_time) if progress was positive and dt is valid;
            otherwise None.
        """
        t = time.time() if now is None else float(now)
        d_count = int(count) - int(self._last_count)
        d_t = float(t - self._last_t)

        # Only report meaningful deltas.
        if d_count > 0 and d_t > 1e-9:
            return float(d_count) / d_t
        return None

    def commit(self, *, count: int, now: Optional[float] = None) -> None:
        """
        Commit the current counter/time as the new delta baseline.
        """
        t = time.time() if now is None else float(now)
        self._last_count = int(count)
        self._last_t = t


@dataclass
class DeltaGate:
    """
    A "delta" gate: triggers when (current - last) >= every.

    Why delta gating?
    -----------------
    This gate is robust to:
      - counter jumps (e.g., step increments in batches)
      - irregular callback invocation frequency

    Usage pattern
    -------------
    - reset(current=...)
    - if ready(current=...): do work, then commit(current=...)
    """
    every: int
    _last: int = 0
    _inited: bool = False

    def reset(self, *, current: int) -> None:
        """Initialize the gate at the current counter value."""
        self._last = int(current)
        self._inited = True

    def ready(self, *, current: int) -> bool:
        """
        Return True when enough progress has accumulated since last commit.

        Notes
        -----
        - On first call (not initialized), it initializes and returns False.
        - If every <= 0, it never triggers.
        """
        if self.every <= 0:
            return False

        cur = int(current)
        if not self._inited:
            self.reset(current=cur)
            return False

        return (cur - self._last) >= self.every

    def commit(self, *, current: int) -> None:
        """Commit the current counter value as the new baseline."""
        self._last = int(current)
        self._inited = True


class TimingCallback(BaseCallback):
    """
    Log coarse throughput and timing signals for regression / bottleneck detection.

    What it measures (best-effort)
    ------------------------------
    - env_steps_per_sec_avg      : average env-step throughput since train start/reset
    - env_steps_per_sec_delta    : env-step throughput since last log (if meaningful)
    - updates_per_sec_avg        : average update throughput since train start/reset
    - updates_per_sec_delta      : update throughput since last log (if meaningful)
    - step_time_ms_mean          : mean wall gap between on_step calls (approx)
    - update_time_ms_mean        : mean wall gap between on_update calls (approx)

    Notes / limitations
    -------------------
    - This does NOT precisely time env.step() or optimizer computation.
      It uses callback-to-callback wall gaps, which is usually enough to catch
      regressions (e.g., slowdown due to logging, environment, I/O).
    - Counters are read via `safe_int_attr(trainer)`, so this assumes the trainer
      exposes meaningful step/update counters through that helper.
    """

    def __init__(
        self,
        *,
        log_every_steps: int = 5_000,
        log_every_updates: int = 200,
        log_prefix: str = "perf/",
        max_gap_sec: float = 600.0,
    ):
        # Prefix for all emitted metrics.
        self.log_prefix = str(log_prefix)

        # Independent delta gates for step-logging and update-logging.
        self._step_gate = DeltaGate(every=int(log_every_steps))
        self._upd_gate = DeltaGate(every=int(log_every_updates))

        # Gap averagers track callback-to-callback wall-time gaps.
        self._step_gap = GapAverager(max_gap_sec=float(max_gap_sec))
        self._upd_gap = GapAverager(max_gap_sec=float(max_gap_sec))

        # Throughput meters track counter throughput (avg + delta).
        self._step_tp = ThroughputMeter()
        self._upd_tp = ThroughputMeter()

    def on_train_start(self, trainer: Any) -> bool:
        """
        Initialize all gates and meters at the current trainer counters.

        This supports resuming:
        - If the trainer resumes from a checkpoint with non-zero counters,
          baselines are aligned to current counts and current wall time.
        """
        now = time.time()

        # NOTE: These readouts currently use the same helper for both.
        # This assumes safe_int_attr(trainer) returns the relevant counter in each context.
        # If your trainer separates env_step vs update_step, wire those explicitly.
        step = max(0, int(safe_int_attr(trainer)))
        upd = max(0, int(safe_int_attr(trainer)))

        # Reset gates at current counters so the next log triggers after "every" progress.
        self._step_gate.reset(current=step)
        self._upd_gate.reset(current=upd)

        # Reset throughput meters baseline.
        self._step_tp.reset(count=step, now=now)
        self._upd_tp.reset(count=upd, now=now)

        # Reset gap averagers baseline.
        self._step_gap.reset(now=now)
        self._upd_gap.reset(now=now)
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Step hook: periodically log env-step throughput and coarse step timing.

        - Ticks step gap averager each invocation.
        - Uses DeltaGate so logging is robust to step jumps.
        """
        now = time.time()
        self._step_gap.tick(now=now)

        step = int(safe_int_attr(trainer))
        if step <= 0:
            return True

        # Only log when enough env-step progress occurred since last log.
        if not self._step_gate.ready(current=step):
            return True

        payload: Dict[str, Any] = {
            # Average env-step throughput since reset.
            "env_steps_per_sec_avg": self._step_tp.avg_per_sec(count=step, now=now),
        }

        # Delta throughput since last commit (if meaningful).
        d = self._step_tp.delta_per_sec(count=step, now=now)
        if d is not None:
            payload["env_steps_per_sec_delta"] = d

        # Optional cross-signal: update throughput average.
        # Useful to see whether updates keep up with environment collection.
        upd = int(safe_int_attr(trainer))
        if upd > 0:
            payload["updates_per_sec_avg"] = self._upd_tp.avg_per_sec(count=upd, now=now)

        # Mean callback-to-callback gap for the on_step hook (coarse).
        m = self._step_gap.mean_ms()
        if m is not None:
            payload["step_time_ms_mean"] = m

        # Use trainer-derived step for logging alignment.
        log(trainer, payload, step=infer_step(trainer), prefix=self.log_prefix)

        # Commit baselines for the next delta window.
        self._step_gate.commit(current=step)
        self._step_tp.commit(count=step, now=now)
        return True

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update hook: periodically log update throughput and coarse update timing.

        - Ticks update gap averager each invocation.
        - Uses DeltaGate so logging is robust to update jumps.
        """
        now = time.time()
        self._upd_gap.tick(now=now)

        upd = int(safe_int_attr(trainer))
        if upd <= 0:
            return True

        # Only log when enough update progress occurred since last log.
        if not self._upd_gate.ready(current=upd):
            return True

        payload: Dict[str, Any] = {
            # Average update throughput since reset.
            "updates_per_sec_avg": self._upd_tp.avg_per_sec(count=upd, now=now),
        }

        # Delta throughput since last commit (if meaningful).
        d = self._upd_tp.delta_per_sec(count=upd, now=now)
        if d is not None:
            payload["updates_per_sec_delta"] = d

        # Optional cross-signal: env-step throughput average.
        step = int(safe_int_attr(trainer))
        if step > 0:
            payload["env_steps_per_sec_avg"] = self._step_tp.avg_per_sec(count=step, now=now)

        # Mean callback-to-callback gap for the on_update hook (coarse).
        m = self._upd_gap.mean_ms()
        if m is not None:
            payload["update_time_ms_mean"] = m

        log(trainer, payload, step=infer_step(trainer), prefix=self.log_prefix)

        # Commit baselines for the next delta window.
        self._upd_gate.commit(current=upd)
        self._upd_tp.commit(count=upd, now=now)
        return True