from __future__ import annotations

from typing import Any, Dict, Optional, Deque, Sequence, List, Mapping
from collections import deque
import math

from .base_callback import BaseCallback
from ..utils.log_utils import log
from ..utils.callback_utils import infer_step


class EpisodeStatsCallback(BaseCallback):
    """
    Episode-level statistics from step transitions (framework-agnostic).

    This callback consumes per-step `transition` payloads (best-effort) and logs
    episode metrics such as return, length, and truncation rate.

    Key capability
    --------------
    - Single-env (B=1): accurately reconstructs episodic return/length by accumulating
      reward until `done=True`.
    - Batched env (B>1): without per-env episode summaries (like per-env return/len),
      accurate reconstruction is generally impossible. In that case, this callback
      logs only conservative batch-level completion/truncation signals.

    Logged metrics
    --------------
    Single-env (accurate):
      - (optional raw) episode/return, episode/len, episode/truncated
      - rolling window aggregates:
          return_mean, return_std, len_mean, trunc_rate

    Batched env limitation (conservative):
      - episode/batched_done_count       : number of envs finishing at the step
      - episode/batched_trunc_rate       : truncation rate among finished envs only

    Notes
    -----
    - This callback intentionally avoids numpy dependency and implements small math
      utilities in pure Python.
    - Truncation is inferred best-effort from typical Gym/Gymnasium info keys.
    """

    # =========================================================================
    # Internal: normalized transition representation
    # =========================================================================
    class _NormalizedTransition:
        """
        Normalized step transition representation.

        The goal is to map heterogeneous trainer/env step payloads into one
        consistent interface.

        Fields
        ------
        rewards   : List[float]  (len=B)
        dones     : List[bool]   (len=B)   (done = terminated OR truncated)
        truncated : List[bool]   (len=B)   best-effort inference
        infos     : List[Any]    (len=B)   raw info payloads (if provided)
        is_batched: bool         indicates the original payload appeared batched

        Single-env example:
          rewards=[r], dones=[done], truncated=[trunc], infos=[info]

        Batched env example:
          rewards=[r0, r1, ...], dones=[d0, d1, ...], truncated=[t0, t1, ...]
        """

        def __init__(
            self,
            *,
            rewards: List[float],
            dones: List[bool],
            truncated: List[bool],
            infos: List[Any],
            is_batched: bool,
        ) -> None:
            self.rewards = rewards
            self.dones = dones
            self.truncated = truncated
            self.infos = infos
            self.is_batched = is_batched

    def __init__(
        self,
        *,
        window: int = 100,
        log_every_episodes: int = 10,
        log_prefix: str = "rollout/",
        log_raw_episode: bool = False,
    ) -> None:
        # Rolling window size used for mean/std aggregation.
        self.window = int(window)

        # Aggregate metrics are logged every N episodes (0 disables aggregate logging).
        self.log_every_episodes = int(log_every_episodes)

        # Prefix for all logged keys to keep namespaces clean.
        self.log_prefix = str(log_prefix)

        # If True, log per-episode raw values for each finished episode.
        # (Can be noisy but useful for debugging.)
        self.log_raw_episode = bool(log_raw_episode)

        # ---------------------------------------------------------------------
        # Single-env episodic accumulators (only valid/used in single-env path).
        # ---------------------------------------------------------------------
        self._ep_return: float = 0.0
        self._ep_len: int = 0
        self._ep_count: int = 0

        # Rolling buffers for aggregates. maxlen=None => unbounded (not recommended).
        maxlen = self.window if self.window > 0 else None
        self._returns: Deque[float] = deque(maxlen=maxlen)
        self._lengths: Deque[int] = deque(maxlen=maxlen)
        self._trunc_flags: Deque[int] = deque(maxlen=maxlen)  # 1 if truncated else 0

    # =========================================================================
    # Small utility helpers (avoid numpy dependency)
    # =========================================================================
    @staticmethod
    def _is_sequence(x: Any) -> bool:
        """Return True if x looks like a list/tuple."""
        return isinstance(x, (list, tuple))

    @staticmethod
    def _to_float(x: Any, default: float = 0.0) -> float:
        """Best-effort float conversion with fallback."""
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _to_bool(x: Any, default: bool = False) -> bool:
        """Best-effort bool conversion with fallback."""
        try:
            return bool(x)
        except Exception:
            return default

    @staticmethod
    def _mean(xs: Sequence[float]) -> float:
        """Safe mean for empty sequences."""
        if not xs:
            return 0.0
        return float(sum(xs) / len(xs))

    @staticmethod
    def _std(xs: Sequence[float]) -> float:
        """
        Population standard deviation (ddof=0).

        Notes
        -----
        - For small windows, population std is a stable choice.
        - Returns 0.0 for n<=1 to avoid divide-by-zero.
        """
        n = len(xs)
        if n <= 1:
            return 0.0
        m = sum(xs) / n
        var = sum((x - m) * (x - m) for x in xs) / n
        return float(math.sqrt(var))

    # =========================================================================
    # Truncation inference from info payload
    # =========================================================================
    def _infer_truncated_from_info(self, info: Any) -> bool:
        """
        Best-effort truncation detection from info-like objects.

        Gym/Gymnasium conventions:
        - Gym: may store "TimeLimit.truncated" in info.
        - Some wrappers: may store "truncated" boolean.

        If info is not dict-like or missing expected keys, returns False.
        """
        try:
            if info is None:
                return False
            d = dict(info)  # may raise if info isn't dict-like
            if self._to_bool(d.get("TimeLimit.truncated", False)):
                return True
            if self._to_bool(d.get("truncated", False)):
                return True
            return False
        except Exception:
            return False

    # =========================================================================
    # Transition normalization
    # =========================================================================
    def _ensure_list(self, x: Any, *, n: int, default: Any) -> List[Any]:
        """
        Broadcast or coerce `x` into a list of length n.

        Rules
        -----
        - If x is a list/tuple:
            * If len == n: return as list
            * If len < n: pad with `default`
            * If len > n: truncate to n
        - Else:
            * Broadcast scalar to length n (using default for None)

        This lets us align fields (terminated/truncated/infos) to B = batch size.
        """
        if self._is_sequence(x):
            xs = list(x)
            if len(xs) == n:
                return xs
            if len(xs) < n:
                xs = xs + [default] * (n - len(xs))
            return xs[:n]
        return [x if x is not None else default] * n

    def _normalize_transition(self, transition: Mapping[str, Any]) -> Optional[_NormalizedTransition]:
        """
        Normalize heterogeneous trainer transition payloads.

        Supported input patterns
        ------------------------
        Single-env:
          - {"reward": r, "done": done, "info": info}
          - {"reward": r, "terminated": term, "truncated": trunc, "info": info}

        Batched env:
          - {"rewards": [...], "dones": [...], "infos": [...]}
          - {"rewards": [...], "terminated": [...], "truncated": [...], "infos": [...]}
          - Mixed cases with scalar reward but vector info/infos (best-effort)

        Output
        ------
        Returns a _NormalizedTransition with aligned lists of length B.

        Returns None if transition is empty/unusable.
        """
        if not transition:
            return None

        # Common plural keys (vectorized envs)
        rewards = transition.get("rewards", None)
        dones = transition.get("dones", None)
        infos = transition.get("infos", None)

        # Gymnasium style (terminated/truncated)
        terminated = transition.get("terminated", None)
        truncated = transition.get("truncated", None)

        # Decide "batched" based on whether key fields are sequences.
        is_batched = (
            self._is_sequence(rewards)
            or self._is_sequence(dones)
            or self._is_sequence(terminated)
            or self._is_sequence(truncated)
        )

        # --- rewards ---
        # Prefer "rewards" list if present; else fall back to scalar "reward".
        if self._is_sequence(rewards):
            r_list = [self._to_float(x, 0.0) for x in rewards]
        else:
            r_list = [self._to_float(transition.get("reward", 0.0), 0.0)]

        # Batch size B is determined by rewards list length (primary anchor).
        B = len(r_list)

        # --- dones + truncated ---
        # Priority:
        # 1) explicit dones list (already combined done flags)
        # 2) terminated/truncated lists combined (done = terminated OR truncated)
        # 3) scalar done or scalar terminated|truncated broadcast to B
        if self._is_sequence(dones):
            d_list = [self._to_bool(x) for x in dones]
            # Start truncated flags as False; refine later using "truncated"/info.
            trunc_list = [False] * len(d_list)
        else:
            if self._is_sequence(terminated) or self._is_sequence(truncated):
                term_list = self._ensure_list(terminated, n=B, default=False)
                tru_list = self._ensure_list(truncated, n=B, default=False)
                term_b = [self._to_bool(x) for x in term_list]
                tru_b = [self._to_bool(x) for x in tru_list]
                d_list = [t or tr for t, tr in zip(term_b, tru_b)]
                trunc_list = list(tru_b)
            else:
                # Scalar fallback: use "done" if available, else combine scalar terminated/truncated.
                done_single = transition.get("done", None)
                if done_single is None and (terminated is not None or truncated is not None):
                    done_single = self._to_bool(terminated) or self._to_bool(truncated)

                d_list = [self._to_bool(done_single, default=False)] * B
                trunc_list = (
                    [self._to_bool(truncated, default=False)] * B
                    if truncated is not None
                    else [False] * B
                )

        # --- infos ---
        # Prefer plural infos; else fall back to scalar info, broadcast to B.
        if infos is None:
            info_alt = transition.get("info", None)
            if self._is_sequence(info_alt):
                infos_list = list(info_alt)
            else:
                infos_list = [info_alt] * B
        else:
            infos_list = self._ensure_list(infos, n=B, default=None)

        # Refine truncation flags using info fields (TimeLimit.truncated, etc.).
        trunc_from_info = [self._infer_truncated_from_info(infos_list[i]) for i in range(B)]
        trunc_list = [bool(trunc_list[i]) or bool(trunc_from_info[i]) for i in range(B)]

        return self._NormalizedTransition(
            rewards=r_list,
            dones=d_list,
            truncated=trunc_list,
            infos=infos_list,
            is_batched=is_batched,
        )

    # =========================================================================
    # Episode aggregation + logging (single-env accurate path)
    # =========================================================================
    def _record_episode(self, trainer: Any, ep_return: float, ep_len: int, truncated: bool) -> None:
        """
        Commit a finished episode into rolling buffers and emit logs (raw + aggregates).

        This is called only when we can accurately reconstruct episode boundaries,
        i.e., the single-env path where we accumulate reward/len until done.
        """
        self._ep_count += 1

        # Store into rolling buffers (bounded by `window`).
        self._returns.append(float(ep_return))
        self._lengths.append(int(ep_len))
        self._trunc_flags.append(1 if truncated else 0)

        # Optional: log raw per-episode sample (can be high-frequency).
        if self.log_raw_episode:
            log(
                trainer,
                {
                    "episode/return": float(ep_return),
                    "episode/len": int(ep_len),
                    "episode/truncated": 1.0 if truncated else 0.0,
                    "episode/count": float(self._ep_count),
                },
                step=infer_step(trainer),
                prefix=self.log_prefix,
            )

        # Periodic aggregates over rolling window:
        # - return_mean/std over recent episodes
        # - len_mean over recent episodes
        # - trunc_rate as fraction of truncated episodes in window
        if (
            self.log_every_episodes > 0
            and (self._ep_count % self.log_every_episodes) == 0
            and len(self._returns) > 0
        ):
            rets = list(self._returns)
            lens = [float(x) for x in self._lengths]
            tr = [float(x) for x in self._trunc_flags]

            log(
                trainer,
                {
                    "return_mean": self._mean(rets),
                    "return_std": self._std(rets),
                    "len_mean": self._mean(lens),
                    "trunc_rate": self._mean(tr),
                    "episodes_window": int(len(self._returns)),
                    "episodes_total": float(self._ep_count),
                },
                step=infer_step(trainer),
                prefix=self.log_prefix,
            )

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Consume one environment step transition.

        - Normalizes the transition into unified representation
        - If batched: logs conservative batch-level done/trunc stats only
        - If single-env: accumulates reward/len until done, then records episode
        """
        if not transition:
            return True

        nt = self._normalize_transition(transition)
        if nt is None:
            return True

        # ---------------------------------------------------------------------
        # Batched path (conservative):
        #
        # In vectorized envs, `transition` often contains per-env rewards/dones.
        # However, without per-env accumulators (or explicit per-env episode return/len),
        # we cannot reconstruct episodic return/len correctly because each env has its
        # own episode boundaries and partial sums.
        #
        # Therefore:
        # - Count how many envs finished at this step
        # - Compute truncation rate among the finished envs only
        # ---------------------------------------------------------------------
        if nt.is_batched and len(nt.rewards) > 1:
            finished = [i for i, d in enumerate(nt.dones) if d]
            if finished:
                truncs = [1.0 if nt.truncated[i] else 0.0 for i in finished]
                log(
                    trainer,
                    {
                        "episode/batched_done_count": float(len(finished)),
                        "episode/batched_trunc_rate": self._mean(truncs) if truncs else 0.0,
                    },
                    step=infer_step(trainer),
                    prefix=self.log_prefix,
                )
            return True

        # ---------------------------------------------------------------------
        # Single-env path (accurate):
        #
        # Maintain running sums:
        #   ep_return += reward
        #   ep_len    += 1
        #
        # When done=True:
        #   - commit episode stats
        #   - reset accumulators for next episode
        # ---------------------------------------------------------------------
        r = float(nt.rewards[0]) if nt.rewards else 0.0
        done = bool(nt.dones[0]) if nt.dones else False
        trunc = bool(nt.truncated[0]) if nt.truncated else False

        self._ep_return += r
        self._ep_len += 1

        if not done:
            return True

        self._record_episode(trainer, self._ep_return, self._ep_len, trunc)

        # Reset for next episode
        self._ep_return = 0.0
        self._ep_len = 0
        return True