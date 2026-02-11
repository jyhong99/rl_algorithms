from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import _infer_step
from ..utils.common_utils import _is_sequence, _mean, _std


class EpisodeStatsCallback(BaseCallback):
    """
    Episode-level statistics reconstructed from per-step transitions (framework-agnostic).

    This callback consumes per-step transition payloads (best-effort) and logs
    episode metrics such as return, length, and truncation rate.

    Key capabilities
    ----------------
    **Single environment (B=1):**
        Accurately reconstructs episodic return and episode length by accumulating
        reward until ``done=True`` is observed.

    **Vectorized/batched environments (B>1):**
        Without per-environment running accumulators (or explicit per-env episode
        summaries), accurate episodic return/length reconstruction is generally
        impossible because each environment has independent episode boundaries.
        In that case, this callback logs only conservative, step-local signals.

    Parameters
    ----------
    window:
        Rolling window size (in episodes) for computing aggregate statistics.
        If ``window <= 0``, rolling buffers become unbounded (not recommended for
        long runs).
    log_every_episodes:
        Log aggregate statistics every N finished episodes. If ``<= 0``, aggregate
        logging is disabled.
    log_prefix:
        Prefix passed to :meth:`BaseCallback.log` for all metrics emitted by this
        callback (e.g., ``"rollout/"`` or ``"train/"``).
    log_raw_episode:
        If True, log per-episode raw values for each finished episode (can be noisy).

    Logged metrics
    --------------
    Single-env path (accurate):
        Optionally per episode:
        - ``episode/return``
        - ``episode/len``
        - ``episode/truncated``
        - ``episode/count``

        Aggregates over the rolling window:
        - ``return_mean``, ``return_std``
        - ``len_mean``
        - ``trunc_rate``
        - ``episodes_window``, ``episodes_total``

    Batched path (conservative):
        Step-local metrics only (emitted when at least one env finishes at the step):
        - ``episode/batched_done_count``:
            number of envs that finished at the step (count of ``done=True``)
        - ``episode/batched_trunc_rate``:
            truncation rate among those finished envs only

    Notes
    -----
    - Truncation is inferred best-effort from common Gym/Gymnasium conventions:
      ``info["TimeLimit.truncated"]`` or ``info["truncated"]``.
    - The callback avoids hard dependency on numpy; it relies on simple utilities
      (``mean``, ``std``) from your ``common_utils``.
    """

    # =========================================================================
    # Internal: normalized transition representation
    # =========================================================================
    class _NormalizedTransition:
        """
        Normalized step transition representation.

        This class maps heterogeneous transition payload formats into a consistent
        interface that downstream logic can consume without branching on trainer/env
        conventions.

        Parameters
        ----------
        rewards:
            Rewards as a list of floats (length B).
        dones:
            Done flags as a list of bools (length B). Here, ``done`` corresponds to
            episode end (terminated OR truncated).
        truncated:
            Truncation flags as a list of bools (length B), inferred best-effort.
        infos:
            Raw info payloads aligned to length B (may contain dict-like objects).
        is_batched:
            Whether the original payload appears batched (vectorized).

        Attributes
        ----------
        rewards:
            Reward list (length B).
        dones:
            Done list (length B).
        truncated:
            Truncated list (length B).
        infos:
            Info list (length B).
        is_batched:
            Boolean indicating batched origin.

        Examples
        --------
        Single-env:
            ``rewards=[r]``, ``dones=[done]``, ``truncated=[trunc]``, ``infos=[info]``

        Batched env:
            ``rewards=[r0, r1, ...]``, ``dones=[d0, d1, ...]``, ``truncated=[t0, t1, ...]``
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
        self.window = int(window)
        self.log_every_episodes = int(log_every_episodes)
        self.log_prefix = str(log_prefix)
        self.log_raw_episode = bool(log_raw_episode)

        # Single-env accumulators (used only in the single-env path).
        self._ep_return: float = 0.0
        self._ep_len: int = 0
        self._ep_count: int = 0

        maxlen = self.window if self.window > 0 else None
        self._returns: Deque[float] = deque(maxlen=maxlen)
        self._lengths: Deque[int] = deque(maxlen=maxlen)
        self._trunc_flags: Deque[int] = deque(maxlen=maxlen)  # 1 if truncated else 0

    # =========================================================================
    # Truncation inference
    # =========================================================================
    def _infer_truncated_from_info(self, info: Any) -> bool:
        """
        Infer truncation status from an ``info`` payload (best-effort).

        Gym/Gymnasium conventions include:
        - ``info["TimeLimit.truncated"]`` (Gym TimeLimit wrapper)
        - ``info["truncated"]`` (some wrappers / custom envs)

        Parameters
        ----------
        info:
            Info-like object (typically dict-like). If not dict-like or missing
            expected keys, returns False.

        Returns
        -------
        bool
            True if truncation is inferred; otherwise False.
        """
        try:
            if info is None:
                return False
            d = dict(info)  # may raise if not dict-like
            return bool(d.get("TimeLimit.truncated", False) or d.get("truncated", False))
        except Exception:
            return False

    # =========================================================================
    # Transition normalization
    # =========================================================================
    def _ensure_list(self, x: Any, *, n: int, default: Any) -> List[Any]:
        """
        Broadcast or coerce ``x`` into a list of length ``n``.

        Rules
        -----
        - If ``x`` is a sequence (list/tuple/etc. per ``is_sequence``):
            - If ``len(x) == n``: return as list
            - If ``len(x) < n``: pad with ``default``
            - If ``len(x) > n``: truncate to ``n``
        - Otherwise:
            - Broadcast scalar to length ``n`` (if ``x is None`` use ``default``)

        Parameters
        ----------
        x:
            Value to coerce or broadcast.
        n:
            Target length.
        default:
            Padding / broadcast value when missing.

        Returns
        -------
        List[Any]
            List of length ``n``.
        """
        if _is_sequence(x):
            xs = list(x)
            if len(xs) == n:
                return xs
            if len(xs) < n:
                xs = xs + [default] * (n - len(xs))
            return xs[:n]
        return [x if x is not None else default] * n

    def _normalize_transition(self, transition: Mapping[str, Any]) -> Optional[_NormalizedTransition]:
        """
        Normalize a heterogeneous transition payload into a consistent representation.

        Supported input patterns
        ------------------------
        Single-env:
            - ``{"reward": r, "done": done, "info": info}``
            - ``{"reward": r, "terminated": term, "truncated": trunc, "info": info}``

        Batched env:
            - ``{"rewards": [...], "dones": [...], "infos": [...]}``
            - ``{"rewards": [...], "terminated": [...], "truncated": [...], "infos": [...]}``
            - Mixed cases are handled best-effort.

        Output semantics
        ----------------
        - The output lists have length ``B`` (batch size), inferred primarily from
          the reward vector length.
        - ``done`` is interpreted as ``terminated OR truncated`` when both are present.
        - ``truncated`` is refined using info keys such as ``TimeLimit.truncated``.

        Parameters
        ----------
        transition:
            Dict-like mapping containing step results.

        Returns
        -------
        Optional[_NormalizedTransition]
            Normalized transition, or None if the payload is empty/unusable.
        """
        if not transition:
            return None

        rewards = transition.get("rewards", None)
        dones = transition.get("dones", None)
        infos = transition.get("infos", None)

        terminated = transition.get("terminated", None)
        truncated = transition.get("truncated", None)

        is_batched = (
            _is_sequence(rewards)
            or _is_sequence(dones)
            or _is_sequence(terminated)
            or _is_sequence(truncated)
        )

        # ---- rewards (anchor for batch size B) ----
        if _is_sequence(rewards):
            r_list = [float(x) for x in rewards]
        else:
            r_list = [float(transition.get("reward", 0.0))]

        B = len(r_list)

        # ---- dones + truncated flags ----
        if _is_sequence(dones):
            d_list = [bool(x) for x in dones]
            trunc_list = [False] * len(d_list)
        else:
            if _is_sequence(terminated) or _is_sequence(truncated):
                term_list = self._ensure_list(terminated, n=B, default=False)
                tru_list = self._ensure_list(truncated, n=B, default=False)
                term_b = [bool(x) for x in term_list]
                tru_b = [bool(x) for x in tru_list]
                d_list = [t or tr for t, tr in zip(term_b, tru_b)]
                trunc_list = list(tru_b)
            else:
                done_single = transition.get("done", None)
                if done_single is None and (terminated is not None or truncated is not None):
                    done_single = bool(terminated) or bool(truncated)

                d_list = [bool(done_single)] * B
                trunc_list = [bool(truncated)] * B if truncated is not None else [False] * B

        # ---- infos ----
        if infos is None:
            info_alt = transition.get("info", None)
            if _is_sequence(info_alt):
                infos_list = list(info_alt)
            else:
                infos_list = [info_alt] * B
        else:
            infos_list = self._ensure_list(infos, n=B, default=None)

        # Refine truncation using info (TimeLimit.truncated, etc.).
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
        Commit a finished episode into rolling buffers and emit logs.

        This method is used only in the single-env path where we can accurately
        reconstruct episode boundaries by accumulating reward and length until
        ``done=True``.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed), used only for logging.
        ep_return:
            Episode return (sum of rewards over the episode).
        ep_len:
            Episode length (number of environment steps in the episode).
        truncated:
            Whether the episode ended via truncation (best-effort).

        Notes
        -----
        - Rolling buffers are bounded by ``window`` (unless ``window <= 0``).
        - Aggregate statistics are emitted every ``log_every_episodes`` episodes.
        """
        self._ep_count += 1

        self._returns.append(float(ep_return))
        self._lengths.append(int(ep_len))
        self._trunc_flags.append(1 if truncated else 0)

        if self.log_raw_episode:
            self.log(
                trainer,
                {
                    "episode/return": float(ep_return),
                    "episode/len": int(ep_len),
                    "episode/truncated": 1.0 if truncated else 0.0,
                    "episode/count": float(self._ep_count),
                },
                step=_infer_step(trainer),
                prefix=self.log_prefix,
            )

        if (
            self.log_every_episodes > 0
            and (self._ep_count % self.log_every_episodes) == 0
            and len(self._returns) > 0
        ):
            rets = list(self._returns)
            lens = [float(x) for x in self._lengths]
            tr = [float(x) for x in self._trunc_flags]

            self.log(
                trainer,
                {
                    "return_mean": _mean(rets),
                    "return_std": _std(rets),
                    "len_mean": _mean(lens),
                    "trunc_rate": _mean(tr),
                    "episodes_window": int(len(self._returns)),
                    "episodes_total": float(self._ep_count),
                },
                step=_infer_step(trainer),
                prefix=self.log_prefix,
            )

    # =========================================================================
    # Callback hook
    # =========================================================================
    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Consume one environment step transition.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed), used for logging.
        transition:
            Step transition payload produced by the training loop (best-effort).
            Expected to contain some combination of reward/done/info fields
            (single-env or batched).

        Returns
        -------
        bool
            Always True (this callback never requests early stop).

        Notes
        -----
        Processing:
        - Normalizes the transition into a unified representation.
        - If batched (B>1): logs conservative batch-level completion/truncation
          signals only.
        - If single-env (B==1): accumulates episode return/length until done,
          then records the finished episode and resets accumulators.
        """
        if not transition:
            return True

        nt = self._normalize_transition(transition)
        if nt is None:
            return True

        # ---------------------------------------------------------------------
        # Batched path (conservative):
        # ---------------------------------------------------------------------
        if nt.is_batched and len(nt.rewards) > 1:
            finished = [i for i, d in enumerate(nt.dones) if d]
            if finished:
                truncs = [1.0 if nt.truncated[i] else 0.0 for i in finished]
                self.log(
                    trainer,
                    {
                        "episode/batched_done_count": float(len(finished)),
                        "episode/batched_trunc_rate": _mean(truncs) if truncs else 0.0,
                    },
                    step=_infer_step(trainer),
                    prefix=self.log_prefix,
                )
            return True

        # ---------------------------------------------------------------------
        # Single-env path (accurate):
        # ---------------------------------------------------------------------
        r = float(nt.rewards[0]) if nt.rewards else 0.0
        done = bool(nt.dones[0]) if nt.dones else False
        trunc = bool(nt.truncated[0]) if nt.truncated else False

        self._ep_return += r
        self._ep_len += 1

        if not done:
            return True

        self._record_episode(trainer, self._ep_return, self._ep_len, trunc)

        # Reset accumulators for the next episode.
        self._ep_return = 0.0
        self._ep_len = 0
        return True
