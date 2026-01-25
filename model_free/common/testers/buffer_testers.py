from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


def _bootstrap_sys_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        # If parent contains a "model_free" package dir, add parent to sys.path
        if os.path.isdir(os.path.join(parent, "model_free")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        cur = parent

    # Fallback: add grandparent (often works when tests/ is inside package)
    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_bootstrap_sys_path()

from model_free.common.testers.test_utils import (
    TestFailure,
    run_tests,
    assert_eq,
    assert_close,
    assert_true,
    assert_allclose,
    assert_raises
)

from model_free.common.buffers.rollout_buffer import RolloutBuffer
from model_free.common.buffers.replay_buffer import ReplayBuffer
from model_free.common.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


# =============================================================================
# Tests: RolloutBuffer
# =============================================================================
def test_rolloutbuffer_add_full_and_size():
    buf = RolloutBuffer(buffer_size=4, obs_shape=(3,), action_shape=(2,), device="cpu")
    assert_eq(buf.size, 0)
    assert_true(buf.full is False)

    for t in range(4):
        obs = np.ones((3,), dtype=np.float32) * (t + 1)
        act = np.zeros((2,), dtype=np.float32) + t
        buf.add(obs, act, reward=1.0, done=False, value=0.5, log_prob=-0.1)

    assert_eq(buf.size, 4)
    assert_true(buf.full is True)

    # adding one more should error
    def _add_overflow():
        buf.add(np.zeros((3,), np.float32), np.zeros((2,), np.float32), 0.0, False, 0.0, 0.0)
    assert_raises(RuntimeError, _add_overflow)


def test_rolloutbuffer_compute_returns_simple_case_no_done():
    # gamma=1, lambda=1, dones all 0, last_done False
    # rewards=1 each step, values=0 => advantages should be [4,3,2,1], returns same.
    buf = RolloutBuffer(
        buffer_size=4,
        obs_shape=(1,),
        action_shape=(1,),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        device="cpu",
    )

    for _ in range(4):
        buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), reward=1.0, done=False, value=0.0, log_prob=0.0)

    assert_true(buf.full is True)
    buf.compute_returns_and_advantage(last_value=0.0, last_done=False)

    assert_allclose(buf.advantages, np.array([4, 3, 2, 1], dtype=np.float32), msg="GAE advantages mismatch")
    assert_allclose(buf.returns, np.array([4, 3, 2, 1], dtype=np.float32), msg="returns mismatch")


def test_rolloutbuffer_compute_returns_with_done_cut():
    # If done at t=1, future should cut.
    # Setup: T=3, gamma=1, lambda=1, values=0, rewards=[1,1,1], dones=[0,1,0], last_done=False
    # For t=0: return should be 1+1 (since done at t=1 stops after r1), so adv0=2
    # t=1: done => adv1=1
    # t=2: after done, independent => adv2=1
    buf = RolloutBuffer(
        buffer_size=3,
        obs_shape=(1,),
        action_shape=(1,),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        device="cpu",
    )

    buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), reward=1.0, done=False, value=0.0, log_prob=0.0)
    buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), reward=1.0, done=True,  value=0.0, log_prob=0.0)
    buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), reward=1.0, done=False, value=0.0, log_prob=0.0)

    buf.compute_returns_and_advantage(last_value=0.0, last_done=False)

    assert_allclose(buf.advantages, np.array([2, 1, 1], dtype=np.float32))
    assert_allclose(buf.returns, np.array([2, 1, 1], dtype=np.float32))


def test_rolloutbuffer_normalize_advantages_zero_mean_unit_std():
    buf = RolloutBuffer(
        buffer_size=4,
        obs_shape=(1,),
        action_shape=(1,),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=True,
        adv_eps=1e-8,
        device="cpu",
    )
    # Make non-constant advantages
    # rewards= [1,2,3,4], values=0 => advantages are cumulative sums reversed [10,9,7,4]
    for r in [1.0, 2.0, 3.0, 4.0]:
        buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), reward=r, done=False, value=0.0, log_prob=0.0)

    buf.compute_returns_and_advantage(last_value=0.0, last_done=False)
    adv = buf.advantages.astype(np.float64)
    m = float(np.mean(adv))
    s = float(np.std(adv))
    assert_close(m, 0.0, atol=1e-5, msg="normalized advantages mean not ~0")
    assert_close(s, 1.0, atol=1e-5, msg="normalized advantages std not ~1")


def test_rolloutbuffer_sample_batches_shapes_and_device():
    buf = RolloutBuffer(buffer_size=5, obs_shape=(3,), action_shape=(2,), device="cpu")
    for t in range(5):
        buf.add(np.ones((3,), np.float32) * t, np.ones((2,), np.float32), reward=1.0, done=False, value=0.0, log_prob=-0.2)
    buf.compute_returns_and_advantage(last_value=0.0, last_done=False)

    batches = list(buf.sample(batch_size=2, shuffle=False))
    # 5 -> batches: 2,2,1
    assert_eq(len(batches), 3)

    b0 = batches[0]
    assert_eq(tuple(b0.observations.shape), (2, 3))
    assert_eq(tuple(b0.actions.shape), (2, 2))
    assert_true(b0.observations.device.type == "cpu")
    assert_true(b0.actions.device.type == "cpu")


def test_rolloutbuffer_sample_requires_full():
    buf = RolloutBuffer(buffer_size=3, obs_shape=(1,), action_shape=(1,), device="cpu")
    buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), 1.0, False, 0.0, 0.0)
    assert_raises(RuntimeError, lambda: list(buf.sample(2)))


def test_rolloutbuffer_compute_requires_full():
    buf = RolloutBuffer(buffer_size=3, obs_shape=(1,), action_shape=(1,), device="cpu")
    buf.add(np.array([0.0], np.float32), np.array([0.0], np.float32), 1.0, False, 0.0, 0.0)
    assert_raises(RuntimeError, lambda: buf.compute_returns_and_advantage(0.0, False))


# =============================================================================
# Tests: ReplayBuffer
# =============================================================================
def _fill_replay(buf: ReplayBuffer, n: int, *, done_at: Optional[int] = None, behavior: bool = False, probs: bool = False) -> None:
    for t in range(n):
        obs = np.ones(buf.obs_shape, np.float32) * t
        act = np.ones(buf.action_shape, np.float32) * (t + 0.5)
        nxt = np.ones(buf.obs_shape, np.float32) * (t + 1)
        done = (done_at is not None and t == done_at)
        kwargs: Dict[str, Any] = {}
        if behavior:
            kwargs["behavior_logp"] = -0.1 * (t + 1)
        if probs:
            # uniform probs
            A = int(buf.n_actions or 0)
            p = np.ones((A,), np.float32) / float(A)
            kwargs["behavior_probs"] = p
        buf.add(obs, act, reward=float(t), next_obs=nxt, done=done, **kwargs)


def test_replaybuffer_add_and_len_and_shapes():
    buf = ReplayBuffer(capacity=5, obs_shape=(3,), action_shape=(2,), device="cpu")
    assert_eq(buf.size, 0)

    _fill_replay(buf, 3)
    assert_eq(buf.size, 3)
    assert_eq(buf.observations.shape, (5, 3))
    assert_eq(buf.actions.shape, (5, 2))
    assert_eq(buf.rewards.shape, (5, 1))
    assert_eq(buf.dones.shape, (5, 1))


def test_replaybuffer_add_shape_mismatch_raises():
    buf = ReplayBuffer(capacity=3, obs_shape=(3,), action_shape=(2,), device="cpu")

    def _bad_obs():
        buf.add(np.zeros((4,), np.float32), np.zeros((2,), np.float32), 1.0, np.zeros((3,), np.float32), False)
    assert_raises(ValueError, _bad_obs)

    def _bad_act():
        buf.add(np.zeros((3,), np.float32), np.zeros((3,), np.float32), 1.0, np.zeros((3,), np.float32), False)
    assert_raises(ValueError, _bad_act)


def test_replaybuffer_sample_shapes_and_device():
    buf = ReplayBuffer(capacity=20, obs_shape=(4,), action_shape=(1,), device="cpu")
    _fill_replay(buf, 10)

    batch = buf.sample(batch_size=7)
    assert_eq(tuple(batch.observations.shape), (7, 4))
    assert_eq(tuple(batch.actions.shape), (7, 1))
    assert_eq(tuple(batch.rewards.shape), (7, 1))
    assert_true(batch.observations.device.type == "cpu")


def test_replaybuffer_behavior_logp_enabled_requires_value():
    buf = ReplayBuffer(capacity=5, obs_shape=(2,), action_shape=(1,), device="cpu", store_behavior_logp=True)

    def _missing():
        buf.add(np.zeros((2,), np.float32), np.zeros((1,), np.float32), 1.0, np.zeros((2,), np.float32), False)
    assert_raises(ValueError, _missing)

    # ok when provided
    buf.add(np.zeros((2,), np.float32), np.zeros((1,), np.float32), 1.0, np.zeros((2,), np.float32), False, behavior_logp=-0.3)
    b = buf.sample(1)
    assert_true(b.behavior_logp is not None)
    assert_eq(tuple(b.behavior_logp.shape), (1, 1))


def test_replaybuffer_behavior_probs_enabled_requires_n_actions_and_probs():
    # missing n_actions -> init error
    assert_raises(ValueError, lambda: ReplayBuffer(capacity=5, obs_shape=(2,), action_shape=(1,), store_behavior_probs=True))

    buf = ReplayBuffer(capacity=5, obs_shape=(2,), action_shape=(1,), device="cpu", store_behavior_probs=True, n_actions=3)

    def _missing_probs():
        buf.add(np.zeros((2,), np.float32), np.zeros((1,), np.float32), 1.0, np.zeros((2,), np.float32), False)
    assert_raises(ValueError, _missing_probs)

    # ok
    buf.add(
        np.zeros((2,), np.float32),
        np.zeros((1,), np.float32),
        1.0,
        np.zeros((2,), np.float32),
        False,
        behavior_probs=np.array([0.2, 0.3, 0.5], np.float32),
    )
    b = buf.sample(1)
    assert_true(b.behavior_probs is not None)
    assert_eq(tuple(b.behavior_probs.shape), (1, 3))


def test_replaybuffer_n_step_writes_returns_and_flush_on_done():
    # n_step=3, gamma=1.0, rewards: 0,1,2 then done at t=2
    # For idx0 (t=0): R=0+1+2=3, done_n=True, next_obs_n from t=2 next_obs
    buf = ReplayBuffer(capacity=10, obs_shape=(1,), action_shape=(1,), device="cpu", n_step=3, gamma=1.0)
    _fill_replay(buf, 3, done_at=2)

    assert_true(buf.n_step_returns is not None)
    assert_true(buf.n_step_dones is not None)
    assert_true(buf.n_step_next_observations is not None)

    # idx0 corresponds to first inserted transition (pos advanced, but storage idx0 is 0)
    assert_close(float(buf.n_step_returns[0, 0]), 3.0, msg="n_step return mismatch")
    assert_close(float(buf.n_step_dones[0, 0]), 1.0, msg="n_step done mismatch")


# =============================================================================
# Tests: PrioritizedReplayBuffer
# =============================================================================
def test_per_add_sets_priority_and_sample_returns_weights_and_indices():
    buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), action_shape=(1,), device="cpu", alpha=0.6, beta=0.4)

    # Add with explicit priorities
    for t, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        obs = np.ones((2,), np.float32) * t
        act = np.ones((1,), np.float32)
        nxt = np.ones((2,), np.float32) * (t + 1)
        buf.add(obs, act, reward=1.0, next_obs=nxt, done=False, priority=p)

    batch = buf.sample(batch_size=3)
    assert_true(isinstance(batch.indices, np.ndarray))
    assert_true(th.is_tensor(batch.weights))
    assert_eq(tuple(batch.weights.shape), (3, 1))
    # weights are normalized, should be in (0, 1]
    w = batch.weights.detach().cpu().numpy()
    assert_true(np.all(w > 0.0) and np.all(w <= 1.0 + 1e-6))


def test_per_sample_empty_raises():
    buf = PrioritizedReplayBuffer(capacity=8, obs_shape=(1,), action_shape=(1,), device="cpu")
    assert_raises(RuntimeError, lambda: buf.sample(1))


def test_per_update_priorities_changes_max_priority():
    buf = PrioritizedReplayBuffer(capacity=8, obs_shape=(1,), action_shape=(1,), device="cpu")
    for t in range(4):
        buf.add(np.array([t], np.float32), np.array([0.0], np.float32), 0.0, np.array([t + 1], np.float32), False, priority=1.0)

    old = float(buf.max_priority)
    buf.update_priorities(np.array([1, 2], np.int64), np.array([10.0, 5.0], np.float32))
    assert_true(buf.max_priority >= old)
    assert_true(buf.max_priority >= 10.0)


def test_per_update_priorities_shape_mismatch_raises():
    buf = PrioritizedReplayBuffer(capacity=8, obs_shape=(1,), action_shape=(1,), device="cpu")
    for t in range(3):
        buf.add(np.array([t], np.float32), np.array([0.0], np.float32), 0.0, np.array([t + 1], np.float32), False)

    def _bad():
        buf.update_priorities(np.array([1, 2], np.int64), np.array([1.0], np.float32))
    assert_raises(ValueError, _bad)


# =============================================================================
# Simple runner (same style as your callbacks/writers)
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    # RolloutBuffer
    ("rollout_add_full_and_size", test_rolloutbuffer_add_full_and_size),
    ("rollout_compute_returns_simple_no_done", test_rolloutbuffer_compute_returns_simple_case_no_done),
    ("rollout_compute_returns_with_done_cut", test_rolloutbuffer_compute_returns_with_done_cut),
    ("rollout_normalize_advantages", test_rolloutbuffer_normalize_advantages_zero_mean_unit_std),
    ("rollout_sample_batches", test_rolloutbuffer_sample_batches_shapes_and_device),
    ("rollout_sample_requires_full", test_rolloutbuffer_sample_requires_full),
    ("rollout_compute_requires_full", test_rolloutbuffer_compute_requires_full),

    # ReplayBuffer
    ("replay_add_len_shapes", test_replaybuffer_add_and_len_and_shapes),
    ("replay_add_shape_mismatch_raises", test_replaybuffer_add_shape_mismatch_raises),
    ("replay_sample_shapes_device", test_replaybuffer_sample_shapes_and_device),
    ("replay_behavior_logp", test_replaybuffer_behavior_logp_enabled_requires_value),
    ("replay_behavior_probs", test_replaybuffer_behavior_probs_enabled_requires_n_actions_and_probs),
    ("replay_n_step_flush_on_done", test_replaybuffer_n_step_writes_returns_and_flush_on_done),

    # PrioritizedReplayBuffer
    ("per_add_sample_weights_indices", test_per_add_sets_priority_and_sample_returns_weights_and_indices),
    ("per_sample_empty_raises", test_per_sample_empty_raises),
    ("per_update_priorities_changes_max", test_per_update_priorities_changes_max_priority),
    ("per_update_priorities_shape_mismatch_raises", test_per_update_priorities_shape_mismatch_raises),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="buffers")

if __name__ == "__main__":
    raise SystemExit(main())