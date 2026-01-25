from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Tuple

import numpy as np

def _bootstrap_sys_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        if os.path.isdir(os.path.join(parent, "model_free")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        cur = parent

    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)

_bootstrap_sys_path()

from model_free.common.testers.test_harness import DummyEnvGym4, DummyEnvGymnasium5
from model_free.common.testers.test_utils import (
    seed_all,
    run_tests,
    assert_eq,
    assert_true,
    assert_shape,
    assert_finite,
    assert_allclose,
    assert_allclose_dict,
)
from model_free.common.wrappers.normalize_wrapper import NormalizeWrapper


def test_reset_step_gym4(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), norm_obs=True, norm_reward=True, training=True)

    obs = w.reset()
    assert_shape(obs, (3,), "gym4 reset obs shape")
    assert_eq(np.asarray(obs).dtype, np.dtype(np.float32), "gym4 reset obs dtype")

    obs, r, done, info = w.step(np.array([0.0, 0.0], dtype=np.float32))
    assert_shape(obs, (3,), "gym4 step obs shape")
    assert_true(isinstance(r, float), "gym4 reward should be float")
    assert_true(isinstance(done, (bool, np.bool_)), "gym4 done type")
    assert_true(isinstance(info, dict), "gym4 info should be dict")
    assert_finite(obs, "gym4 obs finite")
    assert_finite(r, "gym4 reward finite")


def test_reset_step_gymnasium5(NormalizeWrapper):
    env = DummyEnvGymnasium5(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), norm_obs=True, norm_reward=True, training=True)

    obs, info = w.reset()
    assert_shape(obs, (3,), "gymnasium reset obs shape")
    assert_true(isinstance(info, dict), "gymnasium reset info dict")

    obs, r, terminated, truncated, info = w.step(np.array([0.0, 0.0], dtype=np.float32))
    assert_shape(obs, (3,), "gymnasium step obs shape")
    assert_true(isinstance(r, float), "gymnasium reward float")
    assert_true(isinstance(terminated, (bool, np.bool_)), "terminated type")
    assert_true(isinstance(truncated, (bool, np.bool_)), "truncated type")
    assert_true(isinstance(info, dict), "info dict")
    assert_finite(obs, "gymnasium obs finite")
    assert_finite(r, "gymnasium reward finite")


def test_action_rescale_maps_to_bounds(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), action_rescale=True, clip_action=0.0)

    a_min = np.array([-1.0, -1.0], dtype=np.float32)
    a_max = np.array([ 1.0,  1.0], dtype=np.float32)

    fm = w._format_action(a_min)
    fx = w._format_action(a_max)

    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    assert_true(np.allclose(fm, low), "action_rescale -1 -> low")
    assert_true(np.allclose(fx, high), "action_rescale +1 -> high")


def test_clip_action(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), action_rescale=False, clip_action=1.0)

    a = np.array([100.0, -100.0], dtype=np.float32)
    fa = w._format_action(a)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    assert_true(np.all(fa <= high + 1e-6) and np.all(fa >= low - 1e-6), "clip_action clamps into bounds")


def test_training_flag_controls_rms_update(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), norm_obs=True, norm_reward=False, training=True)

    w.reset()
    for _ in range(3):
        w.step(np.zeros((2,), dtype=np.float32))

    s1 = w.state_dict()
    obs_rms_1 = s1["obs_rms"]

    w.set_training(False)
    for _ in range(3):
        w.step(np.zeros((2,), dtype=np.float32))

    s2 = w.state_dict()
    obs_rms_2 = s2["obs_rms"]

    # count는 스칼라라 eq로 OK
    assert_eq(obs_rms_1["count"], obs_rms_2["count"], "obs_rms count should not change when training=False")

    # mean/var는 배열일 수 있으므로 allclose로 검사
    assert_allclose_dict(
        obs_rms_1,
        obs_rms_2,
        "obs_rms stats should not change when training=False",
        keys=("mean", "var"),
        rtol=0.0,
        atol=0.0,   # 완전 동일 기대
    )


def test_max_episode_steps_gym4_truncation(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w = NormalizeWrapper(env, obs_shape=(3,), max_episode_steps=2)

    w.reset()
    _, _, done1, _info1 = w.step(np.zeros((2,), dtype=np.float32))
    assert_true(not done1, "should not be done at step 1")

    _, _, done2, info2 = w.step(np.zeros((2,), dtype=np.float32))
    assert_true(done2, "should be forced done at max_episode_steps")
    assert_true(bool(info2.get("TimeLimit.truncated", False)), "TimeLimit.truncated should be set")


def test_state_dict_roundtrip(NormalizeWrapper):
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w1 = NormalizeWrapper(env, obs_shape=(3,), norm_obs=True, norm_reward=True, training=True)

    w1.reset()
    for _ in range(2):
        w1.step(np.zeros((2,), dtype=np.float32))

    st = w1.state_dict()

    env2 = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    w2 = NormalizeWrapper(env2, obs_shape=(3,), norm_obs=True, norm_reward=True, training=False)
    w2.load_state_dict(st)

    st2 = w2.state_dict()
    assert_eq(st2["obs_dtype"], st["obs_dtype"], "obs_dtype restored")
    assert_eq(st2["running_return"], st["running_return"], "running_return restored")
    assert_true("obs_rms" in st2 and "ret_rms" in st2, "rms states restored")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    # --- wrapper tests (added) ---
    ("normalize_reset_step_gym4", lambda: test_reset_step_gym4(NormalizeWrapper)),
    ("normalize_reset_step_gymnasium5", lambda: test_reset_step_gymnasium5(NormalizeWrapper)),
    ("normalize_action_rescale_maps_to_bounds", lambda: test_action_rescale_maps_to_bounds(NormalizeWrapper)),
    ("normalize_clip_action", lambda: test_clip_action(NormalizeWrapper)),
    ("normalize_training_flag_controls_rms_update", lambda: test_training_flag_controls_rms_update(NormalizeWrapper)),
    ("normalize_max_episode_steps_gym4_truncation", lambda: test_max_episode_steps_gym4_truncation(NormalizeWrapper)),
    ("normalize_state_dict_roundtrip", lambda: test_state_dict_roundtrip(NormalizeWrapper)),
]


def main(argv=None) -> int:
    seed_all(0)
    return run_tests(TESTS, argv=argv, suite_name="wrappers")


if __name__ == "__main__":
    raise SystemExit(main())
