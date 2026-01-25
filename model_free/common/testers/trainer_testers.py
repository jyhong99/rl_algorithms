from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, List, Callable

import os
import sys
import numpy as np

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

from model_free.common.trainers.trainer import Trainer
from model_free.common.trainers.trainer_builder import build_trainer

from model_free.common.testers.test_harness import DummyEnvGym4, DummyAlgo
from model_free.common.testers.test_utils import (
    run_tests,
    assert_true,
    assert_eq,
)


# =============================================================================
# Tests
# =============================================================================
def test_trainer_single_env_smoke() -> None:
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    algo = DummyAlgo(action_shape=(2,), warmup_steps=2)

    t = Trainer(
        train_env=env,
        eval_env=DummyEnvGym4(obs_shape=(3,), action_shape=(2,)),
        algo=algo,
        total_env_steps=10,
        n_envs=1,
        log_every_steps=0,  # disable sys logger cadence for test
        normalize=False,
    )
    t.train()

    assert_true(t.global_env_step == 10, "trainer should reach total_env_steps")
    assert_true(t.global_update_step >= 1, "should have performed at least one update")


def test_trainer_normalize_wraps_envs() -> None:
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    algo = DummyAlgo(action_shape=(2,), warmup_steps=2)

    t = Trainer(
        train_env=env,
        eval_env=DummyEnvGym4(obs_shape=(3,), action_shape=(2,)),
        algo=algo,
        total_env_steps=5,
        n_envs=1,
        log_every_steps=0,
        normalize=True,
        obs_shape=(3,),
        norm_obs=True,
        norm_reward=True,
    )

    # 확인: 래핑 여부
    assert_true(t.train_env.__class__.__name__ == "NormalizeWrapper", "train_env should be NormalizeWrapper")
    assert_true(t.eval_env.__class__.__name__ == "NormalizeWrapper", "eval_env should be NormalizeWrapper")

    t.train()
    assert_eq(t.global_env_step, 5, "env steps after train")


def test_checkpoint_roundtrip_smoke(tmp_dir: Optional[str] = None) -> None:
    # tmp_dir을 안 쓰는 harness라면 run_dir를 고정된 로컬 폴더로 둬도 됨
    run_dir = "./_tmp_test_runs/trainer_ckpt"
    env = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    algo = DummyAlgo(action_shape=(2,), warmup_steps=1)

    t = Trainer(
        train_env=env,
        eval_env=DummyEnvGym4(obs_shape=(3,), action_shape=(2,)),
        algo=algo,
        total_env_steps=3,
        n_envs=1,
        log_every_steps=0,
        run_dir=run_dir,
        normalize=False,
    )

    t.train()
    step_before = t.global_env_step
    upd_before = t.global_update_step

    ckpt_path = t.save_checkpoint()
    assert_true(isinstance(ckpt_path, str) and len(ckpt_path) > 0, "checkpoint should be saved")

    # 새 trainer에 로드
    env2 = DummyEnvGym4(obs_shape=(3,), action_shape=(2,))
    algo2 = DummyAlgo(action_shape=(2,), warmup_steps=999)  # 로드로 덮일 것
    t2 = Trainer(
        train_env=env2,
        eval_env=DummyEnvGym4(obs_shape=(3,), action_shape=(2,)),
        algo=algo2,
        total_env_steps=3,
        n_envs=1,
        log_every_steps=0,
        run_dir=run_dir,
        normalize=False,
    )

    t2.load_checkpoint(str(ckpt_path))

    assert_eq(t2.global_env_step, step_before, "restored global_env_step")
    assert_eq(t2.global_update_step, upd_before, "restored global_update_step")


def test_build_trainer_factory_smoke() -> None:
    def make_env() -> Any:
        return DummyEnvGym4(obs_shape=(3,), action_shape=(2,))

    algo = DummyAlgo(action_shape=(2,), warmup_steps=2)

    t = build_trainer(
        make_train_env=make_env,
        make_eval_env=make_env,
        algo=algo,
        total_env_steps=5,
        enable_logger=False,
        enable_evaluator=False,
        normalize=True,
        obs_shape=(3,),
        norm_obs=True,
        norm_reward=False,
        n_envs=1,
        log_every_steps=0,
        dump_config=False,
        enable_eval_callback=False,
        enable_ckpt_callback=False,
        enable_episode_stats=False,
        enable_timing=False,
        enable_nan_guard=False,
        enable_lr_logging=False,
        enable_config_env_info=False,
    )

    t.train()
    assert_eq(t.global_env_step, 5, "build_trainer should create runnable trainer")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("trainer_single_env_smoke", test_trainer_single_env_smoke),
    ("trainer_normalize_wraps_envs", test_trainer_normalize_wraps_envs),
    ("trainer_checkpoint_roundtrip_smoke", test_checkpoint_roundtrip_smoke),
    ("build_trainer_factory_smoke", test_build_trainer_factory_smoke),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="trainers")


if __name__ == "__main__":
    raise SystemExit(main())
