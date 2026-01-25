from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn

import os
import sys

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

from model_free.common.testers.test_harness import (
    FakeTrainer, 
    CallbackRunner
)

from model_free.common.testers.test_utils import (
    run_tests,
    assert_close,
    assert_eq,
    assert_in,
    assert_ge,
    assert_true,
    mk_tmp_dir
)




# =============================================================================
# Individual tests (converted from your pytest-style tests)
# =============================================================================
def test_config_and_env_info_callback_logs_once() -> None:
    from model_free.common.callbacks.config_and_env_info_callback import ConfigAndEnvInfoCallback

    cb = ConfigAndEnvInfoCallback(log_prefix="sys/", log_once=True)
    tr = FakeTrainer()

    tr.seed = 123
    tr.total_env_steps = 1000
    tr.n_envs = 4
    tr.device = "cpu"

    class Spec:
        id = "DummyEnv-v0"

    class DummyEnv:
        spec = Spec()
        num_envs = 8
        norm_obs = True

    tr.train_env = DummyEnv()

    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    recs = [r for r in tr.logger.records if r.prefix == "sys/"]
    assert_true(len(recs) >= 1, "expected at least one sys/ log record")
    merged: Dict[str, Any] = {}
    for r in recs:
        merged.update(r.metrics)

    assert_eq(merged.get("seed"), 123)
    assert_eq(merged.get("env_id"), "DummyEnv-v0")
    assert_in(merged.get("n_envs"), (4, 8))

    n0 = len(tr.logger.records)
    cb.on_train_start(tr)
    assert_eq(len(tr.logger.records), n0, "log_once=True should not log twice")


def test_config_and_env_info_callback_allows_repeat_when_log_once_false() -> None:
    from model_free.common.callbacks.config_and_env_info_callback import ConfigAndEnvInfoCallback

    cb = ConfigAndEnvInfoCallback(log_prefix="sys/", log_once=False)
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])

    runner.train_start()
    n0 = len(tr.logger.records)
    cb.on_train_start(tr)
    assert_true(len(tr.logger.records) >= n0 + 1, "log_once=False should allow repeated logs")


def test_episode_stats_single_env_accumulates_and_logs() -> None:
    from model_free.common.callbacks.episode_stats_callback import EpisodeStatsCallback

    cb = EpisodeStatsCallback(window=100, log_every_episodes=1, log_prefix="rollout/", log_raw_episode=True)
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    tr.global_env_step = 1
    runner.step({"reward": 1.0, "done": False, "info": {}})
    tr.global_env_step = 2
    runner.step({"reward": 2.0, "done": False, "info": {}})
    tr.global_env_step = 3
    runner.step({"reward": 3.0, "done": True, "info": {"TimeLimit.truncated": False}})

    recs = [r for r in tr.logger.records if r.prefix == "rollout/"]
    assert_true(any("episode/return" in r.metrics for r in recs), "missing raw episode log")
    assert_true(any("return_mean" in r.metrics for r in recs), "missing aggregate log")

    last = [r for r in recs if "return_mean" in r.metrics][-1].metrics
    assert_close(last["return_mean"], 6.0)
    assert_close(last["len_mean"], 3.0)
    assert_close(last["trunc_rate"], 0.0)


def test_episode_stats_single_env_truncation_inferred_from_info() -> None:
    from model_free.common.callbacks.episode_stats_callback import EpisodeStatsCallback

    cb = EpisodeStatsCallback(window=10, log_every_episodes=1, log_prefix="rollout/", log_raw_episode=False)
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    tr.global_env_step = 1
    runner.step({"reward": 1.0, "done": False, "info": {}})
    tr.global_env_step = 2
    runner.step({"reward": 1.0, "done": True, "info": {"TimeLimit.truncated": True}})

    recs = [r for r in tr.logger.records if r.prefix == "rollout/"]
    assert_true(len(recs) >= 1, "expected rollout/ logs")
    last = recs[-1].metrics
    assert_true("trunc_rate" in last, "missing trunc_rate")
    assert_close(last["trunc_rate"], 1.0)


def test_episode_stats_batched_logs_only_batched_counts() -> None:
    from model_free.common.callbacks.episode_stats_callback import EpisodeStatsCallback

    cb = EpisodeStatsCallback(window=100, log_every_episodes=1, log_prefix="rollout/")
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    tr.global_env_step = 10
    runner.step({
        "rewards": [1.0, 2.0, 3.0],
        "dones": [True, False, True],
        "infos": [{}, {}, {"TimeLimit.truncated": True}],
    })

    recs = [r for r in tr.logger.records if r.prefix == "rollout/"]
    assert_true(len(recs) >= 1)
    m = recs[-1].metrics
    assert_close(m["episode/batched_done_count"], 2.0)
    assert_close(m["episode/batched_trunc_rate"], 0.5)


class _TinyAlgoForNorm:
    def __init__(self):
        self.head = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))

    def get_modules_for_logging(self):
        return {"head": self.head}


def test_grad_param_norm_callback_logs_global_norms() -> None:
    from model_free.common.callbacks.grad_param_norm_callback import GradParamNormCallback

    cb = GradParamNormCallback(
        log_every_updates=1,
        log_prefix="debug/",
        include_param_norm=True,
        include_grad_norm=True,
        norm_type=2.0,
        per_module=False,
    )

    tr = FakeTrainer()
    tr.algo = _TinyAlgoForNorm()
    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    x = th.randn(5, 3)
    y = tr.algo.head(x).sum()
    y.backward()

    tr.global_update_step = 1
    ok = runner.update(metrics={"loss": 1.0})
    assert_true(ok is True)

    recs = [r for r in tr.logger.records if r.prefix == "debug/"]
    assert_true(len(recs) >= 1)
    m = recs[-1].metrics
    assert_true("param_norm" in m)
    assert_true("grad_norm" in m)
    assert_ge(m["param_norm"], 0.0)
    assert_ge(m["grad_norm"], 0.0)


def test_grad_param_norm_callback_per_module_keys() -> None:
    from model_free.common.callbacks.grad_param_norm_callback import GradParamNormCallback

    cb = GradParamNormCallback(log_every_updates=1, log_prefix="debug/", per_module=True)
    tr = FakeTrainer()
    tr.algo = _TinyAlgoForNorm()
    runner = CallbackRunner(tr, [cb])

    x = th.randn(2, 3)
    tr.algo.head(x).sum().backward()

    tr.global_update_step = 1
    runner.update(metrics={"loss": 1.0})

    recs = [r for r in tr.logger.records if r.prefix == "debug/"]
    assert_true(len(recs) >= 1)
    m = recs[-1].metrics
    assert_true("head/param_norm" in m)
    assert_true("head/grad_norm" in m)


def test_grad_param_norm_callback_no_algo_no_log() -> None:
    from model_free.common.callbacks.grad_param_norm_callback import GradParamNormCallback

    cb = GradParamNormCallback(log_every_updates=1, log_prefix="debug/")
    tr = FakeTrainer()
    tr.algo = None  # type: ignore
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    runner.update(metrics={"loss": 1.0})

    recs = [r for r in tr.logger.records if r.prefix == "debug/"]
    assert_eq(len(recs), 0, "should not log without algo")


class _TinyAlgoForLR:
    def __init__(self):
        self.m = nn.Linear(2, 2)
        self.optimizers = {"actor": th.optim.Adam(self.m.parameters(), lr=1e-3)}
        self.schedulers = {}

    def get_lr_dict(self) -> Dict[str, float]:
        return {"lr/actor": 1e-3}


def test_lr_logging_callback_prefers_get_lr_dict() -> None:
    from model_free.common.callbacks.lr_logging_callback import LRLoggingCallback

    cb = LRLoggingCallback(log_every_updates=1, log_prefix="train/")
    tr = FakeTrainer()
    tr.algo = _TinyAlgoForLR()
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    runner.update(metrics={"loss": 1.0})

    recs = [r for r in tr.logger.records if r.prefix == "train/"]
    assert_true(len(recs) >= 1)
    m = recs[-1].metrics
    assert_true(any(str(k).endswith("lr/actor") for k in m.keys()), f"unexpected lr keys: {list(m.keys())}")


def test_lr_logging_callback_fallback_to_optimizer_param_groups() -> None:
    from model_free.common.callbacks.lr_logging_callback import LRLoggingCallback

    cb = LRLoggingCallback(log_every_updates=1, log_prefix="train/")

    class AlgoNoGetDict:
        def __init__(self):
            self.m = nn.Linear(2, 2)
            self.optimizers = {"opt": th.optim.SGD(self.m.parameters(), lr=0.01)}
            self.schedulers = {}

    tr = FakeTrainer()
    tr.algo = AlgoNoGetDict()
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    runner.update(metrics={"loss": 1.0})

    recs = [r for r in tr.logger.records if r.prefix == "train/"]
    assert_true(len(recs) >= 1)
    m = recs[-1].metrics
    assert_true(any(str(k).startswith("lr/opt") for k in m.keys()), f"unexpected lr keys: {list(m.keys())}")


def test_nan_guard_triggers_on_nan_scalar() -> None:
    from model_free.common.callbacks.nan_guard_callback import NaNGuardCallback

    cb = NaNGuardCallback(log_prefix="sys/")
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])
    runner.train_start()

    tr.global_update_step = 1
    ok = runner.update(metrics={"loss": float("nan")})
    assert_true(ok is False, "NaNGuard should stop training on NaN")

    recs = [r for r in tr.logger.records if r.prefix == "sys/"]
    assert_true(any(float(r.metrics.get("nan_guard/triggered", 0.0)) == 1.0 for r in recs), "missing nan_guard log")


def test_nan_guard_ignores_finite_values() -> None:
    from model_free.common.callbacks.nan_guard_callback import NaNGuardCallback

    cb = NaNGuardCallback(log_prefix="sys/")
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    ok = runner.update(metrics={"loss": 1.0, "aux": [1.0, 2.0]})
    assert_true(ok is True)

    recs = [r for r in tr.logger.records if r.prefix == "sys/"]
    assert_true(not any(float(r.metrics.get("nan_guard/triggered", 0.0)) == 1.0 for r in recs),
                "nan_guard should not trigger on finite values")


def test_nan_guard_triggers_on_nested_list() -> None:
    from model_free.common.callbacks.nan_guard_callback import NaNGuardCallback

    cb = NaNGuardCallback(log_prefix="sys/", max_depth=3)
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    ok = runner.update(metrics={"vals": [1.0, [2.0, float("inf")]]})
    assert_true(ok is False, "NaNGuard should stop on nested non-finite")


def test_ray_report_callback_no_ray_is_noop() -> None:
    from model_free.common.callbacks.ray_report_callback import RayReportCallback

    cb = RayReportCallback(report_on_update=True, report_on_eval=True)
    tr = FakeTrainer()
    runner = CallbackRunner(tr, [cb])

    tr.global_update_step = 1
    assert_true(runner.update(metrics={"loss": 1.0}) is True)
    assert_true(runner.eval_end({"return_mean": 1.0}) is True)


def test_ray_tune_checkpoint_callback_no_ray_is_noop() -> None:
    from model_free.common.callbacks.ray_tune_checkpoint_callback import RayTuneCheckpointCallback

    cb = RayTuneCheckpointCallback(report_empty_metrics=True)
    tr = FakeTrainer()
    tr.ckpt_dir = mk_tmp_dir("ckpt_")
    _ = CallbackRunner(tr, [cb])

    p = tr.save_checkpoint("dummy_ckpt_dir")
    assert_true(cb.on_checkpoint(tr, p) is True)


# =============================================================================
# Main runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("config_env_info_logs_once", test_config_and_env_info_callback_logs_once),
    ("config_env_info_repeat_when_log_once_false", test_config_and_env_info_callback_allows_repeat_when_log_once_false),
    ("episode_stats_single_env", test_episode_stats_single_env_accumulates_and_logs),
    ("episode_stats_trunc_infer", test_episode_stats_single_env_truncation_inferred_from_info),
    ("episode_stats_batched", test_episode_stats_batched_logs_only_batched_counts),
    ("grad_param_norm_global", test_grad_param_norm_callback_logs_global_norms),
    ("grad_param_norm_per_module", test_grad_param_norm_callback_per_module_keys),
    ("grad_param_norm_no_algo", test_grad_param_norm_callback_no_algo_no_log),
    ("lr_logging_prefers_get_dict", test_lr_logging_callback_prefers_get_lr_dict),
    ("lr_logging_fallback_optimizer", test_lr_logging_callback_fallback_to_optimizer_param_groups),
    ("nan_guard_triggers_nan", test_nan_guard_triggers_on_nan_scalar),
    ("nan_guard_ignores_finite", test_nan_guard_ignores_finite_values),
    ("nan_guard_triggers_nested", test_nan_guard_triggers_on_nested_list),
    ("ray_report_noop", test_ray_report_callback_no_ray_is_noop),
    ("ray_tune_ckpt_noop", test_ray_tune_checkpoint_callback_no_ray_is_noop),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="callbacks")

if __name__ == "__main__":
    raise SystemExit(main())
