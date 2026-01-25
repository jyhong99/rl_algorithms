from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import importlib
from typing import Callable, Dict, List, Tuple


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
    TestSkip,
    run_tests,
    assert_eq,
    assert_in,
    assert_true,
    assert_file_exists,
)
from model_free.common.testers.test_harness import TempDir, MemoryWriter




# =============================================================================
# Import targets (your package)
# =============================================================================
def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise TestFailure(f"import failed for {name}: {type(e).__name__}: {e}") from e


# =============================================================================
# Tests: JSONLWriter
# =============================================================================
def jsonl_writer_writes_lines() -> None:
    mod = _import("model_free.common.loggers.jsonl_writer")
    JSONLWriter = getattr(mod, "JSONLWriter")

    with TempDir() as d:
        w = JSONLWriter(d, filename="m.jsonl")
        w.write({"a": 1.0, "step": 3.0})
        w.write({"b": 2.5, "step": 4.0})
        w.flush()
        w.close()

        p = os.path.join(d, "m.jsonl")
        assert_file_exists(p)

        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        assert_eq(len(lines), 2, "jsonl line count mismatch")
        o0 = json.loads(lines[0])
        o1 = json.loads(lines[1])
        assert_eq(o0["a"], 1.0)
        assert_eq(o0["step"], 3.0)
        assert_eq(o1["b"], 2.5)
        assert_eq(o1["step"], 4.0)


# =============================================================================
# Tests: CSVWriter (wide)
# =============================================================================
def csv_writer_wide_schema_and_rows() -> None:
    mod = _import("model_free.common.loggers.csv_writer")
    CSVWriter = getattr(mod, "CSVWriter")

    with TempDir() as d:
        w = CSVWriter(
            d,
            wide=True,
            long=False,
            wide_filename="wide.csv",
            long_filename="long.csv",
        )
        w.write({"step": 1.0, "wall_time": 0.0, "timestamp": 0.0, "k1": 10.0, "k2": 20.0})
        w.write({"step": 2.0, "wall_time": 0.0, "timestamp": 0.0, "k1": 11.0, "k2": 21.0, "NEW": 999.0})
        w.flush()
        w.close()

        p = os.path.join(d, "wide.csv")
        assert_file_exists(p)

        with open(p, "r", encoding="utf-8") as f:
            rows = [r.strip() for r in f.readlines() if r.strip()]

        # header + 2 data rows
        assert_true(len(rows) >= 3, "wide csv should have header + 2 rows")
        header = rows[0].split(",")
        assert_in("k1", header)
        assert_in("k2", header)
        # NEW should be ignored (not in schema), but header fixed from first row
        assert_true("NEW" not in header, "wide schema must be fixed on first row")


# =============================================================================
# Tests: CSVWriter (long)
# =============================================================================
def csv_writer_long_schema_and_rows() -> None:
    mod = _import("model_free.common.loggers.csv_writer")
    CSVWriter = getattr(mod, "CSVWriter")

    with TempDir() as d:
        w = CSVWriter(
            d,
            wide=False,
            long=True,
            wide_filename="wide.csv",
            long_filename="long.csv",
        )
        # include meta + two metrics
        w.write({"step": 7.0, "wall_time": 1.0, "timestamp": 2.0, "loss": 0.1, "acc": 0.9})
        w.flush()
        w.close()

        p = os.path.join(d, "long.csv")
        assert_file_exists(p)

        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        # header + 2 metric rows expected
        assert_true(len(lines) >= 3, "long csv should have header + rows")
        assert_eq(lines[0], "step,wall_time,timestamp,key,value", "long csv header mismatch")

        # parse metric rows (order not guaranteed)
        data = [ln.split(",") for ln in lines[1:]]
        keys = [x[3] for x in data if len(x) >= 5]
        assert_in("loss", keys)
        assert_in("acc", keys)


# =============================================================================
# Tests: SafeWriter
# =============================================================================
def safe_writer_swallows_exceptions() -> None:
    base_mod = _import("model_free.common.loggers.base_writer")
    SafeWriter = getattr(base_mod, "SafeWriter")
    Writer = getattr(base_mod, "Writer")

    class BadWriter(Writer):
        def write(self, row: Dict[str, float]) -> None:
            raise RuntimeError("boom")
        def flush(self) -> None:
            raise RuntimeError("boom")
        def close(self) -> None:
            raise RuntimeError("boom")

    w = SafeWriter(BadWriter(), name="bad")
    # should not raise
    w.write({"a": 1.0})
    w.flush()
    w.close()


# =============================================================================
# Tests: TensorBoardWriter (stub-based)
# =============================================================================
def tensorboard_writer_stub_smoke() -> None:
    tb_mod = _import("model_free.common.loggers.tensorboard_writer")

    # If module hard-failed to import SummaryWriter, we still can test by stubbing it
    calls = {"add_scalar": 0, "flush": 0, "close": 0}

    class DummySummaryWriter:
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir
        def add_scalar(self, k: str, v: float, global_step: int) -> None:
            calls["add_scalar"] += 1
        def flush(self) -> None:
            calls["flush"] += 1
        def close(self) -> None:
            calls["close"] += 1

    # Monkeypatch module attribute SummaryWriter
    old = getattr(tb_mod, "SummaryWriter", None)
    setattr(tb_mod, "SummaryWriter", DummySummaryWriter)
    try:
        TensorBoardWriter = getattr(tb_mod, "TensorBoardWriter")

        with TempDir() as d:
            w = TensorBoardWriter(d)
            w.write({"step": 5.0, "wall_time": 0.0, "timestamp": 0.0, "train/loss": 1.23})
            w.flush()
            w.close()

        assert_true(calls["add_scalar"] >= 1, "TensorBoardWriter must call add_scalar")
        assert_true(calls["flush"] >= 1, "TensorBoardWriter must flush")
        assert_true(calls["close"] >= 1, "TensorBoardWriter must close")
    finally:
        setattr(tb_mod, "SummaryWriter", old)


# =============================================================================
# Tests: WandBWriter (offline smoke)
# =============================================================================
def wandb_writer_offline_smoke() -> None:
    wb_mod = _import("model_free.common.loggers.wandb_writer")
    WandBWriter = getattr(wb_mod, "WandBWriter")
    wandb_obj = getattr(wb_mod, "wandb", None)

    if wandb_obj is None:
        raise TestSkip("wandb not installed; skipping")

    # Force offline, silence, and local dir usage
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")

    with TempDir() as d:
        # Some wandb versions still want a "project" name
        w = WandBWriter(run_dir=d, project="writers_test_project", mode="offline")
        w.write({"step": 1.0, "wall_time": 0.0, "timestamp": 0.0, "loss": 0.123})
        w.flush()
        w.close()

    # If it didn't raise, it's enough for a smoke test


# =============================================================================
# Tests: Logger core semantics (memory writer)
# =============================================================================
def logger_core_semantics() -> None:
    log_mod = _import("model_free.common.loggers.logger")
    Logger = getattr(log_mod, "Logger")

    mw = MemoryWriter()

    with TempDir() as d:
        # console_every=0 to avoid stdout dependence; flush_every=1 to exercise flush
        lg = Logger(
            log_dir=d,
            exp_name="exp",
            writers=[mw],
            console_every=0,
            flush_every=1,
            drop_non_finite=True,
            strict=True,
        )

        # Fake trainer for bind_trainer step inference
        class T:
            global_env_step = 0
        t = T()
        lg.bind_trainer(t)

        # 1) prefix join + meta keys
        t.global_env_step = 10
        lg.log({"loss": 1.0}, prefix="train")
        assert_true(len(mw.rows) >= 1, "logger should write at least one row")
        row0 = mw.rows[-1]
        assert_in("train/loss", row0, "prefix join failed")
        assert_in("step", row0)
        assert_in("wall_time", row0)
        assert_in("timestamp", row0)
        assert_eq(int(row0["step"]), 10, "step inference mismatch")

        # 2) drop_non_finite should drop NaN/Inf metrics but still write meta
        t.global_env_step = 11
        lg.log({"bad": float("nan"), "ok": 2.0, "inf": float("inf")}, prefix="train")
        row1 = mw.rows[-1]
        assert_in("train/ok", row1)
        assert_true("train/bad" not in row1, "NaN should be dropped when drop_non_finite=True")
        assert_true("train/inf" not in row1, "Inf should be dropped when drop_non_finite=True")

        # 3) key throttling: only log train/loss every 2 steps
        lg.set_key_every({"train/loss": 2})
        t.global_env_step = 12
        lg.log({"loss": 5.0}, prefix="train")  # step 12 => should log
        row2 = mw.rows[-1]
        assert_in("train/loss", row2)

        t.global_env_step = 13
        lg.log({"loss": 6.0}, prefix="train")  # step 13 => should NOT log that key
        row3 = mw.rows[-1]
        assert_true("train/loss" not in row3, "throttled key should not appear on odd step")

        lg.close()

    assert_true(mw.flush_calls >= 1, "logger should flush writers (flush_every=1)")
    assert_true(mw.close_calls >= 1, "logger should close writers")


# =============================================================================
# Register tests
# =============================================================================
TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("jsonl_writer_writes_lines", jsonl_writer_writes_lines),
    ("csv_writer_wide_schema_and_rows", csv_writer_wide_schema_and_rows),
    ("csv_writer_long_schema_and_rows", csv_writer_long_schema_and_rows),
    ("safe_writer_swallows_exceptions", safe_writer_swallows_exceptions),
    ("tensorboard_writer_stub_smoke", tensorboard_writer_stub_smoke),
    ("wandb_writer_offline_smoke", wandb_writer_offline_smoke),
    ("logger_core_semantics", logger_core_semantics),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="loggers")

if __name__ == "__main__":
    raise SystemExit(main())
