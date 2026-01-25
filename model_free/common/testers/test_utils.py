from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Mapping

import os
import math
import tempfile
import sys
import traceback
import numpy as np
import torch as th


class Color:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def enable_ansi_colors_windows() -> None:
    """
    Best-effort: enable ANSI escape sequences on Windows terminals.
    Safe to call on non-Windows platforms.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        # If it fails, just fall back to no-color output
        return


def colorize(text: str, color: str, *, enable: bool = True) -> str:
    if not enable:
        return text
    return f"{color}{text}{Color.RESET}"



# =============================================================================
# Mini test framework (no pytest)
# =============================================================================
class TestFailure(AssertionError):
    pass

class TestSkip(Exception):
    """Raised to mark a skipped test."""
    pass

def assert_true(cond: bool, msg: str = "") -> None:
    if not cond:
        raise TestFailure(msg or "assert_true failed")


def assert_eq(a: Any, b: Any, msg: str = "") -> None:
    if a != b:
        raise TestFailure(msg or f"assert_eq failed: {a!r} != {b!r}")


def assert_in(x: Any, xs: Any, msg: str = "") -> None:
    if x not in xs:
        raise TestFailure(msg or f"assert_in failed: {x!r} not in {xs!r}")


def assert_ge(a: float, b: float, msg: str = "") -> None:
    if float(a) < float(b):
        raise TestFailure(msg or f"assert_ge failed: {a} < {b}")


def assert_close(a: float, b: float, rel_tol: float = 1e-6, msg: str = "") -> None:
    if not math.isclose(float(a), float(b), rel_tol=rel_tol):
        raise TestFailure(msg or f"assert_close failed: {a} vs {b} (rel_tol={rel_tol})")


def mk_tmp_dir(prefix: str = "cbtests_") -> str:
    return tempfile.mkdtemp(prefix=prefix)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise TestFailure(f"file not found: {path}")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def assert_close(a: float, b: float, *, rtol: float = 1e-6, atol: float = 1e-8, msg: str = "assert_close failed") -> None:
    if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
        raise TestFailure(f"{msg}: {a} vs {b} (rtol={rtol}, atol={atol})")


def assert_allclose(
    a: Any,
    b: Any,
    msg: str = "",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """
    Assert two numeric objects are close (supports scalar / ndarray / torch tensor).

    - torch.Tensor -> torch.allclose
    - array-like   -> np.allclose
    - scalar       -> abs diff
    """
    # torch tensors
    if th.is_tensor(a) or th.is_tensor(b):
        ta = a if th.is_tensor(a) else th.as_tensor(a)
        tb = b if th.is_tensor(b) else th.as_tensor(b)
        ok = bool(th.allclose(ta, tb, rtol=rtol, atol=atol))
        if not ok:
            raise TestFailure(msg or f"assert_allclose failed: {ta} != {tb}")
        return

    # numpy/array-like
    try:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        ok = bool(np.allclose(aa, bb, rtol=rtol, atol=atol))
        if not ok:
            raise TestFailure(msg or f"assert_allclose failed: {aa} != {bb}")
        return
    except Exception:
        pass

    # scalar fallback
    fa = float(a)
    fb = float(b)
    if not (abs(fa - fb) <= atol + rtol * abs(fb)):
        raise TestFailure(msg or f"assert_allclose failed: {fa} != {fb}")


def assert_allclose_dict(
    d1: Any,
    d2: Any,
    msg: str = "",
    *,
    keys: Optional[Sequence[str]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """
    Assert mapping values are allclose for specified keys.
    """
    if not isinstance(d1, Mapping) or not isinstance(d2, Mapping):
        raise TestFailure(msg or "assert_allclose_dict failed: inputs are not mappings")

    if keys is None:
        # intersect keys
        keys = sorted(set(d1.keys()) & set(d2.keys()))

    for k in keys:
        if k not in d1 or k not in d2:
            raise TestFailure(msg or f"assert_allclose_dict missing key: {k}")

        assert_allclose(
            d1[k],
            d2[k],
            msg=(msg or f"assert_allclose_dict failed") + f" (key={k})",
            rtol=rtol,
            atol=atol,
        )


def assert_raises(exc_type: type, fn: Callable[[], Any], *, msg: str = "assert_raises failed") -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as e:
        raise TestFailure(f"{msg}: expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise TestFailure(f"{msg}: expected {exc_type.__name__} but no exception raised")


def _shape_of(x: Any) -> Tuple[int, ...]:
    # torch tensor
    if th.is_tensor(x):
        return tuple(int(d) for d in x.shape)

    # numpy / array-like
    try:
        arr = np.asarray(x)
        return tuple(int(d) for d in arr.shape)
    except Exception:
        # scalar-like
        return ()


def assert_shape(x: Any, shape: Sequence[int], msg: str = "") -> None:
    got = _shape_of(x)
    exp = tuple(int(s) for s in shape)
    if got != exp:
        raise TestFailure(msg or f"assert_shape failed: got {got}, expected {exp}")


def assert_finite(x: Any, msg: str = "") -> None:
    """
    Assert all values are finite (no NaN/Inf).

    Works for:
      - torch tensors
      - numpy arrays / array-like
      - Python scalars
    """
    # torch tensor
    if th.is_tensor(x):
        ok = bool(th.isfinite(x).all().item())
        if not ok:
            raise TestFailure(msg or "assert_finite failed: tensor has NaN/Inf")
        return

    # numpy / scalar
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        # last resort: try float conversion
        try:
            v = float(x)
        except Exception:
            raise TestFailure(msg or f"assert_finite failed: cannot convert {type(x)} to numeric")
        if not np.isfinite(v):
            raise TestFailure(msg or f"assert_finite failed: {v} is not finite")
        return

    if not np.all(np.isfinite(arr)):
        raise TestFailure(msg or f"assert_finite failed: array has NaN/Inf, shape={arr.shape}")
    

def seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)



def run_tests(
    tests: Sequence[Tuple[str, Callable[[], Any]]],
    *,
    argv: Optional[List[str]] = None,
    suite_name: str = "writers",
) -> int:
    """
    Run a list of zero-arg test callables and print colored PASS/FAIL + summary.

    Parameters
    ----------
    tests : Sequence[Tuple[str, Callable[[], Any]]]
        List of (test_name, test_fn). Each test_fn must be callable with no args.
    argv : Optional[List[str]]
        CLI args (excluding program name). If None, uses sys.argv[1:].
        If argv[0] exists, it is used as a substring filter on test names.
    suite_name : str
        Label used in console output, e.g. "writers" or "callbacks".

    Returns
    -------
    int
        0 if all passed, 1 if any failed, 2 if filter matched no tests.
    """
    argv = sys.argv[1:] if argv is None else argv

    # optional: filter tests by substring
    filt = argv[0] if argv else ""

    selected = [(n, f) for (n, f) in tests if (not filt or filt in n)]
    if not selected:
        print(f"[{suite_name}] No tests matched filter: {filt!r}")
        return 2

    passed: List[str] = []
    failed: List[Tuple[str, str]] = []  # (test_name, error_summary)

    print(f"[{suite_name}] Running {len(selected)} tests" + (f" (filter={filt!r})" if filt else ""))

    for name, fn in selected:
        try:
            fn()
            passed.append(name)
            print(colorize(f" [ PASS ] {name}", Color.GREEN))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failed.append((name, err))
            print(colorize(f" [ FAIL ] {name}: {err}", Color.RED))
            traceback.print_exc()

    # -----------------------------
    # Summary (grouped results)
    # -----------------------------
    print()
    print(colorize(f"[{suite_name}] ========================= Summary =========================", Color.CYAN))

    if passed:
        print(colorize(f"[{suite_name}] Passed ({len(passed)}):", Color.GREEN))
        for n in passed:
            print(colorize(f"  - {n}", Color.GREEN))
    else:
        print(colorize(f"[{suite_name}] Passed (0)", Color.GREEN))

    print()

    if failed:
        print(colorize(f"[{suite_name}] Failed ({len(failed)}):", Color.RED))
        for n, err in failed:
            print(colorize(f"  - {n}", Color.RED))
            print(colorize(f"      {err}", Color.RED))
    else:
        print(colorize(f"[{suite_name}] Failed (0)", Color.GREEN))

    print()

    return 0 if len(failed) == 0 else 1
