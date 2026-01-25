from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


def _bootstrap_sys_path() -> None:
    """
    Ensure `model_free` package is importable when running this file directly.

    Walks up parent dirs looking for a `model_free/` folder and prepends that
    directory to sys.path.
    """
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

from model_free.common.testers.test_utils import (  # noqa: E402
    seed_all,
    run_tests,
    assert_eq,
    assert_true,
    TestFailure,
)

from model_free.baselines.q_learning.qrdqn.qrdqn import qrdqn  # noqa: E402
from model_free.baselines.q_learning.qrdqn.head import QRDQNHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
class DummyDiscreteSpace:
    """Minimal Discrete(n) space with sample()."""

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = int(n)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> int:
        return int(self._rng.randint(0, self.n))


class DummyBoxSpace:
    """Minimal Box(shape,) space with sample()."""

    def __init__(self, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.low = np.full(self.shape, float(low), dtype=np.float32)
        self.high = np.full(self.shape, float(high), dtype=np.float32)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> np.ndarray:
        return self._rng.uniform(low=self.low, high=self.high).astype(np.float32)


@dataclass
class DummyCartPoleLikeEnv:
    """
    A tiny episodic env mimicking CartPole-like shapes:
      - obs: (4,) float32
      - actions: {0,1}
    Episode ends after `horizon` steps.
    Reward = 1.0 per step.
    """

    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(4,), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyDiscreteSpace(n=2, seed=self.seed + 123)
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((4,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()  # re-init RNG/spaces deterministically
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)

        noise = self._rng.randn(4).astype(np.float32) * 0.01
        delta = (0.05 if a == 1 else -0.05)
        self._obs = np.clip(self._obs + delta + noise, -1.0, 1.0).astype(np.float32)

        self._t += 1
        reward = 1.0
        terminated = False
        truncated = self._t >= int(self.horizon)
        info: Dict[str, Any] = {"TimeLimit.truncated": bool(truncated)}
        return self._obs.copy(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        return


# =============================================================================
# Local test helpers
# =============================================================================
def _assert_raises_any(exc_types: Tuple[type, ...], fn: Callable[[], Any], msg: str = "") -> None:
    """
    Like assert_raises, but accepts multiple exception types.

    This is useful because OffPolicyAlgorithm.on_env_step may raise KeyError
    (missing dict key access) or ValueError (explicit validation), depending on
    how strictly input validation is implemented.
    """
    try:
        fn()
    except exc_types:
        return
    except Exception as e:
        raise TestFailure(f"{msg}: expected {exc_types}, got {type(e).__name__}: {e}") from e
    raise TestFailure(f"{msg}: expected {exc_types}, got no exception")


def _state_vector(module: th.nn.Module) -> th.Tensor:
    """Flatten module parameters into a single CPU float32 vector."""
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(*, device: str = "cpu", use_per: bool = True):
    """
    Build a small QRDQN algo for smoke/integration tests.

    The dummy env uses obs_dim=4, n_actions=2.
    """
    return qrdqn(
        obs_dim=4,
        n_actions=2,
        device=device,
        n_quantiles=32,
        hidden_sizes=(64, 64),
        buffer_size=10_000,
        batch_size=32,
        warmup_env_steps=50,
        update_after=50,
        update_every=1,
        utd=1.0,
        gradient_steps=1,
        max_updates_per_call=10,
        use_per=bool(use_per),
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_final=0.8,
        per_beta_anneal_steps=1_000,
        # core
        gamma=0.99,
        target_update_interval=50,
        tau=0.0,
        double_dqn=True,
        max_grad_norm=10.0,
        use_amp=False,
        # exploration (may or may not be used inside OffPolicyAlgorithm.act)
        exploration_eps=0.2,
        exploration_eps_final=0.05,
        exploration_eps_anneal_steps=1_000,
        exploration_eval_eps=0.0,
    )


def _rollout_steps(
    algo: Any,
    env: Any,
    *,
    n_steps: int,
    seed: int = 0,
    do_updates: bool = True,
) -> Dict[str, Any]:
    """
    Run n_steps of interaction and optionally train.

    This function assumes:
      - algo.act(obs, deterministic=False) is supported
      - algo.on_env_step({...}) consumes standard transition fields

    Returns:
      {"num_updates": int, "last_metrics": Any}
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)
        a_int = int(a)

        next_obs, r, terminated, truncated, _info = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_int,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
            }
        )

        if do_updates and algo.ready_to_update():
            last_metrics = algo.update()
            num_updates += 1

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return {"num_updates": int(num_updates), "last_metrics": last_metrics}


# =============================================================================
# Tests: QRDQNHead
# =============================================================================
def test_qrdqn_head_quantiles_and_q_values_shapes():
    seed_all(0)

    head = QRDQNHead(
        obs_dim=4,
        n_actions=2,
        n_quantiles=32,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # quantiles: (B,N,A)
    z = head.quantiles(obs)
    assert_true(th.is_tensor(z), "head.quantiles should return torch.Tensor")
    assert_eq(tuple(z.shape), (1, 32, 2))

    # expected Q: (B,A)
    q = head.q_values(obs)
    assert_true(th.is_tensor(q), "head.q_values should return torch.Tensor")
    assert_eq(tuple(q.shape), (1, 2))


def test_qrdqn_head_act_shape_and_range():
    seed_all(0)

    head = QRDQNHead(
        obs_dim=4,
        n_actions=2,
        n_quantiles=16,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # act() should return a discrete action index (int-like)
    a = head.act(obs, deterministic=True)
    a_int = int(a)
    assert_true(a_int in (0, 1), f"action out of range: {a_int}")


def test_qrdqn_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = QRDQNHead(
        obs_dim=4,
        n_actions=2,
        n_quantiles=16,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        device="cpu",
    )

    _ = head1.q_values(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "qrdqn_head_ckpt")
        head1.save(path)

        head2 = QRDQNHead(
            obs_dim=4,
            n_actions=2,
            n_quantiles=16,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            dueling_mode=False,
            device="cpu",
        )
        head2.load(path)

        sd1 = head1.q.state_dict()
        sd2 = head2.q.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"online q mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration
# =============================================================================
def test_qrdqn_on_env_step_requires_standard_fields():
    """
    Missing required transition keys must raise (KeyError or ValueError),
    depending on how strictly OffPolicyAlgorithm validates inputs.
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=20, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)
    a = int(env.action_space.sample())
    next_obs, r, terminated, truncated, _ = env.step(a)
    done = bool(terminated or truncated)

    # Missing 'next_obs' and 'done'
    transition_missing = {
        "obs": obs,
        "action": a,
        "reward": float(r),
    }

    _assert_raises_any((KeyError, ValueError), lambda: algo.on_env_step(transition_missing),
                       msg="on_env_step should fail on missing required keys")

    # Missing 'obs'
    transition_missing2 = {
        "action": a,
        "reward": float(r),
        "next_obs": next_obs,
        "done": done,
    }
    _assert_raises_any((KeyError, ValueError), lambda: algo.on_env_step(transition_missing2),
                       msg="on_env_step should fail on missing obs")

    env.close()


def test_qrdqn_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    out = _rollout_steps(algo, env, n_steps=1500, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # basic bookkeeping keys from OffPolicyAlgorithm.update()
    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")

    # core metrics
    assert_true("loss/q" in m, "missing loss/q")
    assert_true(np.isfinite(float(m["loss/q"])), "loss/q not finite")


def test_qrdqn_smoke_run_with_per_reports_beta():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=True)
    out = _rollout_steps(algo, env, n_steps=1800, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # PER bookkeeping from update_once() path
    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")


def test_qrdqn_replay_batch_contract():
    """
    Basic replay contract for discrete off-policy:
      - batch has observations/actions/rewards/next_observations/dones
      - shapes are consistent with QRDQNCore expectations
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    _ = _rollout_steps(algo, env, n_steps=400, seed=0, do_updates=False)

    assert_true(algo.buffer is not None, "buffer should be initialized")
    assert_true(algo.buffer.size >= algo.batch_size, "buffer must have enough samples")

    batch = algo.buffer.sample(algo.batch_size)

    assert_true(hasattr(batch, "observations"), "batch missing observations")
    assert_true(hasattr(batch, "actions"), "batch missing actions")
    assert_true(hasattr(batch, "rewards"), "batch missing rewards")
    assert_true(hasattr(batch, "next_observations"), "batch missing next_observations")
    assert_true(hasattr(batch, "dones"), "batch missing dones")

    # minimal shape checks
    obs = batch.observations
    act = batch.actions
    rew = batch.rewards
    nxt = batch.next_observations
    done = batch.dones

    if th.is_tensor(obs):
        assert_true(tuple(obs.shape) == (algo.batch_size, 4), f"obs shape bad: {tuple(obs.shape)}")
    if th.is_tensor(nxt):
        assert_true(tuple(nxt.shape) == (algo.batch_size, 4), f"next_obs shape bad: {tuple(nxt.shape)}")

    # actions may be (B,) or (B,1) depending on buffer implementation
    if th.is_tensor(act):
        shp = tuple(act.shape)
        assert_true(shp in ((algo.batch_size,), (algo.batch_size, 1)), f"action shape bad: {shp}")

    if th.is_tensor(rew):
        shp = tuple(rew.shape)
        assert_true(shp in ((algo.batch_size,), (algo.batch_size, 1)), f"reward shape bad: {shp}")

    if th.is_tensor(done):
        shp = tuple(done.shape)
        assert_true(shp in ((algo.batch_size,), (algo.batch_size, 1)), f"done shape bad: {shp}")

    env.close()


def test_qrdqn_parameters_change_after_update():
    """
    Ensure at least one update actually changes parameters.
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_int = int(a)

        next_obs, r, terminated, truncated, _ = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_int,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
            }
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    # online quantile network is head.q
    q_before = _state_vector(algo.head.q)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    q_after = _state_vector(algo.head.q)

    dq = float(th.linalg.vector_norm(q_after - q_before).item()) if q_before.numel() else 0.0
    assert_true(dq > 0.0, f"expected params to change; dq={dq}")

    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("qrdqn_head_quantiles_and_q_values_shapes", test_qrdqn_head_quantiles_and_q_values_shapes),
    ("qrdqn_head_act_shape_and_range", test_qrdqn_head_act_shape_and_range),
    ("qrdqn_head_save_and_load_roundtrip", test_qrdqn_head_save_and_load_roundtrip),
    ("qrdqn_on_env_step_requires_standard_fields", test_qrdqn_on_env_step_requires_standard_fields),
    ("qrdqn_smoke_run_no_per_runs_and_updates", test_qrdqn_smoke_run_no_per_runs_and_updates),
    ("qrdqn_smoke_run_with_per_reports_beta", test_qrdqn_smoke_run_with_per_reports_beta),
    ("qrdqn_replay_batch_contract", test_qrdqn_replay_batch_contract),
    ("qrdqn_parameters_change_after_update", test_qrdqn_parameters_change_after_update),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="qrdqn")


if __name__ == "__main__":
    raise SystemExit(main())
