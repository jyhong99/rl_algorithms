from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
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
    assert_raises,
)

from model_free.baselines.policy_gradients.off_policy.acer.acer import acer  # noqa: E402
from model_free.baselines.policy_gradients.off_policy.acer.head import ACERHead  # noqa: E402


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
    A tiny episodic env mimicking CartPole shapes:
      - obs: (4,) float32
      - actions: {0,1}
    Episode ends after horizon steps.
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
            self.__post_init__()  # re-init rng/spaces deterministically
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0  # start near zero
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)
        # simple deterministic-ish dynamics with noise
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
# Helpers
# =============================================================================
def _uniform_behavior(n_actions: int, action: int) -> Tuple[float, np.ndarray]:
    probs = np.full((int(n_actions),), 1.0 / float(n_actions), dtype=np.float32)
    a = int(action)
    a = max(0, min(a, int(n_actions) - 1))
    logp = float(np.log(probs[a] + 1e-8))
    return logp, probs


def _behavior_from_policy(head: Any, obs: Any, action: int) -> Tuple[float, np.ndarray]:
    with th.no_grad():
        p = head.probs(obs)
        if th.is_tensor(p):
            p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    p = np.clip(p, 1e-8, 1.0)
    p = p / float(p.sum())
    a = int(action)
    a = max(0, min(a, p.shape[0] - 1))
    logp = float(np.log(p[a] + 1e-8))
    return logp, p.astype(np.float32)


def _state_vector(module: th.nn.Module) -> th.Tensor:
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(*, device: str = "cpu", use_per: bool = True):
    # dummy env uses obs_dim=4, n_actions=2
    return acer(
        obs_dim=4,
        n_actions=2,
        device=device,
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
        critic_is=False,
        entropy_coef=0.0,
        max_grad_norm=10.0,
        use_amp=False,
    )


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)
        a_int = int(a)

        if algo.warmup_steps > 0 and algo._env_steps < algo.warmup_steps:
            beh_logp, beh_probs = _uniform_behavior(algo.n_actions, a_int)
        else:
            beh_logp, beh_probs = _behavior_from_policy(algo.head, obs, a_int)

        next_obs, r, terminated, truncated, _info = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_int,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
                "behavior_logp": float(beh_logp),
                "behavior_probs": beh_probs,
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
# Tests: ACERHead
# =============================================================================
def test_acer_head_probs_and_logp_shapes():
    seed_all(0)

    # If this fails with unexpected kwarg 'activation', fix ACERHead to pass the
    # correct keyword to QNetwork/DoubleQNetwork (likely activation_fn=...).
    head = ACERHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        double_q=True,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    p = head.probs(obs)
    assert_true(th.is_tensor(p), "head.probs should return torch.Tensor")
    assert_eq(tuple(p.shape), (1, 2))
    assert_true(abs(float(p.sum().detach().cpu().item()) - 1.0) < 1e-5, "probs must sum to 1")

    act = th.tensor([0], dtype=th.int64)
    lp = head.logp(obs, act)
    assert_true(th.is_tensor(lp), "head.logp should return torch.Tensor")
    assert_eq(tuple(lp.shape), (1, 1))


def test_acer_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = ACERHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        double_q=True,
        device="cpu",
    )

    _ = head1.probs(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "acer_head_ckpt")
        head1.save(path)

        head2 = ACERHead(
            obs_dim=4,
            n_actions=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            dueling_mode=False,
            double_q=True,
            device="cpu",
        )
        head2.load(path)

        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration
# =============================================================================
def test_acer_requires_behavior_fields_when_storage_enabled():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=20, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)
    a = int(env.action_space.sample())
    next_obs, r, terminated, truncated, _ = env.step(a)
    done = bool(terminated or truncated)

    transition_missing = {
        "obs": obs,
        "action": a,
        "reward": float(r),
        "next_obs": next_obs,
        "done": done,
    }
    assert_raises(ValueError, lambda: algo.on_env_step(transition_missing))
    env.close()


def test_acer_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    out = _rollout_steps(algo, env, n_steps=1500, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")
    assert_true("loss/actor" in m, "missing loss/actor")
    assert_true("loss/critic" in m, "missing loss/critic")
    assert_true(np.isfinite(float(m["loss/actor"])), "loss/actor not finite")
    assert_true(np.isfinite(float(m["loss/critic"])), "loss/critic not finite")


def test_acer_smoke_run_with_per_reports_beta():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=True)
    out = _rollout_steps(algo, env, n_steps=1800, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")


def test_acer_replay_batch_contract_contains_behavior_fields():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    _ = _rollout_steps(algo, env, n_steps=400, seed=0, do_updates=False)

    assert_true(algo.buffer is not None, "buffer should be initialized")
    assert_true(algo.buffer.size >= algo.batch_size, "buffer must have enough samples")

    batch = algo.buffer.sample(algo.batch_size)

    assert_true(
        hasattr(batch, "behavior_logp") or hasattr(batch, "logp"),
        "batch must expose behavior_logp (preferred) or logp",
    )
    assert_true(hasattr(batch, "behavior_probs"), "batch must expose behavior_probs")

    blp = getattr(batch, "behavior_logp", None)
    if blp is None:
        blp = getattr(batch, "logp", None)
    assert_true(blp is not None, "behavior logp missing unexpectedly")
    if th.is_tensor(blp):
        shp = tuple(blp.shape)
        assert_true(shp == (algo.batch_size,) or shp == (algo.batch_size, 1), f"behavior_logp shape bad: {shp}")

    bp = getattr(batch, "behavior_probs", None)
    assert_true(bp is not None, "behavior_probs missing unexpectedly")
    if th.is_tensor(bp):
        shp = tuple(bp.shape)
        assert_true(
            len(shp) == 2 and shp[0] == algo.batch_size and shp[1] == int(algo.n_actions),
            f"behavior_probs shape bad: {shp} (A={algo.n_actions})",
        )

    env.close()


def test_acer_parameters_change_after_update():
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

        if algo.warmup_steps > 0 and algo._env_steps < algo.warmup_steps:
            beh_logp, beh_probs = _uniform_behavior(algo.n_actions, a_int)
        else:
            beh_logp, beh_probs = _behavior_from_policy(algo.head, obs, a_int)

        next_obs, r, terminated, truncated, _ = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_int,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
                "behavior_logp": float(beh_logp),
                "behavior_probs": beh_probs,
            }
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    actor_before = _state_vector(algo.head.actor)
    critic_before = _state_vector(algo.head.critic)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    actor_after = _state_vector(algo.head.actor)
    critic_after = _state_vector(algo.head.critic)

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    dc = float(th.linalg.vector_norm(critic_after - critic_before).item()) if critic_before.numel() else 0.0

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("acer_head_probs_and_logp_shapes", test_acer_head_probs_and_logp_shapes),
    ("acer_head_save_and_load_roundtrip", test_acer_head_save_and_load_roundtrip),
    ("acer_requires_behavior_fields_when_storage_enabled", test_acer_requires_behavior_fields_when_storage_enabled),
    ("acer_smoke_run_no_per_runs_and_updates", test_acer_smoke_run_no_per_runs_and_updates),
    ("acer_smoke_run_with_per_reports_beta", test_acer_smoke_run_with_per_reports_beta),
    ("acer_replay_batch_contract_contains_behavior_fields", test_acer_replay_batch_contract_contains_behavior_fields),
    ("acer_parameters_change_after_update", test_acer_parameters_change_after_update),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="acer")


if __name__ == "__main__":
    raise SystemExit(main())
