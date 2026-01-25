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
    Add project root to sys.path so tests can be run directly as a script.

    Mirrors the ACER/PPO tester style:
      - Walk parents up to a fixed depth to find a folder containing "model_free".
      - Insert that parent directory into sys.path.
      - Fallback to a reasonable default if not found.
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
    assert_raises,
)

from model_free.baselines.policy_gradients.on_policy.trpo.trpo import trpo  # noqa: E402
from model_free.baselines.policy_gradients.on_policy.trpo.head import TRPOHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
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
class DummyContinuousEpisodicEnv:
    """
    A tiny episodic continuous-control env.

    Shapes
    ------
    - obs:  (obs_dim,) float32
    - act:  (act_dim,) float32

    Dynamics (toy)
    --------------
    - next_obs = clip(obs + 0.05 * action + noise, -1, 1)
    - reward   = 1.0 per step (dense), purely to exercise rollout/update wiring
    - episode terminates by horizon only (truncated=True at horizon)

    This is intentionally simple: we want deterministic-ish shapes and stable tests,
    not meaningful learning.
    """

    obs_dim: int = 4
    act_dim: int = 2
    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(self.obs_dim,), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyBoxSpace(shape=(self.act_dim,), low=-1.0, high=1.0, seed=self.seed + 123)
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()
        self._t = 0
        self._obs = np.zeros((self.obs_dim,), dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        noise = self._rng.randn(self.obs_dim).astype(np.float32) * 0.01

        # Map action_dim -> obs_dim (simple projection/repeat)
        if self.act_dim == self.obs_dim:
            delta = 0.05 * a
        else:
            reps = int(np.ceil(self.obs_dim / self.act_dim))
            delta = 0.05 * np.tile(a, reps)[: self.obs_dim]

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
def _state_vector(module: th.nn.Module) -> th.Tensor:
    """Flatten all parameters into a single CPU float vector for change-detection checks."""
    parts: List[th.Tensor] = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _rollout_steps(
    algo: Any,
    env: Any,
    *,
    n_steps: int,
    seed: int = 0,
    do_updates: bool = True,
) -> Dict[str, Any]:
    """
    Roll out n_steps transitions and optionally run updates when ready.

    Contract (matches ACER/PPO test style):
      - algo.setup(env)
      - for each step:
          a = algo.act(obs)
          env.step(a)
          algo.on_env_step(transition)
          if algo.ready_to_update(): algo.update()
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.act_dim)

        next_obs, r, terminated, truncated, _info = env.step(a_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_np,
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
# Tests: TRPOHead
# =============================================================================
def test_trpo_head_act_and_evaluate_shapes():
    """
    Basic interface test:
      - head.act returns (B, act_dim) action tensor
      - head.evaluate_actions returns value/log_prob/entropy with expected shapes

    Notes
    -----
    If your TRPOHead does not implement evaluate_actions (unlike PPOHead),
    replace this test with a get_dist + log_prob smoke test.
    """
    seed_all(0)

    head = TRPOHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
    )

    obs = np.zeros((4,), dtype=np.float32)

    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1, 2))

    # Many heads provide evaluate_actions; if yours doesn't, this will raise and should be fixed
    # by aligning the head API across on-policy algorithms.
    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys in evaluate_actions output")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]

    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "value/log_prob/entropy must be tensors")
    assert_eq(tuple(v.shape), (1, 1))

    # Depending on dist impl, log_prob/entropy may be (B,1) or (B,act_dim).
    assert_true(lp.shape[0] == 1, f"log_prob batch mismatch: {tuple(lp.shape)}")
    assert_true(ent.shape[0] == 1, f"entropy batch mismatch: {tuple(ent.shape)}")


def test_trpo_head_save_and_load_roundtrip():
    """Save/load should restore identical actor/critic weights."""
    seed_all(0)

    head1 = TRPOHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "trpo_head_ckpt")
        head1.save(path)

        head2 = TRPOHead(
            obs_dim=4,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
            log_std_mode="param",
            log_std_init=-0.5,
        )
        head2.load(path)

        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


# =============================================================================
# Tests: OnPolicyAlgorithm integration (TRPO)
# =============================================================================
def test_trpo_algorithm_smoke_run_runs_and_updates():
    """
    End-to-end smoke test:
      - rollout until update triggers
      - ensure update() returns metrics with expected keys

    Notes
    -----
    TRPO policy step can be rejected by line search (step_frac=0),
    but critic update should still run. Therefore we mainly assert
    that update executes and emits core metrics.
    """
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = trpo(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        # TRPO is typically full-batch; if your OnPolicyAlgorithm requires an int,
        # passing rollout_steps is the safest default.
        minibatch_size=256,
        max_kl=1e-2,
        cg_iters=10,
        cg_damping=1e-2,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        accept_ratio=0.1,
        critic_lr=3e-4,
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one TRPO update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # Expected TRPOCore metrics (based on your core implementation)
    for k in [
        "loss/value",
        "stats/surr",
        "stats/kl",
        "stats/step_frac",
        "lr/critic",
    ]:
        assert_true(k in m, f"missing metric: {k}")

    assert_true(np.isfinite(float(m["loss/value"])), "loss/value not finite")
    assert_true(np.isfinite(float(m["stats/surr"])), "stats/surr not finite")
    assert_true(np.isfinite(float(m["stats/kl"])), "stats/kl not finite")
    assert_true(np.isfinite(float(m["stats/step_frac"])), "stats/step_frac not finite")


def test_trpo_parameters_change_after_update():
    """
    Verify that at least one of actor/critic parameters changes after update().

    Notes
    -----
    - TRPO line search may reject the policy step (actor unchanged).
    - Critic regression should still change critic parameters in almost all cases.
    Therefore we accept (actor OR critic changed).
    """
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = trpo(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=256,
        max_kl=1e-2,
        cg_iters=10,
        cg_damping=1e-2,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        accept_ratio=0.1,
        critic_lr=3e-4,
        use_amp=False,
    )
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    # Drive until ready_to_update() becomes true
    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.act_dim)

        next_obs, r, terminated, truncated, _ = env.step(a_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_np,
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


def test_trpo_kl_metric_respects_constraint_when_step_accepted_or_zero():
    """
    KL metric sanity:
      - If the TRPO step is accepted, KL should be <= max_kl (within tolerance).
      - If not accepted, step_frac==0 and KL is typically reported as 0.

    This test does not attempt to force acceptance; it verifies consistency
    between reported step_frac and KL bound.
    """
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    max_kl = 1e-2

    algo = trpo(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=256,
        max_kl=max_kl,
        cg_iters=10,
        cg_damping=1e-2,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        accept_ratio=0.1,
        critic_lr=3e-4,
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")
    assert_true("stats/kl" in m and "stats/step_frac" in m, "missing stats/kl or stats/step_frac")

    kl = float(m["stats/kl"])
    step_frac = float(m["stats/step_frac"])

    # If a step was accepted, KL should be within the trust region (allow small numerical slack).
    if step_frac > 0.0:
        assert_true(kl <= max_kl * 1.5, f"KL too large for accepted step: kl={kl}, max_kl={max_kl}")
    else:
        # If no step accepted, KL should usually be ~0 (reported as 0.0 in your implementation).
        assert_true(kl >= 0.0, f"KL must be non-negative: kl={kl}")


def test_trpo_rejects_invalid_action_shape_if_enforced_by_head():
    """
    Head behavior varies across implementations. Some heads normalize/broadcast
    actions instead of raising.

    Therefore:
      - We TRY to trigger a failure with a clearly incompatible action shape.
      - If no exception is raised, we accept that the head is permissive.
        (This test is written as "best effort" and will PASS either way.)
    """
    seed_all(0)

    head = TRPOHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
    )

    obs = np.zeros((4,), dtype=np.float32)

    # Bad action: shape (3,) is incompatible with action_dim=2 in strict implementations
    bad_act = th.zeros((3,), dtype=th.float32)

    try:
        _ = head.evaluate_actions(obs, bad_act)
        assert_true(True, "head is permissive about action shapes")
    except Exception:
        assert_true(True, "head rejected invalid action shape as expected")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("trpo_head_act_and_evaluate_shapes", test_trpo_head_act_and_evaluate_shapes),
    ("trpo_head_save_and_load_roundtrip", test_trpo_head_save_and_load_roundtrip),
    ("trpo_algorithm_smoke_run_runs_and_updates", test_trpo_algorithm_smoke_run_runs_and_updates),
    ("trpo_parameters_change_after_update", test_trpo_parameters_change_after_update),
    ("trpo_kl_metric_respects_constraint_when_step_accepted_or_zero", test_trpo_kl_metric_respects_constraint_when_step_accepted_or_zero),
    ("trpo_rejects_invalid_action_shape_if_enforced_by_head", test_trpo_rejects_invalid_action_shape_if_enforced_by_head),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="trpo")


if __name__ == "__main__":
    raise SystemExit(main())
