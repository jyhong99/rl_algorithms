# td3_testers.py
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th


# =============================================================================
# Path bootstrap (same style as your other testers)
# =============================================================================
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

from model_free.baselines.policy_gradients.off_policy.td3.td3 import td3  # noqa: E402
from model_free.baselines.policy_gradients.off_policy.td3.head import TD3Head  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
class DummyBoxSpace:
    """
    Minimal Box space supporting:
      - low/high/shape
      - sample()
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.low = np.full(self.shape, float(low), dtype=np.float32)
        self.high = np.full(self.shape, float(high), dtype=np.float32)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> np.ndarray:
        return self._rng.uniform(low=self.low, high=self.high).astype(np.float32)


@dataclass
class DummyContinuousEnv:
    """
    Tiny continuous-control env with Box obs/action, no gym dependency.

    Shapes
    ------
    - obs:    (obs_dim,) float32
    - action: (act_dim,) float32, expected within [action_low, action_high]

    Episode ends after `horizon` steps.
    Reward is a simple negative quadratic cost on state and action.
    """

    obs_dim: int = 3
    act_dim: int = 2
    horizon: int = 200
    seed: int = 0
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(int(self.obs_dim),), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyBoxSpace(
            shape=(int(self.act_dim),),
            low=float(self.action_low),
            high=float(self.action_high),
            seed=self.seed + 123,
        )
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((int(self.obs_dim),), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()  # deterministic re-init
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0
        return self._obs.copy(), {}

    def step(self, action: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(int(self.act_dim))
        a = np.clip(a, float(self.action_low), float(self.action_high)).astype(np.float32)

        # Stable-ish dynamics: x <- 0.98 x + 0.05 a + noise
        noise = (self._rng.randn(int(self.obs_dim)).astype(np.float32)) * 0.01
        a_pad = np.resize(a, (int(self.obs_dim),)).astype(np.float32)

        self._obs = (0.98 * self._obs + 0.05 * a_pad + noise).astype(np.float32)
        self._obs = np.clip(self._obs, -1.0, 1.0).astype(np.float32)

        reward = -float(np.mean(self._obs**2) + 0.05 * np.mean(a**2))

        self._t += 1
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
    parts: List[th.Tensor] = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(
    *,
    device: str = "cpu",
    use_per: bool = True,
    exploration_noise: Optional[str] = None,
) -> Any:
    # Dummy env: obs_dim=3, act_dim=2
    return td3(
        obs_dim=3,
        action_dim=2,
        device=device,
        hidden_sizes=(64, 64),
        # bounds (match DummyContinuousEnv)
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        # head-side exploration noise
        exploration_noise=exploration_noise,  # e.g. "gaussian" to force stochastic act()
        noise_sigma=0.2,
        # scheduling / replay
        buffer_size=10_000,
        batch_size=32,
        warmup_env_steps=50,
        update_after=50,
        update_every=1,
        utd=1.0,
        gradient_steps=1,
        max_updates_per_call=10,
        # PER
        use_per=bool(use_per),
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_final=0.8,
        per_beta_anneal_steps=1_000,
        # core
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        target_update_interval=1,
        max_grad_norm=10.0,
        use_amp=False,
    )


def _rollout_steps(
    algo: Any,
    env: Any,
    *,
    n_steps: int,
    seed: int = 0,
    do_updates: bool = True,
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)

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
# Tests: TD3Head
# =============================================================================
def test_td3_head_actor_output_shape_and_bounds():
    """
    TD3Head is deterministic. We verify:
    - act() returns correct shape (B, action_dim)
    - actions are within [action_low, action_high] if bounds are provided
    """
    seed_all(0)

    head = TD3Head(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        noise=None,  # no exploration noise for this test
    )

    obs = np.zeros((3,), dtype=np.float32)
    a = head.act(obs, deterministic=True)
    a_np = np.asarray(a, dtype=np.float32)

    # Normalize to (B, A)
    if a_np.ndim == 1:
        a_np = a_np.reshape(1, -1)

    assert_eq(tuple(a_np.shape), (1, 2))

    # DeterministicPolicyNetwork clamps to provided bounds
    assert_true(float(a_np.max()) <= 1.0 + 1e-5, "actor action above +1 (bounds violated)")
    assert_true(float(a_np.min()) >= -1.0 - 1e-5, "actor action below -1 (bounds violated)")


def test_td3_head_target_action_shape_and_bounds():
    """
    target_action() should:
    - return shape (B, action_dim)
    - apply smoothing noise then clamp to bounds (if bounds provided)
    """
    seed_all(0)

    head = TD3Head(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        noise=None,
    )

    nxt = np.zeros((3,), dtype=np.float32)
    a2 = head.target_action(nxt, noise_std=0.2, noise_clip=0.5)
    assert_true(th.is_tensor(a2), "expected tensor from target_action()")
    assert_eq(tuple(a2.shape), (1, 2))

    a2_np = a2.detach().cpu().numpy()
    assert_true(float(a2_np.max()) <= 1.0 + 1e-5, "target_action above +1 (bounds violated)")
    assert_true(float(a2_np.min()) >= -1.0 - 1e-5, "target_action below -1 (bounds violated)")


def test_td3_head_save_and_load_roundtrip():
    """
    Save/load roundtrip should reproduce actor/critic + targets exactly.
    """
    seed_all(0)

    head1 = TD3Head(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        noise=None,
    )

    _ = head1.act(np.zeros((3,), dtype=np.float32), deterministic=True)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "td3_head_ckpt")
        head1.save(path)

        head2 = TD3Head(
            obs_dim=3,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            init_type="orthogonal",
            gain=1.0,
            bias=0.0,
            device="cpu",
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
            noise=None,
        )
        head2.load(path)

        # Actor compare
        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")

        # Critic compare
        cd1 = head1.critic.state_dict()
        cd2 = head2.critic.state_dict()
        assert_eq(set(cd1.keys()), set(cd2.keys()))
        for k in cd1.keys():
            assert_true(th.allclose(cd1[k], cd2[k]), f"critic mismatch after load: {k}")

        # Targets compare
        at1 = head1.actor_target.state_dict()
        at2 = head2.actor_target.state_dict()
        assert_eq(set(at1.keys()), set(at2.keys()))
        for k in at1.keys():
            assert_true(th.allclose(at1[k], at2[k]), f"actor_target mismatch after load: {k}")

        ct1 = head1.critic_target.state_dict()
        ct2 = head2.critic_target.state_dict()
        assert_eq(set(ct1.keys()), set(ct2.keys()))
        for k in ct1.keys():
            assert_true(th.allclose(ct1[k], ct2[k]), f"critic_target mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration (TD3)
# =============================================================================
def test_td3_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=False, exploration_noise="gaussian")
    out = _rollout_steps(algo, env, n_steps=1500, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # OffPolicyAlgorithm common metrics
    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")

    # TD3 core metrics
    assert_true("loss/actor" in m, "missing loss/actor")
    assert_true("loss/critic" in m, "missing loss/critic")
    assert_true("td3/did_actor_update" in m, "missing td3/did_actor_update")
    assert_true(np.isfinite(float(m["loss/critic"])), "loss/critic not finite")


def test_td3_smoke_run_with_per_reports_beta():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=True, exploration_noise="gaussian")
    out = _rollout_steps(algo, env, n_steps=1800, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")


def test_td3_replay_batch_contract():
    """
    Verify replay batch fields and basic shapes.
    """
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=False, exploration_noise="gaussian")
    algo.setup(env)

    # Fill buffer without updating
    _ = _rollout_steps(algo, env, n_steps=400, seed=0, do_updates=False)

    assert_true(algo.buffer is not None, "buffer should be initialized")
    assert_true(algo.buffer.size >= algo.batch_size, "buffer must have enough samples")

    batch = algo.buffer.sample(algo.batch_size)
    assert_true(hasattr(batch, "observations"), "batch missing observations")
    assert_true(hasattr(batch, "actions"), "batch missing actions")
    assert_true(hasattr(batch, "rewards"), "batch missing rewards")
    assert_true(hasattr(batch, "next_observations"), "batch missing next_observations")
    assert_true(hasattr(batch, "dones"), "batch missing dones")

    obs = batch.observations
    act = batch.actions
    assert_true(th.is_tensor(obs) and th.is_tensor(act), "batch tensors expected")
    assert_eq(tuple(obs.shape), (algo.batch_size, 3))
    assert_eq(tuple(act.shape), (algo.batch_size, 2))

    env.close()


def test_td3_parameters_change_after_update():
    """
    After at least one update, critic parameters should change.
    Actor parameters may change depending on policy_delay.
    """
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=False, exploration_noise="gaussian")
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)

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

    # Critic should essentially always change; actor might be delayed.
    assert_true(dc > 0.0 or da > 0.0, f"expected params to change; da={da}, dc={dc}")
    env.close()


def test_td3_exploration_noise_changes_actions_when_enabled():
    """
    TD3 is deterministic, but exploration noise (head-side) should make
    deterministic=False produce different actions from the same obs with high probability.
    """
    seed_all(0)

    algo = td3(
        obs_dim=3,
        action_dim=2,
        device="cpu",
        hidden_sizes=(32, 32),
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        exploration_noise="gaussian",
        noise_sigma=0.2,
        buffer_size=1000,
        batch_size=16,
        warmup_env_steps=0,
        update_after=0,
        use_per=False,
        policy_delay=2,
    )

    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=10, seed=0, action_low=-1.0, action_high=1.0)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    a1 = algo.act(obs, deterministic=False)
    a2 = algo.act(obs, deterministic=False)

    a1 = np.asarray(a1, dtype=np.float32).reshape(-1)
    a2 = np.asarray(a2, dtype=np.float32).reshape(-1)

    assert_true(not np.allclose(a1, a2), "expected noisy actions to differ (deterministic=False)")

    # deterministic=True should suppress exploration noise
    d1 = np.asarray(algo.act(obs, deterministic=True), dtype=np.float32).reshape(-1)
    d2 = np.asarray(algo.act(obs, deterministic=True), dtype=np.float32).reshape(-1)
    assert_true(np.allclose(d1, d2), "expected deterministic actions to be identical (deterministic=True)")

    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("td3_head_actor_output_shape_and_bounds", test_td3_head_actor_output_shape_and_bounds),
    ("td3_head_target_action_shape_and_bounds", test_td3_head_target_action_shape_and_bounds),
    ("td3_head_save_and_load_roundtrip", test_td3_head_save_and_load_roundtrip),
    ("td3_smoke_run_no_per_runs_and_updates", test_td3_smoke_run_no_per_runs_and_updates),
    ("td3_smoke_run_with_per_reports_beta", test_td3_smoke_run_with_per_reports_beta),
    ("td3_replay_batch_contract", test_td3_replay_batch_contract),
    ("td3_parameters_change_after_update", test_td3_parameters_change_after_update),
    ("td3_exploration_noise_changes_actions_when_enabled", test_td3_exploration_noise_changes_actions_when_enabled),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="td3")


if __name__ == "__main__":
    raise SystemExit(main())
