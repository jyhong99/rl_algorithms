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

from model_free.baselines.policy_gradients.off_policy.ddpg.ddpg import ddpg  # noqa: E402
from model_free.baselines.policy_gradients.off_policy.ddpg.head import DDPGHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
class DummyBoxSpace:
    """
    Minimal Box space supporting:
      - low/high/shape
      - sample()
    """

    def __init__(self, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
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
    Reward is a simple negative quadratic cost on state and action (so learning has signal).
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

        # Simple stable-ish dynamics: x <- 0.98 x + 0.05 a + noise
        noise = (self._rng.randn(int(self.obs_dim)).astype(np.float32)) * 0.01
        # Broadcast action into obs_dim if needed (repeat/crop)
        a_pad = np.resize(a, (int(self.obs_dim),)).astype(np.float32)

        self._obs = (0.98 * self._obs + 0.05 * a_pad + noise).astype(np.float32)
        self._obs = np.clip(self._obs, -1.0, 1.0).astype(np.float32)

        # Reward: negative quadratic cost (encourages near-zero state/action)
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
    exploration_noise: Optional[str] = "gaussian",
) -> Any:
    # Dummy env: obs_dim=3, act_dim=2
    # Important: For DDPG, provide action_low/high so actor outputs match env bounds.
    action_low = np.full((2,), -1.0, dtype=np.float32)
    action_high = np.full((2,), 1.0, dtype=np.float32)

    return ddpg(
        obs_dim=3,
        action_dim=2,
        device=device,
        hidden_sizes=(64, 64),
        action_low=action_low,
        action_high=action_high,
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
        # noise
        exploration_noise=exploration_noise,
        noise_sigma=0.1,
        # core
        gamma=0.99,
        tau=0.005,
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
# Tests: DDPGHead
# =============================================================================
def test_ddpg_head_actor_output_shape_and_bounds():
    seed_all(0)

    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)

    head = DDPGHead(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        action_low=low,
        action_high=high,
        noise=None,
        noise_clip=None,
    )

    obs = np.zeros((3,), dtype=np.float32)
    a = head.act(obs, deterministic=True)
    a_np = np.asarray(a, dtype=np.float32).reshape(1, 2) if np.asarray(a).ndim == 1 else np.asarray(a, dtype=np.float32)

    # Depending on your head.act() contract you may return (action, info) or just action.
    # Best-effort normalize:
    if isinstance(a, tuple) and len(a) >= 1:
        a0 = a[0]
        a_np = np.asarray(a0, dtype=np.float32)
        if a_np.ndim == 1:
            a_np = a_np.reshape(1, -1)

    assert_eq(tuple(a_np.shape), (1, 2))

    # Bounds check (loose): allow small numerical eps
    assert_true(float(a_np.max()) <= 1.0 + 1e-4, "actor action above high bound")
    assert_true(float(a_np.min()) >= -1.0 - 1e-4, "actor action below low bound")


def test_ddpg_head_save_and_load_roundtrip():
    seed_all(0)

    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)

    head1 = DDPGHead(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        action_low=low,
        action_high=high,
        noise=None,
        noise_clip=None,
    )

    # Touch forward to ensure params initialized similarly across codepaths
    _ = head1.act(np.zeros((3,), dtype=np.float32), deterministic=True)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ddpg_head_ckpt")
        head1.save(path)

        head2 = DDPGHead(
            obs_dim=3,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            init_type="orthogonal",
            gain=1.0,
            bias=0.0,
            device="cpu",
            action_low=low,
            action_high=high,
            noise=None,
            noise_clip=None,
        )
        head2.load(path)

        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")

        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration (DDPG)
# =============================================================================
def test_ddpg_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=False, exploration_noise="gaussian")
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


def test_ddpg_smoke_run_with_per_reports_beta():
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


def test_ddpg_replay_batch_contract():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=50, seed=0, action_low=-1.0, action_high=1.0)

    algo = _make_small_algo(device="cpu", use_per=False, exploration_noise=None)
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

    # Best-effort shape assertions
    obs = batch.observations
    act = batch.actions
    assert_true(th.is_tensor(obs) and th.is_tensor(act), "batch tensors expected")
    assert_eq(tuple(obs.shape), (algo.batch_size, 3))
    # action may be stored as (B,2) float
    assert_eq(tuple(act.shape), (algo.batch_size, 2))

    env.close()


def test_ddpg_parameters_change_after_update():
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

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")
    env.close()


def test_ddpg_noise_effect_non_deterministic_act_changes_actions():
    seed_all(0)

    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)

    # Build a head with OU or Gaussian noise; deterministic=False should inject noise.
    algo = ddpg(
        obs_dim=3,
        action_dim=2,
        device="cpu",
        hidden_sizes=(32, 32),
        action_low=low,
        action_high=high,
        exploration_noise="gaussian",
        noise_sigma=0.2,
        buffer_size=1000,
        batch_size=16,
        warmup_env_steps=0,   # disable warmup so head.act is used immediately
        update_after=0,
        use_per=False,
    )

    env = DummyContinuousEnv(obs_dim=3, act_dim=2, horizon=10, seed=0, action_low=-1.0, action_high=1.0)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    a1 = algo.act(obs, deterministic=False)
    a2 = algo.act(obs, deterministic=False)

    a1 = np.asarray(a1, dtype=np.float32).reshape(-1)
    a2 = np.asarray(a2, dtype=np.float32).reshape(-1)

    # Not guaranteed, but should differ with high probability if noise is applied.
    assert_true(not np.allclose(a1, a2), "expected noisy actions to differ (deterministic=False)")
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("ddpg_head_actor_output_shape_and_bounds", test_ddpg_head_actor_output_shape_and_bounds),
    ("ddpg_head_save_and_load_roundtrip", test_ddpg_head_save_and_load_roundtrip),

    ("ddpg_smoke_run_no_per_runs_and_updates", test_ddpg_smoke_run_no_per_runs_and_updates),
    ("ddpg_smoke_run_with_per_reports_beta", test_ddpg_smoke_run_with_per_reports_beta),
    ("ddpg_replay_batch_contract", test_ddpg_replay_batch_contract),
    ("ddpg_parameters_change_after_update", test_ddpg_parameters_change_after_update),
    ("ddpg_noise_effect_non_deterministic_act_changes_actions", test_ddpg_noise_effect_non_deterministic_act_changes_actions),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="ddpg")


if __name__ == "__main__":
    raise SystemExit(main())
