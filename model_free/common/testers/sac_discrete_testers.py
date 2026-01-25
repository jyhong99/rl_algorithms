# sac_discrete_testers.py
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

from model_free.baselines.policy_gradients.off_policy.sac_discrete.sac_discrete import (  # noqa: E402
    sac_discrete,
)
from model_free.baselines.policy_gradients.off_policy.sac_discrete.head import (  # noqa: E402
    SACDiscreteHead,
)


# =============================================================================
# Minimal discrete spaces/env (no gym dependency)
# =============================================================================
class DummyDiscreteSpace:
    """
    Minimal discrete action space supporting:
      - n
      - sample()
    """

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = int(n)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> int:
        return int(self._rng.randint(0, self.n))


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
class DummyDiscreteEnv:
    """
    Tiny discrete-control env with Box obs + Discrete actions, no gym dependency.

    Shapes
    ------
    - obs:       (obs_dim,) float32
    - action:    int in [0, n_actions)

    Episode ends after `horizon` steps.
    Reward is a simple quadratic shaping + action-dependent signal (so learning has signal).
    """

    obs_dim: int = 4
    n_actions: int = 5
    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(int(self.obs_dim),), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyDiscreteSpace(n=int(self.n_actions), seed=self.seed + 123)
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((int(self.obs_dim),), dtype=np.float32)

        # A simple "preferred action" that depends on state sign pattern.
        # This makes reward somewhat learnable.
        self._w = self._rng.randn(int(self.obs_dim)).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()  # deterministic re-init
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0
        return self._obs.copy(), {}

    def step(self, action: Union[int, np.integer]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)
        a = max(0, min(a, int(self.n_actions) - 1))

        # Stable-ish dynamics: x <- 0.98 x + noise
        noise = (self._rng.randn(int(self.obs_dim)).astype(np.float32)) * 0.02
        self._obs = (0.98 * self._obs + noise).astype(np.float32)
        self._obs = np.clip(self._obs, -1.0, 1.0).astype(np.float32)

        # Reward:
        # - encourage small state magnitude
        # - plus a mild action preference tied to sign(w·obs)
        score = float(np.dot(self._w, self._obs))
        preferred = 0 if score < 0 else 1
        reward = -float(np.mean(self._obs**2)) + (0.1 if a == preferred else 0.0)

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
) -> Any:
    # Dummy env: obs_dim=4, n_actions=5
    return sac_discrete(
        obs_dim=4,
        n_actions=5,
        device=device,
        actor_hidden_sizes=(64, 64),
        critic_hidden_sizes=(64, 64),
        dueling_mode=False,
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
        target_update_interval=1,
        max_grad_norm=10.0,
        use_amp=False,
        auto_alpha=True,
        alpha_init=0.2,
        target_entropy=None,
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
        # Discrete SAC: expect int action
        a = algo.act(obs, deterministic=False)
        a_int = int(np.asarray(a).reshape(-1)[0]) if not isinstance(a, (int, np.integer)) else int(a)

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
# Tests: SACDiscreteHead
# =============================================================================
def test_sac_discrete_head_act_output_shape_and_range():
    """
    SACDiscreteHead.act() should return a discrete action index.
    We verify:
      - output is scalar-like (or (B,) where B=1)
      - within [0, n_actions)
    """
    seed_all(0)

    head = SACDiscreteHead(
        obs_dim=4,
        n_actions=5,
        actor_hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    a = head.act(obs, deterministic=False)
    a_np = np.asarray(a).reshape(-1)
    assert_true(a_np.size in (1,), "act() should return a scalar-like action for single obs")

    a0 = int(a_np[0])
    assert_true(0 <= a0 < 5, f"action out of range: {a0}")


def test_sac_discrete_head_q_shapes():
    """
    q_values / q_values_target should return (q1, q2) with shape (B, n_actions).
    """
    seed_all(0)

    head = SACDiscreteHead(
        obs_dim=4,
        n_actions=5,
        actor_hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    q1, q2 = head.q_values_pair(obs)
    assert_true(th.is_tensor(q1) and th.is_tensor(q2), "q_values must return tensors")
    assert_eq(tuple(q1.shape), (1, 5))
    assert_eq(tuple(q2.shape), (1, 5))

    q1t, q2t = head.q_values_target_pair(obs)
    assert_true(th.is_tensor(q1t) and th.is_tensor(q2t), "q_values_target must return tensors")
    assert_eq(tuple(q1t.shape), (1, 5))
    assert_eq(tuple(q2t.shape), (1, 5))


def test_sac_discrete_head_save_and_load_roundtrip():
    """
    Save/load roundtrip should reproduce actor + critic + critic_target weights exactly.
    """
    seed_all(0)

    head1 = SACDiscreteHead(
        obs_dim=4,
        n_actions=5,
        actor_hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
    )

    # Touch forward paths
    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)
    _ = head1.q_values(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sac_discrete_head_ckpt")
        head1.save(path)

        head2 = SACDiscreteHead(
            obs_dim=4,
            n_actions=5,
            actor_hidden_sizes=(32, 32),
            critic_hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            dueling_mode=False,
            init_type="orthogonal",
            gain=1.0,
            bias=0.0,
            device="cpu",
        )
        head2.load(path)

        # Actor
        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")

        # Critic
        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")

        # Target critic
        sd1t = head1.critic_target.state_dict()
        sd2t = head2.critic_target.state_dict()
        assert_eq(set(sd1t.keys()), set(sd2t.keys()))
        for k in sd1t.keys():
            assert_true(th.allclose(sd1t[k], sd2t[k]), f"critic_target mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration (Discrete SAC)
# =============================================================================
def test_sac_discrete_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyDiscreteEnv(obs_dim=4, n_actions=5, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    out = _rollout_steps(algo, env, n_steps=2000, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # OffPolicyAlgorithm common metrics
    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")

    # Core metrics (your SACDiscreteCore returns these keys)
    assert_true("loss/actor" in m, "missing loss/actor")
    assert_true("loss/critic" in m, "missing loss/critic")
    # alpha naming: in your core it's "stats/alpha"
    assert_true(("stats/alpha" in m) or ("alpha" in m), "missing alpha metric (stats/alpha or alpha)")

    assert_true(np.isfinite(float(m["loss/actor"])), "loss/actor not finite")
    assert_true(np.isfinite(float(m["loss/critic"])), "loss/critic not finite")


def test_sac_discrete_smoke_run_with_per_reports_beta():
    seed_all(0)
    env = DummyDiscreteEnv(obs_dim=4, n_actions=5, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=True)
    out = _rollout_steps(algo, env, n_steps=2200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")


def test_sac_discrete_replay_batch_contract():
    """
    Verify replay batch fields and basic shapes.
    Actions should be integer-like stored in a tensor of shape (B,) or (B,1) depending on buffer.
    We'll accept either, but enforce that it can be viewed to (B,).
    """
    seed_all(0)
    env = DummyDiscreteEnv(obs_dim=4, n_actions=5, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    # Fill buffer without updating
    _ = _rollout_steps(algo, env, n_steps=500, seed=0, do_updates=False)

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
    assert_eq(tuple(obs.shape), (algo.batch_size, 4))

    # accept (B,) or (B,1)
    assert_true(
        (act.ndim == 1 and act.shape[0] == algo.batch_size) or (act.ndim == 2 and act.shape[0] == algo.batch_size),
        f"unexpected action batch shape: {tuple(act.shape)}",
    )

    # Ensure actions are in valid range (best-effort)
    act_flat = act.view(-1).detach().cpu().numpy()
    assert_true(np.all(act_flat >= 0) and np.all(act_flat < 5), "sampled actions out of range [0,n_actions)")

    env.close()


def test_sac_discrete_parameters_change_after_update():
    """
    After at least one update, actor parameters (and typically critics) should change.
    """
    seed_all(0)
    env = DummyDiscreteEnv(obs_dim=4, n_actions=5, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    max_drive = 8000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_int = int(np.asarray(a).reshape(-1)[0]) if not isinstance(a, (int, np.integer)) else int(a)

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


def test_sac_discrete_non_deterministic_act_changes_actions():
    """
    Discrete SAC uses a categorical stochastic policy. With deterministic=False,
    successive actions from the same obs should differ with high probability.

    Note: Not guaranteed; if it flakes, increase trials or adjust policy temperature/initialization.
    """
    seed_all(0)

    algo = sac_discrete(
        obs_dim=4,
        n_actions=5,
        device="cpu",
        actor_hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
        buffer_size=1000,
        batch_size=16,
        warmup_env_steps=0,
        update_after=0,
        use_per=False,
        auto_alpha=True,
        alpha_init=0.2,
    )

    env = DummyDiscreteEnv(obs_dim=4, n_actions=5, horizon=10, seed=0)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    a1 = algo.act(obs, deterministic=False)
    a2 = algo.act(obs, deterministic=False)

    a1 = int(np.asarray(a1).reshape(-1)[0]) if not isinstance(a1, (int, np.integer)) else int(a1)
    a2 = int(np.asarray(a2).reshape(-1)[0]) if not isinstance(a2, (int, np.integer)) else int(a2)

    assert_true(a1 != a2, "expected stochastic actions to differ (deterministic=False)")
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("sac_discrete_head_act_output_shape_and_range", test_sac_discrete_head_act_output_shape_and_range),
    ("sac_discrete_head_q_shapes", test_sac_discrete_head_q_shapes),
    ("sac_discrete_head_save_and_load_roundtrip", test_sac_discrete_head_save_and_load_roundtrip),

    ("sac_discrete_smoke_run_no_per_runs_and_updates", test_sac_discrete_smoke_run_no_per_runs_and_updates),
    ("sac_discrete_smoke_run_with_per_reports_beta", test_sac_discrete_smoke_run_with_per_reports_beta),
    ("sac_discrete_replay_batch_contract", test_sac_discrete_replay_batch_contract),
    ("sac_discrete_parameters_change_after_update", test_sac_discrete_parameters_change_after_update),
    ("sac_discrete_non_deterministic_act_changes_actions", test_sac_discrete_non_deterministic_act_changes_actions),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="sac_discrete")


if __name__ == "__main__":
    raise SystemExit(main())
