from __future__ import annotations

import os
import sys

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn


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
    seed_all,
    run_tests,
    assert_eq,
    assert_true,
    assert_raises
)

from model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm
from model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm

from model_free.common.testers.test_harness import (
    DummyHeadForCore,
    DummyCore,
    DummyACHead,
    DummyQLHead,
    FakeEnv,
    FakeReplayBuffer,
    FakeRolloutBuffer,
    StatefulCore,
    StatefulHead,
    AlgoForSaveLoad,
    DummyOffPolicyCore,
    DummyOffPolicyHead,
    ACHeadForCore,
    QHeadForCore,
    ConcreteActorCriticCore,
    ConcreteQLearningCore,
    DummyDetHead,
    DummyOffPolicyACHead,
    MinimalPolicyAlgo,
    ConcreteBaseHead,
    DummyOnPolicyHead,
    DummyOnPolicyCore
)


    
# =============================================================================
# Tests: BaseCore (target update + freeze + state_dict)
# =============================================================================
def test_basecore_maybe_update_target_hard_and_freeze():
    seed_all(0)
    head = DummyHeadForCore()
    core = DummyCore(head=head, use_amp=False)

    src = nn.Linear(4, 3)
    tgt = nn.Linear(4, 3)

    # force different params
    with th.no_grad():
        for p in src.parameters():
            p.add_(1.0)

    # update_calls=0 => should update when interval=1
    core._maybe_update_target(target=tgt, source=src, interval=1, tau=0.0)

    # after hard update, params should match
    for ps, pt in zip(src.parameters(), tgt.parameters()):
        assert_true(th.allclose(ps, pt), "hard_update failed to copy parameters")

    # frozen
    assert_true(not any(p.requires_grad for p in tgt.parameters()), "target not frozen after update")
    assert_true(tgt.training is False, "target not set to eval() after freeze")


def test_basecore_maybe_update_target_soft_tau_and_interval():
    seed_all(0)
    head = DummyHeadForCore()
    core = DummyCore(head=head, use_amp=False)

    src = nn.Linear(4, 3)
    tgt = nn.Linear(4, 3)

    # set deterministic params
    with th.no_grad():
        for p in src.parameters():
            p.fill_(2.0)
        for p in tgt.parameters():
            p.fill_(0.0)

    # interval=2: at update_calls=0 => update happens
    core._maybe_update_target(target=tgt, source=src, interval=2, tau=0.5)

    # tgt should be halfway between 0 and 2 => 1.0
    for pt in tgt.parameters():
        assert_true(th.allclose(pt, th.ones_like(pt)), "soft_update tau=0.5 not applied correctly")

    # bump once: update_calls=0 still (we didn't bump), so emulate update calls
    core._bump()  # update_calls=1
    # interval=2 => should NOT update at call 1
    with th.no_grad():
        for p in src.parameters():
            p.fill_(10.0)
    before = [p.detach().clone() for p in tgt.parameters()]
    core._maybe_update_target(target=tgt, source=src, interval=2, tau=0.5)
    after = [p.detach().clone() for p in tgt.parameters()]
    assert_true(all(th.allclose(a, b) for a, b in zip(before, after)), "interval gating failed")


def test_basecore_tau_validation():
    head = DummyHeadForCore()
    core = DummyCore(head=head, use_amp=False)
    src = nn.Linear(1, 1)
    tgt = nn.Linear(1, 1)
    assert_raises(ValueError, lambda: core._maybe_update_target(target=tgt, source=src, interval=1, tau=2.0))


def test_basecore_state_dict_roundtrip():
    head = DummyHeadForCore()
    core = DummyCore(head=head, use_amp=False)
    core._bump()
    core._bump()
    sd = core.state_dict()

    core2 = DummyCore(head=head, use_amp=False)
    core2.load_state_dict(sd)
    assert_eq(core2.update_calls, 2, "BaseCore state_dict/load_state_dict failed")


# =============================================================================
# Tests: Heads (basic acting / evaluate_actions / q_values shape)
# =============================================================================
def test_onpolicy_head_evaluate_actions_shapes():
    seed_all(0)
    head = DummyACHead(device="cpu")
    obs = np.random.randn(4).astype(np.float32)
    action = np.random.randn(2).astype(np.float32)

    out = head.evaluate_actions(obs, action, as_scalar=False)
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys in evaluate_actions")
    assert_eq(tuple(out["value"].shape), (1, 1), "value shape should be (B,1)")
    assert_eq(tuple(out["log_prob"].shape), (1, 2), "Normal.log_prob gives per-dim; your code keeps (B,A) unless summed elsewhere")
    assert_eq(tuple(out["entropy"].shape), (1, 2), "entropy should be (B,A) for factorized Normal")

    # as_scalar requires B=1
    out_s = head.evaluate_actions(obs, action, as_scalar=True)
    assert_true(all(isinstance(v, float) for v in out_s.values()), "as_scalar should return floats")


def test_qlearning_head_act_epsilon_greedy():
    seed_all(0)
    head = DummyQLHead(device="cpu", n_actions=5)

    obs = np.random.randn(4).astype(np.float32)
    a_det = head.act(obs, epsilon=0.0, deterministic=True)
    assert_true(th.is_tensor(a_det), "act should return tensor")
    assert_true(a_det.dtype == th.long, "discrete action should be long")
    assert_true(int(a_det.item()) in range(5), "action out of range")

    # stochastic epsilon
    a_eps = head.act(obs, epsilon=1.0, deterministic=False)
    assert_true(int(a_eps.item()) in range(5), "epsilon action out of range")


def test_basealgorithm_save_load_roundtrip(tmp_path: Optional[str] = None):
    seed_all(0)
    head = StatefulHead()
    core = StatefulCore()
    algo = AlgoForSaveLoad(head=head, core=core, device="cpu")

    # mutate state
    with th.no_grad():
        head.w.add_(2.0)
    core.k = 7

    # save
    path = "tmp_algo_ckpt.pt" if tmp_path is None else os.path.join(tmp_path, "tmp_algo_ckpt.pt")
    try:
        algo.save(path)

        # new instances
        head2 = StatefulHead()
        core2 = StatefulCore()
        algo2 = AlgoForSaveLoad(head=head2, core=core2, device="cpu")
        algo2.load(path)

        assert_true(float(head2.w.item()) == float(head.w.item()), "head state_dict not restored")
        assert_eq(core2.k, 7, "core state not restored")
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def test_offpolicy_update_budget_and_sys_num_updates():
    seed_all(0)
    head = DummyOffPolicyHead()
    core = DummyOffPolicyCore()
    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        buffer_size=100,
        batch_size=4,
        warmup_steps=2,
        update_after=3,
        update_every=1,
        utd=1.0,
        gradient_steps=3,
        max_updates_per_call=10,
        use_per=False,
        reset_noise_on_done=True,
    )

    # Bypass real setup & buffer types: inject fake env shapes via setup but then swap buffer
    env = FakeEnv(obs_dim=4, act_dim=2)
    algo.setup(env)
    algo.buffer = FakeReplayBuffer()  # type: ignore

    # Warmup action path
    a0 = algo.act(np.zeros(4, np.float32), deterministic=False)
    assert_true(isinstance(a0, np.ndarray), "warmup action should be sampled")
    # ingest transitions to accumulate budget and buffer size
    for t in range(5):
        algo.on_env_step(
            {"obs": np.zeros(4, np.float32), "action": np.zeros(2, np.float32), "reward": 1.0, "next_obs": np.zeros(4, np.float32), "done": (t == 2)}
        )

    # after env_steps=5, update_after=3 => eligible, budget should be >= 5 (utd=1)
    assert_true(algo.ready_to_update(), "should be ready to update")

    out = algo.update()
    # owed=int(budget) ~= 5, but capped by max_updates_per_call=10 => 5 update units
    # sys/num_updates = update_units * gradient_steps = 5*3=15
    assert_eq(int(out.get("sys/num_updates", -1.0)), 15, "sys/num_updates accounting mismatch")
    assert_eq(int(out.get("offpolicy/update_units", -1.0)), 5, "offpolicy/update_units mismatch")
    assert_eq(int(out.get("offpolicy/grad_steps", -1.0)), 3, "offpolicy/grad_steps mismatch")


def test_onpolicy_num_updates_equals_num_minibatches():
    seed_all(0)
    head = DummyOnPolicyHead()
    core = DummyOnPolicyCore()
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=10,
        update_epochs=2,
        minibatch_size=4,
        device="cpu",
    )

    # setup: then swap rollout buffer with fake deterministic rollout
    env = FakeEnv(obs_dim=4, act_dim=2)
    algo.setup(env)
    algo.rollout = FakeRolloutBuffer(buffer_size=10)  # type: ignore

    # fill rollout
    for t in range(10):
        algo.on_env_step(
            {"obs": np.zeros(4, np.float32), "action": np.zeros(2, np.float32), "reward": 1.0, "done": (t == 9), "next_obs": np.zeros(4, np.float32)}
        )

    assert_true(algo.ready_to_update(), "rollout should be full")

    out = algo.update()
    num_minibatches = int(out.get("onpolicy/num_minibatches", -1.0))
    sys_updates = int(out.get("sys/num_updates", -1.0))

    assert_eq(sys_updates, num_minibatches, "onpolicy sys/num_updates should equal num_minibatches")
    assert_eq(core.calls, num_minibatches, "core.update_from_batch call count mismatch")


def test_actorcriticcore_optimizer_steps_params():
    seed_all(0)
    head = ACHeadForCore(device="cpu")
    core = ConcreteActorCriticCore(
        head=head,
        actor_optim_name="adamw", critic_optim_name="adamw",
        actor_lr=1e-3, critic_lr=1e-3,
        actor_sched_name="none", critic_sched_name="none",
    )

    # snapshot params
    before_actor = [p.detach().clone() for p in head.actor.parameters()]
    before_critic = [p.detach().clone() for p in head.critic.parameters()]

    out = core.update_from_batch(batch={})

    after_actor = [p.detach().clone() for p in head.actor.parameters()]
    after_critic = [p.detach().clone() for p in head.critic.parameters()]

    assert_true(any(not th.allclose(a, b) for a, b in zip(before_actor, after_actor)),
                "ActorCriticCore: actor parameters did not change after update")
    assert_true(any(not th.allclose(a, b) for a, b in zip(before_critic, after_critic)),
                "ActorCriticCore: critic parameters did not change after update")
    assert_true(core.update_calls == 1, "ActorCriticCore: update_calls not bumped")
    assert_true("loss/actor" in out and "loss/critic" in out, "ActorCriticCore: missing metrics")


def test_qlearningcore_optimizer_steps_params():
    seed_all(0)
    head = QHeadForCore(device="cpu", n_actions=5)
    core = ConcreteQLearningCore(
        head=head,
        n_actions=5,
        optim_name="adamw",
        lr=1e-3,
        sched_name="none",
    )

    before = [p.detach().clone() for p in head.q.parameters()]
    out = core.update_from_batch(batch={})
    after = [p.detach().clone() for p in head.q.parameters()]

    assert_true(any(not th.allclose(a, b) for a, b in zip(before, after)),
                "QLearningCore: q parameters did not change after update")
    assert_true(core.update_calls == 1, "QLearningCore: update_calls not bumped")
    assert_true("loss/q" in out, "QLearningCore: missing metrics")


def test_basehead_tensor_helpers_and_target_utils():
    seed_all(0)
    head = ConcreteBaseHead(device="cpu")

    x1 = np.ones((4,), np.float32)   # (D,)
    t1 = head._to_tensor_batched(x1)
    assert_eq(tuple(t1.shape), (1, 4), "_to_tensor_batched should add batch dim for 1D input")

    x2 = np.ones((3, 4), np.float32)  # already batched
    t2 = head._to_tensor_batched(x2)
    assert_eq(tuple(t2.shape), (3, 4), "_to_tensor_batched should keep batched input")

    # target utils: hard/soft/freeze should run
    src = nn.Linear(4, 3)
    tgt = nn.Linear(4, 3)
    with th.no_grad():
        for p in src.parameters():
            p.add_(1.0)

    head.hard_update(tgt, src)
    assert_true(all(th.allclose(ps, pt) for ps, pt in zip(src.parameters(), tgt.parameters())),
                "BaseHead.hard_update failed")

    # soft update moves towards src
    with th.no_grad():
        for p in src.parameters():
            p.fill_(2.0)
        for p in tgt.parameters():
            p.zero_()

    head.soft_update(tgt, src, tau=0.5)
    for p in tgt.parameters():
        assert_true(th.allclose(p, th.ones_like(p)), "BaseHead.soft_update tau=0.5 failed")

    head.freeze_target(tgt)
    assert_true(not any(p.requires_grad for p in tgt.parameters()), "BaseHead.freeze_target failed")


def test_offpolicy_actorcritic_head_sample_action_and_logp():
    seed_all(0)
    head = DummyOffPolicyACHead(device="cpu")

    obs = np.random.randn(4).astype(np.float32)
    a, logp = head.sample_action_and_logp(obs)

    assert_true(th.is_tensor(a) and th.is_tensor(logp), "sample_action_and_logp should return tensors")
    assert_eq(tuple(a.shape), (1, 2), "action should be (B,A)")
    assert_eq(tuple(logp.shape), (1,), "logp should be (B,)")

    # q_values interfaces
    q1, q2 = head.q_values(obs, a)
    assert_eq(tuple(q1.shape), (1, 1), "q1 shape should be (B,1)")
    assert_eq(tuple(q2.shape), (1, 1), "q2 shape should be (B,1)")


def test_deterministic_actorcritic_head_action_clamp():
    seed_all(0)
    head = DummyDetHead(device="cpu")

    # Create obs that produces potentially out-of-range action
    obs = np.ones((4,), np.float32) * 100.0
    a = head.act(obs, deterministic=True)  # deterministic path
    assert_eq(tuple(a.shape), (1, 2), "action should be (B,A)")

    # Ensure clamped within specified bounds
    a_np = a.detach().cpu().numpy().reshape(-1)
    assert_true(np.all(a_np <= head.action_high + 1e-6), "action not clamped to action_high")
    assert_true(np.all(a_np >= head.action_low - 1e-6), "action not clamped to action_low")

    # q_values_target exists
    qt = head.q_values_target(obs, a)
    assert_eq(tuple(qt.shape), (1, 1), "q_values_target should be (B,1)")


def test_basepolicyalgorithm_env_steps_and_protocol():
    head = StatefulHead()
    core = StatefulCore()
    algo = MinimalPolicyAlgo(head=head, core=core, device="cpu")

    assert_eq(algo.env_steps, 0, "env_steps should start at 0")
    algo.setup(env=None)

    for _ in range(2):
        algo.on_env_step({"obs": 0, "action": 0, "reward": 0, "next_obs": 0, "done": False})
    assert_eq(algo.env_steps, 2, "env_steps not incremented")
    assert_true(not algo.ready_to_update(), "should not be ready before 3 steps")

    algo.on_env_step({"obs": 0, "action": 0, "reward": 0, "next_obs": 0, "done": False})
    assert_eq(algo.env_steps, 3, "env_steps not incremented to 3")
    assert_true(algo.ready_to_update(), "should be ready at 3 steps")

    out = algo.update()
    assert_true("dummy/update" in out, "update should return metrics")
    assert_true(not algo.ready_to_update(), "update should reset readiness")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("basecore_maybe_update_target_hard_and_freeze", test_basecore_maybe_update_target_hard_and_freeze),
    ("basecore_maybe_update_target_soft_tau_and_interval", test_basecore_maybe_update_target_soft_tau_and_interval),
    ("basecore_tau_validation", test_basecore_tau_validation),
    ("basecore_state_dict_roundtrip", test_basecore_state_dict_roundtrip),
    ("onpolicy_head_evaluate_actions_shapes", test_onpolicy_head_evaluate_actions_shapes),
    ("qlearning_head_act_epsilon_greedy", test_qlearning_head_act_epsilon_greedy),
    ("basealgorithm_save_load_roundtrip", test_basealgorithm_save_load_roundtrip),
    ("offpolicy_update_budget_and_sys_num_updates", test_offpolicy_update_budget_and_sys_num_updates),
    ("onpolicy_num_updates_equals_num_minibatches", test_onpolicy_num_updates_equals_num_minibatches),
    ("actorcriticcore_optimizer_steps_params", test_actorcriticcore_optimizer_steps_params),
    ("qlearningcore_optimizer_steps_params", test_qlearningcore_optimizer_steps_params),
    ("basehead_tensor_helpers_and_target_utils", test_basehead_tensor_helpers_and_target_utils),
    ("offpolicy_actorcritic_head_sample_action_and_logp", test_offpolicy_actorcritic_head_sample_action_and_logp),
    ("deterministic_actorcritic_head_action_clamp", test_deterministic_actorcritic_head_action_clamp),
    ("basepolicyalgorithm_env_steps_and_protocol", test_basepolicyalgorithm_env_steps_and_protocol),
]

def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="policies")

if __name__ == "__main__":
    raise SystemExit(main())