"""Gym の Snapshot（robosuite / LIBERO）。シミュレータを同じプロセスで動かして中身を比べる。

- 保存した時点の内部状態（物理・コントローラ・グリッパ・観測の cache・乱数・API の warm start）が、
  別に作った env に**そのまま**戻る（`gym_state.capture` の結果が一致する）
- 戻した env で同じコードを実行すると、元の env と同じ結果になる（ビット単位）
- MuJoCo の物理状態だけを戻したのでは、続きが一致しない（Python 側の状態も要る）
- 戻すときも、観測を作り直すときも、物理を進めない
- Snapshot を取っても、元の trial の結果は変わらない
- モデルが違う env には戻さない

robosuite か LIBERO が入った環境でだけ動く（無ければ skip）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_LIBERO = importlib.util.find_spec("libero") is not None
HAS_ROBOSUITE = importlib.util.find_spec("robosuite") is not None and not HAS_LIBERO
pytestmark = pytest.mark.skipif(not (HAS_LIBERO or HAS_ROBOSUITE), reason="robosuite も LIBERO も無い")

STEP1 = """\
import numpy as np
home = get_joints()
target = home + np.array([0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
def helper(scale):
    return home + (target - home) * scale
move_to_joints(helper(0.5))
close_gripper()
"""
STEP2 = """\
move_to_joints(helper(1.0))
open_gripper()
print(repr(get_joints().tolist()))
"""

ROBOSUITE_TASKS = [
    "franka_robosuite_cubes_low_level",
    "franka_robosuite_cube_lift_low_level",
    "franka_robosuite_cubes_restack_low_level",
    "franka_robosuite_nut_assembly_low_level",
    "franka_robosuite_spill_wipe_low_level",
    "two_arm_lift_robosuite",
]


def _low_level(name: str):
    if name == "libero":
        from capx.envs.simulators.libero import FrankaLiberoEnv

        return FrankaLiberoEnv("libero_object", 0, privileged=True)
    from capx.envs.base import _ENV_FACTORIES

    # `get_env` は同じインスタンスを使い回す（lru_cache）。比べる 2 つの env が同じ
    # シミュレータを共有しないよう、毎回作る
    return _ENV_FACTORIES[name](privileged=True, enable_render=True, viser_debug=False)


TASKS = ["libero"] if HAS_LIBERO else ROBOSUITE_TASKS


def _make(name: str):
    import snapshot_support

    return snapshot_support.make_env(_low_level(name))


def _close(*envs) -> None:
    for env in envs:
        env.close()
        env.inner._process_executor.close()


def _state(env) -> dict:
    """比べる用の、Gym の内部状態（`capture` そのもの）。"""
    from capx.snapshot import gym_state

    state, _info = gym_state.capture(env.inner)
    return state


def _assert_same(a, b, path="") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: {set(a) ^ set(b)}"
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), path
    else:
        assert a == b or (a != a and b != b), f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("task", TASKS)
def test_restore_is_exact_and_the_replay_matches(task) -> None:
    import snapshot_support as ss

    a = _make(task)
    a.reset_for_trial(1)
    assert a.step(STEP1).ok
    data, _timing = a.snapshot()
    saved = _state(a)
    second_a = a.step(STEP2)
    after_a = ss.physics(a)
    outcome_a = a.evaluate()

    b = _make(task)
    info = b.restore(data)
    assert info["agent_step"] == 1 and "helper" in info["namespace_names"]
    _assert_same(saved, _state(b))  # 内部状態がそのまま戻る
    second_b = b.step(STEP2)
    assert second_b.stdout == second_a.stdout
    for key, value in after_a.items():
        assert np.array_equal(value, ss.physics(b)[key]), key
    outcome_b = b.evaluate()
    assert (outcome_b.reward, outcome_b.task_completed) == (outcome_a.reward, outcome_a.task_completed)
    assert len(b.recorded_steps) == 2, "予算（残り Step 数）も戻る"
    _close(a, b)


def test_mujoco_state_alone_is_not_enough() -> None:
    """物理状態だけを戻すと、コントローラ・グリッパの指令値などが reset 直後のままで、続きがずれる。"""
    import snapshot_support as ss
    from capx.snapshot import blob, gym_state

    task = TASKS[0]
    a = _make(task)
    a.reset_for_trial(1)
    a.step(STEP1)
    data, _ = a.snapshot()
    a.step(STEP2)

    b = _make(task)
    b.restore(data)
    # Python 側の状態を、reset 直後のものに戻してしまう（物理だけを戻した状態を作る）
    fresh = _make(task)
    fresh.reset_for_trial(1)
    fresh_state = _state(fresh)
    snap = blob.unpack(data)
    partial = dict(fresh_state, mujoco=snap.gym["env"]["mujoco"])
    gym_state.restore(b.inner, partial, snap.manifest["gym"])
    b.step(STEP2)
    assert not np.array_equal(ss.physics(a)["qpos"], ss.physics(b)["qpos"])
    _close(a, b, fresh)


@pytest.mark.parametrize("task", TASKS[:1])
def test_restore_and_observation_do_not_step_physics(task) -> None:
    import snapshot_support as ss

    a = _make(task)
    a.reset_for_trial(1)
    a.step(STEP1)
    data, _ = a.snapshot()
    b = _make(task)
    b.restore(data)
    before = ss.physics(b)
    assert b.render()
    obs = b.inner.low_level_env.get_observation()
    assert any("rgb" in v.get("images", {}) for v in obs.values() if isinstance(v, dict)), "画像を作り直せる"
    after = ss.physics(b)
    for key in before:
        assert np.array_equal(before[key], after[key]), key
    assert ss.physics(a)["time"] == after["time"]
    low_a, low_b = a.inner.low_level_env, b.inner.low_level_env
    if hasattr(low_a, "_current_obs"):  # LIBERO の観測の cache（画像以外）も一致する
        for key, value in low_a._current_obs.items():
            if not key.endswith(("_image", "_depth")):
                assert np.array_equal(np.asarray(value), np.asarray(low_b._current_obs[key])), key
    _close(a, b)


def test_taking_snapshots_does_not_change_the_trial() -> None:
    import snapshot_support as ss

    task = TASKS[0]
    seed_env = _make(task)
    seed_env.reset_for_trial(1)
    start, _ = seed_env.snapshot()  # 2 つの env を同じ状態から始める
    on, off = _make(task), _make(task)
    on.restore(start)
    off.restore(start)
    for code in (STEP1, STEP2):
        on.step(code)
        on.snapshot()
        on.snapshot()
        off.step(code)
    for key, value in ss.physics(on).items():
        assert np.array_equal(value, ss.physics(off)[key]), key
    assert on.evaluate().reward == off.evaluate().reward
    _close(seed_env, on, off)


def test_an_incompatible_model_is_refused() -> None:
    from capx.snapshot import blob, gym_state

    task = TASKS[0]
    a = _make(task)
    a.reset_for_trial(1)
    snap = blob.unpack(a.snapshot()[0])
    info = dict(snap.manifest["gym"], model_signature="sha256:other")
    with pytest.raises(gym_state.GymSnapshotError, match="モデル"):
        gym_state.restore(a.inner, snap.gym["env"], info)
    _close(a)


def test_an_unsupported_executor_is_refused() -> None:
    """生成コードを同じプロセスで動かす env（ローカル実行）は、Snapshot を取らない。"""
    from capx.agent_api import Budget
    from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
    from capx.local_env import LocalAgentEnv
    from capx.snapshot.gym_state import GymSnapshotError

    import snapshot_support as ss

    env = LocalAgentEnv(
        CodeExecutionEnvBase(CodeExecEnvConfig(low_level=_low_level(TASKS[0]), apis=[ss.API_NAME], prompt="p")),
        budget=Budget(),
        task_id="t",
    )
    env.reset_for_trial(1)
    with pytest.raises(GymSnapshotError, match="ローカル実行は未対応"):
        env.snapshot()
