"""Snapshot / Resume / Fork を、backend・別プロセスの worker・保存先を本当に動かして確かめる。

受け入れのシナリオ（docs/Snapshot.md）をそのまま追う。

1. Snapshot を有効にして trial を回す。Step 1 でヘルパー関数を定義し、Step 2 で使い、Step 3 で失敗する
2. 元の worker を終わらせる
3. Step 2 の Snapshot から、新しい worker で再開する。Step 1 の関数と変数がそのまま使える
4. 元の Step 3 と違うコードを実行し、そのまま次の Step も続ける（Multi-turn）
5. 同じ Snapshot から別の worker を起こし、元の Step 3 を実行し直すと、元と同じ結果になる
6. Fork から取った Snapshot から、さらに分岐できる
7. 元の Snapshot は変わらない。backend を起こし直しても再開できる
8. 他人の Snapshot・壊れた Snapshot・不完全な Snapshot からは再開できない

Docker は使わない（tests/snapshot_harness.py）。シミュレータが要るので、robosuite か LIBERO が
入った環境でだけ動く（無ければ skip）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


def _runtime() -> str | None:
    if importlib.util.find_spec("libero") is not None:
        return "libero"
    if importlib.util.find_spec("robosuite") is not None:
        return "robosuite"
    return None


RUNTIME = _runtime()
pytestmark = pytest.mark.skipif(RUNTIME is None, reason="robosuite も LIBERO も無い")

STEP1 = """\
import numpy as np
home = get_joints()
target = home + np.array([0.15, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
def helper(scale):
    return home + (target - home) * scale
class Memo:
    def __init__(self):
        self.log = []
memo = Memo()
memo.log.append("step1")
print(repr(get_joints().tolist()))
"""
STEP2 = """\
move_to_joints(helper(0.5))
close_gripper()
memo.log.append("step2")
print(repr(get_joints().tolist()))
"""
# 元の Step 3: 動かしすぎて、最後に失敗する
STEP3_ORIGINAL = """\
move_to_joints(helper(2.0))
print(repr(get_joints().tolist()))
raise RuntimeError("missed the object")
"""
# 直した Step 3
STEP3_FIXED = """\
move_to_joints(helper(1.0))
open_gripper()
memo.log.append("fixed3")
print(memo.log, repr(get_joints().tolist()))
"""
STEP4 = """\
memo.log.append("step4")
print(memo.log)
"""

TOKENS = {"tokA": "alice", "tokB": "bob"}


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from snapshot_harness import Backend

    base = tmp_path_factory.mktemp("snapshot_e2e")
    (base / "logs").mkdir()
    server = Backend(base / "store", base / "logs", tokens=TOKENS).start()
    yield server
    server.stop()


@pytest.fixture
def as_user(monkeypatch):
    def use(token: str) -> None:
        monkeypatch.setenv("CAPX_ENV_SERVER_TOKEN", token)

    use("tokA")
    return use


def _env(backend, **kw):
    from capx.remote_env.client import RemoteAgentEnv

    return RemoteAgentEnv(server_url=backend.url, task_id=f"snap_{RUNTIME}", timeout_s=900, **kw)


def _blob_sha(backend, snapshot_id: str) -> str:
    path = next(Path(backend.store_dir).rglob(f"{snapshot_id}.snap"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scenario_fork_from_a_past_failure(backend, as_user) -> None:
    from capx.remote_env.client import SnapshotError

    # 1. Snapshot を有効にして回す
    original = _env(backend, snapshot_enabled=True, record_video=True)
    task = original.reset_for_trial(1)
    assert original.initial_snapshot_id
    results = [original.step(code) for code in (STEP1, STEP2, STEP3_ORIGINAL)]
    assert [r.ok for r in results] == [True, True, False]
    assert all(r.snapshot_id for r in results), "例外で終わった Step でも Snapshot は取れる"
    assert len({original.initial_snapshot_id, *(r.snapshot_id for r in results)}) == 4
    step2_id = results[1].snapshot_id
    outcome = original.evaluate()
    step2_sha = _blob_sha(backend, step2_id)

    # 2. 元の worker を終わらせる
    worker_names = set(backend.docker.procs)
    original.close()
    assert not (worker_names & set(backend.docker.procs)), "元の worker は消えている"

    # 3. Step 2 の Snapshot から、新しい worker で再開する（reset は呼ばない）
    fork = _env(backend, resume_from=step2_id, snapshot_enabled=True, record_video=True)
    assert fork.task.instruction == task.instruction and fork.task.task_id == task.task_id
    assert fork.agent_step == 2 and [h["code"] for h in fork.history] == [STEP1, STEP2]
    assert "helper" in fork.resume_info["namespace_names"]

    # 4. 違うコードを実行し、そのまま続ける
    fixed = fork.step(STEP3_FIXED)
    assert fixed.ok, fixed.stderr
    assert fixed.stdout.startswith("['step1', 'step2', 'fixed3']"), "Step 1 の関数・変数がそのまま使える"
    assert fixed.snapshot_id and fork.agent_step == 3
    assert fixed.video, "再開後も動画が録れる"
    assert fork.render(), "再開後も画像が取れる"
    step4 = fork.step(STEP4)
    assert step4.stdout.strip() == "['step1', 'step2', 'fixed3', 'step4']"
    fork.evaluate()
    fork.close()

    # 5. 同じ Snapshot から別の worker で、元の Step 3 をやり直すと、元と同じ結果になる
    replay = _env(backend, resume_from=step2_id)
    again = replay.step(STEP3_ORIGINAL)
    assert again.stdout == results[2].stdout, "関節角が元の Step 3 と一致する"
    assert again.stderr == results[2].stderr
    replayed = replay.evaluate()
    assert replayed.reward == pytest.approx(outcome.reward, abs=1e-9)
    assert replayed.task_completed == outcome.task_completed
    replay.close()

    # 6. Fork の Snapshot から、さらに分岐する
    deeper = _env(backend, resume_from=fixed.snapshot_id)
    assert deeper.agent_step == 3
    assert deeper.step(STEP4).stdout.strip() == "['step1', 'step2', 'fixed3', 'step4']"
    deeper.close()

    # 7. 元の Snapshot は変わらない。backend を起こし直しても再開できる
    assert _blob_sha(backend, step2_id) == step2_sha
    backend.restart()
    after_restart = _env(backend, resume_from=step2_id)
    assert after_restart.step("print(memo.log)").stdout.strip() == "['step1', 'step2']"
    after_restart.close()

    # 8. 他人の Snapshot からは再開できない（存在も教えない）
    as_user("tokB")
    with pytest.raises(SnapshotError, match="404"):
        _env(backend, resume_from=step2_id)
    as_user("tokA")

    # 壊れた Snapshot・不完全な Snapshot は読まない
    initial = next(Path(backend.store_dir).rglob(f"{results[0].snapshot_id}.snap"))
    data = bytearray(initial.read_bytes())
    data[len(data) // 2] ^= 0xFF
    initial.write_bytes(bytes(data))
    with pytest.raises(SnapshotError, match="壊れている"):
        _env(backend, resume_from=results[0].snapshot_id)
    record = next(Path(backend.store_dir).rglob(f"{results[2].snapshot_id}.json"))
    record.unlink()  # 記録が無い = 完成していない
    with pytest.raises(SnapshotError, match="404"):
        _env(backend, resume_from=results[2].snapshot_id)
    assert not backend.docker.procs, "再開できなかった worker は残らない"


def test_snapshot_off_sends_nothing_and_changes_nothing(backend, as_user, monkeypatch) -> None:
    monkeypatch.delenv("CAPX_SNAPSHOT_ENABLED", raising=False)
    env = _env(backend)
    assert env.snapshot_enabled is False
    env.reset_for_trial(1)
    result = env.step(STEP1)
    assert result.ok and result.snapshot_id is None and env.initial_snapshot_id is None
    env.close()


@pytest.mark.skipif(RUNTIME != "libero", reason="LIBERO の初期状態は試行番号で決まるので、ON / OFF を比べられる")
def test_snapshot_on_and_off_give_the_same_result(backend, as_user) -> None:
    outputs = []
    for enabled in (True, False):
        env = _env(backend, snapshot_enabled=enabled)
        env.reset_for_trial(1)
        steps = [env.step(code) for code in (STEP1, STEP2, STEP3_ORIGINAL)]
        outcome = env.evaluate()
        outputs.append(([s.stdout for s in steps], outcome.reward, outcome.task_completed))
        env.close()
    assert outputs[0] == outputs[1]


def test_env_file_setting_turns_snapshots_on(backend, as_user, monkeypatch) -> None:
    monkeypatch.setenv("CAPX_SNAPSHOT_ENABLED", "true")
    env = _env(backend)
    assert env.snapshot_enabled
    env.close()
    off = _env(backend, snapshot_enabled=False)  # 引数が環境変数より優先
    assert not off.snapshot_enabled
    off.close()
    os.environ.pop("CAPX_SNAPSHOT_ENABLED", None)


# -- remote/forking.py とサンプル --------------------------------------------------


@pytest.fixture
def remote_dir(monkeypatch, backend):
    import sys

    remote = Path(__file__).resolve().parent.parent / "remote"
    monkeypatch.syspath_prepend(str(remote))
    monkeypatch.setenv("CAPX_ENV_SERVER_URL", backend.url)
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield remote
    for name in ("forking", "tracing", "trial", "examples"):
        sys.modules.pop(name, None)


def _trial_with_snapshots(backend) -> list[str]:
    env = _env(backend, snapshot_enabled=True)
    env.reset_for_trial(1)
    ids = [env.step(code).snapshot_id for code in (STEP1, STEP2)]
    env.close()
    return ids


def test_run_fork_saves_results_like_a_trial(backend, as_user, remote_dir, tmp_path) -> None:
    import forking

    _step1, step2 = _trial_with_snapshots(backend)
    env = forking.open_fork(step2, f"snap_{RUNTIME}")
    try:
        assert forking.remaining_budget(env).max_steps == 8
        summary = forking.run_fork(
            env, forking.RunCode([STEP3_FIXED, STEP4]), fork_type="human_edit", output_dir=str(tmp_path), model="m"
        )
    finally:
        env.close()
    assert summary.num_code_blocks == 2 and summary.success
    result = next(tmp_path.rglob("result.json"))
    assert "fork_snap_" in str(result), "Fork の結果は元の trial と別のフォルダ"
    import json

    snapshots = json.loads(result.read_text())["snapshots"]
    assert snapshots["resumed_from"] == step2 and snapshots["resumed_at_step"] == 2
    assert len(snapshots["steps"]) == 2 and all(snapshots["steps"])
    code = (result.parent / "code.py").read_text()
    assert "fixed3" in code and "step4" in code


def test_resume_through_an_s3_store(tmp_path, monkeypatch) -> None:
    """保存先が S3（MinIO と同じ API）のとき。backend を起こし直しても再開できる。"""
    pytest.importorskip("moto.server")
    import socket

    from moto.server import ThreadedMotoServer
    from snapshot_harness import Backend

    from capx.snapshot.s3 import S3Client
    from capx.snapshot.store import S3Store

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    s3 = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    s3.start()
    endpoint = f"http://127.0.0.1:{port}"
    S3Client(endpoint, "k", "s")._request("PUT", "capx-snapshots")
    store = S3Store(endpoint=endpoint, bucket="capx-snapshots", access_key_id="k", secret_access_key="s")
    (tmp_path / "logs").mkdir()
    server = Backend(tmp_path / "unused", tmp_path / "logs", store=store).start()
    monkeypatch.delenv("CAPX_ENV_SERVER_TOKEN", raising=False)
    try:
        env = _env(server, snapshot_enabled=True)
        env.reset_for_trial(1)
        step1 = env.step(STEP1)
        env.close()
        assert {r["snapshot_id"] for r in store.list("dev")} == {env.initial_snapshot_id, step1.snapshot_id}
        server.restart()
        resumed = _env(server, resume_from=step1.snapshot_id)
        assert resumed.step("print(memo.log)").stdout.strip() == "['step1']"
        resumed.close()
    finally:
        server.stop()
        s3.stop()
