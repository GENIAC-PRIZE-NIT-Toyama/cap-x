"""backend の Snapshot（保存・再開・権限）。docker と worker は偽物。

見たいのは、
- Snapshot の id は保存先に置き終わってから返す（途中で失敗したら id を返さない）
- 再開は新しい worker で行い、戻せなければ worker を片付けて失敗を返す
- 他人の Snapshot は保存も再開も一覧もできない
- backend は pickle を解かない（bytes の形と sha256 だけを見る）
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import zmq

from capx.remote_env.server import docker as dk
from capx.remote_env.server.probe import WorkerCallError
from capx.remote_env.server.sessions import Config, SessionError, SessionManager
from capx.snapshot import blob
from capx.snapshot.store import DirStore


def _key() -> str:
    return zmq.curve_keypair()[0].decode("ascii")


def _blob(task_id: str = "cube_stack", agent_step: int = 2) -> bytes:
    return blob.pack(
        {"task_id": task_id, "agent_step": agent_step, "trial": 1, "seed": 1},
        gym={"env": {"qpos": np.zeros(3)}, "local_env": {}},
        # 本物の pickle ではない。backend が解こうとすれば失敗する（解かないことの確認）
        namespace=b"\x80\x04not-unpickled-by-the-backend",
        history=[],
    )


class FakeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def __call__(self, cmd):
        self.commands.append(cmd)
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 1, "", ""
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true", ""
        return 0, "", ""


class FakeWorker:
    """backend から呼ばれる worker の代わり。"""

    def __init__(self, data: bytes | None = None, restore_error: str | None = None) -> None:
        self.data = data or _blob()
        self.restore_error = restore_error
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, session, operation, timeout_s, **payload):
        self.calls.append((operation, payload))
        if operation == "snapshot_fetch":
            if payload["token"] != "tok":
                raise WorkerCallError("その Snapshot は無い")
            return {"data": self.data}
        if operation == "restore":
            if self.restore_error:
                raise WorkerCallError(self.restore_error)
            return {
                "resume": {"task": {"task_id": "cube_stack"}, "agent_step": 2, "history": []},
                "restore_s": 0.1,
            }
        raise AssertionError(operation)


async def _alive(_session):
    return {"busy": False, "idle_s": 0.0}


def _manager(tmp_path, worker=None, store=True) -> tuple[SessionManager, FakeDocker]:
    docker = FakeDocker()
    manager = SessionManager(
        Config(public_host="gpu.example", port_start=19500, port_end=19510),
        runner=docker,
        prober=_alive,
        caller=worker or FakeWorker(),
        store=DirStore(tmp_path) if store else None,
    )
    return manager, docker


def run(coro):
    return asyncio.run(coro)


def test_snapshot_is_saved_then_its_id_is_returned(tmp_path) -> None:
    manager, _ = _manager(tmp_path)
    session = run(manager.create("alice", "cube_stack", _key()))
    record = run(manager.persist_snapshot(session, "tok", {"trace_id": "t1", "observation_id": "o1"}))

    assert record["snapshot_id"].startswith("snap_") and record["agent_step"] == 2
    assert record["meta"] == {"trace_id": "t1", "observation_id": "o1"}
    data, stored = manager.store.get("alice", record["snapshot_id"])
    assert data == _blob() and stored["owner"] == "alice" and stored["task_id"] == "cube_stack"


def test_nothing_is_published_when_saving_fails(tmp_path) -> None:
    class Broken(DirStore):
        def put(self, *a, **kw):
            from capx.snapshot.store import SnapshotStoreError

            raise SnapshotStoreError("MinIO に届かない")

    manager, _ = _manager(tmp_path)
    manager.store = Broken(tmp_path)
    session = run(manager.create("alice", "cube_stack", _key()))
    with pytest.raises(SessionError) as exc:
        run(manager.persist_snapshot(session, "tok"))
    assert exc.value.status == 502 and manager.store.list("alice") == []


@pytest.mark.parametrize(
    "data, status",
    [(b"garbage", 502), (_blob(task_id="nut_assembly"), 502)],
)
def test_a_bad_or_foreign_snapshot_from_the_worker_is_not_stored(tmp_path, data, status) -> None:
    manager, _ = _manager(tmp_path, FakeWorker(data))
    session = run(manager.create("alice", "cube_stack", _key()))
    with pytest.raises(SessionError) as exc:
        run(manager.persist_snapshot(session, "tok"))
    assert exc.value.status == status and manager.store.list("alice") == []


def test_unknown_token_and_oversized_meta(tmp_path) -> None:
    manager, _ = _manager(tmp_path)
    session = run(manager.create("alice", "cube_stack", _key()))
    with pytest.raises(SessionError) as exc:
        run(manager.persist_snapshot(session, "nope"))
    assert exc.value.status == 409
    with pytest.raises(SessionError, match="大きすぎる"):
        run(manager.persist_snapshot(session, "tok", {"agent_state": "x" * 70_000}))


def test_resume_starts_a_new_locked_worker_and_restores_it(tmp_path) -> None:
    worker = FakeWorker()
    manager, docker = _manager(tmp_path, worker)
    first = run(manager.create("alice", "cube_stack", _key()))
    snapshot_id = run(manager.persist_snapshot(first, "tok"))["snapshot_id"]

    resumed = run(manager.create("alice", "cube_stack", _key(), resume_from=snapshot_id))
    assert resumed.session_id != first.session_id, "元のセッションは使わない"
    assert resumed.resume["snapshot_id"] == snapshot_id and resumed.resume["agent_step"] == 2
    assert worker.calls[-1] == ("restore", {"data": _blob()})
    run_cmd = dk.run_command(resumed.spec)
    assert "--resume" in run_cmd, "復元が済むまで step を受け付けない worker として起こす"
    assert f"CAPX_BACKEND_PUBLIC_KEY={manager._probe_public_key()}" in run_cmd
    assert "--resume" not in dk.run_command(first.spec)


def test_a_failed_restore_cleans_up_the_new_worker(tmp_path) -> None:
    worker = FakeWorker()
    manager, docker = _manager(tmp_path, worker)
    snapshot_id = run(manager.persist_snapshot(run(manager.create("alice", "cube_stack", _key())), "tok"))[
        "snapshot_id"
    ]
    worker.restore_error = "GymSnapshotError: モデルが違う"
    with pytest.raises(SessionError, match="復元できなかった.*モデルが違う") as exc:
        run(manager.create("alice", "cube_stack", _key(), resume_from=snapshot_id))
    assert exc.value.status == 500
    assert len(manager.list()) == 1, "失敗した worker は片付けた"
    removed = [c for c in docker.commands if c[:3] == ["docker", "rm", "-f"]]
    assert removed, "コンテナを消した"


def test_resume_is_refused_before_starting_anything(tmp_path) -> None:
    manager, docker = _manager(tmp_path)
    snapshot_id = run(manager.persist_snapshot(run(manager.create("alice", "cube_stack", _key())), "tok"))[
        "snapshot_id"
    ]
    runs_before = len([c for c in docker.commands if c[:2] == ["docker", "run"]])
    cases = [
        ("bob", "cube_stack", snapshot_id, 404),  # 他人のもの
        ("alice", "nut_assembly", snapshot_id, 400),  # タスクが違う
        ("alice", "cube_stack", "snap_" + "0" * 24, 404),  # 無い
        ("alice", "cube_stack", "../../etc/passwd", 404),  # id の形が違う
    ]
    for owner, task, sid, status in cases:
        with pytest.raises(SessionError) as exc:
            run(manager.create(owner, task, _key(), resume_from=sid))
        assert exc.value.status == status, (owner, task, sid)
    assert len([c for c in docker.commands if c[:2] == ["docker", "run"]]) == runs_before


def test_without_a_store_snapshots_are_unavailable(tmp_path) -> None:
    manager, _ = _manager(tmp_path, store=False)
    session = run(manager.create("alice", "cube_stack", _key()))
    with pytest.raises(SessionError) as exc:
        run(manager.persist_snapshot(session, "tok"))
    assert exc.value.status == 503


# -- HTTP ----------------------------------------------------------------------


@pytest.fixture
def http(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from capx.remote_env.server.app import create_app, parse_tokens

    manager, _ = _manager(tmp_path)
    with TestClient(create_app(manager, parse_tokens("alice:tokA,bob:tokB"), reap=False)) as client:
        yield client


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_http_save_list_resume_and_ownership(http) -> None:
    created = http.post("/sessions", json={"task_id": "cube_stack", "client_public_key": _key()}, headers=_auth("tokA"))
    session_id = created.json()["session_id"]

    # bob は alice のセッションの Snapshot を保存できない
    assert http.post(f"/sessions/{session_id}/snapshots", json={"token": "tok"}, headers=_auth("tokB")).status_code == 404
    saved = http.post(f"/sessions/{session_id}/snapshots", json={"token": "tok"}, headers=_auth("tokA"))
    assert saved.status_code == 200
    snapshot_id = saved.json()["snapshot_id"]
    assert "data" not in saved.json(), "本体はクライアントに返さない"

    assert [s["snapshot_id"] for s in http.get("/snapshots", headers=_auth("tokA")).json()["snapshots"]] == [snapshot_id]
    assert http.get("/snapshots", headers=_auth("tokB")).json()["snapshots"] == []
    assert http.get(f"/snapshots/{snapshot_id}", headers=_auth("tokB")).status_code == 404

    resumed = http.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key(), "resume_from": snapshot_id},
        headers=_auth("tokA"),
    )
    assert resumed.status_code == 200 and resumed.json()["resume"]["snapshot_id"] == snapshot_id
    stolen = http.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key(), "resume_from": snapshot_id},
        headers=_auth("tokB"),
    )
    assert stolen.status_code == 404

    assert http.delete(f"/snapshots/{snapshot_id}", headers=_auth("tokB")).status_code == 404
    assert http.delete(f"/snapshots/{snapshot_id}", headers=_auth("tokA")).status_code == 200
    assert http.get(f"/snapshots/{snapshot_id}", headers=_auth("tokA")).status_code == 404
    assert http.get("/health").json()["snapshots"] is True
