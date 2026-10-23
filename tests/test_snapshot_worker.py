"""worker の Snapshot の口。実際の ZMQ（CURVE）と偽の env で確かめる。

- Snapshot の取り出し・復元は backend の鍵の接続だけができる（クライアントは中身を読めず、
  自分で作った bytes を戻させることもできない）
- `--resume` の worker は、復元が済むまで step などを受け付けない
- Snapshot が取れなくても step は成功のまま返る（理由は `snapshot_error`）
- 古いクライアントが知らない項目（`snapshot_id`）を送らない
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import zmq

from capx.agent_api import StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env import protocol
from capx.remote_env.server.probe import WorkerCallError, _call
from capx.remote_env.worker.server import WorkerServer


class FakeEnv:
    def __init__(self, snapshot_error: Exception | None = None) -> None:
        self.codes: list[str] = []
        self.restored: list[bytes] = []
        self.snapshot_error = snapshot_error

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        self.codes.append(code)
        return StepResult(request_id="r", ok=True, stdout="out", stderr="")

    def render(self, camera: str = "main") -> bytes:
        return b"img"

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        return TaskSpec(task_id="fake", seed=trial, instruction="", api_docs="")

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(0.25, False, False, False, 0)

    def snapshot(self, **_kw):
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return f"blob-{len(self.codes)}".encode(), {"gym_s": 0.0}

    def restore(self, data: bytes) -> dict:
        self.restored.append(data)
        if data == b"bad":
            raise RuntimeError("壊れている")
        return {"task": {"task_id": "fake"}, "agent_step": 2, "history": []}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def worker(request):
    """CURVE の worker。backend の鍵とクライアントの鍵を許可する。"""
    options = getattr(request, "param", {})
    server_pub, server_sec = zmq.curve_keypair()
    backend, client = zmq.curve_keypair(), zmq.curve_keypair()
    env = FakeEnv(snapshot_error=options.get("snapshot_error"))
    port = _free_port()
    server = WorkerServer(env, backend_key=backend[0], restore_pending=options.get("resume", False))
    threading.Thread(
        target=server.serve,
        kwargs={
            "port": port,
            "host": "127.0.0.1",
            "curve_secret_key": server_sec,
            "allowed_client_keys": {backend[0], client[0]},
        },
        daemon=True,
    ).start()
    time.sleep(0.5)

    def call(keypair, operation, **payload):
        return _call(f"tcp://127.0.0.1:{port}", server_pub.decode(), keypair, operation, 10.0, payload)

    yield {"env": env, "server": server, "backend": backend, "client": client, "call": call}
    server._stop.set()
    time.sleep(0.2)


def test_step_returns_a_token_and_only_the_backend_can_fetch_it(worker) -> None:
    call = worker["call"]
    reply = call(worker["client"], "step", code="x = 1", snapshot=True)
    token = reply["snapshot"]["token"]
    assert reply["snapshot"]["bytes"] == len(b"blob-1")
    assert "snapshot_id" not in reply["result"], "古いクライアントは知らない項目で壊れる"

    with pytest.raises(WorkerCallError, match="backend だけ"):
        call(worker["client"], "snapshot_fetch", token=token)
    assert call(worker["backend"], "snapshot_fetch", token=token)["data"] == b"blob-1"
    with pytest.raises(WorkerCallError, match="無い"):
        call(worker["backend"], "snapshot_fetch", token=token)  # 1 回だけ取り出せる


def test_reset_can_snapshot_and_without_the_flag_nothing_is_kept(worker) -> None:
    call = worker["call"]
    assert "snapshot" in call(worker["client"], "reset", trial=1, snapshot=True)
    plain = call(worker["client"], "step", code="y = 2")
    assert "snapshot" not in plain and "snapshot_error" not in plain


def test_a_client_cannot_restore_its_own_bytes(worker) -> None:
    with pytest.raises(WorkerCallError, match="backend だけ"):
        worker["call"](worker["client"], "restore", data=b"forged")
    assert worker["env"].restored == []


@pytest.mark.parametrize("worker", [{"resume": True}], indirect=True)
def test_a_resume_worker_is_locked_until_restored(worker) -> None:
    call = worker["call"]
    for operation in ("step", "render", "reset", "evaluate"):
        with pytest.raises(WorkerCallError, match="復元が済んでいない"):
            call(worker["client"], operation, code="x = 1")
    with pytest.raises(WorkerCallError, match="壊れている"):
        call(worker["backend"], "restore", data=b"bad")
    with pytest.raises(WorkerCallError, match="復元が済んでいない"):
        call(worker["client"], "step", code="x = 1")  # 失敗した後も使えない
    assert call(worker["backend"], "restore", data=b"good")["resume"]["agent_step"] == 2
    assert call(worker["client"], "step", code="x = 1")["result"]["ok"]
    assert worker["env"].codes == ["x = 1"]


@pytest.mark.parametrize("worker", [{"snapshot_error": RuntimeError("cannot pickle 'generator'")}], indirect=True)
def test_a_failed_snapshot_does_not_fail_the_step(worker) -> None:
    reply = worker["call"](worker["client"], "step", code="x = 1", snapshot=True)
    assert reply["result"]["ok"] and "snapshot" not in reply
    assert "generator" in reply["snapshot_error"]


def test_only_a_few_pending_snapshots_are_kept(worker) -> None:
    call = worker["call"]
    tokens = [call(worker["client"], "step", code=f"x = {i}", snapshot=True)["snapshot"]["token"] for i in range(4)]
    with pytest.raises(WorkerCallError):
        call(worker["backend"], "snapshot_fetch", token=tokens[0])  # 古いものは捨てた
    assert call(worker["backend"], "snapshot_fetch", token=tokens[-1])["data"] == b"blob-4"


def test_ping_still_answers_while_restore_is_pending() -> None:
    server = WorkerServer(FakeEnv(), restore_pending=True)
    reply = server._dispatch(protocol.request("ping"), b"id")
    assert reply.kind == "response"
