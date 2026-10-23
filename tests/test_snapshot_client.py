"""クライアント側の Snapshot（ON/OFF の決まり、ID の受け取り、保存に失敗したとき、再開）。

backend の HTTP は `requests.post` の差し替えで、worker は本物の `WorkerServer`（偽の env）で動かす。
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from capx.agent_api import StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env import client as client_module
from capx.remote_env.client import RemoteAgentEnv, SnapshotError, snapshot_enabled_from_env
from capx.remote_env.worker.server import WorkerServer


class FakeEnv:
    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        return StepResult(request_id="r", ok=True, stdout=code, stderr="")

    def render(self, camera: str = "main") -> bytes:
        return b"img"

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        return TaskSpec(task_id="fake", seed=trial, instruction="do", api_docs="")

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(0.0, False, False, False, 0)

    def snapshot(self, **_kw):
        return b"blob", {}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeBackend:
    """`requests.post` の代わり。セッション作成と Snapshot の保存に答える。"""

    def __init__(self, endpoint: str, fail_save: bool = False, resume: dict | None = None) -> None:
        self.endpoint = endpoint
        self.fail_save = fail_save
        self.resume = resume
        self.saved: list[dict] = []
        self.created: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        if url.endswith("/sessions"):
            self.created.append(json)
            if json.get("resume_from") == "snap_missing":
                return _response(404, {"detail": "その Snapshot は無い"})
            body = {"zmq_endpoint": self.endpoint, "session_id": "s1", "server_public_key": "k" * 40}
            if json.get("resume_from"):
                body["resume"] = self.resume
            return _response(200, body)
        if url.endswith("/snapshots"):
            self.saved.append(json)
            if self.fail_save:
                return _response(502, {"detail": "MinIO に届かない"})
            return _response(200, {"snapshot_id": f"snap_{len(self.saved):024x}", "timing": {}})
        raise AssertionError(url)


def _response(status: int, body: dict):
    return SimpleNamespace(status_code=status, json=lambda: body, text=str(body))


@pytest.fixture
def worker():
    port = _free_port()
    server = WorkerServer(FakeEnv())
    threading.Thread(target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True).start()
    time.sleep(0.4)
    yield f"tcp://127.0.0.1:{port}"
    server._stop.set()
    time.sleep(0.2)


def _connect(monkeypatch, backend: FakeBackend, **kw) -> RemoteAgentEnv:
    import requests

    monkeypatch.setattr(requests, "post", backend.post)
    env = RemoteAgentEnv(server_url="http://backend", task_id="fake", **kw)
    # テストの worker は CURVE を使わないので、鍵なしで繋ぎ直す
    env._socket.close()
    env._curve_server_key = None
    env._connect()
    return env


def test_setting_precedence(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_SNAPSHOT_ENABLED", raising=False)
    assert snapshot_enabled_from_env(None) is False
    monkeypatch.setenv("CAPX_SNAPSHOT_ENABLED", "true")
    assert snapshot_enabled_from_env(None) is True
    assert snapshot_enabled_from_env(False) is False, "引数が環境変数より優先"
    monkeypatch.setenv("CAPX_SNAPSHOT_ENABLED", "0")
    assert snapshot_enabled_from_env(True) is True


def test_direct_endpoints_never_snapshot(monkeypatch, worker) -> None:
    monkeypatch.setenv("CAPX_SNAPSHOT_ENABLED", "true")
    env = RemoteAgentEnv(endpoint=worker)
    assert env.snapshot_enabled is False, "保存する backend が無い"
    env.close()
    with pytest.raises(ValueError, match="server_url"):
        RemoteAgentEnv(endpoint=worker, snapshot_enabled=True)
    with pytest.raises(ValueError, match="server_url"):
        RemoteAgentEnv(endpoint=worker, resume_from="snap_x")


def test_reset_and_every_step_get_an_id(monkeypatch, worker) -> None:
    backend = FakeBackend(worker)
    env = _connect(monkeypatch, backend, snapshot_enabled=True)
    env.snapshot_context.update(trace_id="t1", observation_id="o1")
    env.reset_for_trial(1)
    first, second = env.step("a = 1"), env.step("b = 2")
    assert env.initial_snapshot_id and first.snapshot_id and second.snapshot_id
    assert len({env.initial_snapshot_id, first.snapshot_id, second.snapshot_id}) == 3
    assert env.last_snapshot_id == second.snapshot_id and env.agent_step == 2
    assert [s["meta"]["agent_step"] for s in backend.saved] == [0, 1, 2]
    assert backend.saved[1]["meta"]["trace_id"] == "t1"
    assert len(env.snapshot_timings) == 3
    env.close()


def test_a_failed_save_keeps_the_trial_going(monkeypatch, worker, capsys) -> None:
    env = _connect(monkeypatch, FakeBackend(worker, fail_save=True), snapshot_enabled=True)
    env.reset_for_trial(1)
    result = env.step("a = 1")
    assert result.ok and result.snapshot_id is None
    assert "MinIO に届かない" in env.last_snapshot_error
    assert "保存できなかった" in capsys.readouterr().out
    env.close()


def test_snapshot_off_sends_no_flag(monkeypatch, worker) -> None:
    monkeypatch.delenv("CAPX_SNAPSHOT_ENABLED", raising=False)
    backend = FakeBackend(worker)
    env = _connect(monkeypatch, backend)
    env.reset_for_trial(1)
    assert env.step("a = 1").snapshot_id is None and backend.saved == []
    env.close()


def test_resume_needs_no_reset(monkeypatch, worker) -> None:
    resume = {
        "task": {"task_id": "fake", "seed": 1, "instruction": "do", "api_docs": "", "cameras": ["main"]},
        "agent_step": 2,
        "history": [{"code": "a = 1", "ok": True, "stdout": "", "stderr": ""}] * 2,
        "meta": {"trace_id": "t0", "observation_id": "o0", "agent_state": {"plan": "x"}},
    }
    backend = FakeBackend(worker, resume=resume)
    env = _connect(monkeypatch, backend, resume_from="snap_" + "1" * 24, snapshot_enabled=True)
    assert backend.created[0]["resume_from"] == "snap_" + "1" * 24
    assert env.task.instruction == "do" and env.task.cameras == ("main",)
    assert env.agent_step == 2 and len(env.history) == 2
    assert env.last_snapshot_id == "snap_" + "1" * 24
    result = env.step("c = 3")
    assert env.agent_step == 3 and result.snapshot_id
    assert backend.saved[-1]["meta"]["fork_of"] == "snap_" + "1" * 24
    env.close()


def test_resume_failure_is_a_clear_error(monkeypatch, worker) -> None:
    import requests

    monkeypatch.setattr(requests, "post", FakeBackend(worker).post)
    with pytest.raises(SnapshotError, match="snap_missing.*404"):
        RemoteAgentEnv(server_url="http://backend", task_id="fake", resume_from="snap_missing")


def test_agent_state_rides_along(monkeypatch, worker) -> None:
    backend = FakeBackend(worker)
    env = _connect(monkeypatch, backend, snapshot_enabled=True)
    env.reset_for_trial(1)
    env.set_agent_state({"turn": 1})
    env.step("a = 1")
    assert backend.saved[-1]["meta"]["agent_state"] == {"turn": 1}
    env.close()
