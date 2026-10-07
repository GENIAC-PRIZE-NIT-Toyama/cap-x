"""同じ Agent を Local と Remote で走らせて、結果が同じ形になることを見る。

「Remote でも同じように動く」という約束の根拠。Agent からは `AgentEnv` の
`step` / `render` しか見えず、どちらか判別する手段が無い。ここでは実際に同じ
Agent と同じ env を 2 通りの経路（直接 / ZMQ）で走らせて突き合わせる。

あわせて、切断・応答なし・別クライアントでの再接続を見る。
"""

from __future__ import annotations

import dataclasses
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from capx.agent_api import AgentResult, Budget, EnvUnavailable
from capx.bench import run_trial
from capx.local_env import LocalAgentEnv
from capx.remote_env.client import RemoteAgentEnv
from capx.remote_env.worker.server import WorkerServer


class FakeCodeEnv:
    """`CodeExecutionEnvBase` の代役。コードに "fail" があれば失敗する。"""

    def __init__(self) -> None:
        self.low_level_env = SimpleNamespace(_sim_step_count=0, render_camera_names=("main",))
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return {}, {"task_prompt": "Goal: stack the cubes"}

    def step(self, code):
        self._steps += 1
        self.low_level_env._sim_step_count += 10
        failed = "fail" in code
        info = {
            "sandbox_rc": 1 if failed else 0,
            "stdout": "" if failed else f"ran {self._steps}",
            "stderr": "boom" if failed else "",
            "task_completed": self._steps >= 2 and not failed,
        }
        return {}, 0.25 * self._steps, False, False, info

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class StepThenFailAgent:
    def run(self, env, task, budget):
        first = env.step("move()")
        env.step("fail()")
        env.step("move()")
        env.render()
        return AgentResult(notes={"first_ok": first.ok})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(local_env: LocalAgentEnv) -> tuple[WorkerServer, int]:
    port = _free_port()
    server = WorkerServer(local_env, session_id="s")
    threading.Thread(
        target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
    ).start()
    time.sleep(0.4)
    return server, port


def _summary(s) -> dict:
    keep = ("trial", "success", "reward", "terminated", "truncated", "task_completed", "num_code_blocks")
    return {k: getattr(s, k, None) for k in keep}


def test_the_same_agent_gets_the_same_result_locally_and_remotely() -> None:
    config = {"output_dir": None, "record_video": False}

    local = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    local_summary = run_trial(local, StepThenFailAgent(), 1, config, seed=1)

    served = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    server, port = _serve(served)
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
    try:
        remote_summary = run_trial(remote, StepThenFailAgent(), 1, config, seed=1)
        remote_steps = remote.recorded_steps
    finally:
        remote.close()
        server._stop.set()

    assert _summary(remote_summary) == _summary(local_summary)
    assert [(s.code, s.ok) for s in remote_steps] == [
        (s.code, s.ok) for s in local.recorded_steps
    ]


def test_step_results_have_the_same_fields_locally_and_remotely() -> None:
    local = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    local.reset_for_trial(1, 1)
    local_result = local.step("move()")

    served = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    server, port = _serve(served)
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
    try:
        remote.reset_for_trial(1, 1)
        remote_result = remote.step("move()")
    finally:
        remote.close()
        server._stop.set()

    fields = [f.name for f in dataclasses.fields(local_result)]
    assert [f.name for f in dataclasses.fields(remote_result)] == fields
    for name in ("ok", "stdout", "stderr", "truncated", "truncation_reason"):
        assert getattr(remote_result, name) == getattr(local_result, name), name
    assert "reward" not in fields and "task_completed" not in fields


def test_a_dead_worker_becomes_env_unavailable_not_a_hang() -> None:
    served = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    server, port = _serve(served)
    remote = RemoteAgentEnv(
        endpoint=f"tcp://127.0.0.1:{port}", session_id="s", timeout_s=1.5
    )
    try:
        remote.reset_for_trial(1, 1)
        server._stop.set()  # worker が落ちる
        time.sleep(0.5)
        with pytest.raises(EnvUnavailable):
            remote.step("move()")
    finally:
        remote.close()


def test_nobody_listening_becomes_env_unavailable() -> None:
    remote = RemoteAgentEnv(
        endpoint=f"tcp://127.0.0.1:{_free_port()}", session_id="s", timeout_s=1.0
    )
    try:
        with pytest.raises(EnvUnavailable):
            remote.reset_for_trial(1, 1)
    finally:
        remote.close()


def test_a_new_client_can_connect_after_the_first_one_leaves() -> None:
    served = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    server, port = _serve(served)
    try:
        first = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
        first.reset_for_trial(1, 1)
        first.step("move()")
        # close() は worker を止めない経路もある。同じ端点へ別クライアントで再接続する
        first._socket.close(0)
        first._socket = None

        second = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
        assert second.reset_for_trial(2, 2).seed == 2
    finally:
        server._stop.set()


def test_the_trial_wall_clock_stops_the_next_step_locally_and_remotely() -> None:
    """LLM 待ちを含む trial 全体の時間が上限を超えたら、次の step で止める。"""
    from capx.agent_api import Budget, BudgetExceeded

    budget = Budget(trial_wall_clock_s=0.5)

    local = LocalAgentEnv(FakeCodeEnv(), budget=budget, task_id="fake")
    local.reset_for_trial(1, 1)
    local.step("move()")
    time.sleep(0.6)  # LLM を待っているつもり
    with pytest.raises(BudgetExceeded) as exc:
        local.step("move()")
    assert exc.value.reason == "wall_clock"

    served = LocalAgentEnv(FakeCodeEnv(), budget=budget, task_id="fake")
    server, port = _serve(served)
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
    try:
        remote.reset_for_trial(1, 1)
        remote.step("move()")
        time.sleep(0.6)
        with pytest.raises(BudgetExceeded) as exc:
            remote.step("move()")
        assert exc.value.reason == "wall_clock", "Remote でも予算超過の内訳が届く"
    finally:
        remote.close()
        server._stop.set()


def test_a_worker_that_never_answers_is_detected_even_if_sends_succeed(monkeypatch) -> None:
    """接続は受け付けるが返事をしない相手（SSH 転送の先で worker が落ちた状態）を検出する。

    送れたかどうかだけを見ていると、転送口が受け付けてしまうので、落ちたことに気づけない。
    返事が来ない時間で判断する。
    """
    import socket as socket_mod

    from capx.remote_env import client as client_mod

    monkeypatch.setattr(client_mod, "HEARTBEAT_INTERVAL_S", 0.2)
    monkeypatch.setattr(client_mod, "HEARTBEAT_FAILURES_ALLOWED", 2)

    listener = socket_mod.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    held: list = []

    def accept_and_ignore() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            held.append(conn)  # 受け付けるだけで、何も返さない

    threading.Thread(target=accept_and_ignore, daemon=True).start()
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s", timeout_s=30)
    started = time.monotonic()
    try:
        with pytest.raises(EnvUnavailable, match="返事が無い"):
            remote.reset_for_trial(1, 1)
    finally:
        remote.close()
        listener.close()
        for conn in held:
            conn.close()
    assert time.monotonic() - started < 5, "timeout（30 秒）を待たずに気づく"
