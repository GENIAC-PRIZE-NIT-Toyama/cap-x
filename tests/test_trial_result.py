"""result.json と、試行フォルダに出る成果物。Agent が違っても同じ形で並ぶこと。"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from capx.agent_api import AgentResult
from capx.bench import run_trial
from capx.bench.schema import exec_ok
from capx.llm import client as llm_client
from capx.llm.recorder import recording
from capx.local_env import LocalAgentEnv
from capx.remote_env.client import RemoteAgentEnv
from capx.remote_env.worker.server import WorkerServer


class FakeCodeEnv:
    def __init__(self) -> None:
        self.low_level_env = SimpleNamespace(_sim_step_count=0, render_camera_names=("main",))

    def reset(self, *, seed=None, options=None):
        return {}, {"task_prompt": "Goal: stack"}

    def step(self, code):
        stderr = "executing action in terminated episode" if "late" in code else ""
        rc = 1 if stderr or "fail" in code else 0
        info = {"sandbox_rc": rc, "stdout": "out", "stderr": stderr or ("boom" if rc else ""),
                "task_completed": True}
        return {}, 1.0, True, False, info

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class ChattyAgent:
    """同梱の LLM クライアントを使う Agent（偽のサーバの応答を返させる）。"""

    def run(self, env, task, budget):
        env.step("move()")
        env.render()
        return AgentResult(artifacts={"notes.txt": "hello"})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_exec_ok_does_not_count_the_terminated_episode_message_as_a_failure() -> None:
    assert exec_ok(0, "") == (True, "")
    ok, note = exec_ok(1, "executing action in terminated episode")
    assert ok and "成功として数えた" in note, "上書きしたことが記録に残る"
    assert exec_ok(1, "Traceback ...")[0] is False


def test_the_recorder_collects_prompts_responses_and_tokens(monkeypatch) -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    message = SimpleNamespace(content="hi", reasoning_content=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )
    monkeypatch.setattr(llm_client, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(llm_client, "_resolve_connection", lambda args: ("http://x", "k", "chat"))
    monkeypatch.setattr(llm_client, "_build_chat_kwargs", lambda args, prompt: {})

    with recording() as rec:
        out = llm_client.query_model(SimpleNamespace(), [{"role": "user", "content": "hello"}])

    assert out["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert rec.totals() == {"calls": 1, "tokens_in": 11, "tokens_out": 7}
    assert rec.entries[0]["content"] == "hi"


def test_a_remote_trial_writes_the_same_folder_layout_as_local(tmp_path) -> None:
    served = LocalAgentEnv(FakeCodeEnv(), task_id="fake")
    port = _free_port()
    server = WorkerServer(served, session_id="s")
    threading.Thread(
        target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
    ).start()
    time.sleep(0.4)
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s", record_video=True)
    try:
        summary = run_trial(
            remote, ChattyAgent(), 1, {"output_dir": str(tmp_path), "record_video": True},
            seed=3, meta={"agent_spec": "agents/chatty.py", "model": "m"},
            infrastructure_retries=1,
        )
    finally:
        remote.close()
        server._stop.set()

    (folder,) = list(tmp_path.glob("trial_01_*"))
    for name in ("code.py", "summary.txt", "all_responses.json", "result.json"):
        assert (folder / name).exists(), name
    for name in ("prompts_and_responses", "steps", "artifacts", "images"):
        assert (folder / name).is_dir(), name
    assert (folder / "artifacts/notes.txt").read_text() == "hello"
    assert (folder / "steps/step_01.py").read_text() == "move()"
    assert len(list((folder / "images").glob("*.jpg"))) == 1

    result = json.loads((folder / "result.json").read_text())
    assert result["task_completed"] is True and result["exec_ok"] is True
    assert result["seed"] == 3 and result["infrastructure_retries"] == 1
    assert result["agent"]["spec"] == "agents/chatty.py" and result["agent"]["model"] == "m"
    assert result["budget"]["max_steps"] == 10
    assert result["steps_used"] == 1 and result["schema_version"] == 1
    assert summary.result.task_completed is True


class NumpyBoolEnv(FakeCodeEnv):
    """LIBERO のように、task_completed を numpy の真偽値で返す。"""

    def step(self, code):
        obs, reward, terminated, truncated, info = super().step(code)
        info["task_completed"] = np.bool_(False)
        info["sandbox_rc"] = np.int64(0)
        return obs, np.float64(0.0), terminated, truncated, info


def test_numpy_values_from_the_simulator_become_python_values_locally() -> None:
    env = LocalAgentEnv(NumpyBoolEnv(), task_id="fake")
    env.reset_for_trial(1, 1)
    env.step("move()")
    outcome = env.evaluate()
    assert type(outcome.task_completed) is bool and outcome.task_completed is False
    assert type(outcome.sandbox_rc) is int
    assert type(env.recorded_steps[0].task_completed) is bool


def test_numpy_values_arrive_as_python_values_over_zmq() -> None:
    served = LocalAgentEnv(NumpyBoolEnv(), task_id="fake")
    port = _free_port()
    server = WorkerServer(served, session_id="s")
    threading.Thread(
        target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
    ).start()
    time.sleep(0.4)
    remote = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s")
    try:
        remote.reset_for_trial(1, 1)
        remote.step("move()")
        outcome = remote.evaluate()
    finally:
        remote.close()
        server._stop.set()
    assert type(outcome.task_completed) is bool


def test_result_json_accepts_numpy_values_but_not_unknown_types(tmp_path) -> None:
    import pytest

    from capx.bench.artifacts import save_trial_extras

    save_trial_extras(
        str(tmp_path / "a"), steps=[], llm_entries=[], agent_artifacts={}, images=[],
        result={"task_completed": np.bool_(True), "reward": np.float32(0.5), "xs": np.arange(2)},
    )
    data = json.loads((tmp_path / "a/result.json").read_text())
    assert data == {"task_completed": True, "reward": 0.5, "xs": [0, 1]}

    with pytest.raises(TypeError):
        save_trial_extras(
            str(tmp_path / "b"), steps=[], llm_entries=[], agent_artifacts={}, images=[],
            result={"x": object()},
        )
