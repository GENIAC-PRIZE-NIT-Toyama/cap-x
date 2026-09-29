"""worker と RemoteAgentEnv が実際に ZMQ で話せるかを見る。

偽の env を使うので simulator は要らない。見たいのは配管——`ROUTER`/`DEALER`、
msgpack、`request_id` の突き合わせ、長時間 `step` 中の heartbeat。

これが通っていれば、Docker を被せる前に通信の形は確定している。
"""

from __future__ import annotations

import threading
import time

import pytest

from capx.agent_api import StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env.client import RemoteAgentEnv
from capx.remote_env.worker.server import WorkerServer


class FakeEnv:
    """`LocalAgentEnv` の代役。worker が呼ぶメソッドだけ持つ。"""

    def __init__(self, step_delay_s: float = 0.0) -> None:
        self.codes: list[str] = []
        self.step_delay_s = step_delay_s

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        self.codes.append(code)
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        return StepResult(
            request_id="fixed", ok=True, stdout=f"ran {code}", stderr=""
        )

    def render(self, camera: str = "main") -> bytes:
        return b"\xff\xd8JPEG" + camera.encode()

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        return TaskSpec(
            task_id="fake", seed=trial, instruction="do it", api_docs="docs"
        )

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(
            reward=0.5,
            task_completed=False,
            terminated=False,
            truncated=False,
            sandbox_rc=0,
        )


@pytest.fixture
def connected(request):
    """worker を立てて、繋がったクライアントを返す。"""
    delay = getattr(request, "param", 0.0)
    env = FakeEnv(step_delay_s=delay)
    port = _free_port()
    server = WorkerServer(env, session_id="s1")
    threading.Thread(
        target=server.serve,
        kwargs={"port": port, "host": "127.0.0.1"},
        daemon=True,
    ).start()
    time.sleep(0.4)

    client = RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s1")
    try:
        yield client, env, server
    finally:
        try:
            client.close()
        except Exception:
            pass
        server._stop.set()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_reset_returns_task_spec(connected) -> None:
    client, _env, _server = connected
    task = client.reset_for_trial(3)
    assert task.task_id == "fake"
    assert task.seed == 3
    assert task.instruction == "do it"


def test_step_reaches_the_env_and_comes_back(connected) -> None:
    client, env, _server = connected
    result = client.step("open_gripper()")
    assert result.ok
    assert result.stdout == "ran open_gripper()"
    assert env.codes == ["open_gripper()"], "worker 側の env に届いていない"


def test_render_returns_bytes_unchanged(connected) -> None:
    """画像は base64 を経由しない。bytes のまま往復する。"""
    client, _env, _server = connected
    assert client.render() == b"\xff\xd8JPEGmain"


def test_evaluate_comes_from_the_worker(connected) -> None:
    """採点はクライアントが送らず、worker から取る。改ざんさせない。"""
    client, _env, _server = connected
    outcome = client.evaluate()
    assert outcome.reward == 0.5
    assert outcome.sandbox_rc == 0


def test_step_result_carries_no_reward(connected) -> None:
    """Agent 側に採点が漏れていないことを、実際の往復でも確かめる。"""
    client, _env, _server = connected
    result = client.step("x = 1")
    assert not hasattr(result, "reward")
    assert not hasattr(result, "task_completed")


@pytest.mark.parametrize("connected", [1.5], indirect=True)
def test_long_step_stays_connected(connected) -> None:
    """`step` が長引いても切れない。

    worker の I/O スレッドと実行スレッドが分かれていないと、実行中は
    `ping` に応答できず、生存確認が落ちて EnvUnavailable になる。
    """
    client, env, _server = connected
    started = time.monotonic()
    result = client.step("slow()")
    assert result.ok
    assert time.monotonic() - started >= 1.5
    assert env.codes == ["slow()"]


def test_ping_answers_while_a_step_runs() -> None:
    """実行中でも `ping` が即答されることを直接見る。

    これが返らないと backend が worker を「無応答」と判断して、走っている
    コンテナを kill してしまう。
    """
    env = FakeEnv(step_delay_s=1.0)
    port = _free_port()
    server = WorkerServer(env, session_id="s1")
    threading.Thread(
        target=server.serve,
        kwargs={"port": port, "host": "127.0.0.1"},
        daemon=True,
    ).start()
    time.sleep(0.4)

    import zmq

    from capx.remote_env import protocol

    context = zmq.Context.instance()
    stepper = context.socket(zmq.DEALER)
    stepper.setsockopt(zmq.LINGER, 0)
    stepper.connect(f"tcp://127.0.0.1:{port}")
    pinger = context.socket(zmq.DEALER)
    pinger.setsockopt(zmq.LINGER, 0)
    pinger.connect(f"tcp://127.0.0.1:{port}")

    try:
        stepper.send(protocol.encode(protocol.request("step", code="slow()")))
        time.sleep(0.3)  # 実行スレッドが掴んでいる最中

        pinger.send(protocol.encode(protocol.request("ping")))
        assert pinger.poll(2000), "step 実行中に ping が返らなかった"
        reply = protocol.decode(pinger.recv())
        assert reply.operation == "ping"
        assert reply.payload["busy"] is True
    finally:
        stepper.close()
        pinger.close()
        server._stop.set()
