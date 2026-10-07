"""worker の ZMQ ポートが、許可した鍵の持ち主にしか開かないことを確かめる。

このポートは任意の Python を実行する入口で、LAN に開ける。暗号化と認証が
無ければ、到達できる人は誰でも実行できる。だから「鍵を渡したら通る」より
「鍵が違う・鍵が無いなら通らない」の方を重く見る。

simulator は要らない。偽の env と実際の ZMQ ソケットで確かめる。
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import zmq

from capx.agent_api import EnvUnavailable, StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env import protocol
from capx.remote_env.client import RemoteAgentEnv
from capx.remote_env.worker.server import WorkerServer


class FakeEnv:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        self.codes.append(code)
        return StepResult(request_id="r", ok=True, stdout="", stderr="")

    def render(self, camera: str = "main") -> bytes:
        return b"img"

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        return TaskSpec(task_id="fake", seed=trial, instruction="", api_docs="")

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(0.0, False, False, False, 0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def secured():
    """CURVE で守られた worker と、許可済みの鍵ペア。"""
    server_pub, server_sec = zmq.curve_keypair()
    allowed_pub, allowed_sec = zmq.curve_keypair()
    env = FakeEnv()
    port = _free_port()
    server = WorkerServer(env)
    threading.Thread(
        target=server.serve,
        kwargs={
            "port": port,
            "host": "127.0.0.1",
            "curve_secret_key": server_sec,
            "allowed_client_keys": {allowed_pub},
        },
        daemon=True,
    ).start()
    time.sleep(0.6)
    yield {
        "env": env,
        "endpoint": f"tcp://127.0.0.1:{port}",
        "server_pub": server_pub,
        "allowed": (allowed_pub, allowed_sec),
    }
    server._stop.set()
    time.sleep(0.2)


def test_allowed_key_can_drive_the_env(secured) -> None:
    client = RemoteAgentEnv(
        endpoint=secured["endpoint"],
        curve_server_key=secured["server_pub"],
        curve_keypair=secured["allowed"],
    )
    try:
        assert client.step("x = 1").ok
        assert secured["env"].codes == ["x = 1"]
    finally:
        client.close()


def test_unregistered_key_never_reaches_the_env(secured) -> None:
    """鍵ペアを自分で作って名乗っても、登録されていなければ実行できない。

    許可リストに無い鍵は、メッセージが worker に届く前に遮断される。
    ここが破れると、LAN 上の誰かが GPU マシンで任意のコードを走らせられる。
    """
    intruder = zmq.curve_keypair()
    client = RemoteAgentEnv(
        endpoint=secured["endpoint"],
        curve_server_key=secured["server_pub"],
        curve_keypair=intruder,
        timeout_s=3.0,
    )
    try:
        with pytest.raises(EnvUnavailable):
            client.step("import os; os.system('id')")
    finally:
        client.close()

    assert secured["env"].codes == [], "許可外の鍵のコードが実行されてしまった"


def test_plaintext_connection_never_reaches_the_env(secured) -> None:
    """暗号化なしで繋いでも、コードは届かない。"""
    context = zmq.Context.instance()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(secured["endpoint"])
    try:
        dealer.send(protocol.encode(protocol.request("step", code="evil()")))
        assert not dealer.poll(1500), "平文の要求に応答が返った"
    finally:
        dealer.close()

    assert secured["env"].codes == []


def test_wrong_server_key_is_refused(secured) -> None:
    """別のサーバになりすまされていないかを、クライアント側でも確かめる。"""
    impostor_pub, _ = zmq.curve_keypair()
    client = RemoteAgentEnv(
        endpoint=secured["endpoint"],
        curve_server_key=impostor_pub,
        curve_keypair=secured["allowed"],
        timeout_s=3.0,
    )
    try:
        with pytest.raises(EnvUnavailable):
            client.step("x = 1")
    finally:
        client.close()
    assert secured["env"].codes == []


def test_ping_reports_idle_time(secured) -> None:
    """backend が idle 回収の判断に使う。step の最中は busy になる。"""
    pub, sec = secured["allowed"]
    context = zmq.Context.instance()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.curve_serverkey = secured["server_pub"]
    dealer.curve_publickey = pub
    dealer.curve_secretkey = sec
    dealer.connect(secured["endpoint"])
    try:
        time.sleep(0.5)
        dealer.send(protocol.encode(protocol.request("ping")))
        assert dealer.poll(3000)
        reply = protocol.decode(dealer.recv())
        assert reply.payload["busy"] is False
        assert reply.payload["idle_s"] >= 0.4
    finally:
        dealer.close()
