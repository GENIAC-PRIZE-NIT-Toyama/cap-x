"""ストリーミング。購読したときだけフレームが届き、購読しなければエンコードしない。"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from capx.agent_api import StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env import protocol
from capx.remote_env.client import RemoteAgentEnv
from capx.remote_env.worker import server as worker_server
from capx.remote_env.worker.server import WorkerServer


class StreamingEnv:
    """step の間、シミュレータのようにフレームを録り続ける。"""

    def __init__(self, seconds: float = 0.8) -> None:
        self.seconds = seconds
        self._listener = None

    def set_frame_listener(self, listener) -> None:
        self._listener = listener

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        end = time.monotonic() + self.seconds
        i = 0
        while time.monotonic() < end:
            if self._listener:
                self._listener(np.full((16, 16, 3), i % 256, dtype=np.uint8))
            i += 1
            time.sleep(0.005)  # 200 fps で録る
        return StepResult(request_id="x", ok=True, stdout="", stderr="")

    def render(self, camera: str = "main") -> bytes:
        return b"jpeg"

    def reset_for_trial(self, trial, seed=None) -> TaskSpec:
        return TaskSpec(task_id="s", seed=1, instruction="", api_docs="")

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(0.0, False, False, False, 0)


def _start(monkeypatch):
    encoded = []
    real = worker_server._encode_jpeg
    monkeypatch.setattr(
        worker_server, "_encode_jpeg", lambda f, q: (encoded.append(1), real(f, q))[1]
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = WorkerServer(StreamingEnv(), session_id="s")
    threading.Thread(
        target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
    ).start()
    time.sleep(0.4)
    return server, RemoteAgentEnv(endpoint=f"tcp://127.0.0.1:{port}", session_id="s"), encoded


def test_without_a_subscription_nothing_is_encoded(monkeypatch) -> None:
    server, client, encoded = _start(monkeypatch)
    try:
        client.step("move()")
        assert encoded == [], "購読なしでエンコードが走った"
    finally:
        client.close()
        server._stop.set()


def test_subscribing_delivers_throttled_jpeg_frames(monkeypatch, tmp_path) -> None:
    server, client, encoded = _start(monkeypatch)
    received = []
    try:
        client.subscribe_frames(lambda seq, jpeg: received.append((seq, jpeg)), save_dir=str(tmp_path))
        client.step("move()")  # 0.8 秒の間に 200 fps で録る
        # 10 fps に間引かれる（0.8 秒で高々 8〜9 枚）。溜めずに最新だけを送る。
        assert 2 <= len(received) <= 10, len(received)
        assert all(jpeg[:2] == b"\xff\xd8" for _seq, jpeg in received), "JPEG でない"
        assert len(encoded) <= 10
        assert len(list(tmp_path.glob("frame_*.jpg"))) == len(received)

        client.unsubscribe_frames()
        before = len(encoded)
        client.step("move()")
        assert len(encoded) == before, "解除したあともエンコードしている"
    finally:
        client.close()
        server._stop.set()


def test_event_messages_are_not_mistaken_for_replies() -> None:
    message = protocol.event("frame", seq=1, image=b"x")
    assert message.kind == "event" and message.request_id == ""
    assert protocol.decode(protocol.encode(message)).kind == "event"
