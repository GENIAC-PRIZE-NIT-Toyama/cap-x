"""手元PC から GPU マシンの worker を呼ぶ。

`AgentEnv` としては `LocalAgentEnv` と区別がつかない。Agent は同じ
`step` / `render` / `close` しか見ないので、同じ Agent ファイルが両方で動く。

backend は経路に入らない。セッションを作るのは HTTP だが、コードと画像と
動画はここから worker へ直接 ZMQ で流れる。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from capx.agent_api import (
    Budget,
    BudgetExceeded,
    EnvUnavailable,
    StepResult,
    TaskSpec,
)

logger = logging.getLogger("capx.remote")

#: 1 回の要求を待つ上限。`step` は最長で予算いっぱいかかるので長く取る。
#: これを過ぎても応答が無ければ、worker が死んだとみなす。
DEFAULT_TIMEOUT_S = 1200.0

#: 応答待ちの間に heartbeat を挟む間隔。worker が生きているかを確かめる。
HEARTBEAT_INTERVAL_S = 15.0

#: heartbeat が続けて失敗してよい回数。
HEARTBEAT_FAILURES_ALLOWED = 3


class RemoteAgentEnv:
    """ZMQ `DEALER` で worker の `LocalAgentEnv` を呼ぶ。"""

    def __init__(
        self,
        endpoint: str = "",
        *,
        server_url: str | None = None,
        task_id: str | None = None,
        budget: Budget | None = None,
        session_id: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        curve_server_key: bytes | None = None,
        curve_keypair: tuple[bytes, bytes] | None = None,
    ) -> None:
        """
        Args:
            endpoint: `tcp://host:19500`。直接指定するとき用。
            server_url: backend の URL。渡すとセッション作成から行う。
            task_id: backend に渡すタスク名。`env_config` は送らない
                （任意の `_target_` を受けると任意 import の入口になる）。
            budget: 予算。強制するのは worker 側。
            curve_server_key: worker の公開鍵。渡すと CURVE で繋ぐ。
            curve_keypair: 自分の (公開鍵, 秘密鍵)。秘密鍵はこの PC から出ない。
        """
        self._budget = budget or Budget()
        self._timeout_s = timeout_s
        self._session_id = session_id
        self._sequence = 0
        self._socket: Any = None
        self._server_url: str | None = None
        self._curve_server_key = curve_server_key
        self._curve_keypair = curve_keypair
        self._task: TaskSpec | None = None
        self._steps: list = []
        self._frame_handler: Callable[[int, bytes], None] | None = None

        if server_url:
            if not task_id:
                raise ValueError("server_url を使うときは task_id が要る")
            # 鍵はここで作り、公開鍵だけを backend に渡す。秘密鍵が PC から
            # 出ないので、HTTP が平文でも盗聴で CURVE が破られない。
            import zmq

            self._curve_keypair = zmq.curve_keypair()
            self._server_url = server_url
            endpoint, self._session_id, self._curve_server_key = _create_session(
                server_url, task_id, self._curve_keypair[0]
            )

        if not endpoint:
            raise ValueError("endpoint か server_url のどちらかが要る")
        self._endpoint = endpoint
        self._connect()

    # -- AgentEnv ----------------------------------------------------------

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        if len(code.encode("utf-8")) > self._budget.max_code_bytes:
            raise BudgetExceeded("output_limit", "コードが上限を超えた")

        payload = self._call("step", code=code, capture_video=capture_video)
        result = StepResult(**payload["result"])

        # Bench は Agent の自己申告ではなく、`step()` を通ったものを記録する。
        # Remote では worker ではなくここが手元側の Bench なので、ここで控える。
        from capx.local_env import RecordedStep

        self._steps.append(
            RecordedStep(
                code=code,
                ok=result.ok,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_s=result.execution_time_used_s,
                truncated=result.truncated,
            )
        )
        return result

    def render(self, camera: str = "main") -> bytes:
        payload = self._call("render", camera=camera)
        image = payload.get("image") or b""
        return bytes(image)

    # -- ストリーミング ----------------------------------------------------

    def subscribe_frames(
        self,
        on_frame: Callable[[int, bytes], None] | None = None,
        *,
        save_dir: str | None = None,
    ) -> None:
        """実行中のフレーム（JPEG、最大 10 fps）を受け取る。既定は off。

        フレームは `step()` を待っている間に届き、この関数を呼んだスレッドで
        `on_frame(seq, jpeg)` が呼ばれる。`save_dir` を渡すと `frame_000001.jpg`
        の名前で保存する。購読しなければ、worker はエンコードもしない。
        """
        directory = Path(save_dir) if save_dir else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

        def handler(seq: int, jpeg: bytes) -> None:
            if directory is not None:
                (directory / f"frame_{seq:06d}.jpg").write_bytes(jpeg)
            if on_frame is not None:
                on_frame(seq, jpeg)

        self._frame_handler = handler
        self._call("subscribe")

    def unsubscribe_frames(self) -> None:
        self._call("unsubscribe")
        self._frame_handler = None

    def _handle_event(self, event: Any) -> None:
        if event.operation == "frame" and self._frame_handler is not None:
            self._frame_handler(
                int(event.payload.get("seq", 0)), bytes(event.payload.get("image", b""))
            )

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._call("close", timeout_s=2.0)  # 届かない相手を待たない
        except Exception:
            pass  # 閉じるときの失敗は握りつぶす。どのみち捨てる接続
        finally:
            self._socket.close()
            self._socket = None
            self._delete_session()

    def _delete_session(self) -> None:
        """backend にセッションを返す。失敗しても握りつぶす。

        Ctrl-C で抜けると呼ばれない。そのときは backend の idle 回収に任せる。
        """
        if not self._server_url or not self._session_id:
            return
        try:
            import requests

            requests.delete(
                f"{self._server_url.rstrip('/')}/sessions/{self._session_id}",
                headers=_auth_headers(),
                timeout=30,
            )
        except Exception:
            pass

    # -- Bench 専用 --------------------------------------------------------

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        payload = self._call("reset", trial=trial, seed=seed)
        self._steps.clear()
        self._task = TaskSpec(**payload["task"])
        return self._task

    def evaluate(self):
        """採点を worker から取る。**クライアントは値を送らない。**

        送れる設計にすると改ざんできる。計算するのは worker 側だけ。
        """
        from capx.local_env import TrialOutcome

        payload = self._call("evaluate")
        return TrialOutcome(
            reward=float(payload["reward"]),
            task_completed=payload["task_completed"],
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            sandbox_rc=int(payload["sandbox_rc"]),
            steps=[],
        )

    @property
    def recorded_steps(self) -> list:
        """このクライアントを通った `step()` の記録。"""
        return list(self._steps)

    @property
    def inner(self):
        raise AttributeError(
            "RemoteAgentEnv は env の実体を持たない。"
            " 動画の書き出しなどは worker 側で行う"
        )

    # -- 内部 --------------------------------------------------------------

    def _connect(self) -> None:
        import zmq

        from capx.remote_env import protocol

        context = zmq.Context.instance()
        self._socket = context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.MAXMSGSIZE, protocol.MAX_MESSAGE_BYTES)
        self._socket.setsockopt(zmq.HEARTBEAT_IVL, protocol.HEARTBEAT_IVL_MS)
        self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, protocol.HEARTBEAT_TIMEOUT_MS)
        if self._curve_server_key is not None:
            if self._curve_keypair is None:
                raise ValueError("curve_server_key には curve_keypair も要る")
            self._socket.curve_serverkey = self._curve_server_key
            self._socket.curve_publickey, self._socket.curve_secretkey = self._curve_keypair
        self._socket.connect(self._endpoint)
        logger.info("connected to %s", self._endpoint)

    def _call(self, operation: str, timeout_s: float | None = None, **payload: Any):
        """1 往復する。応答が来るまで heartbeat で生存を確かめ続ける。

        `step` は数百秒かかりうる。その間ずっと黙って待つと、worker が死んだ
        場合に timeout まで気づけない。`ping` は worker の I/O スレッドが
        即答するので、実行中でも返ってくる。
        """
        import zmq

        from capx.remote_env import protocol

        if self._socket is None:
            raise EnvUnavailable("closed env")

        self._sequence += 1
        message = protocol.request(operation, session_id=self._session_id, **payload)
        message.sequence = self._sequence
        self._send(message)

        deadline = timeout_s or self._timeout_s
        # poll の刻みは heartbeat の間隔と残り時間の短い方。短い timeout を
        # 渡されたのに 1 回目の poll が 15 秒ブロックする、を避ける。
        slice_s = min(HEARTBEAT_INTERVAL_S, deadline)
        waited = 0.0
        missed_heartbeats = 0

        while waited < deadline:
            step_s = min(slice_s, deadline - waited)
            if self._socket.poll(int(step_s * 1000)):
                raw = self._socket.recv()
                reply = protocol.decode(raw)

                if reply.kind == "event":
                    self._handle_event(reply)
                    continue  # ストリームのフレーム。本命の応答を待ち続ける
                if reply.operation == "ping":
                    continue  # 割り込んだ heartbeat の応答。本命を待ち続ける
                if reply.request_id and reply.request_id != message.request_id:
                    continue  # 取り違え。捨てて待つ

                if reply.kind == "error":
                    self._raise(reply)
                return reply.payload

            waited += step_s
            # heartbeat は「長く待っているとき」だけ。短い呼び出しでは要らない。
            if step_s < HEARTBEAT_INTERVAL_S:
                continue
            if not self._ping():
                missed_heartbeats += 1
                if missed_heartbeats >= HEARTBEAT_FAILURES_ALLOWED:
                    raise EnvUnavailable(
                        f"worker が {missed_heartbeats} 回連続で応答しない"
                        f"（{self._endpoint}）"
                    )
            else:
                missed_heartbeats = 0

        raise EnvUnavailable(f"{operation} が {deadline}s で応答しなかった")

    def _send(self, message: Any) -> None:
        """送れなければ待たずに `EnvUnavailable` にする。

        `DEALER` は接続が確立するまで送信をキューに積む。許可されていない鍵
        で繋いだときなど、確立しない相手には積み続けて、キューが満杯になると
        `send` 自体が戻らなくなる。ブロックさせずに、届かないことを伝える。
        """
        import zmq

        from capx.remote_env import protocol

        try:
            self._socket.send(protocol.encode(message), flags=zmq.NOBLOCK)
        except zmq.Again as exc:
            raise EnvUnavailable(
                f"{self._endpoint} に送れない（接続できていない）。"
                " 鍵が違うか、worker が居ない"
            ) from exc

    def _ping(self) -> bool:
        from capx.remote_env import protocol

        try:
            self._send(protocol.request("ping"))
            return True
        except EnvUnavailable:
            return False

    @staticmethod
    def _raise(reply: Any) -> None:
        kind = reply.payload.get("failure_kind", "agent_error")
        message = reply.payload.get("message", "worker error")
        if kind == "budget":
            raise BudgetExceeded("execution_budget", message)
        if kind == "infrastructure":
            raise EnvUnavailable(message)
        raise RuntimeError(message)


def _auth_headers() -> dict[str, str]:
    import os

    token = os.environ.get("CAPX_ENV_SERVER_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _create_session(
    server_url: str, task_id: str, client_public_key: bytes
) -> tuple[str, str, bytes]:
    """backend にセッションを作らせる。

    Returns:
        (zmq_endpoint, session_id, worker の公開鍵)
    """
    import requests

    response = requests.post(
        f"{server_url.rstrip('/')}/sessions",
        json={
            "task_id": task_id,
            "client_public_key": client_public_key.decode("ascii"),
        },
        headers=_auth_headers(),
        timeout=600,  # worker の環境構築を待つ
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        from capx.agent_api import CapacityFull, EnvStartFailed

        message = f"セッションを作れなかった（{response.status_code}）: {detail}"
        if response.status_code == 429:
            raise CapacityFull(message)
        raise EnvStartFailed(message)
    data = response.json()
    return data["zmq_endpoint"], data["session_id"], data["server_public_key"].encode("ascii")
