"""手元PC から GPU マシンの worker を呼ぶ。

`AgentEnv` としては `LocalAgentEnv` と区別がつかない。Agent は同じ
`step` / `render` / `close` しか見ないので、同じ Agent ファイルが両方で動く。

backend は経路に入らない。セッションを作るのは HTTP だが、コードと画像と
動画はここから worker へ直接 ZMQ で流れる。
"""

from __future__ import annotations

import logging
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
    ) -> None:
        """
        Args:
            endpoint: `tcp://host:19500`。直接指定するとき用。
            server_url: backend の URL。渡すとセッション作成から行う。
            task_id: backend に渡すタスク名。`env_config` は送らない
                （任意の `_target_` を受けると任意 import の入口になる）。
            budget: 予算。強制するのは worker 側。
        """
        self._budget = budget or Budget()
        self._timeout_s = timeout_s
        self._session_id = session_id
        self._sequence = 0
        self._socket: Any = None
        self._task: TaskSpec | None = None
        self._steps: list = []

        if server_url:
            if not task_id:
                raise ValueError("server_url を使うときは task_id が要る")
            endpoint, self._session_id = _create_session(server_url, task_id)

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

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._call("close", timeout_s=10.0)
        except Exception:
            pass  # 閉じるときの失敗は握りつぶす。どのみち捨てる接続
        finally:
            self._socket.close()
            self._socket = None

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
        self._socket.send(protocol.encode(message))

        deadline = (timeout_s or self._timeout_s)
        waited = 0.0
        missed_heartbeats = 0

        while waited < deadline:
            if self._socket.poll(int(HEARTBEAT_INTERVAL_S * 1000)):
                raw = self._socket.recv()
                reply = protocol.decode(raw)

                if reply.operation == "ping":
                    continue  # 割り込んだ heartbeat の応答。本命を待ち続ける
                if reply.request_id and reply.request_id != message.request_id:
                    continue  # 取り違え。捨てて待つ

                if reply.kind == "error":
                    self._raise(reply)
                return reply.payload

            waited += HEARTBEAT_INTERVAL_S
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

    def _ping(self) -> bool:
        from capx.remote_env import protocol

        try:
            self._socket.send(protocol.encode(protocol.request("ping")))
            return True
        except Exception:
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


def _create_session(server_url: str, task_id: str) -> tuple[str, str]:
    """backend にセッションを作らせ、`(zmq_endpoint, session_id)` を得る。"""
    import requests

    response = requests.post(
        f"{server_url.rstrip('/')}/sessions",
        json={"task_id": task_id},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["zmq_endpoint"], data["session_id"]
