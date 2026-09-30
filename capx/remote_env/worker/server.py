"""コンテナの中で `LocalAgentEnv` を ZMQ 越しに見せる。

    uv run --no-sync --active -m capx.remote_env.worker.server \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \\
        --port 19500

**I/O スレッドと実行スレッドを分ける。** `step()` は MuJoCo を回すので数十秒から
数百秒ブロックする。同じスレッドで ZMQ を回すと、その間 `ping` に応答できず、
backend が「無応答」と判断して実行中のコンテナを kill してしまう。

`ROUTER` ソケットは I/O スレッドだけが触る（ZMQ のソケットはスレッドを
またげない）。実行は別スレッドに渡し、結果をキュー経由で受け取る。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("capx.worker")

#: 実行スレッドの結果を待つ間、I/O ループが回る間隔。
POLL_MS = 100


@dataclass
class _Job:
    message: Any
    identity: bytes


class WorkerServer:
    """1 セッション = 1 プロセス = 1 環境。

    同時に走る `step` は 1 つだけ。2 つ目が来たら拒否する——同じ
    シミュレータを 2 つの要求が同時に進めると、どちらの結果も信用できない。
    """

    def __init__(self, env: Any, session_id: str = "") -> None:
        self._env = env
        self._session_id = session_id
        self._last_activity = time.monotonic()
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._results: queue.Queue[tuple[bytes, Any]] = queue.Queue()
        self._done: dict[str, Any] = {}  # request_id -> 応答（再送対策）
        self._busy = False
        self._stop = threading.Event()

    # -- 実行スレッド ------------------------------------------------------

    def _execute_loop(self) -> None:
        """`step` / `render` を順に処理する。ここだけが env を触る。"""
        from capx.remote_env import protocol

        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue

            msg = job.message
            try:
                payload = self._handle(msg)
                reply = protocol.response(msg, **payload)
            except Exception as exc:
                logger.exception("%s に失敗", msg.operation)
                reply = protocol.error(msg, repr(exc), kind=_failure_kind(exc))

            self._done[msg.request_id] = reply
            self._results.put((job.identity, reply))
            self._busy = False

    def _handle(self, msg: Any) -> dict[str, Any]:
        from dataclasses import asdict

        if msg.operation == "step":
            code = msg.payload.get("code", "")
            capture = bool(msg.payload.get("capture_video", False))
            result = self._env.step(code, capture_video=capture)
            return {"result": asdict(result)}

        if msg.operation == "render":
            camera = msg.payload.get("camera", "main")
            return {"image": self._env.render(camera), "media_type": "image/jpeg"}

        if msg.operation == "reset":
            trial = int(msg.payload.get("trial", 1))
            task = self._env.reset_for_trial(trial, msg.payload.get("seed"))
            return {"task": asdict(task)}

        if msg.operation == "evaluate":
            outcome = self._env.evaluate()
            return {
                "reward": outcome.reward,
                "task_completed": outcome.task_completed,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                "sandbox_rc": outcome.sandbox_rc,
            }

        raise ValueError(f"unknown operation: {msg.operation}")

    # -- I/O スレッド ------------------------------------------------------

    def serve(
        self,
        port: int,
        host: str = "0.0.0.0",
        *,
        curve_secret_key: bytes | None = None,
        allowed_client_keys: set[bytes] | None = None,
    ) -> None:
        """待ち受ける。鍵を渡すと CURVE で暗号化し、許可した公開鍵だけ通す。

        鍵なしで起動できるのは開発用。ポートを LAN に開ける本番では必ず渡す
        ——このソケットは任意の Python を実行する入口なので、暗号化と認証が
        無いと、到達できる人は誰でも実行できてしまう。

        許可外の鍵と、暗号化なしの接続は、メッセージが `recv` に届く前に
        遮断される（`tests/test_remote_curve.py`）。
        """
        import zmq

        from capx.remote_env import protocol

        context = zmq.Context.instance()
        authenticator = None
        socket = context.socket(zmq.ROUTER)
        if curve_secret_key is not None:
            from zmq.auth.thread import ThreadAuthenticator

            allowed = set(allowed_client_keys or ())

            class _AllowList:
                def callback(self, domain: str, key: Any) -> bool:
                    raw = key if isinstance(key, bytes) else str(key).encode()
                    return raw in allowed

            authenticator = ThreadAuthenticator(context)
            authenticator.start()
            authenticator.configure_curve_callback(
                domain="*", credentials_provider=_AllowList()
            )
            socket.curve_server = True
            socket.curve_secretkey = curve_secret_key
            logger.info("CURVE on: %d client key(s) allowed", len(allowed))
        else:
            logger.warning("CURVE OFF — 開発用。ポートを開けるときは鍵を渡す")
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, protocol.MAX_MESSAGE_BYTES)
        socket.setsockopt(zmq.SNDHWM, 16)
        socket.setsockopt(zmq.RCVHWM, 16)
        socket.bind(f"tcp://{host}:{port}")
        logger.info("listening on tcp://%s:%d", host, port)

        worker = threading.Thread(target=self._execute_loop, daemon=True)
        worker.start()

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        try:
            while not self._stop.is_set():
                # 実行スレッドからの結果を先に流す
                while True:
                    try:
                        identity, reply = self._results.get_nowait()
                    except queue.Empty:
                        break
                    socket.send_multipart([identity, protocol.encode(reply)])

                if not poller.poll(POLL_MS):
                    continue

                identity, raw = socket.recv_multipart()
                try:
                    msg = protocol.decode(raw)
                except protocol.ProtocolError as exc:
                    socket.send_multipart(
                        [identity, protocol.encode(protocol.error(None, str(exc)))]
                    )
                    continue

                reply = self._dispatch(msg, identity)
                if reply is not None:
                    socket.send_multipart([identity, protocol.encode(reply)])
        finally:
            self._stop.set()
            socket.close()
            if authenticator is not None:
                authenticator.stop()

    def _dispatch(self, msg: Any, identity: bytes) -> Any:
        """I/O スレッドで即答できるものは返し、重いものはキューへ。

        Returns:
            すぐ返す応答。`None` なら実行スレッドが後で返す。
        """
        from capx.remote_env import protocol

        # ping は env を触らないので、step の最中でも即答できる。
        # これが返らないと backend に「死んだ」と誤判定される。
        if msg.operation == "ping":
            # backend が idle 回収に使う。step の最中は busy なので回収されない。
            return protocol.response(
                msg,
                busy=self._busy,
                idle_s=time.monotonic() - self._last_activity,
            )

        self._last_activity = time.monotonic()

        if msg.operation == "close":
            self._stop.set()
            return protocol.response(msg, closed=True)

        # 同じ request_id が再送された場合、実行し直さず前の応答を返す。
        # 応答だけが失われたのか実行されなかったのか、送信側には区別できない。
        if msg.request_id in self._done:
            return self._done[msg.request_id]

        if self._busy:
            return protocol.error(
                msg, "already running a step", kind="infrastructure"
            )

        self._busy = True
        self._jobs.put(_Job(message=msg, identity=identity))
        return None


def _failure_kind(exc: Exception) -> str:
    from capx.agent_api import BudgetExceeded

    if isinstance(exc, BudgetExceeded):
        return "budget"
    return "agent_error"


def main(
    config_path: str,
    /,
    *,
    port: int = 19500,
    host: str = "0.0.0.0",
    session_id: str = "",
    record_video: bool = True,
    execution_time_s: float = 1000.0,
    max_steps: int = 10,
) -> None:
    """worker を起動する。環境を作り終えてから listen する。

    構築（robosuite / MuJoCo / EGL の初期化）を listen より前に済ませるので、
    接続を受け付けた時点で必ず使える。backend の readiness 判定は
    「繋がるか」だけでよくなる。
    """
    from capx.agent_api import Budget
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader
    from capx.local_env import LocalAgentEnv

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"{config_path} に `env` がない")

    logger.info("building env from %s ...", config_path)
    env = LocalAgentEnv(
        instantiate(configs_dict["env"]),
        budget=Budget(execution_time_s=execution_time_s, max_steps=max_steps),
        record_video=record_video,
        task_id=os.path.splitext(os.path.basename(config_path))[0],
    )
    logger.info("env ready")

    secret = os.environ.get("CAPX_CURVE_SERVER_SECRET")
    allowed = os.environ.get("CAPX_CURVE_ALLOWED_CLIENT_KEYS", "")
    # 秘密鍵は引数ではなく環境変数で受ける（`ps` に出さないため）
    WorkerServer(env, session_id=session_id).serve(
        port=port,
        host=host,
        curve_secret_key=secret.encode() if secret else None,
        allowed_client_keys={k.strip().encode() for k in allowed.split(",") if k.strip()},
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
