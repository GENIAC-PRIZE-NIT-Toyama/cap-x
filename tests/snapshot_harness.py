"""Docker を使わずに、backend・worker・保存先を手元で本当に動かす（Snapshot の E2E 用）。

`docker run` の代わりに、worker を手元の別プロセスとして起こす。backend（FastAPI）は
uvicorn で本当に listen し、クライアント（`RemoteAgentEnv`）は HTTP と ZMQ（CURVE）で
繋ぐ。Snapshot は backend が worker から取り出して、ディレクトリ（`DirStore`）に置く。
Docker の隔離（`--internal`・読み取り専用など）は確かめない（それは test_server_docker.py）。

`scripts/snapshot_bench.py` も使う。
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

#: E2E 用のタスク（tests/data の設定。知覚・IK のサーバに頼らない API）。
TEST_TASKS = {
    "snap_robosuite": ("tests/data/snapshot_robosuite.yaml", "robosuite"),
    "snap_libero": ("tests/data/snapshot_libero.yaml", "libero"),
}

#: worker を起こすときの入口。テスト用の API を登録してから worker を起動する。
#: macOS の OpenGL（MUJOCO_GL=cgl）は、メインスレッド以外から描画すると GIL を握ったまま
#: 止まる。本番（Linux の EGL）は env を別スレッドで動かすが、ここでは入れ替えて、ZMQ の
#: I/O を別スレッドに、env の実行をメインスレッドに置く。
_BOOT = """
import sys, threading
sys.path.insert(0, {tests!r})
import snapshot_support, tyro
from capx.remote_env.worker import server

if sys.platform == "darwin":
    _Thread, _serve = threading.Thread, server.WorkerServer.serve

    def serve(self, *args, **kwargs):
        loops = []

        class _Swap(_Thread):
            def __init__(self, target=None, **kw):
                if getattr(target, "__name__", "") == "_execute_loop":
                    loops.append(target)
                    target = lambda: None
                super().__init__(target=target, **kw)

        server.threading.Thread = _Swap
        io = _Thread(target=_serve, args=(self, *args), kwargs=kwargs, daemon=True)
        io.start()
        while not loops and io.is_alive():
            io.join(0.05)
        if loops:
            loops[0]()
        io.join()

    server.WorkerServer.serve = serve
tyro.cli(server.main)
"""

_FLAGS_WITH_VALUE = {
    "--name", "--network", "--label", "--cap-drop", "--security-opt", "--pids-limit", "--memory",
    "--cpus", "-e", "--tmpfs", "--user", "--ulimit", "--runtime", "-p",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_port_range(n: int) -> int:
    """連続して空いているポートの先頭。"""
    for _ in range(50):
        start = free_port()
        ok = True
        for port in range(start, start + n):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    ok = False
                    break
        if ok:
            return start
    raise RuntimeError("空きポートが見つからない")


class LocalDocker:
    """`docker ...` を、手元のプロセスで代わりに行う。"""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self.runs: list[list[str]] = []

    async def __call__(self, cmd: list[str]) -> tuple[int, str, str]:
        if cmd[:2] == ["docker", "run"]:
            return self._run(cmd)
        if cmd[:3] == ["docker", "rm", "-f"]:
            proc = self.procs.pop(cmd[3], None)
            if proc is not None:
                proc.kill()
                await asyncio.to_thread(proc.wait)
            return 0, "", ""
        if cmd[:2] == ["docker", "inspect"]:
            proc = self.procs.get(cmd[-1])
            return 0, "true" if proc is not None and proc.poll() is None else "false", ""
        if cmd[:2] == ["docker", "logs"]:
            path = self.log_dir / f"{cmd[-1]}.log"
            return 0, path.read_text() if path.exists() else "", ""
        return 0, "", ""

    def _run(self, cmd: list[str]) -> tuple[int, str, str]:
        self.runs.append(cmd)
        env = dict(os.environ)
        name, host_port, args = "", "", []
        i = 2
        while i < len(cmd):
            token = cmd[i]
            if token in _FLAGS_WITH_VALUE:
                value = cmd[i + 1]
                if token == "-e":
                    key, _, val = value.partition("=")
                    if not key.endswith("_NUM_THREADS"):
                        env[key] = val
                elif token == "--name":
                    name = value
                elif token == "-p":
                    host_port = value.split(":")[1]
                i += 2
            elif token.startswith("-") and not args:
                i += 1  # -d / --read-only など値の無い印
            else:
                args.append(token)
                i += 1
        # worker は構築後に /tmp へ移る。capx を相対パスで見つけている環境でも、生成コードの
        # プロセスが capx を import できるよう、絶対パスにする
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))])
        worker_args = args[1:]  # 先頭はイメージ名
        worker_args[worker_args.index("--port") + 1] = host_port
        worker_args[worker_args.index("--host") + 1] = "127.0.0.1"
        log = open(self.log_dir / f"{name}.log", "w")  # noqa: SIM115 — プロセスが書き続ける
        self.procs[name] = subprocess.Popen(
            [sys.executable, "-c", _BOOT.format(tests=str(TESTS)), *worker_args],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return 0, name, ""

    def kill_all(self) -> None:
        for proc in self.procs.values():
            proc.kill()
        self.procs.clear()


class Backend:
    """backend を uvicorn で動かす。`restart()` で台帳（メモリ）を捨てて起こし直せる。"""

    def __init__(
        self, store_dir: Path, log_dir: Path, tokens: dict[str, str] | None = None, store: Any = None
    ) -> None:
        self.store_dir = store_dir
        #: 保存先。省略すると `store_dir` のディレクトリ（DirStore）。
        self.store = store
        self.log_dir = log_dir
        self.tokens = tokens
        self.docker = LocalDocker(log_dir)
        self.port = free_port()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.manager: Any = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> Backend:
        import uvicorn

        from capx.remote_env.server import tasks
        from capx.remote_env.server.app import create_app
        from capx.remote_env.server.probe import make_caller, make_prober
        from capx.remote_env.server.sessions import Config, SessionManager
        from capx.snapshot.store import DirStore

        for name, (path, runtime) in TEST_TASKS.items():
            tasks.TASKS[name] = tasks.TaskEntry(path, runtime=runtime)
        start = free_port_range(8)
        self.manager = SessionManager(
            Config(public_host="127.0.0.1", port_start=start, port_end=start + 8, ready_timeout_s=600),
            runner=self.docker,
            store=self.store if self.store is not None else DirStore(self.store_dir),
        )
        self.manager._probe = make_prober(self.manager.probe_keypair)
        self.manager._call = make_caller(self.manager.probe_keypair)
        config = uvicorn.Config(
            create_app(self.manager, self.tokens, reap=False), host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("backend が起動しない")
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._thread.join(timeout=60)
        self.docker.kill_all()

    def restart(self) -> None:
        """backend を落として起こし直す（台帳は消え、保存先は残る）。"""
        self.stop()
        self.docker = LocalDocker(self.log_dir)
        self.port = free_port()
        self.start()
