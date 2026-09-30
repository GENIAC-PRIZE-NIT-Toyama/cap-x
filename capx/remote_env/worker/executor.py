"""生成コードを別プロセスで走らせ、API 呼び出しだけを受け付ける。

Gym（このプロセス）にはシミュレータ・reward・採点がある。policy プロセスには
それが無く、API の関数名のスタブだけがある（`policy_process.py`）。生成コードが
`env` を探しても、`gc.get_objects()` で全オブジェクトを漁っても、reward には
辿り着けない——別のプロセスにあるから。

**限界:** policy プロセスは同じユーザー、同じコンテナで動く。コンテナは
`--cap-drop ALL` で ptrace 等が使えないが、別ユーザーに分けてはいない。
悪意のある参加者への完全な防御ではなく、誤って・LLM が近道として reward に
触れるのを防ぐ境界（docs/RemoteDevelopment.md §5）。

時間切れは policy プロセスを kill して止める。同じプロセスで `exec` していた
ときは、`while True` を止める手段が無かった。
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

from capx.remote_env.worker import rpc

logger = logging.getLogger("capx.worker")

#: policy プロセスに渡す環境変数。CURVE の秘密鍵などは渡さない。
_CHILD_ENV_KEYS = ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "MPLCONFIGDIR", "VIRTUAL_ENV")


class ProcessExecutor:
    def __init__(self, step_timeout_s: float = 1000.0) -> None:
        self._timeout_s = step_timeout_s
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._function_names: list[str] = []
        self._inputs: dict[str, Any] = {}

    # -- プロセスの管理 ----------------------------------------------------

    def _start(self) -> None:
        parent, child = socket.socketpair()
        env = {k: os.environ[k] for k in _CHILD_ENV_KEYS if k in os.environ}
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "capx.remote_env.worker.policy_process", str(child.fileno())],
            pass_fds=[child.fileno()],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        child.close()
        self._sock = parent
        rpc.send(
            parent,
            {"op": "reset", "inputs": self._inputs, "functions": self._function_names},
        )
        rpc.recv(parent)  # ready

    def _stop(self) -> None:
        if self._sock is not None:
            try:
                rpc.send(self._sock, {"op": "exit"})
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- CodeExecutionEnvBase から呼ばれる ---------------------------------

    def reset(self, inputs: dict[str, Any], function_names: list[str]) -> None:
        """新しいエピソード。globals を作り直す（前のエピソードの変数を持ち越さない）。"""
        self._inputs = inputs
        self._function_names = list(function_names)
        self._stop()
        self._start()

    def run(
        self, code: str, obs: dict[str, Any], functions: dict[str, Callable[..., Any]]
    ) -> dict[str, Any]:
        if not self._alive():
            # 前のステップで時間切れ・異常終了した。globals は失われる。
            self._function_names = list(functions)
            self._stop()
            self._start()

        assert self._sock is not None
        rpc.send(
            self._sock,
            {"op": "exec", "code": code, "obs": _sendable(obs), "functions": list(functions)},
        )
        deadline = time.monotonic() + self._timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._killed(f"実行時間の上限（{self._timeout_s:.0f}s）に達した")
            self._sock.settimeout(remaining)
            try:
                message = rpc.recv(self._sock)
            except TimeoutError:
                return self._killed(f"実行時間の上限（{self._timeout_s:.0f}s）に達した")
            except (rpc.RpcClosed, OSError):
                return self._killed("生成コードのプロセスが異常終了した")

            op = message.get("op")
            if op == "call":
                self._serve_call(message, functions)
            elif op == "done":
                self._sock.settimeout(None)
                return {
                    "ok": bool(message["ok"]),
                    "stdout": message.get("stdout", ""),
                    "stderr": message.get("stderr", ""),
                    "result": message.get("result"),
                }

    def _serve_call(self, message: dict, functions: dict[str, Callable[..., Any]]) -> None:
        assert self._sock is not None
        name = message.get("name")
        try:
            if name not in functions:
                raise KeyError(f"{name!r} は使えない API")
            value = functions[name](*message.get("args", ()), **message.get("kwargs", {}))
            reply = {"op": "return", "value": value}
            rpc.pack(reply)  # 運べない型を返す関数なら、ここで分かる
        except Exception as exc:
            logger.info("API %s が失敗: %r", name, exc)
            reply = {"op": "raise", "error": f"{type(exc).__name__}: {exc}"}
        rpc.send(self._sock, reply)

    def _killed(self, reason: str) -> dict[str, Any]:
        logger.warning("policy プロセスを終了: %s", reason)
        self._stop()
        return {"ok": False, "stdout": "", "stderr": reason + "\n", "result": None}

    def close(self) -> None:
        self._stop()


def _sendable(obs: dict[str, Any]) -> dict[str, Any]:
    """運べない値（メッセージにできない型）は落とす。"""
    out = {}
    for key, value in obs.items():
        try:
            rpc.pack(value)
        except Exception:
            continue
        out[key] = value
    return out
