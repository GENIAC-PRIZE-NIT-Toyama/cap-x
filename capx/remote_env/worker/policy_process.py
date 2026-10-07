"""生成コードを実行するプロセス。Gym とは別。

    python -m capx.remote_env.worker.policy_process <fd>

ここには reward も採点もシミュレータも無い。API の関数は、名前だけを持つ
スタブで、呼ぶと Gym 側に RPC で頼む。**capx の環境やシミュレータを import
しない**（numpy と msgpack だけ）。import すると、その中身がこのプロセスから
見えてしまう。

globals はステップをまたいで持ち越す（現行の `CodeExecutionEnvBase` と同じ）。
"""

from __future__ import annotations

import contextlib
import io
import socket
import sys
import traceback
from typing import Any

from capx.remote_env.worker import rpc


class _Session:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._globals: dict[str, Any] = {}

    def _stub(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            rpc.send(self._sock, {"op": "call", "name": name, "args": args, "kwargs": kwargs})
            reply = rpc.recv(self._sock)
            if reply.get("op") == "raise":
                raise RuntimeError(reply.get("error", "API call failed"))
            return reply.get("value")

        call.__name__ = name
        return call

    def _reset(self, message: dict) -> None:
        self._globals = {
            "__name__": "__main__",
            "INPUTS": message.get("inputs") or {},
            "RESULT": None,
        }
        for name in message.get("functions", []):
            self._globals[name] = self._stub(name)

    def _exec(self, message: dict) -> dict:
        g = self._globals
        g["obs"] = message.get("obs") or {}
        for name in message.get("functions", []):
            g.setdefault(name, self._stub(name))

        out, err = io.StringIO(), io.StringIO()
        ok = True
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(message["code"], g, g)  # noqa: S102 — ここが生成コードの実行
        except BaseException:
            ok = False
            traceback.print_exc(file=err)

        result = g.get("RESULT")
        try:
            rpc.pack(result)
        except Exception:
            result = None
        return {
            "op": "done",
            "ok": ok,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "result": result,
        }

    def serve(self) -> None:
        while True:
            try:
                message = rpc.recv(self._sock)
            except rpc.RpcClosed:
                return
            op = message.get("op")
            if op == "reset":
                self._reset(message)
                rpc.send(self._sock, {"op": "ready"})
            elif op == "exec":
                rpc.send(self._sock, self._exec(message))
            elif op == "exit":
                return


def main() -> None:
    sock = socket.socket(fileno=int(sys.argv[1]))
    _Session(sock).serve()


if __name__ == "__main__":
    main()
