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

_CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"


class _ApiError(Exception):
    """Gym 側で API が投げた例外。中身は Gym 側が作った「型: メッセージ」。"""


def _format_error(exc: BaseException, seen: set[int] | None = None) -> str:
    """生成コードの中で起きた部分だけの traceback。

    このファイルのフレーム（`exec` を呼ぶ `_exec`、API のスタブ）は外す。`-m` で起動するので
    `__name__` は生成コードと同じ `"__main__"` になる。見分けるのはファイル名で。
    API の例外は、Gym 側から届いた文字列をそのまま最後の行にする（包み直した型は出さない）。
    """
    seen = seen if seen is not None else set()
    seen.add(id(exc))
    text = ""
    if exc.__cause__ is not None and id(exc.__cause__) not in seen:
        text = _format_error(exc.__cause__, seen) + _CAUSE
    elif (
        exc.__context__ is not None
        and not exc.__suppress_context__
        and id(exc.__context__) not in seen
    ):
        text = _format_error(exc.__context__, seen) + _CONTEXT

    frames = [
        (frame, lineno)
        for frame, lineno in traceback.walk_tb(exc.__traceback__)
        if frame.f_code.co_filename != __file__
    ]
    if frames:
        text += "Traceback (most recent call last):\n"
        text += "".join(traceback.StackSummary.extract(frames).format())
    if isinstance(exc, _ApiError):
        return text + f"{exc}\n"
    return text + "".join(traceback.format_exception_only(type(exc), exc))


class _Session:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._globals: dict[str, Any] = {}

    def _stub(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            rpc.send(self._sock, {"op": "call", "name": name, "args": args, "kwargs": kwargs})
            reply = rpc.recv(self._sock)
            if reply.get("op") == "raise":
                raise _ApiError(reply.get("error", "API call failed"))
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
        except BaseException as exc:
            ok = False
            err.write(_format_error(exc))

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
