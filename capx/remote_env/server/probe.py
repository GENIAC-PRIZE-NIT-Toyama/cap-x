"""backend から worker への生存確認と、Snapshot の受け渡し。

生存確認は `ping` だけ送り、busy と idle を読む。backend 専用の鍵を worker が許可して
いるので、CURVE のまま話せる。

Snapshot の取り出し（`snapshot_fetch`）と復元（`restore`）も同じ鍵で呼ぶ。worker は
この 2 つを backend の鍵の接続にだけ許す。Snapshot の bytes はクライアントを通らない。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from capx.remote_env import protocol

PROBE_TIMEOUT_S = 3.0


def make_prober(keypair: tuple[bytes, bytes], host: str = "127.0.0.1"):
    """`Session` を受けて ping し、応答の payload か None を返す関数を作る。"""

    async def probe(session: Any) -> dict | None:
        return await asyncio.to_thread(
            _ping, f"tcp://{host}:{session.spec.host_port}", session.server_public_key, keypair
        )

    return probe


def _ping(endpoint: str, server_public: str, keypair: tuple[bytes, bytes]) -> dict | None:
    import zmq

    context = zmq.Context.instance()
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.curve_serverkey = server_public.encode("ascii")
    sock.curve_publickey, sock.curve_secretkey = keypair
    try:
        sock.connect(endpoint)
        sock.send(protocol.encode(protocol.request("ping")), flags=zmq.NOBLOCK)
        if not sock.poll(int(PROBE_TIMEOUT_S * 1000)):
            return None
        reply = protocol.decode(sock.recv())
        return reply.payload if reply.kind == "response" else None
    except Exception:
        return None
    finally:
        sock.close()


class WorkerCallError(RuntimeError):
    """worker が失敗を返した・応答しない。"""


def make_caller(keypair: tuple[bytes, bytes], host: str = "127.0.0.1"):
    """`call(session, operation, timeout_s, **payload) -> payload` を作る。"""

    async def call(session: Any, operation: str, timeout_s: float, **payload: Any) -> dict:
        return await asyncio.to_thread(
            _call,
            f"tcp://{host}:{session.spec.host_port}",
            session.server_public_key,
            keypair,
            operation,
            timeout_s,
            payload,
        )

    return call


def _call(
    endpoint: str,
    server_public: str,
    keypair: tuple[bytes, bytes],
    operation: str,
    timeout_s: float,
    payload: dict,
) -> dict:
    import zmq

    context = zmq.Context.instance()
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.MAXMSGSIZE, protocol.MAX_MESSAGE_BYTES)
    sock.curve_serverkey = server_public.encode("ascii")
    sock.curve_publickey, sock.curve_secretkey = keypair
    message = protocol.request(operation, **payload)
    try:
        sock.connect(endpoint)
        sock.send(protocol.encode(message))
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not sock.poll(int(min(remaining, 5.0) * 1000)):
                if remaining <= 0:
                    raise WorkerCallError(f"worker が {operation} に {timeout_s:.0f}s で答えなかった")
                continue
            reply = protocol.decode(sock.recv())
            if reply.request_id != message.request_id:
                continue  # 別の要求への応答（ストリームのフレームなど）
            if reply.kind == "error":
                raise WorkerCallError(str(reply.payload.get("message", "worker error")))
            return reply.payload
    finally:
        sock.close()
