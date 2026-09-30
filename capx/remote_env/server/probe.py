"""backend から worker への生存確認。

データは通さない。`ping` だけ送り、busy と idle を読む。backend 専用の鍵を
worker が許可しているので、CURVE のまま話せる。
"""

from __future__ import annotations

import asyncio
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
