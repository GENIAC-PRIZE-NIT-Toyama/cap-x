"""Gym プロセスと policy プロセスの間の RPC。

生成コードを動かすプロセス（policy）は、Gym（シミュレータ・reward・採点）と
別にする。境界を越えるのは API の呼び出しと結果だけ。

**pickle は使わない。** policy 側は信用できないコードが動くので、そこから届く
バイト列を pickle で解くと、Gym 側で任意のコードが実行できてしまう。msgpack は
データしか運ばない。tuple は list になってしまうので、印を付けて戻す。
"""

from __future__ import annotations

import socket
import struct
from typing import Any

#: 1 フレームの上限。観測（画像・深度）を含んでも収まる値。
MAX_FRAME_BYTES = 256 * 1024 * 1024

_TUPLE = "__tuple__"
_HEADER = struct.Struct("!Q")


class RpcClosed(ConnectionError):
    """相手が閉じた・落ちた。"""


def _tag_tuples(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return {_TUPLE: [_tag_tuples(x) for x in obj]}
    if isinstance(obj, list):
        return [_tag_tuples(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tag_tuples(v) for k, v in obj.items()}
    return obj


def pack(obj: Any) -> bytes:
    import msgpack
    import msgpack_numpy

    return msgpack.packb(
        _tag_tuples(obj), default=msgpack_numpy.encode, use_bin_type=True
    )


def unpack(raw: bytes) -> Any:
    import msgpack
    import msgpack_numpy

    def hook(obj: Any) -> Any:
        if len(obj) == 1 and _TUPLE in obj:
            return tuple(obj[_TUPLE])
        return msgpack_numpy.decode(obj)

    return msgpack.unpackb(raw, object_hook=hook, raw=False, strict_map_key=False)


def send(sock: socket.socket, obj: Any) -> None:
    raw = pack(obj)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError(f"RPC メッセージが上限を超えた: {len(raw)}")
    sock.sendall(_HEADER.pack(len(raw)) + raw)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        chunk = sock.recv(min(n, 1 << 20))
        if not chunk:
            raise RpcClosed("相手が閉じた")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def recv(sock: socket.socket) -> Any:
    """1 メッセージ受ける。`sock.settimeout` で待ち時間を切れる（超えたら `TimeoutError`）。"""
    (length,) = _HEADER.unpack(_read_exact(sock, _HEADER.size))
    if length > MAX_FRAME_BYTES:
        # 壊れた長さや悪意のある長さで巨大な確保をしない
        raise RpcClosed(f"RPC フレームが大きすぎる: {length}")
    return unpack(_read_exact(sock, length))
