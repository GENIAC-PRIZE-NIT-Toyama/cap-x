"""手元PC と worker が交わすメッセージの契約を固定する。

ここが崩れると、参加者の手元と GPU マシンで解釈がずれる。片方だけ更新された
ときに黙って動き続けるより、version 不一致で落ちる方がよい。

simulator も ZMQ の接続も要らないので手元PC で走る。
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.remote_env import protocol


def test_round_trip_keeps_fields() -> None:
    message = protocol.request("step", session_id="s1", code="x = 1", capture_video=True)
    message.sequence = 7

    decoded = protocol.decode(protocol.encode(message))

    assert decoded.kind == "request"
    assert decoded.operation == "step"
    assert decoded.session_id == "s1"
    assert decoded.sequence == 7
    assert decoded.payload["code"] == "x = 1"
    assert decoded.payload["capture_video"] is True
    assert decoded.request_id == message.request_id


def test_string_keys_survive() -> None:
    """`raw=True` だと str の key が bytes になる。

    既存の `msgpack_server_client_utils` がそうなっていて、受け取った側で
    `payload["code"]` が引けない。ここを取り違えると通信できているのに
    KeyError で落ちる。
    """
    decoded = protocol.decode(protocol.encode(protocol.request("step", code="x")))
    assert all(isinstance(k, str) for k in decoded.payload)


def test_bytes_survive_unchanged() -> None:
    """画像と動画は bytes のまま運ぶ。base64 にしないので 33% 膨らまない。"""
    blob = bytes(range(256)) * 10
    decoded = protocol.decode(
        protocol.encode(protocol.response(protocol.request("render"), image=blob))
    )
    assert decoded.payload["image"] == blob


def test_numpy_arrays_survive() -> None:
    """観測は送らない契約だが、経路としては numpy を運べる必要がある。"""
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded = protocol.decode(
        protocol.encode(protocol.response(protocol.request("step"), arr=array))
    )
    assert np.array_equal(decoded.payload["arr"], array)


def test_version_mismatch_is_rejected() -> None:
    """片方だけ更新されたら、黙って動き続けずに落ちる。"""
    message = protocol.request("step")
    message.version = protocol.PROTOCOL_VERSION + 1
    with pytest.raises(protocol.ProtocolError, match="version"):
        protocol.decode(protocol.encode(message))


def test_garbage_is_rejected() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(b"not msgpack at all")


def test_non_dict_is_rejected() -> None:
    import msgpack

    with pytest.raises(protocol.ProtocolError, match="dict"):
        protocol.decode(msgpack.packb([1, 2, 3]))


def test_oversized_message_is_refused_on_encode() -> None:
    """壊れた長さで巨大なメモリを確保しないための上限。"""
    huge = protocol.response(
        protocol.request("step"), blob=b"\0" * (protocol.MAX_MESSAGE_BYTES + 1)
    )
    with pytest.raises(protocol.ProtocolError, match="上限"):
        protocol.encode(huge)


def test_error_carries_failure_kind() -> None:
    """リトライするかどうかを文字列判定させない。

    budget 超過は Agent の責任なのでリトライしない。infrastructure は
    作り直してリトライする。区別がつかないと、両方同じ扱いになる。
    """
    original = protocol.request("step")
    for kind in ("budget", "infrastructure", "agent_error"):
        decoded = protocol.decode(
            protocol.encode(protocol.error(original, "boom", kind=kind))
        )
        assert decoded.kind == "error"
        assert decoded.payload["failure_kind"] == kind
        assert decoded.request_id == original.request_id


def test_response_matches_request_id() -> None:
    """応答の取り違えを防ぐ鍵。再送の判定にも同じものを使う。"""
    original = protocol.request("step")
    reply = protocol.response(original, result={})
    assert reply.request_id == original.request_id
    assert reply.operation == original.operation
