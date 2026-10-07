"""手元PC と worker の間で交わすメッセージ。

msgpack でバイナリのまま運ぶ。画像も mp4 も base64 にしないので 33% の膨張が
ない。ZMQ の multipart は使わず 1 フレームに収める——`ROUTER`/`DEALER` の
封筒（identity frame）と混ざると読みにくいため。

既存の `capx/utils/msgpack_server_client_utils.py` は長さ付き TCP で、
受信長を無検証で読む・schema が無い・`raw=True` で str key が bytes になる、
といった違いがある。着想だけ借りて別に作った。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

#: 1 メッセージの上限。動画を積んでも収まり、壊れた長さで巨大確保しない値。
MAX_MESSAGE_BYTES = 256 * 1024 * 1024

#: ZMQ 層の heartbeat（アプリ層の `ping` とは別）。送信間隔と、応答が無いと
#: 切断とみなすまでの時間。step の最中も I/O スレッドが回っているので誤検出しない。
HEARTBEAT_IVL_MS = 5_000
HEARTBEAT_TIMEOUT_MS = 30_000


class ProtocolError(RuntimeError):
    """受け取ったメッセージが契約に合わない。"""


@dataclass
class Message:
    """1 往復ぶんの中身。

    Attributes:
        kind: ``request`` / ``response`` / ``event`` / ``error``。
            ``event`` はサーバからの push（ストリームのフレーム）で、
            対応する request を持たない。
        operation: ``step`` / ``render`` / ``ping`` / ``close`` / ``frame``。
        request_id: 応答を突き合わせる鍵。再送の判定にも使う——同じ id が
            二度来たら、実行し直さず前の応答を返す。
        sequence: 送信側の連番。順序の乱れを検出する。
    """

    kind: str
    operation: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    sequence: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "operation": self.operation,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "payload": self.payload,
        }


def _packer():
    import msgpack
    import msgpack_numpy

    # numpy 配列をそのまま運べるようにする。`patch()` はグローバルに効くので
    # 呼ばず、pack/unpack にフックを渡す。
    return msgpack, msgpack_numpy


def encode(message: Message) -> bytes:
    """`Message` を bytes にする。"""
    msgpack, msgpack_numpy = _packer()
    raw = msgpack.packb(
        message.to_dict(), default=msgpack_numpy.encode, use_bin_type=True
    )
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"メッセージが上限を超えた: {len(raw)} > {MAX_MESSAGE_BYTES}"
        )
    return raw


def decode(raw: bytes) -> Message:
    """bytes を `Message` にする。契約に合わなければ `ProtocolError`。

    受け取ったものを検証してから使う。壊れた長さで巨大なメモリを確保したり、
    知らない version のメッセージを解釈し続けたりしない。
    """
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"メッセージが上限を超えた: {len(raw)}")

    msgpack, msgpack_numpy = _packer()
    try:
        data = msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)
    except Exception as exc:
        raise ProtocolError(f"msgpack を解けない: {exc}") from exc

    if not isinstance(data, dict):
        raise ProtocolError(f"dict でない: {type(data).__name__}")

    version = data.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version が違う: {version} != {PROTOCOL_VERSION}。"
            " 手元PC と worker のどちらかが古い"
        )

    for key in ("kind", "operation"):
        if not isinstance(data.get(key), str):
            raise ProtocolError(f"{key} が無いか文字列でない")

    payload = data.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise ProtocolError("payload が dict でない")

    return Message(
        kind=data["kind"],
        operation=data["operation"],
        request_id=str(data.get("request_id", "")),
        session_id=str(data.get("session_id", "")),
        sequence=int(data.get("sequence", 0)),
        payload=payload or {},
        version=version,
    )


def request(operation: str, session_id: str = "", **payload: Any) -> Message:
    return Message(
        kind="request", operation=operation, session_id=session_id, payload=payload
    )


def response(to: Message, **payload: Any) -> Message:
    return Message(
        kind="response",
        operation=to.operation,
        request_id=to.request_id,
        session_id=to.session_id,
        payload=payload,
    )


def event(operation: str, session_id: str = "", **payload: Any) -> Message:
    """サーバからの push。対応する request を持たない（ストリームのフレーム）。"""
    return Message(
        kind="event", operation=operation, session_id=session_id, request_id="", payload=payload
    )


#: ストリームのフレームを送る間隔の下限（10 fps）と JPEG の品質。
STREAM_INTERVAL_S = 0.1
STREAM_JPEG_QUALITY = 80


def error(
    to: Message | None, message: str, kind: str = "agent_error", reason: str | None = None
) -> Message:
    """失敗を返す。`kind` は `capx.agent_api.FailureKind` に対応させる。

    クライアント側がこれを見て、リトライするか（infrastructure）
    しないか（budget）を決める。文字列判定にしないための区別。
    `reason` は予算超過の内訳（`execution_budget` / `wall_clock` など）。
    """
    payload: dict[str, Any] = {"message": message, "failure_kind": kind}
    if reason:
        payload["reason"] = reason
    return Message(
        kind="error",
        operation=to.operation if to else "unknown",
        request_id=to.request_id if to else "",
        session_id=to.session_id if to else "",
        payload=payload,
    )
