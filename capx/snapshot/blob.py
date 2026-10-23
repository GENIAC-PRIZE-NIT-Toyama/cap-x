"""Snapshot 1 つの形式。worker が作り、backend が保存し、別の worker が戻す。

    msgpack の map:
        magic            "capx-snapshot"
        schema           SCHEMA_VERSION
        manifest         msgpack の bytes（タスク・Step 数・各セクションの sha256 など。データだけ）
        manifest_sha256  manifest の bytes の sha256
        sections         {"gym": bytes, "namespace": bytes, "history": bytes}（各 zlib 圧縮）

どの 1 バイトが壊れても読まない（manifest は自分の sha256 で、各セクションは manifest に
書いた sha256 で確かめる）。

    sections の中身:
        gym        Gym の状態（`gym_state.capture`）と、LocalAgentEnv の記録（予算・最後の採点）。msgpack
        namespace  生成コードの globals（cloudpickle）。**policy プロセスの中でだけ解く**
        history    それまでの Step のコード・stdout・stderr。msgpack

**1 つの bytes にまとめる。** Gym だけ・Namespace だけが保存された Snapshot が生まれない
（どちらかが作れなければ、bytes 自体を作らない）。backend は 1 つの bytes を丸ごと置く。

backend はこの形式を msgpack で読んで、形と各セクションの sha256 を確かめる。pickle は
解かない（namespace は bytes のまま）。
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Any

MAGIC = "capx-snapshot"
#: 形式を変えたら上げる。読めない版は拒否する。
SCHEMA_VERSION = 1
#: zlib の圧縮レベル。6 で、MuJoCo の状態と Namespace は 1/3〜1/10 になる（docs/Snapshot.md の実測）。
COMPRESS_LEVEL = 6
SECTIONS = ("gym", "namespace", "history")


class SnapshotFormatError(RuntimeError):
    """Snapshot の bytes が壊れている・形が違う・版が違う。"""


@dataclass
class Snapshot:
    manifest: dict[str, Any]
    gym: dict[str, Any]
    namespace: bytes
    history: list[dict[str, Any]]


def _msgpack():
    import msgpack
    import msgpack_numpy

    return msgpack, msgpack_numpy


def pack_data(value: Any) -> bytes:
    msgpack, msgpack_numpy = _msgpack()
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpack_data(raw: bytes) -> Any:
    msgpack, msgpack_numpy = _msgpack()
    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False, strict_map_key=False)


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def pack(
    manifest: dict[str, Any], *, gym: dict[str, Any], namespace: bytes, history: list[dict[str, Any]]
) -> bytes:
    """1 つの bytes にする。manifest に各セクションの大きさと sha256 を書き足す。"""
    raw_sections = {"gym": pack_data(gym), "namespace": namespace, "history": pack_data(history)}
    sections = {name: zlib.compress(raw, COMPRESS_LEVEL) for name, raw in raw_sections.items()}
    manifest = {
        **manifest,
        "sections": {
            name: {
                "sha256": sha256(sections[name]),
                "compressed_bytes": len(sections[name]),
                "raw_bytes": len(raw_sections[name]),
                "compression": "zlib",
            }
            for name in SECTIONS
        },
    }
    msgpack, _ = _msgpack()
    manifest_raw = msgpack.packb(manifest, use_bin_type=True)
    return msgpack.packb(
        {
            "magic": MAGIC,
            "schema": SCHEMA_VERSION,
            "manifest": manifest_raw,
            "manifest_sha256": sha256(manifest_raw),
            "sections": sections,
        },
        use_bin_type=True,
    )


def read(blob: bytes, *, max_bytes: int | None = None) -> tuple[dict[str, Any], dict[str, bytes]]:
    """外枠を読み、形・版・各セクションの sha256 を確かめる。pickle は解かない。

    backend（保存前と復元前）と worker（復元前）が使う。

    Returns:
        (manifest, 圧縮されたままの sections)
    """
    if max_bytes is not None and len(blob) > max_bytes:
        raise SnapshotFormatError(f"Snapshot が上限より大きい: {len(blob)} > {max_bytes}")
    msgpack, _ = _msgpack()
    try:
        outer = msgpack.unpackb(blob, raw=False, max_bin_len=len(blob), max_str_len=1 << 20)
    except Exception as exc:
        raise SnapshotFormatError(f"Snapshot を読めない（壊れている）: {exc}") from None
    if not isinstance(outer, dict) or outer.get("magic") != MAGIC:
        raise SnapshotFormatError("CaP-X の Snapshot ではない")
    if outer.get("schema") != SCHEMA_VERSION:
        raise SnapshotFormatError(
            f"Snapshot の形式の版が違う（{outer.get('schema')}。この版が読めるのは {SCHEMA_VERSION}）"
        )
    manifest_raw, sections = outer.get("manifest"), outer.get("sections")
    if not isinstance(manifest_raw, bytes) or not isinstance(sections, dict):
        raise SnapshotFormatError("Snapshot の形が違う（manifest / sections）")
    if sha256(manifest_raw) != outer.get("manifest_sha256"):
        raise SnapshotFormatError("Snapshot の manifest が壊れている（sha256 が合わない）")
    try:
        manifest = msgpack.unpackb(manifest_raw, raw=False)
    except Exception as exc:
        raise SnapshotFormatError(f"Snapshot の manifest を読めない: {exc}") from None
    if not isinstance(manifest, dict):
        raise SnapshotFormatError("Snapshot の manifest の形が違う")
    described = manifest.get("sections") or {}
    for name in SECTIONS:
        data = sections.get(name)
        if not isinstance(data, bytes) or name not in described:
            raise SnapshotFormatError(f"Snapshot に {name} が無い（不完全）")
        if sha256(data) != described[name].get("sha256"):
            raise SnapshotFormatError(f"Snapshot の {name} が壊れている（sha256 が合わない）")
    return manifest, sections


def unpack(blob: bytes, *, max_bytes: int | None = None) -> Snapshot:
    """worker が復元に使う。namespace は bytes のまま返す（解くのは policy プロセス）。"""
    manifest, sections = read(blob, max_bytes=max_bytes)
    try:
        raw = {name: zlib.decompress(sections[name]) for name in SECTIONS}
    except zlib.error as exc:
        raise SnapshotFormatError(f"Snapshot を展開できない（壊れている）: {exc}") from None
    for name in SECTIONS:
        if len(raw[name]) != manifest["sections"][name].get("raw_bytes"):
            raise SnapshotFormatError(f"Snapshot の {name} の大きさが合わない（壊れている）")
    return Snapshot(
        manifest=manifest,
        gym=unpack_data(raw["gym"]),
        namespace=raw["namespace"],
        history=unpack_data(raw["history"]),
    )
