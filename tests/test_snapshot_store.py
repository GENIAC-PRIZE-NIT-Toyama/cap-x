"""Snapshot の形式（壊れ・不完全・版違いを読まない）と、保存先（持ち主ごと・変更されない）。"""

from __future__ import annotations

import numpy as np
import pytest

from capx.snapshot import blob
from capx.snapshot.store import (
    DirStore,
    SnapshotNotFound,
    SnapshotStoreError,
    check_id,
    new_id,
    owner_dir,
    store_from_env,
)


def _blob(**manifest) -> bytes:
    return blob.pack(
        {"task_id": "cube_stack", "agent_step": 2, **manifest},
        gym={"env": {"qpos": np.arange(5.0)}, "local_env": {"last_reward": 0.5}},
        namespace=b"pickled-namespace",
        history=[{"code": "x = 1", "ok": True, "stdout": "", "stderr": ""}],
    )


def _record(data: bytes, snapshot_id: str, owner: str = "alice") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "owner": owner,
        "task_id": "cube_stack",
        "size_bytes": len(data),
        "sha256": blob.sha256(data),
    }


def test_round_trip_keeps_every_section() -> None:
    snap = blob.unpack(_blob())
    assert snap.manifest["task_id"] == "cube_stack" and snap.manifest["agent_step"] == 2
    assert np.array_equal(snap.gym["env"]["qpos"], np.arange(5.0))
    assert snap.namespace == b"pickled-namespace"
    assert snap.history[0]["code"] == "x = 1"
    assert set(snap.manifest["sections"]) == {"gym", "namespace", "history"}


def test_any_flipped_byte_is_detected() -> None:
    data = bytearray(_blob())
    for position in range(len(data)):
        broken = bytearray(data)
        broken[position] ^= 0x55
        with pytest.raises(blob.SnapshotFormatError):
            blob.unpack(bytes(broken))


def test_incomplete_or_foreign_data_is_rejected() -> None:
    import msgpack

    data = _blob()
    outer = msgpack.unpackb(data, raw=False)
    del outer["sections"]["namespace"]  # Gym だけの Snapshot
    with pytest.raises(blob.SnapshotFormatError, match="namespace が無い"):
        blob.read(msgpack.packb(outer, use_bin_type=True))
    outer = msgpack.unpackb(data, raw=False)
    outer["schema"] = blob.SCHEMA_VERSION + 1
    with pytest.raises(blob.SnapshotFormatError, match="版が違う"):
        blob.read(msgpack.packb(outer, use_bin_type=True))
    with pytest.raises(blob.SnapshotFormatError, match="CaP-X の Snapshot ではない"):
        blob.read(msgpack.packb({"hello": 1}))
    with pytest.raises(blob.SnapshotFormatError, match="上限"):
        blob.read(data, max_bytes=10)


def test_ids_cannot_escape_the_owner_directory() -> None:
    assert check_id(new_id())
    for bad in ("../alice/snap_000000000000000000000000", "snap_xyz", "", "snap_" + "0" * 23):
        with pytest.raises(SnapshotNotFound):
            check_id(bad)
    assert owner_dir("alice") == "alice"
    assert owner_dir("../bob").startswith("h_") and owner_dir("山本").startswith("h_")


def test_dir_store_is_per_owner_and_immutable(tmp_path) -> None:
    store = DirStore(tmp_path)
    data, sid = _blob(), new_id()
    store.put("alice", sid, data, _record(data, sid))

    got, record = store.get("alice", sid)
    assert got == data and record["task_id"] == "cube_stack"
    with pytest.raises(SnapshotNotFound):
        store.get("bob", sid)  # 他人の id を知っていても見つからない
    with pytest.raises(SnapshotStoreError, match="上書きしない"):
        store.put("alice", sid, data, _record(data, sid))
    assert [r["snapshot_id"] for r in store.list("alice")] == [sid] and store.list("bob") == []

    store.delete("alice", sid)
    with pytest.raises(SnapshotNotFound):
        store.get("alice", sid)


def test_dir_store_detects_a_corrupted_body_and_a_missing_record(tmp_path) -> None:
    store = DirStore(tmp_path)
    data, sid = _blob(), new_id()
    store.put("alice", sid, data, _record(data, sid))
    body = next(tmp_path.rglob(f"{sid}.snap"))
    body.write_bytes(data[:-1] + b"\x00")
    with pytest.raises(SnapshotStoreError, match="壊れている"):
        store.get("alice", sid)
    next(tmp_path.rglob(f"{sid}.json")).unlink()  # 記録が無い = 完成していない
    with pytest.raises(SnapshotNotFound):
        store.get("alice", sid)


def test_store_from_env() -> None:
    assert store_from_env({}) is None
    assert isinstance(store_from_env({"CAPX_SNAPSHOT_DIR": "/tmp/x"}), DirStore)
    with pytest.raises(SnapshotStoreError, match="CAPX_SNAPSHOT_S3_BUCKET"):
        store_from_env({"CAPX_SNAPSHOT_S3_ENDPOINT": "http://127.0.0.1:9000"})
