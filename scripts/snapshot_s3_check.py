"""backend のマシンから、Snapshot の保存先（MinIO）に届くかを確かめる（管理者用）。

    uv run --no-sync --active scripts/snapshot_s3_check.py

backend と同じ環境変数（CAPX_SNAPSHOT_S3_*）を読み、bucket に届くか・置けるか・読めるか・
一覧にできるか・消せるかを順に試す。試しに置いたものは最後に消す。
"""

from __future__ import annotations

import sys
import time


def main() -> None:
    from capx.snapshot.store import S3Store, SnapshotStoreError, new_id, store_from_env

    try:
        store = store_from_env()
    except SnapshotStoreError as exc:
        sys.exit(f"NG 設定: {exc}")
    if not isinstance(store, S3Store):
        sys.exit("NG 設定: CAPX_SNAPSHOT_S3_ENDPOINT が無い（S3 / MinIO を使う設定になっていない）")
    print(f"保存先: {store.describe()}")

    from capx.snapshot import blob

    owner, sid = "_check", new_id()
    data = blob.pack({"task_id": "_check"}, gym={}, namespace=b"", history=[])
    record = {"snapshot_id": sid, "owner": owner, "sha256": blob.sha256(data), "size_bytes": len(data)}
    steps = [
        ("bucket に届く", lambda: store.check()),
        ("置ける", lambda: store.put(owner, sid, data, record)),
        ("読める（sha256 が合う）", lambda: store.get(owner, sid)),
        ("一覧にできる", lambda: [r["snapshot_id"] for r in store.list(owner)].index(sid)),
        ("消せる", lambda: store.delete(owner, sid)),
    ]
    for name, fn in steps:
        started = time.monotonic()
        try:
            fn()
        except Exception as exc:
            sys.exit(f"NG {name}: {type(exc).__name__}: {exc}")
        print(f"OK {name}（{(time.monotonic() - started) * 1000:.0f}ms）")


if __name__ == "__main__":
    main()
