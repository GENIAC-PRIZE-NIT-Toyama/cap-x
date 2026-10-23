"""backend が Snapshot を置く場所。S3（MinIO）と、開発・テスト用のディレクトリ。

1 つの Snapshot は 2 つのオブジェクトになる。

    <prefix><持ち主>/<snapshot_id>.snap   本体（`capx.snapshot.blob` の bytes）
    <prefix><持ち主>/<snapshot_id>.json   記録（持ち主・タスク・sha256・Step 数・Trace など）

**本体を先に、記録を後に置く。** 記録があることが「完成した」印で、記録の無い Snapshot
（本体の途中で落ちたもの）は存在しないものとして扱う。id は毎回新しく作り、同じ名前に
上書きしない（Snapshot は変更されない）。

**持ち主ごとに場所を分ける。** 読むときは、要求した人の名前で場所を組み立てる。他人の
Snapshot は、id を知っていても見つからない（backend の認証が有効なとき。認証なしでは
全員が同じ持ち主 `dev` になる）。

backend はここで pickle を解かない。本体は bytes のまま運ぶ。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

#: Snapshot の id。推測されないよう 96 bit の乱数を入れる。
ID_RE = re.compile(r"^snap_[0-9a-f]{24}$")


class SnapshotNotFound(KeyError):
    """無い（他人のもの・消したもの・完成していないもの）。"""


class SnapshotStoreError(RuntimeError):
    """置けない・読めない（保存先の障害）。"""


def new_id() -> str:
    import secrets

    return f"snap_{secrets.token_hex(12)}"


def check_id(snapshot_id: str) -> str:
    if not isinstance(snapshot_id, str) or not ID_RE.match(snapshot_id):
        raise SnapshotNotFound(f"Snapshot の id の形が違う: {snapshot_id!r}")
    return snapshot_id


def owner_dir(owner: str) -> str:
    """持ち主の名前をパスに使える形にする。使えない文字があればハッシュにする。"""
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", owner) and owner not in (".", ".."):
        return owner
    return "h_" + hashlib.sha256(owner.encode()).hexdigest()[:24]


class SnapshotStore(Protocol):
    def put(self, owner: str, snapshot_id: str, blob: bytes, record: dict[str, Any]) -> None: ...

    def get(self, owner: str, snapshot_id: str) -> tuple[bytes, dict[str, Any]]: ...

    def record(self, owner: str, snapshot_id: str) -> dict[str, Any]: ...

    def list(self, owner: str, limit: int = 100) -> list[dict[str, Any]]: ...

    def delete(self, owner: str, snapshot_id: str) -> None: ...

    def describe(self) -> str: ...


def _check_blob(blob: bytes, record: dict[str, Any]) -> None:
    from capx.snapshot.blob import sha256

    if sha256(blob) != record.get("sha256") or len(blob) != record.get("size_bytes"):
        raise SnapshotStoreError(
            f"Snapshot {record.get('snapshot_id')} の本体が壊れている（sha256 か大きさが記録と合わない）"
        )


class DirStore:
    """手元のディレクトリに置く。開発とテスト用（backend を 1 台で動かすときにも使える）。"""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def describe(self) -> str:
        return f"dir:{self._root}"

    def _paths(self, owner: str, snapshot_id: str) -> tuple[Path, Path]:
        base = self._root / owner_dir(owner)
        return base / f"{check_id(snapshot_id)}.snap", base / f"{snapshot_id}.json"

    def put(self, owner: str, snapshot_id: str, blob: bytes, record: dict[str, Any]) -> None:
        body, meta = self._paths(owner, snapshot_id)
        body.parent.mkdir(parents=True, exist_ok=True)
        if meta.exists() or body.exists():
            raise SnapshotStoreError(f"{snapshot_id} はもうある（上書きしない）")
        _atomic_write(body, blob)
        _atomic_write(meta, json.dumps(record, ensure_ascii=False).encode())

    def record(self, owner: str, snapshot_id: str) -> dict[str, Any]:
        _body, meta = self._paths(owner, snapshot_id)
        if not meta.is_file():
            raise SnapshotNotFound(snapshot_id)
        return json.loads(meta.read_text())

    def get(self, owner: str, snapshot_id: str) -> tuple[bytes, dict[str, Any]]:
        record = self.record(owner, snapshot_id)
        body, _meta = self._paths(owner, snapshot_id)
        try:
            blob = body.read_bytes()
        except FileNotFoundError:
            raise SnapshotStoreError(f"{snapshot_id} の本体が無い（記録だけが残っている）") from None
        _check_blob(blob, record)
        return blob, record

    def list(self, owner: str, limit: int = 100) -> list[dict[str, Any]]:
        base = self._root / owner_dir(owner)
        metas = sorted(base.glob("snap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [json.loads(p.read_text()) for p in metas[:limit]]

    def delete(self, owner: str, snapshot_id: str) -> None:
        body, meta = self._paths(owner, snapshot_id)
        if not meta.exists():
            raise SnapshotNotFound(snapshot_id)
        meta.unlink()  # 記録から消す（先に「無い」状態にする）
        body.unlink(missing_ok=True)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


class S3Store:
    """S3 互換のストレージ（MinIO）に置く。Langfuse と同じ MinIO を使ってよいが、bucket は分ける。

    `capx.snapshot.s3`（requests だけで書いた最小のクライアント）で話す。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-1",
        prefix: str = "v1/",
    ) -> None:
        from capx.snapshot.s3 import S3Client

        self._bucket = bucket
        self._prefix = prefix
        self._endpoint = endpoint
        self._client = S3Client(endpoint, access_key_id, secret_access_key, region)

    def describe(self) -> str:
        return f"s3:{self._endpoint}/{self._bucket}/{self._prefix}"

    def _keys(self, owner: str, snapshot_id: str) -> tuple[str, str]:
        base = f"{self._prefix}{owner_dir(owner)}/{check_id(snapshot_id)}"
        return f"{base}.snap", f"{base}.json"

    def _call(self, what: str, fn, *args):
        from capx.snapshot.s3 import S3Error

        try:
            return fn(*args)
        except S3Error as exc:
            raise SnapshotStoreError(f"Snapshot を{what}（{self.describe()}）: {exc}") from None

    def check(self) -> None:
        """bucket に届くか（起動時に確かめる）。"""
        self._call("置く bucket に届かない", self._client.head_bucket, self._bucket)

    def put(self, owner: str, snapshot_id: str, blob: bytes, record: dict[str, Any]) -> None:
        body, meta = self._keys(owner, snapshot_id)
        self._call("保存できない", self._client.put_object, self._bucket, body, blob)
        self._call(
            "保存できない",
            self._client.put_object,
            self._bucket,
            meta,
            json.dumps(record, ensure_ascii=False).encode(),
            "application/json",
        )

    def record(self, owner: str, snapshot_id: str) -> dict[str, Any]:
        _body, meta = self._keys(owner, snapshot_id)
        raw = self._call("読めない", self._client.get_object, self._bucket, meta)
        if raw is None:
            raise SnapshotNotFound(snapshot_id)
        return json.loads(raw)

    def get(self, owner: str, snapshot_id: str) -> tuple[bytes, dict[str, Any]]:
        record = self.record(owner, snapshot_id)
        body, _meta = self._keys(owner, snapshot_id)
        blob = self._call("読めない", self._client.get_object, self._bucket, body)
        if blob is None:
            raise SnapshotStoreError(f"{snapshot_id} の本体が無い（記録だけが残っている）")
        _check_blob(blob, record)
        return blob, record

    def list(self, owner: str, limit: int = 100) -> list[dict[str, Any]]:
        prefix = f"{self._prefix}{owner_dir(owner)}/"
        objects = self._call("一覧にできない", self._client.list_objects, self._bucket, prefix)
        metas = sorted((o for o in objects if o.key.endswith(".json")), key=lambda o: o.last_modified, reverse=True)
        out = []
        for obj in metas[:limit]:
            raw = self._call("読めない", self._client.get_object, self._bucket, obj.key)
            if raw is not None:
                out.append(json.loads(raw))
        return out

    def delete(self, owner: str, snapshot_id: str) -> None:
        body, meta = self._keys(owner, snapshot_id)
        self.record(owner, snapshot_id)  # 無ければ SnapshotNotFound
        self._call("消せない", self._client.delete_object, self._bucket, meta)
        self._call("消せない", self._client.delete_object, self._bucket, body)


def store_from_env(environ: dict[str, str] | None = None) -> SnapshotStore | None:
    """環境変数から保存先を作る。設定が無ければ None（Snapshot は使えない）。

        CAPX_SNAPSHOT_S3_ENDPOINT / CAPX_SNAPSHOT_S3_BUCKET /
        CAPX_SNAPSHOT_S3_ACCESS_KEY_ID / CAPX_SNAPSHOT_S3_SECRET_ACCESS_KEY /
        CAPX_SNAPSHOT_S3_REGION（任意）/ CAPX_SNAPSHOT_S3_PREFIX（任意）
        または CAPX_SNAPSHOT_DIR（手元のディレクトリ。開発用）
    """
    env = os.environ if environ is None else environ
    endpoint = env.get("CAPX_SNAPSHOT_S3_ENDPOINT", "").strip()
    if endpoint:
        missing = [
            key
            for key in (
                "CAPX_SNAPSHOT_S3_BUCKET",
                "CAPX_SNAPSHOT_S3_ACCESS_KEY_ID",
                "CAPX_SNAPSHOT_S3_SECRET_ACCESS_KEY",
            )
            if not env.get(key)
        ]
        if missing:
            raise SnapshotStoreError(f"Snapshot の S3 の設定が足りない: {', '.join(missing)}")
        return S3Store(
            endpoint=endpoint,
            bucket=env["CAPX_SNAPSHOT_S3_BUCKET"],
            access_key_id=env["CAPX_SNAPSHOT_S3_ACCESS_KEY_ID"],
            secret_access_key=env["CAPX_SNAPSHOT_S3_SECRET_ACCESS_KEY"],
            region=env.get("CAPX_SNAPSHOT_S3_REGION") or "us-east-1",
            prefix=env.get("CAPX_SNAPSHOT_S3_PREFIX", "v1/"),
        )
    directory = env.get("CAPX_SNAPSHOT_DIR", "").strip()
    if directory:
        return DirStore(directory)
    return None

