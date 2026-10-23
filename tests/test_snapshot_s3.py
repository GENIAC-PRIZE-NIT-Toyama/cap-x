"""`capx.snapshot.s3`（requests だけで書いた S3 クライアント）。

- 署名（SigV4）が botocore の署名と同じになる（MinIO も AWS と同じ検証をする）
- moto の S3 サーバ相手に、置く・読む・一覧・消すが通る
どちらも、botocore / moto が入った環境でだけ動く（無ければ skip）。
"""

from __future__ import annotations

import datetime as dt
import importlib.util

import pytest

from capx.snapshot.s3 import S3Client

NOW = dt.datetime(2026, 10, 23, 4, 5, 6, tzinfo=dt.timezone.utc)


@pytest.mark.skipif(importlib.util.find_spec("botocore") is None, reason="botocore が無い")
@pytest.mark.parametrize(
    "method, key, query, body",
    [
        ("PUT", "v1/alice/snap_0123.snap", {}, b"\x00\x01binary"),
        ("GET", "v1/alice/snap_0123.json", {}, b""),
        ("GET", "", {"list-type": "2", "prefix": "v1/alice bob/", "continuation-token": "a+b/c=="}, b""),
        ("DELETE", "v1/h_ab/snap_x.snap", {}, b""),
        ("HEAD", "", {}, b""),
    ],
)
def test_signature_matches_botocore(monkeypatch, method, key, query, body) -> None:
    import botocore.auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    client = S3Client("http://192.168.0.5:9000", "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG", "us-east-1")
    from urllib.parse import quote

    path = "/capx-snapshots" + ("/" + quote(key, safe="/~") if key else "")
    ours = client.sign(method, path, query, body, {}, now=NOW)

    class Frozen(dt.datetime):
        @classmethod
        def utcnow(cls):
            return NOW.replace(tzinfo=None)

        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(botocore.auth.datetime, "datetime", Frozen)
    from capx.snapshot.s3 import _canonical_query

    url = "http://192.168.0.5:9000" + path + (f"?{_canonical_query(query)}" if query else "")
    request = AWSRequest(method=method, url=url, data=body)
    request.headers["x-amz-content-sha256"] = ours["x-amz-content-sha256"]
    botocore.auth.S3SigV4Auth(Credentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG"), "s3", "us-east-1").add_auth(request)
    assert ours["authorization"] == request.headers["Authorization"]


@pytest.fixture
def moto_server():
    pytest.importorskip("moto.server")
    from moto.server import ThreadedMotoServer

    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


def test_round_trip_against_an_s3_server(moto_server) -> None:
    from capx.snapshot import blob
    from capx.snapshot.store import S3Store, SnapshotNotFound, new_id

    client = S3Client(moto_server, "test", "test")
    client._request("PUT", "capx-snapshots")  # bucket を作る（MinIO では管理者が作る）
    store = S3Store(endpoint=moto_server, bucket="capx-snapshots", access_key_id="test", secret_access_key="test")
    store.check()

    data = b"\x00snapshot bytes\xff" * 1000
    ids = [new_id() for _ in range(3)]
    for sid in ids:
        store.put("alice", sid, data, {"snapshot_id": sid, "sha256": blob.sha256(data), "size_bytes": len(data)})
    assert store.get("alice", ids[0])[0] == data
    assert {r["snapshot_id"] for r in store.list("alice")} == set(ids)
    assert store.list("bob") == []
    with pytest.raises(SnapshotNotFound):
        store.get("bob", ids[0])
    store.delete("alice", ids[0])
    with pytest.raises(SnapshotNotFound):
        store.record("alice", ids[0])
    assert client.get_object("capx-snapshots", "v1/alice/missing.json") is None


def test_listing_follows_continuation_tokens(moto_server) -> None:
    client = S3Client(moto_server, "test", "test")
    client._request("PUT", "bkt")
    for i in range(1005):  # 1 回の一覧は 1000 件まで
        client.put_object("bkt", f"p/{i:05d}.json", b"{}")
    assert len(client.list_objects("bkt", "p/")) == 1005
