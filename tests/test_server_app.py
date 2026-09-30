"""backend の HTTP と認証。

「トークンが無ければ通らない」「他人のセッションに触れない」を重く見る。
docker は偽物。fastapi が無い環境（参加者側の `remote/`）では skip する
——そこに backend の依存は要らない。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import zmq  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from capx.remote_env.server.app import create_app, parse_tokens  # noqa: E402
from capx.remote_env.server.sessions import Config, SessionManager  # noqa: E402


class FakeDocker:
    async def __call__(self, cmd):
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 1, "", ""
        return 0, "", ""


def _key() -> str:
    return zmq.curve_keypair()[0].decode("ascii")


@pytest.fixture
def client():
    manager = SessionManager(
        Config(public_host="gpu.example", port_start=19500, port_end=19510),
        runner=FakeDocker(),
    )
    tokens = parse_tokens("alice:tokA,bob:tokB")
    with TestClient(create_app(manager, tokens, reap=False)) as c:
        yield c


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_parse_tokens() -> None:
    assert parse_tokens("alice:a,bob:b") == {"a": "alice", "b": "bob"}
    assert parse_tokens("solo") == {"solo": "user"}
    with pytest.raises(ValueError):
        parse_tokens("alice:")


def test_health_needs_no_token(client) -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, _auth("wrong"), {"Authorization": "Basic tokA"}])
def test_no_valid_token_no_access(client, headers) -> None:
    assert client.post(
        "/sessions", json={"task_id": "cube_stack", "client_public_key": _key()}, headers=headers
    ).status_code == 401
    assert client.get("/sessions", headers=headers).status_code == 401


def test_create_returns_endpoint_and_keys(client) -> None:
    r = client.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key()},
        headers=_auth("tokA"),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["zmq_endpoint"].startswith("tcp://gpu.example:195")
    assert len(body["server_public_key"]) == 40
    assert body["session_id"]


def test_secret_key_is_never_in_the_response(client) -> None:
    """worker の秘密鍵は HTTP に載せない。平文で流れる経路なので。"""
    r = client.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key()},
        headers=_auth("tokA"),
    )
    assert not any("secret" in k.lower() for k in r.json())


def test_listing_never_exposes_session_ids(client) -> None:
    """一覧は空き状況だけ。id を返さない。

    認証なしだと全員が同じ持ち主になる。一覧に id を出すと、他人のセッションを
    消せてしまう。id は作った本人だけが知っていればよい。
    """
    created = client.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key()},
        headers=_auth("tokA"),
    ).json()

    listing = client.get("/sessions", headers=_auth("tokB")).json()
    assert listing == {"active": 1, "capacity": 20}
    assert created["session_id"] not in str(listing)


def test_bob_cannot_delete_alices_session(client) -> None:
    created = client.post(
        "/sessions",
        json={"task_id": "cube_stack", "client_public_key": _key()},
        headers=_auth("tokA"),
    ).json()

    # 他人のものは「無い」と答える。存在を教えない
    assert client.delete(f"/sessions/{created['session_id']}", headers=_auth("tokB")).status_code == 404
    assert client.delete(f"/sessions/{created['session_id']}", headers=_auth("tokA")).status_code == 200
    assert client.get("/sessions", headers=_auth("tokA")).json()["active"] == 0


def test_one_person_can_open_several_sessions(client) -> None:
    """ターミナルを 3 つ開けば 3 セッション。互いを閉じない。"""
    ids = [
        client.post(
            "/sessions",
            json={"task_id": "cube_stack", "client_public_key": _key()},
            headers=_auth("tokA"),
        ).json()["session_id"]
        for _ in range(3)
    ]
    assert len(set(ids)) == 3
    assert client.get("/sessions", headers=_auth("tokA")).json()["active"] == 3


def test_unlisted_task_is_400(client) -> None:
    r = client.post(
        "/sessions",
        json={"task_id": "not_a_task", "client_public_key": _key()},
        headers=_auth("tokA"),
    )
    assert r.status_code == 400
    assert "cube_stack" in r.json()["detail"], "選べるタスクを教える"


def test_tasks_lists_the_allowlist(client) -> None:
    tasks = client.get("/tasks", headers=_auth("tokA")).json()["tasks"]
    assert "cube_stack" in tasks


@pytest.fixture
def open_client():
    """認証なし。既定の起動状態。"""
    manager = SessionManager(
        Config(public_host="gpu.example", port_start=19500, port_end=19510),
        runner=FakeDocker(),
    )
    with TestClient(create_app(manager, None, reap=False)) as c:
        yield c


def test_without_tokens_no_header_is_needed(open_client) -> None:
    """既定は認証なし。参加者の設定にトークンは要らない。"""
    r = open_client.post(
        "/sessions", json={"task_id": "cube_stack", "client_public_key": _key()}
    )
    assert r.status_code == 200, r.text


def test_without_tokens_people_still_cannot_end_each_others_sessions(open_client) -> None:
    """認証なしでも、他人のセッションは消せない。id を知らないから。"""
    created = open_client.post(
        "/sessions", json={"task_id": "cube_stack", "client_public_key": _key()}
    ).json()
    guessed = open_client.delete("/sessions/000000000000")
    assert guessed.status_code == 404
    assert open_client.get("/sessions").json()["active"] == 1
    assert created["session_id"] not in str(open_client.get("/sessions").json())
