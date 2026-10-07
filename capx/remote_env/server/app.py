"""backend の HTTP。セッションの作成・削除・一覧。

    POST   /sessions          {task_id, client_public_key} -> ZMQ の宛先と鍵
    DELETE /sessions/{id}     返す
    GET    /sessions          空き状況（使用中の数と上限）。id は返さない
    GET    /tasks             選べるタスク
    GET    /health

認証は任意。`tokens` を渡したときだけ Bearer トークンを求める。既定は認証なしで、
すでに LAN に出ている知覚 API のサーバと同じ扱い。認証の有無に関わらず、
データ面（ZMQ）は CURVE で守られていて、他人の worker には繋がらない——繋ぐには、
その人が作ったクライアント鍵の秘密鍵が要る。
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from capx.remote_env.server.sessions import SessionError, SessionManager
from capx.remote_env.server.tasks import TASKS

logger = logging.getLogger("capx.server")

REAP_INTERVAL_S = 60.0


class CreateSession(BaseModel):
    task_id: str
    client_public_key: str


#: `Host` ヘッダの host 部として受け入れる形。ホスト名・IPv4・角括弧付き IPv6。
_HOST_RE = re.compile(r"^(\[[0-9A-Fa-f:]+\]|[A-Za-z0-9._-]+)$")


def endpoint_host(request: Request, configured: str | None) -> str:
    """クライアントに返す ZMQ の宛先ホストを決める。

    明示の設定があればそれ。無ければ、クライアントがこの backend に HTTP で
    繋いだときの宛先（`Host` ヘッダ）を使う。その宛先で HTTP が通ったのだから、
    同じマシンの ZMQ ポートにも同じ宛先で届く。マシンに IP が複数あっても、
    クライアントが実際に使った側が返る。

    `Host` ヘッダはクライアントが決める値だが、返す先はその同じクライアント
    だけなので、他人に向けた攻撃にはならない。それでも、ホスト名として
    ありえない文字列は受け付けない。
    """
    if configured:
        return configured
    host = request.headers.get("host", "")
    # ポートを外す。IPv6 は "[::1]:8200" の形なので角括弧の外の ":" だけを見る。
    name = host.rsplit(":", 1)[0] if re.search(r"\]?:\d+$", host) else host
    if not _HOST_RE.match(name):
        raise HTTPException(
            400,
            "接続先のホスト名を決められない。--public-host で指定してください",
        )
    return name


def parse_tokens(spec: str) -> dict[str, str]:
    """`alice:tokA,bob:tokB` -> {token: owner}。`tokA` だけなら owner は "user"。

    トークン 1 つが 1 人。owner がセッションの持ち主になり、他人のものは
    消せず、一覧にも出ない。
    """
    tokens: dict[str, str] = {}
    for item in filter(None, (p.strip() for p in spec.split(","))):
        owner, sep, token = item.partition(":")
        if not sep:
            owner, token = "user", item
        if not token:
            raise ValueError(f"トークンが空: {item!r}")
        tokens[token] = owner
    return tokens


def create_app(
    manager: SessionManager,
    tokens: dict[str, str] | None,
    *,
    reap: bool = True,
) -> FastAPI:
    """`tokens` が None なら認証なし（開発用）。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        removed = await manager.reconcile()
        if removed:
            logger.info("前回の残り %d 個を片付けた", removed)
        task = asyncio.create_task(_reaper()) if reap else None
        try:
            yield
        finally:
            if task:
                task.cancel()
            await manager.close_all()

    async def _reaper() -> None:
        while True:
            await asyncio.sleep(REAP_INTERVAL_S)
            try:
                await manager.reap_once()
            except Exception:
                logger.exception("reaper")

    app = FastAPI(title="CaP-X env server", lifespan=lifespan)

    def owner_of(authorization: str | None = Header(default=None)) -> str:
        if tokens is None:
            return "dev"
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() == "bearer" and presented:
            # 全トークンと比較する。どれに一致したかで時間が変わらないように。
            found = None
            for token, owner in tokens.items():
                if hmac.compare_digest(presented.encode(), token.encode()):
                    found = owner
            if found is not None:
                return found
        raise HTTPException(401, "トークンが無いか違う（CAPX_ENV_SERVER_TOKEN）")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "sessions": len(manager.list())}

    @app.get("/tasks")
    async def tasks(_owner: str = Depends(owner_of)) -> dict:
        return {"tasks": sorted(TASKS)}

    @app.post("/sessions")
    async def create(
        body: CreateSession, request: Request, owner: str = Depends(owner_of)
    ) -> dict:
        try:
            session = await manager.create(owner, body.task_id, body.client_public_key)
        except SessionError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        return {
            "session_id": session.session_id,
            "zmq_endpoint": (
                f"tcp://{endpoint_host(request, manager.config.public_host)}"
                f":{session.spec.host_port}"
            ),
            "server_public_key": session.server_public_key,
            "task_id": session.task_id,
        }

    @app.delete("/sessions/{session_id}")
    async def delete(session_id: str, owner: str = Depends(owner_of)) -> dict:
        session = manager.get(session_id)
        # 他人のものは「無い」と答える。存在を教えない。
        if session is None or session.owner != owner:
            raise HTTPException(404, "そのセッションは無い")
        await manager.close(session_id)
        return {"closed": session_id}

    @app.get("/sessions")
    async def occupancy(_owner: str = Depends(owner_of)) -> dict:
        # id は返さない。認証なしだと全員が同じ持ち主になるので、一覧に id を
        # 出すと、他人のセッションを消せてしまう。id は作った本人だけが知る。
        return {
            "active": len(manager.list()),
            "capacity": manager.config.max_sessions,
        }

    return app
