"""セッションの台帳。誰にどのコンテナを貸しているか。

データは通らない。ここがやるのは、コンテナを起こす・鍵を渡す・生きているか見る・
使われなくなったら片付ける、だけ。コードと画像は手元PC と worker の間を
ZMQ で直接流れる。

docker の実行は `runner` として差し替えられる。本番は subprocess だが、
テストでは記録するだけの偽物を渡して、docker 無しでロジックを確かめる。
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import cycle

from capx.remote_env.server import docker as dk
from capx.remote_env.server.tasks import IMAGES, resolve

logger = logging.getLogger("capx.server")

Runner = Callable[[list[str]], Awaitable[tuple[int, str, str]]]
Prober = Callable[["Session"], Awaitable[dict | None]]


async def subprocess_runner(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


class SessionError(RuntimeError):
    """呼び出し側に見せてよい失敗。`status` は HTTP の状態に対応させる。"""

    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    session_id: str
    owner: str
    task_id: str
    spec: dk.RunSpec
    server_public_key: str
    created: float = field(default_factory=time.monotonic)
    missed_pings: int = 0


class PortAllocator:
    """公開ポートの割り当て。確保と解放は同じ lock の下で行う。"""

    def __init__(self, start: int, end: int) -> None:
        self._free = set(range(start, end))
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        async with self._lock:
            if not self._free:
                raise SessionError("空きポートがない。しばらく待って再度試す", 429)
            port = min(self._free)
            self._free.discard(port)
            return port

    async def release(self, port: int) -> None:
        async with self._lock:
            self._free.add(port)


@dataclass
class Config:
    public_host: str
    """クライアントに返す ZMQ の宛先ホスト。手元PC から届く名前か IP。"""

    port_start: int = 19500
    port_end: int = 19600
    max_sessions: int = 20
    """同時セッション数の上限。× 4GB が GPU マシンの RAM を超えないこと。"""

    idle_ttl_s: float = 20 * 60
    max_lifetime_s: float = 2 * 60 * 60
    ready_timeout_s: float = 300.0
    ping_failures_allowed: int = 3
    gpu_devices: list[str] = field(default_factory=lambda: ["all"])


class SessionManager:
    def __init__(
        self,
        config: Config,
        *,
        runner: Runner = subprocess_runner,
        prober: Prober | None = None,
    ) -> None:
        self.config = config
        self._run = runner
        self._probe = prober
        self._sessions: dict[str, Session] = {}
        self._ports = PortAllocator(config.port_start, config.port_end)
        self._gpus = cycle(config.gpu_devices)
        self._lock = asyncio.Lock()
        self._io_ready = False

    # -- 作る --------------------------------------------------------------

    async def create(self, owner: str, task_id: str, client_public_key: str) -> Session:
        try:
            entry = resolve(task_id)
        except KeyError as exc:
            raise SessionError(str(exc.args[0]), 400) from None
        if not _looks_like_curve_key(client_public_key):
            raise SessionError("client_public_key が CURVE の公開鍵の形でない", 400)

        # 同じ人の古いセッションは片付ける。Ctrl-C で抜けた残りに塞がれて
        # 次が作れない、を避ける。
        for old in [s for s in self._sessions.values() if s.owner == owner]:
            logger.info("owner %s の前のセッション %s を閉じる", owner, old.session_id)
            await self.close(old.session_id)

        if len(self._sessions) >= self.config.max_sessions:
            raise SessionError(
                f"同時セッションが上限（{self.config.max_sessions}）。"
                "誰かが終わるまで待って再度試す",
                429,
            )

        import zmq

        server_public, server_secret = zmq.curve_keypair()
        port = await self._ports.acquire()
        session_id = secrets.token_hex(6)
        spec = dk.RunSpec(
            session_id=session_id,
            owner=owner,
            image=IMAGES[entry.runtime],
            config_path=entry.config_path,
            host_port=port,
            gpu_device=next(self._gpus),
            secrets={
                "CAPX_CURVE_SERVER_SECRET": server_secret.decode("ascii"),
                # クライアントの鍵に加え、backend 自身の生存確認用の鍵も許可する
                "CAPX_CURVE_ALLOWED_CLIENT_KEYS": ",".join(
                    [client_public_key, self._probe_public_key()]
                ),
            },
        )
        session = Session(
            session_id=session_id,
            owner=owner,
            task_id=task_id,
            spec=spec,
            server_public_key=server_public.decode("ascii"),
        )
        self._sessions[session_id] = session  # 失敗時に片付けられるよう先に登録

        try:
            await self._start(spec)
            await self._wait_ready(session)
        except Exception:
            await self.close(session_id)
            raise
        logger.info("session %s ready (owner=%s task=%s port=%d)", session_id, owner, task_id, port)
        return session

    async def _start(self, spec: dk.RunSpec) -> None:
        await self._ensure_io_network()
        await self._must(dk.network_create_command(spec), "ネットワークを作れない")
        for cmd in dk.proxy_connect_commands(spec):
            await self._must(cmd, "知覚 API の proxy を繋げない（proxy は起動している？）")
        await self._must(dk.run_command(spec), "コンテナを起動できない")
        # 起動後に繋ぐ。先に繋ぐとデフォルトルートができて外へ出られる。
        await self._must(dk.io_connect_command(spec), "公開ポートを有効にできない")

    async def _ensure_io_network(self) -> None:
        if self._io_ready:
            return
        rc, _out, _err = await self._run(["docker", "network", "inspect", dk.IO_NETWORK])
        if rc != 0:
            await self._must(dk.io_network_create_command(), "IO ネットワークを作れない")
        self._io_ready = True

    async def _must(self, cmd: list[str], message: str) -> str:
        rc, out, err = await self._run(cmd)
        if rc != 0:
            raise SessionError(f"{message}: {err.strip()[:300]}", 500)
        return out

    async def _wait_ready(self, session: Session) -> None:
        """worker が ping に答えるまで待つ。環境を作り終えてから listen するので、
        答えた時点で必ず使える。"""
        if self._probe is None:
            return
        deadline = time.monotonic() + self.config.ready_timeout_s
        while time.monotonic() < deadline:
            if await self._probe(session) is not None:
                return
            rc, out, _err = await self._run(
                ["docker", "inspect", "-f", "{{.State.Running}}", session.spec.container_name]
            )
            if rc != 0 or out.strip() != "true":
                _rc, logs, _ = await self._run(
                    ["docker", "logs", "--tail", "40", session.spec.container_name]
                )
                raise SessionError(
                    f"worker が起動中に終了した:\n{logs[-1500:]}", 500
                )
            await asyncio.sleep(2.0)
        raise SessionError(f"worker が {self.config.ready_timeout_s:.0f}s で立ち上がらなかった", 504)

    # -- 閉じる ------------------------------------------------------------

    async def close(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        for cmd in dk.cleanup_commands(session.spec):
            await self._run(cmd)  # どれかが失敗しても残りを続ける
        await self._ports.release(session.spec.host_port)
        return True

    async def close_all(self) -> None:
        await asyncio.gather(*(self.close(sid) for sid in list(self._sessions)))

    async def reconcile(self) -> int:
        """起動時に、前回の backend が残したコンテナを片付ける。

        backend が落ちるとメモリ上の台帳が消える。ラベルで自分が起動したものを
        見つけて消さないと、ポートも GPU も掴んだまま誰にも使われない。
        """
        rc, out, _err = await self._run(
            ["docker", "ps", "-aq", "--filter", f"label={dk.LABEL_MANAGED}=1"]
        )
        ids = out.split() if rc == 0 else []
        for cid in ids:
            await self._run(["docker", "rm", "-f", cid])
        rc, out, _err = await self._run(
            ["docker", "network", "ls", "-q", "--filter", "name=capx-net-"]
        )
        for nid in (out.split() if rc == 0 else []):
            await self._run(["docker", "network", "rm", nid])
        return len(ids)

    # -- 見張る ------------------------------------------------------------

    async def reap_once(self) -> list[str]:
        """使われなくなったセッションを閉じる。閉じたものの id を返す。

        - 寿命を超えた: 使用中でも閉じる（最終的な上限）
        - worker が応答しない: 続けて N 回で閉じる（コンテナが死んでいる）
        - idle が長い: 閉じる。ただし step の最中（busy）は対象外——
          1000 秒かかる step の間に回収されてはならない
        """
        closed: list[str] = []
        now = time.monotonic()
        for session in list(self._sessions.values()):
            if now - session.created > self.config.max_lifetime_s:
                logger.info("session %s: 寿命", session.session_id)
                await self.close(session.session_id)
                closed.append(session.session_id)
                continue

            if self._probe is None:
                continue
            reply = await self._probe(session)
            if reply is None:
                session.missed_pings += 1
                if session.missed_pings >= self.config.ping_failures_allowed:
                    logger.info("session %s: 応答なし", session.session_id)
                    await self.close(session.session_id)
                    closed.append(session.session_id)
                continue

            session.missed_pings = 0
            if not reply.get("busy") and reply.get("idle_s", 0) > self.config.idle_ttl_s:
                logger.info("session %s: idle %.0fs", session.session_id, reply["idle_s"])
                await self.close(session.session_id)
                closed.append(session.session_id)
        return closed

    # -- 参照 --------------------------------------------------------------

    def list(self, owner: str | None = None) -> list[Session]:
        return [s for s in self._sessions.values() if owner is None or s.owner == owner]

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def _probe_public_key(self) -> str:
        """backend が worker に ping するための鍵。プロセスごとに 1 つ。"""
        if not hasattr(self, "_probe_keys"):
            import zmq

            self._probe_keys = zmq.curve_keypair()
        return self._probe_keys[0].decode("ascii")

    @property
    def probe_keypair(self) -> tuple[bytes, bytes]:
        self._probe_public_key()
        return self._probe_keys


def _looks_like_curve_key(key: str) -> bool:
    """Z85 の 40 文字。形だけ見る（正しさは接続時に CURVE が確かめる）。"""
    return isinstance(key, str) and len(key) == 40 and key.isascii() and key.isprintable()
