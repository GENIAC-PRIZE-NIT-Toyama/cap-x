"""セッションの台帳。誰にどのコンテナを貸しているか。

データは通らない。ここがやるのは、コンテナを起こす・鍵を渡す・生きているか見る・
使われなくなったら片付ける、だけ。コードと画像は手元PC と worker の間を
ZMQ で直接流れる。

例外は Snapshot。worker が作った Snapshot の bytes を backend が取り出して保存先
（MinIO）に置き、再開のときは保存先から読んで新しい worker に渡す。bytes はクライアントを
通らない（クライアントが作った bytes を worker に戻させない）。backend は bytes の形と
sha256 を確かめるだけで、pickle は解かない。

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
from typing import Any

from capx.remote_env.server import docker as dk
from capx.remote_env.server.tasks import IMAGES, limits_for, resolve

logger = logging.getLogger("capx.server")

Runner = Callable[[list[str]], Awaitable[tuple[int, str, str]]]
Prober = Callable[["Session"], Awaitable[dict | None]]
#: `call(session, operation, timeout_s, **payload)`。worker の Snapshot の取り出しと復元。
Caller = Callable[..., Awaitable[dict]]

#: worker から Snapshot を取り出す・worker に戻すのを待つ上限。
SNAPSHOT_FETCH_TIMEOUT_S = 120.0
RESTORE_TIMEOUT_S = 600.0
#: クライアントが Snapshot に添える情報（Trace の id・Agent の状態など）の上限。
MAX_SNAPSHOT_META_BYTES = 64 * 1024


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
    #: Snapshot から再開したセッションなら、その id と、クライアントに渡す再開の情報。
    resumed_from: str | None = None
    resume: dict | None = None


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
    public_host: str | None = None
    """クライアントに返す ZMQ の宛先ホスト。省略すると、クライアントが HTTP で
    この backend に繋いだときの宛先（`Host` ヘッダ）をそのまま使う。"""

    port_start: int = 19500
    port_end: int = 19600
    max_sessions: int = 20
    """同時セッション数の上限。メモリの合計（`memory_budget_bytes`）とは別に効く。"""

    memory_budget_bytes: int = 100 * 1024**3
    """全セッションに貸すメモリの合計の上限。各セッションの上限（タスクの種類ごと。
    `tasks.RUNTIME_MEMORY`）を足して、これを超えるなら新しいセッションを断る。
    GPU マシンの RAM から、OS・backend・中継などの分を残した値にする。"""

    idle_ttl_s: float = 20 * 60
    max_lifetime_s: float = 2 * 60 * 60
    ready_timeout_s: float = 300.0
    ping_failures_allowed: int = 3
    gpu_devices: list[str] = field(default_factory=lambda: ["all"])
    worker_env: dict[str, str] = field(default_factory=dict)
    """全 worker に渡す設定（秘密ではないもの）。例: `MOLMO2_VLM_MODEL`。"""

    snapshot_max_bytes: int = 64 * 1024 * 1024
    """1 つの Snapshot の上限。worker が作るときと、backend が受け取る・読むときに見る。
    生成コードの変数に画像を何枚も持つと大きくなる。worker のメモリ上限（4〜5GB）の中で、
    保存の途中に一時的に数倍のメモリを使うので、大きくしすぎない。"""


class SessionManager:
    def __init__(
        self,
        config: Config,
        *,
        runner: Runner = subprocess_runner,
        prober: Prober | None = None,
        caller: Caller | None = None,
        store: Any = None,
    ) -> None:
        self.config = config
        self._run = runner
        self._probe = prober
        self._call = caller
        #: Snapshot の置き場所（`capx.snapshot.store`）。None なら Snapshot は使えない。
        self.store = store
        self._sessions: dict[str, Session] = {}
        self._ports = PortAllocator(config.port_start, config.port_end)
        self._gpus = cycle(config.gpu_devices)
        self._lock = asyncio.Lock()
        self._io_ready = False

    # -- 作る --------------------------------------------------------------

    async def create(
        self,
        owner: str,
        task_id: str,
        client_public_key: str,
        resume_from: str | None = None,
    ) -> Session:
        """セッションを作る。`resume_from` を渡すと、その Snapshot を新しい worker に戻してから返す。

        再開は、元のセッションを使わない。新しいコンテナを起こし、Snapshot を戻す。
        戻せなければコンテナを片付けて失敗を返す（中途半端な状態のセッションを渡さない）。
        """
        try:
            entry = resolve(task_id)
        except KeyError as exc:
            raise SessionError(str(exc.args[0]), 400) from None
        if not _looks_like_curve_key(client_public_key):
            raise SessionError("client_public_key が CURVE の公開鍵の形でない", 400)

        # 先に Snapshot を読んで確かめる。無い・他人のもの・壊れている・タスクが違うなら、
        # コンテナを起こす前に断る。
        snapshot = await self._load_snapshot(owner, task_id, resume_from) if resume_from else None

        # 同じ人が何個でも同時に作れる。ターミナルを 3 つ開けば 3 セッション動く。
        # 「同じ持ち主が新しく作ったら古いのを閉じる」にすると、それを壊す。
        # Ctrl-C の残りは trial.py の finally が返し、それでも残った分
        # （強制終了・PC のスリープ）は idle 回収が拾う。

        import zmq

        limits = limits_for(entry.runtime)
        server_public, server_secret = zmq.curve_keypair()
        port = await self._ports.acquire()

        # ここから登録まで await を挟まない。挟むと、同時に来た別の作成が同じ空きを
        # 見て、数やメモリの上限を超えて受け付けてしまう。
        rejection = self._admission_error(dk.memory_bytes(limits.memory))
        if rejection is not None:
            await self._ports.release(port)
            raise rejection

        session_id = secrets.token_hex(6)
        spec = dk.RunSpec(
            session_id=session_id,
            owner=owner,
            image=IMAGES[entry.runtime],
            config_path=entry.config_path,
            task_id=task_id,
            overrides=entry.overrides,
            env={
                **self.config.worker_env,
                # Snapshot の取り出しと復元を、この鍵の接続（backend）だけに許す
                "CAPX_BACKEND_PUBLIC_KEY": self._probe_public_key(),
            },
            host_port=port,
            gpu_device=next(self._gpus),
            limits=limits,
            resume=snapshot is not None,
            snapshot_max_mb=max(1, self.config.snapshot_max_bytes // (1024 * 1024)),
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
            resumed_from=resume_from if snapshot else None,
        )
        self._sessions[session_id] = session  # 失敗時に片付けられるよう先に登録

        try:
            await self._start(spec)
            await self._wait_ready(session)
            if snapshot is not None:
                session.resume = await self._restore(session, *snapshot)
        except Exception:
            await self.close(session_id)
            raise
        logger.info(
            "session %s ready (owner=%s task=%s port=%d%s)",
            session_id, owner, task_id, port, f" resume_from={resume_from}" if snapshot else "",
        )
        return session

    # -- Snapshot ----------------------------------------------------------

    def _require_store(self) -> Any:
        if self.store is None:
            raise SessionError(
                "この backend は Snapshot の保存先が設定されていない（管理者に CAPX_SNAPSHOT_S3_* を頼む）",
                503,
            )
        if self._call is None:
            raise SessionError("この backend は worker と Snapshot を受け渡せない", 503)
        return self.store

    async def _load_snapshot(self, owner: str, task_id: str, snapshot_id: str) -> tuple[bytes, dict]:
        from capx.snapshot import blob as snapshot_blob
        from capx.snapshot.store import SnapshotNotFound, SnapshotStoreError

        store = self._require_store()
        try:
            data, record = await asyncio.to_thread(store.get, owner, snapshot_id)
        except SnapshotNotFound:
            raise SessionError(
                f"Snapshot {snapshot_id} は無い（id が違うか、別の人のもの・消したもの）", 404
            ) from None
        except SnapshotStoreError as exc:
            raise SessionError(str(exc), 502) from None
        if record.get("task_id") != task_id:
            raise SessionError(
                f"Snapshot {snapshot_id} はタスク {record.get('task_id')} のもの（指定は {task_id}）", 400
            )
        try:
            manifest, _ = snapshot_blob.read(data, max_bytes=self.config.snapshot_max_bytes)
        except snapshot_blob.SnapshotFormatError as exc:
            raise SessionError(f"Snapshot {snapshot_id} を使えない: {exc}", 422) from None
        if manifest.get("task_id") != task_id:
            raise SessionError(f"Snapshot {snapshot_id} の中身のタスクが記録と違う", 422)
        return data, record

    async def _restore(self, session: Session, data: bytes, record: dict) -> dict:
        from capx.remote_env.server.probe import WorkerCallError

        try:
            reply = await self._call(session, "restore", RESTORE_TIMEOUT_S, data=data)
        except WorkerCallError as exc:
            raise SessionError(f"Snapshot {record['snapshot_id']} から復元できなかった: {exc}", 500) from None
        return {
            **reply["resume"],
            "snapshot_id": record["snapshot_id"],
            "meta": record.get("meta") or {},
            "restore_s": reply.get("restore_s"),
        }

    async def persist_snapshot(self, session: Session, token: str, meta: dict | None = None) -> dict:
        """worker が手元に置いた Snapshot を取り出し、確かめてから保存先に置く。記録を返す。

        保存先に置き終わってから id を返す。途中で失敗したら id は返さない。
        """
        import json

        from capx.remote_env.server.probe import WorkerCallError
        from capx.snapshot import blob as snapshot_blob
        from capx.snapshot.store import SnapshotStoreError, new_id

        store = self._require_store()
        meta = meta or {}
        try:
            meta_bytes = len(json.dumps(meta, ensure_ascii=False))
        except (TypeError, ValueError):
            raise SessionError("Snapshot に添える情報（meta）が JSON にできない", 400) from None
        if meta_bytes > MAX_SNAPSHOT_META_BYTES:
            raise SessionError(f"Snapshot に添える情報（meta）が大きすぎる（{meta_bytes} バイト）", 400)

        started = time.monotonic()
        try:
            reply = await self._call(session, "snapshot_fetch", SNAPSHOT_FETCH_TIMEOUT_S, token=token)
        except WorkerCallError as exc:
            raise SessionError(f"worker から Snapshot を取り出せない: {exc}", 409) from None
        data = bytes(reply["data"])
        fetched = time.monotonic()
        try:
            manifest, _ = snapshot_blob.read(data, max_bytes=self.config.snapshot_max_bytes)
        except snapshot_blob.SnapshotFormatError as exc:
            raise SessionError(f"worker の Snapshot が壊れている: {exc}", 502) from None
        if manifest.get("task_id") != session.task_id:
            raise SessionError("worker の Snapshot のタスクがセッションと違う", 502)

        snapshot_id = new_id()
        record = {
            "snapshot_id": snapshot_id,
            "schema": snapshot_blob.SCHEMA_VERSION,
            "owner": session.owner,
            "task_id": session.task_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size_bytes": len(data),
            "sha256": snapshot_blob.sha256(data),
            "agent_step": manifest.get("agent_step"),
            "trial": manifest.get("trial"),
            "seed": manifest.get("seed"),
            "session_id": session.session_id,
            "resumed_from": session.resumed_from,
            "worker_image": session.spec.image,
            "env_class": manifest.get("env_class"),
            "model_signature": (manifest.get("gym") or {}).get("model_signature"),
            "namespace": manifest.get("namespace"),
            "sections": manifest.get("sections"),
            "versions": manifest.get("versions"),
            "meta": meta,
        }
        try:
            await asyncio.to_thread(store.put, session.owner, snapshot_id, data, record)
        except SnapshotStoreError as exc:
            raise SessionError(str(exc), 502) from None
        stored = time.monotonic()
        logger.info(
            "snapshot %s saved (session=%s step=%s %.0fKB fetch=%.2fs put=%.2fs)",
            snapshot_id, session.session_id, record["agent_step"], len(data) / 1024,
            fetched - started, stored - fetched,
        )
        return {**record, "timing": {"fetch_s": round(fetched - started, 4), "put_s": round(stored - fetched, 4)}}

    def reserved_memory_bytes(self) -> int:
        """使用中のセッションに貸しているメモリの合計（各セッションの上限の和）。"""
        return sum(dk.memory_bytes(s.spec.limits.memory) for s in self._sessions.values())

    def _admission_error(self, need_bytes: int) -> SessionError | None:
        """新しいセッションを受けられないなら、その理由を返す。"""
        if len(self._sessions) >= self.config.max_sessions:
            return SessionError(
                f"同時セッションが上限（{self.config.max_sessions}）。"
                "誰かが終わるまで待って再度試す",
                429,
            )
        used = self.reserved_memory_bytes()
        if used + need_bytes > self.config.memory_budget_bytes:
            gib = 1024**3
            return SessionError(
                f"GPU マシンのメモリの空きが足りない（使用中 {used / gib:.0f}GB、"
                f"このタスクに {need_bytes / gib:.0f}GB 要る、合計の上限 "
                f"{self.config.memory_budget_bytes / gib:.0f}GB）。誰かが終わるまで待って再度試す",
                429,
            )
        return None

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
