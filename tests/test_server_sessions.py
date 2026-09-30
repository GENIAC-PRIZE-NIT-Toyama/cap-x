"""セッション管理のロジックを、docker なしで確かめる。

docker の実行は記録するだけの偽物に差し替える。見たいのは台帳としての正しさ
——誰がいくつ持てるか、片付けが漏れないか、使用中を回収しないか。
"""

from __future__ import annotations

import asyncio

import pytest
import zmq

from capx.remote_env.server import docker as dk
from capx.remote_env.server.sessions import Config, SessionError, SessionManager


def _client_key() -> str:
    return zmq.curve_keypair()[0].decode("ascii")


class FakeDocker:
    """`docker ...` の代わりに記録する。`fail_on` に含まれる語のコマンドは失敗。"""

    def __init__(self, fail_on: str | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_on = fail_on
        self.running = True

    async def __call__(self, cmd: list[str]) -> tuple[int, str, str]:
        self.commands.append(cmd)
        if self.fail_on and self.fail_on in " ".join(cmd):
            return 1, "", "simulated failure"
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 1, "", "no such network"  # 初回は IO ネットワークが無い
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true" if self.running else "false", ""
        return 0, "", ""

    def ran(self, *words: str) -> list[list[str]]:
        return [c for c in self.commands if all(w in " ".join(c) for w in words)]


def _manager(docker: FakeDocker, prober=None, **cfg) -> SessionManager:
    config = Config(public_host="gpu.example", port_start=19500, port_end=19503, **cfg)
    return SessionManager(config, runner=docker, prober=prober)


async def _alive(_session):
    return {"busy": False, "idle_s": 0.0}


def run(coro):
    return asyncio.run(coro)


def test_create_starts_a_container_and_returns_keys() -> None:
    docker = FakeDocker()
    mgr = _manager(docker, prober=_alive)
    session = run(mgr.create("alice", "cube_stack", _client_key()))

    assert session.owner == "alice"
    assert len(session.server_public_key) == 40
    assert docker.ran("docker run", session.spec.container_name)
    # 起動後に IO ネットワークを繋いでいる（先ではない）
    run_at = docker.commands.index(dk.run_command(session.spec))
    io_at = docker.commands.index(dk.io_connect_command(session.spec))
    assert run_at < io_at


def test_unknown_task_is_rejected_before_any_docker_call() -> None:
    """allowlist に無いタスクは、コンテナを起こす前に断る。"""
    docker = FakeDocker()
    mgr = _manager(docker)
    with pytest.raises(SessionError) as exc:
        run(mgr.create("alice", "../../etc/passwd", _client_key()))
    assert exc.value.status == 400
    assert docker.commands == []


def test_malformed_public_key_is_rejected() -> None:
    docker = FakeDocker()
    with pytest.raises(SessionError) as exc:
        run(_manager(docker).create("alice", "cube_stack", "not-a-key"))
    assert exc.value.status == 400
    assert docker.commands == []


def test_a_new_session_replaces_the_owners_old_one() -> None:
    """Ctrl-C で抜けた残りに塞がれて、次が作れない、を避ける。"""
    docker = FakeDocker()
    mgr = _manager(docker, prober=_alive)

    async def scenario():
        first = await mgr.create("alice", "cube_stack", _client_key())
        second = await mgr.create("alice", "cube_stack", _client_key())
        return first, second

    first, second = run(scenario())
    assert mgr.get(first.session_id) is None
    assert mgr.get(second.session_id) is not None
    assert len(mgr.list("alice")) == 1
    assert docker.ran("docker rm -f", first.spec.container_name)


def test_global_limit_returns_429() -> None:
    docker = FakeDocker()
    mgr = _manager(docker, prober=_alive, max_sessions=2)

    async def scenario():
        await mgr.create("a", "cube_stack", _client_key())
        await mgr.create("b", "cube_stack", _client_key())
        await mgr.create("c", "cube_stack", _client_key())

    with pytest.raises(SessionError) as exc:
        run(scenario())
    assert exc.value.status == 429


def test_ports_are_not_reused_while_held_and_are_released_on_close() -> None:
    docker = FakeDocker()
    mgr = _manager(docker, prober=_alive)

    async def scenario():
        a = await mgr.create("a", "cube_stack", _client_key())
        b = await mgr.create("b", "cube_stack", _client_key())
        assert a.spec.host_port != b.spec.host_port
        await mgr.close(a.session_id)
        c = await mgr.create("c", "cube_stack", _client_key())
        return a, c

    a, c = run(scenario())
    assert c.spec.host_port == a.spec.host_port, "解放したポートが再利用されない"


def test_failed_start_cleans_up_and_frees_the_port() -> None:
    """起動に失敗したら、コンテナもネットワークもポートも残さない。"""
    docker = FakeDocker(fail_on="docker run")
    mgr = _manager(docker)

    with pytest.raises(SessionError):
        run(mgr.create("alice", "cube_stack", _client_key()))

    assert mgr.list() == []
    assert docker.ran("docker rm -f")
    assert docker.ran("docker network rm")

    docker.fail_on = None
    run(mgr.create("alice", "cube_stack", _client_key()))  # ポートが戻っている


def test_worker_that_dies_during_startup_reports_its_logs() -> None:
    docker = FakeDocker()
    docker.running = False

    async def never_ready(_s):
        return None

    mgr = _manager(docker, prober=never_ready)
    with pytest.raises(SessionError, match="終了した"):
        run(mgr.create("alice", "cube_stack", _client_key()))
    assert docker.ran("docker logs")


def test_reaper_closes_idle_sessions() -> None:
    docker = FakeDocker()

    async def idle(_s):
        return {"busy": False, "idle_s": 99999.0}

    mgr = _manager(docker, prober=idle, idle_ttl_s=60)

    async def scenario():
        s = await mgr.create("alice", "cube_stack", _client_key())
        return s, await mgr.reap_once()

    s, closed = run(scenario())
    assert closed == [s.session_id]
    assert mgr.list() == []


def test_reaper_never_closes_a_session_that_is_running_a_step() -> None:
    """1000 秒かかる step の最中に回収してはいけない。

    idle_s は最後の要求からの時間なので、長い step の最中は大きくなる。
    busy を見て除外しないと、実行中のコンテナを kill してしまう。
    """
    docker = FakeDocker()

    async def running(_s):
        return {"busy": True, "idle_s": 99999.0}

    mgr = _manager(docker, prober=running, idle_ttl_s=60)

    async def scenario():
        await mgr.create("alice", "cube_stack", _client_key())
        return await mgr.reap_once()

    assert run(scenario()) == []
    assert len(mgr.list()) == 1


def test_reaper_closes_after_repeated_ping_failures_only() -> None:
    """一度の取りこぼしで閉じない。続けて N 回で閉じる。"""
    docker = FakeDocker()
    calls = {"n": 0}

    async def flaky(_s):
        calls["n"] += 1
        return {"busy": False, "idle_s": 0.0} if calls["n"] == 1 else None

    mgr = _manager(docker, prober=flaky, ping_failures_allowed=3)

    async def scenario():
        s = await mgr.create("alice", "cube_stack", _client_key())  # ここで n=1
        first = await mgr.reap_once()  # 失敗 1
        second = await mgr.reap_once()  # 失敗 2
        third = await mgr.reap_once()  # 失敗 3 → 閉じる
        return s, first, second, third

    s, first, second, third = run(scenario())
    assert first == [] and second == []
    assert third == [s.session_id]


def test_reaper_closes_sessions_past_their_lifetime_even_if_busy() -> None:
    docker = FakeDocker()
    mgr = _manager(docker, prober=_alive, max_lifetime_s=-1)

    async def scenario():
        s = await mgr.create("alice", "cube_stack", _client_key())
        return s, await mgr.reap_once()

    s, closed = run(scenario())
    assert closed == [s.session_id]


def test_reconcile_removes_leftovers_from_a_previous_run() -> None:
    """backend が落ちると台帳は消える。ラベルで見つけて片付ける。"""

    class Leftovers(FakeDocker):
        async def __call__(self, cmd):
            self.commands.append(cmd)
            if cmd[:3] == ["docker", "ps", "-aq"]:
                return 0, "c1\nc2\n", ""
            if cmd[:3] == ["docker", "network", "ls"]:
                return 0, "n1\n", ""
            return 0, "", ""

    docker = Leftovers()
    removed = run(_manager(docker).reconcile())
    assert removed == 2
    assert docker.ran("docker rm -f c1")
    assert docker.ran("docker network rm n1")
    assert any(dk.LABEL_MANAGED in " ".join(c) for c in docker.ran("docker ps"))
