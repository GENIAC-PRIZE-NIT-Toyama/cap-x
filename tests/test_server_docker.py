"""worker コンテナの `docker run` に、隔離のフラグが付いているかを確かめる。

隔離の設定は、付け忘れても**動いてしまう**。コンテナは普通に起動し、
セッションも普通に走るので、外へ出られる状態になっていても誰も気づかない。
実行して確かめる前に、組み立てた時点で押さえる。

docker は要らない。組み立てた文字列を見るだけ。
"""

from __future__ import annotations

from capx.remote_env.server import docker as d


def _spec(**kw) -> d.RunSpec:
    base = dict(
        session_id="abc123",
        owner="alice",
        image="capx-worker:latest",
        config_path="env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
        host_port=19503,
        secrets={"CAPX_CURVE_SERVER_SECRET": "SECRETKEYVALUE"},
    )
    base.update(kw)
    return d.RunSpec(**base)


def _flag_values(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, c in enumerate(cmd) if c == flag and i + 1 < len(cmd)]


def test_privileges_are_dropped() -> None:
    cmd = d.run_command(_spec())
    assert _flag_values(cmd, "--cap-drop") == ["ALL"]
    assert "no-new-privileges" in _flag_values(cmd, "--security-opt")


def test_resources_are_capped() -> None:
    cmd = d.run_command(_spec())
    assert _flag_values(cmd, "--memory") == ["4g"]
    assert _flag_values(cmd, "--cpus") == ["2"]
    assert _flag_values(cmd, "--pids-limit") == ["512"]


def test_primary_network_is_the_internal_one() -> None:
    """主ネットワークは --internal 側。IO ネットワークを先に付けると外へ出られる。

    IO ネットワークは起動後に別コマンドで繋ぐ。`docker run` の引数に
    IO ネットワークが入っていたら、デフォルトルートができてしまう。
    """
    spec = _spec()
    cmd = d.run_command(spec)
    assert _flag_values(cmd, "--network") == [spec.network_name]
    assert d.IO_NETWORK not in cmd

    assert d.io_connect_command(spec)[-2:] == [d.IO_NETWORK, spec.container_name]


def test_session_network_is_created_internal() -> None:
    assert "--internal" in d.network_create_command(_spec())


def test_io_network_cannot_masquerade() -> None:
    """公開ポートを通すためだけのネットワーク。外向きの経路にしてはいけない。"""
    joined = " ".join(d.io_network_create_command())
    assert "enable_ip_masquerade=false" in joined
    assert "enable_icc=false" in joined


def test_port_is_published_on_the_allocated_number() -> None:
    cmd = d.run_command(_spec(host_port=19507))
    assert _flag_values(cmd, "-p") == [f"0.0.0.0:19507:{d.CONTAINER_PORT}"]


def test_perception_urls_point_at_the_proxies() -> None:
    """知覚 API は固定宛先の proxy に書き換えて渡す。

    参加者は *_SERVICE_URL を設定しない。生成コードが何を書いても、
    コンテナから届くのはこの 3 つだけ。
    """
    envs = _flag_values(d.run_command(_spec()), "-e")
    assert "SAM3_SERVICE_URL=http://capx-proxy-sam3:8114" in envs
    assert "GRASPNET_SERVICE_URL=http://capx-proxy-graspnet:8115" in envs
    assert "PYROKI_SERVICE_URL=http://capx-proxy-pyroki:8116" in envs


def test_proxies_are_connected_to_the_session_network() -> None:
    spec = _spec()
    connects = d.proxy_connect_commands(spec)
    assert len(connects) == 3
    assert all(c[-2] == spec.network_name for c in connects)


def test_gpu_uses_the_runtime_flag_not_gpus() -> None:
    """`--gpus` は CDI 経由になり、libnvidia-gl の無いホストで失敗する。"""
    cmd = d.run_command(_spec(gpu_device="GPU-abc"))
    assert "--gpus" not in cmd
    assert _flag_values(cmd, "--runtime") == ["nvidia"]
    assert "NVIDIA_VISIBLE_DEVICES=GPU-abc" in _flag_values(cmd, "-e")


def test_containers_are_labelled_for_orphan_recovery() -> None:
    """backend が落ちたあとに、自分が起動したものを見つけて片付けるため。"""
    labels = _flag_values(d.run_command(_spec()), "--label")
    assert f"{d.LABEL_MANAGED}=1" in labels
    assert f"{d.LABEL_SESSION}=abc123" in labels
    assert f"{d.LABEL_OWNER}=alice" in labels


def test_secret_goes_in_the_environment_not_as_an_argument_to_the_worker() -> None:
    """秘密鍵は worker の argv に出さない（`ps` から見えるため）。

    `-e KEY=VALUE` は docker のクライアントの argv には出るが、コンテナ内の
    プロセスの argv には出ない。worker に渡す引数の側に入っていてはいけない。
    """
    cmd = d.run_command(_spec())
    image_at = cmd.index("capx-worker:latest")
    worker_args = cmd[image_at + 1 :]
    assert "SECRETKEYVALUE" not in " ".join(worker_args)
    assert "CAPX_CURVE_SERVER_SECRET=SECRETKEYVALUE" in _flag_values(cmd, "-e")


def test_cleanup_is_one_command_per_step() -> None:
    """どれかが失敗しても残りを続けられるように、別々のコマンドにしてある。"""
    cmds = d.cleanup_commands(_spec())
    assert cmds[0][:3] == ["docker", "rm", "-f"]
    assert cmds[-1][:3] == ["docker", "network", "rm"]
    assert len(cmds) == 1 + len(d.PROXIES) + 1
