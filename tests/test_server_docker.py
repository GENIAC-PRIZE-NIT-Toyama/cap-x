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
    assert _flag_values(cmd, "--pids-limit") == ["4096"]


def test_thread_pools_match_the_allotted_cpus() -> None:
    """数値計算ライブラリがホストのコア数ぶんスレッドを作らないようにする。"""
    envs = _flag_values(d.run_command(_spec()), "-e")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"{var}=2" in envs


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
    assert "MOLMO_SERVICE_URL=http://capx-proxy-molmo:8122/v1" in envs


def test_proxies_are_connected_to_the_session_network() -> None:
    spec = _spec()
    connects = d.proxy_connect_commands(spec)
    assert len(connects) == len(d.PROXIES) == 4
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


def test_compose_proxy_names_match_the_code() -> None:
    """compose のコンテナ名と `PROXIES` がずれると、セッションを作る段階で
    `docker network connect` が失敗する。起動するまで気づけないので、ここで押さえる。
    """
    import re
    from pathlib import Path

    compose = Path(__file__).resolve().parent.parent / "docker/backend/docker-compose.yml"
    names = set(re.findall(r"container_name: (\S+)", compose.read_text(encoding="utf-8")))
    assert names == {name for name, _port in d.PROXIES.values()}


def test_dockerfile_installs_from_the_lock_and_starts_the_worker() -> None:
    """`uv sync --frozen`（lock どおり）と、worker の起動を押さえる。

    --frozen が外れると lock を再解決して、submodule や CUDA の無いビルド環境で
    失敗するか、lock と違うものが入る。ENTRYPOINT が違えば、コンテナは起動して
    何もせず終わる。
    """
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "docker/worker/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "uv sync --frozen" in text
    assert "--extra robosuite" in text
    assert "capx.remote_env.worker.server" in text
    assert f"EXPOSE {d.CONTAINER_PORT}" in text


def test_rootfs_is_read_only_and_the_worker_is_not_root() -> None:
    cmd = d.run_command(_spec())
    assert "--read-only" in cmd
    assert _flag_values(cmd, "--user") == ["10001:10001"]
    tmpfs = _flag_values(cmd, "--tmpfs")
    assert any(t.startswith("/tmp:") for t in tmpfs)
    assert any(t.startswith("/run:") for t in tmpfs)


def test_ulimits_are_set() -> None:
    limits = _flag_values(d.run_command(_spec()), "--ulimit")
    assert "core=0" in limits
    assert any(x.startswith("nofile=") for x in limits)


def test_the_default_seccomp_profile_is_never_switched_off() -> None:
    """docker の既定の seccomp プロフィールに任せる。unconfined にしない。"""
    opts = _flag_values(d.run_command(_spec()), "--security-opt")
    assert not any("seccomp" in o for o in opts)


def test_worker_image_can_be_pinned_by_digest(monkeypatch) -> None:
    import importlib

    from capx.remote_env.server import tasks

    monkeypatch.setenv("CAPX_WORKER_IMAGE", "capx-worker@sha256:abc")
    try:
        assert importlib.reload(tasks).IMAGES["robosuite"] == "capx-worker@sha256:abc"
    finally:
        monkeypatch.delenv("CAPX_WORKER_IMAGE")
        importlib.reload(tasks)


def test_dockerfile_drops_root_and_keeps_oracle_out_of_the_image() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    text = (root / "docker/worker/Dockerfile").read_text(encoding="utf-8")
    assert "USER 10001:10001" in text
    ignore = (root / "docker/worker/Dockerfile.dockerignore").read_text(encoding="utf-8")
    assert "capx/baselines" in ignore
    assert "env_configs/human_oracle_code" in ignore


def test_the_build_step_strips_oracle_code_from_task_modules(tmp_path, monkeypatch) -> None:
    """Dockerfile に埋めた除去スクリプトを、そのまま取り出して実行する。"""
    import subprocess
    import sys
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "docker/worker/Dockerfile").read_text(
        encoding="utf-8"
    )
    script = text.split("RUN python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]

    tasks = tmp_path / "capx/envs/tasks/franka"
    tasks.mkdir(parents=True)
    module = tasks / "t.py"
    module.write_text(
        'ORACLE_CODE = """\nsecret_answer()\n"""\n'
        "\n"
        "class Task:\n"
        "    oracle_code = ORACLE_CODE\n"
        '    other = "keep me"\n'
        "\n"
        "class Other:\n"
        '    oracle_code = "inline_secret()"\n',
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)

    out = module.read_text(encoding="utf-8")
    assert "secret_answer" not in out and "inline_secret" not in out
    assert 'other = "keep me"' in out
    compile(out, "t.py", "exec")  # 壊れていない
    ns: dict = {}
    exec(out, ns)
    assert ns["Task"].oracle_code is None
