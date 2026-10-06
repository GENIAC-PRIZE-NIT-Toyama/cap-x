"""worker コンテナを起動する `docker` コマンドを組み立てる。

実行はしない。文字列のリストを返すだけにして、どのフラグが付くかを
テストで確かめられるようにする——隔離の設定は、付け忘れても動いてしまい、
気づけないため。

ネットワークの作りは cap-x-workshop で検証済みの方式を引き継いでいる。

1. セッションごとに `--internal` のネットワークを作る。ルートを持たないので、
   コンテナから外へは出られない。これを **主** ネットワークにして起動する。
2. 知覚 API（SAM3 / GraspNet / PyRoKi）へは、固定宛先の socat コンテナ
   （proxy）をそのネットワークに繋いで届かせる。宛先が固定なので、生成コードが
   何を書いても、届くのはその 3 つだけ。
3. 起動後に、共有の IO ネットワークにも繋ぐ。`--internal` のネットワークは
   ポート公開を黙って捨てるので、公開のためだけに繋ぐ。masquerade を切って
   あるので、外向きの経路にはならない。

順序が大事で、IO ネットワークを先に（主に）すると、そちらにデフォルトルートが
できて、外へ出られてしまう。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: コンテナ内で worker が待ち受けるポート。ホスト側は割り当てた番号に写す。
CONTAINER_PORT = 19500

#: 共有の IO ネットワーク。公開ポートを通すためだけに繋ぐ。
IO_NETWORK = "capx-io"

#: 知覚 API の proxy。コンテナ名と、コンテナ内から見た環境変数。
PROXIES: dict[str, tuple[str, int]] = {
    "SAM3_SERVICE_URL": ("capx-proxy-sam3", 8114),
    "GRASPNET_SERVICE_URL": ("capx-proxy-graspnet", 8115),
    "PYROKI_SERVICE_URL": ("capx-proxy-pyroki", 8116),
}

LABEL_MANAGED = "capx.managed"
LABEL_SESSION = "capx.session"
LABEL_OWNER = "capx.owner"


@dataclass(frozen=True)
class Limits:
    memory: str = "4g"
    cpus: str = "2"
    pids: int = 512
    tmpfs: str = "/tmp:rw,size=1g"
    run_tmpfs: str = "/run:rw,size=16m"
    nofile: int = 4096
    #: worker イメージの USER と同じ。root で動かさない。
    user: str = "10001:10001"


@dataclass(frozen=True)
class RunSpec:
    session_id: str
    owner: str
    image: str
    config_path: str
    host_port: int
    gpu_device: str = "all"
    limits: Limits = field(default_factory=Limits)
    #: 秘密鍵などをここに入れる。argv ではなく環境変数で渡す（`ps` に出ない）。
    secrets: dict[str, str] = field(default_factory=dict)
    #: 許可リスト上のタスク名。結果に残す名前になる。
    task_id: str = ""
    #: config への上書き（許可リストが決めた値だけ）。
    overrides: tuple[tuple[str, str], ...] = ()

    @property
    def container_name(self) -> str:
        return f"capx-w-{self.session_id}"

    @property
    def network_name(self) -> str:
        return f"capx-net-{self.session_id}"


def network_create_command(spec: RunSpec) -> list[str]:
    return ["docker", "network", "create", "--internal", spec.network_name]


def io_network_create_command() -> list[str]:
    return [
        "docker", "network", "create",
        "-o", "com.docker.network.bridge.enable_ip_masquerade=false",
        "-o", "com.docker.network.bridge.enable_icc=false",
        IO_NETWORK,
    ]


def proxy_connect_commands(spec: RunSpec) -> list[list[str]]:
    return [
        ["docker", "network", "connect", spec.network_name, name]
        for name, _port in PROXIES.values()
    ]


def run_command(spec: RunSpec) -> list[str]:
    """`docker run`。隔離に関わるフラグはここに全部ある。"""
    cmd = [
        "docker", "run", "-d",
        "--name", spec.container_name,
        "--network", spec.network_name,  # 主ネットワークは --internal 側
        "--label", f"{LABEL_MANAGED}=1",
        "--label", f"{LABEL_SESSION}={spec.session_id}",
        "--label", f"{LABEL_OWNER}={spec.owner}",
        # 権限を落とす
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        # 資源の上限
        "--pids-limit", str(spec.limits.pids),
        "--memory", spec.limits.memory,
        "--cpus", spec.limits.cpus,
        "--tmpfs", spec.limits.tmpfs,
        # 書けるのは tmpfs（/tmp と /run）だけ。イメージの中は書き換えられない。
        "--read-only",
        "--tmpfs", spec.limits.run_tmpfs,
        "--user", spec.limits.user,
        "--ulimit", f"nofile={spec.limits.nofile}:{spec.limits.nofile}",
        "--ulimit", "core=0",
        # GPU。`--gpus` ではなく旧来の runtime 指定（workshop で検証済み。
        # `--gpus` は CDI 経由になり、libnvidia-gl が無いホストで失敗する）
        "--runtime", "nvidia",
        "-e", f"NVIDIA_VISIBLE_DEVICES={spec.gpu_device}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
        # LAN に公開する。認証は CURVE。
        "-p", f"0.0.0.0:{spec.host_port}:{CONTAINER_PORT}",
    ]

    # 知覚 API は proxy 経由に書き換えて渡す。参加者はこれらを設定しない。
    for env_var, (name, port) in PROXIES.items():
        cmd += ["-e", f"{env_var}=http://{name}:{port}"]

    for key, value in spec.secrets.items():
        cmd += ["-e", f"{key}={value}"]

    cmd += [
        spec.image,
        spec.config_path,
        "--port", str(CONTAINER_PORT),
        "--host", "0.0.0.0",
        "--session-id", spec.session_id,
    ]
    if spec.task_id:
        cmd += ["--task-id", spec.task_id]
    if spec.overrides:
        cmd += ["--override", *(f"{key}={value}" for key, value in spec.overrides)]
    return cmd


def io_connect_command(spec: RunSpec) -> list[str]:
    """起動後に繋ぐ。先に繋ぐとデフォルトルートができて外へ出られる。"""
    return ["docker", "network", "connect", IO_NETWORK, spec.container_name]


def cleanup_commands(spec: RunSpec) -> list[list[str]]:
    """後始末。どれかが失敗しても残りを続けられるよう、別々のコマンドにする。"""
    cmds = [["docker", "rm", "-f", spec.container_name]]
    for name, _port in PROXIES.values():
        cmds.append(["docker", "network", "disconnect", "-f", spec.network_name, name])
    cmds.append(["docker", "network", "rm", spec.network_name])
    return cmds
