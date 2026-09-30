"""backend を起動する。

    uv run --no-sync --active -m capx.remote_env.server

既定では認証なし。すでに LAN に出ている知覚 API のサーバ（SAM3 など）も認証なしで、
それと同じ扱いにしている。ネットワークの外に出す・不特定多数が繋がる場所で使う
ときだけ、環境変数 CAPX_ENV_SERVER_TOKENS（`名前:トークン` をカンマ区切り）を
設定すると認証が有効になる。

ZMQ の宛先は、クライアントが HTTP で繋いだ宛先（`Host` ヘッダ）から自動で決まる。
マシンに複数の IP があって決め打ちしたいときだけ `--public-host` で指定する。
"""

from __future__ import annotations

import logging
import os

import tyro

from capx.remote_env.server.app import create_app, parse_tokens
from capx.remote_env.server.probe import make_prober
from capx.remote_env.server.sessions import Config, SessionManager


def main(
    *,
    public_host: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8200,
    port_start: int = 19500,
    port_end: int = 19600,
    max_sessions: int = 20,
    idle_ttl_min: float = 20.0,
    max_lifetime_min: float = 120.0,
    gpu_uuids: str = os.environ.get("CAPX_GPU_UUIDS", ""),
) -> None:
    """
    Args:
        public_host: ZMQ の宛先を固定する。省略すると、クライアントが繋いだ宛先を使う。
        max_sessions: 同時セッション数。× 4GB がこのマシンの RAM に収まること。
        gpu_uuids: `nvidia-smi -L` の UUID をカンマ区切りで。セッションに順に割り当てる。
            省略すると環境変数 CAPX_GPU_UUIDS、それも無ければ全 GPU が見える。
            番号ではなく UUID を使うのは、`nvidia-smi` と CUDA で番号の付け方が
            ずれるマシンがあり、番号だと別のカードを掴むため。
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    spec = os.environ.get("CAPX_ENV_SERVER_TOKENS", "")
    tokens = parse_tokens(spec) if spec else None
    if tokens is None:
        logging.warning(
            "認証なしで起動（CAPX_ENV_SERVER_TOKENS 未設定）。繋がれる人は誰でも"
            "セッションを作れる。上限 %d と自動回収で資源を守る",
            max_sessions,
        )

    config = Config(
        public_host=public_host,
        port_start=port_start,
        port_end=port_end,
        max_sessions=max_sessions,
        idle_ttl_s=idle_ttl_min * 60,
        max_lifetime_s=max_lifetime_min * 60,
        gpu_devices=[g.strip() for g in gpu_uuids.split(",") if g.strip()] or ["all"],
    )

    # プローブの鍵は manager が持つ。prober は manager ができてから繋ぐ。
    manager = SessionManager(config)
    manager._probe = make_prober(manager.probe_keypair)

    uvicorn.run(create_app(manager, tokens), host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
