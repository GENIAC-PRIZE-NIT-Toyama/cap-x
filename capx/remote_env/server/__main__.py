"""backend を起動する。

    export CAPX_ENV_SERVER_TOKENS="alice:tok1,bob:tok2"
    uv run --no-sync --active -m capx.remote_env.server --public-host 192.168.0.50

`--public-host` は手元PC から届く GPU マシンの名前か IP。セッションを作った
クライアントに、ZMQ の宛先としてそのまま返す。
"""

from __future__ import annotations

import logging
import os
import sys

import tyro

from capx.remote_env.server.app import create_app, parse_tokens
from capx.remote_env.server.probe import make_prober
from capx.remote_env.server.sessions import Config, SessionManager


def main(
    *,
    public_host: str,
    host: str = "0.0.0.0",
    port: int = 8200,
    port_start: int = 19500,
    port_end: int = 19600,
    max_sessions: int = 20,
    idle_ttl_min: float = 20.0,
    max_lifetime_min: float = 120.0,
    gpu_uuids: str = "",
    no_auth: bool = False,
) -> None:
    """
    Args:
        public_host: 手元PC から届く、この GPU マシンの名前か IP。
        max_sessions: 同時セッション数。× 4GB がこのマシンの RAM に収まること。
        gpu_uuids: `nvidia-smi -L` の UUID をカンマ区切りで。セッションに順に割り当てる。
            省略すると全 GPU が見える（GPU が 1 枚のマシン向け）。
        no_auth: 認証を無効にする。開発用。ポートを開けた状態では使わない。
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if no_auth:
        tokens = None
        logging.warning("認証なし（--no-auth）。開発用")
    else:
        spec = os.environ.get("CAPX_ENV_SERVER_TOKENS", "")
        if not spec:
            sys.exit(
                "CAPX_ENV_SERVER_TOKENS が未設定。例: alice:tok1,bob:tok2\n"
                "（開発で認証を外すなら --no-auth）"
            )
        tokens = parse_tokens(spec)

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
