"""動かないときの点検。

    cd remote
    uv run --env-file .env doctor.py --agent agents/my_agent.py

GPU マシン・認証・空き・LLM・Agent ファイルを順に調べて、直し方まで出す。
"""

from __future__ import annotations

import os
import sys

import tyro

from capx.bench.doctor import render, run_checks


def main(agent: str | None = "agents/example_agent.py") -> None:
    """
    Args:
        agent: 読み込めるか確認する Agent ファイル。
    """
    checks = run_checks(os.environ.get("CAPX_ENV_SERVER_URL"), agent)
    print(render(checks))
    sys.exit(0 if all(c.ok for c in checks) else 1)


if __name__ == "__main__":
    tyro.cli(main)
