"""実行前の点検。`trial.py` を回す前に、動かない原因を 1 回で洗い出す。

1 項目ずつ独立に調べ、失敗しても次へ進む。参加者が「何が動いていないか」を
1 画面で分かるようにするのが目的で、直し方まで書く。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _get(url: str, headers: dict | None = None, timeout: float = 5.0):
    import requests

    return requests.get(url, headers=headers or {}, timeout=timeout)


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 10)
    return Check("Python", ok, f"{v.major}.{v.minor}.{v.micro}", "" if ok else "Python 3.10 以上が要る")


def check_dependencies() -> Check:
    missing = []
    for module in ("capx.bench", "zmq", "msgpack", "openai", "requests"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        return Check("依存パッケージ", False, f"読み込めない: {', '.join(missing)}", "remote/ で `uv sync` を実行する")
    return Check("依存パッケージ", True, "そろっている")


def check_server(server_url: str | None) -> list[Check]:
    if not server_url:
        return [
            Check(
                "GPU マシン",
                False,
                "CAPX_ENV_SERVER_URL が未設定",
                ".env に CAPX_ENV_SERVER_URL を書き、`uv run --env-file .env doctor.py` で実行する",
            )
        ]
    base = server_url.rstrip("/")
    try:
        health = _get(f"{base}/health")
        health.raise_for_status()
    except Exception as exc:
        return [
            Check(
                "GPU マシン",
                False,
                f"{base} に届かない（{type(exc).__name__}）",
                "URL とポートを管理者に確認する。VPN が要る場合は繋ぐ",
            )
        ]
    checks = [Check("GPU マシン", True, f"{base} に届いた")]

    token = os.environ.get("CAPX_ENV_SERVER_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        tasks = _get(f"{base}/tasks", headers)
        if tasks.status_code == 401:
            checks.append(
                Check("認証", False, "トークンが違うか無い", "CAPX_ENV_SERVER_TOKEN を管理者に確認する")
            )
            return checks
        tasks.raise_for_status()
        names = tasks.json().get("tasks", [])
        checks.append(Check("認証", True, "通った"))
        from capx.remote_env.server.tasks import summarize

        checks.append(Check("選べるタスク", True, summarize(names)))
    except Exception as exc:
        checks.append(Check("認証", False, f"確認できない（{exc}）"))
        return checks

    try:
        occupancy = _get(f"{base}/sessions", headers).json()
        active, capacity = occupancy["active"], occupancy["capacity"]
        free = capacity - active
        checks.append(
            Check(
                "セッションの空き",
                free > 0,
                f"{active}/{capacity} 使用中",
                "" if free > 0 else "満員。少し待ってから実行する",
            )
        )
    except Exception as exc:
        checks.append(Check("セッションの空き", False, f"確認できない（{exc}）"))
    return checks


def check_llm() -> list[Check]:
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_BASE_MODEL", "")
    if not base:
        return [Check("LLM", False, "OPENAI_BASE_URL が未設定", ".env に OPENAI_BASE_URL を書く")]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    try:
        response = _get(f"{base.rstrip('/')}/models", headers, timeout=10.0)
        if response.status_code in (401, 403):
            return [Check("LLM", False, "認証に失敗", ".env の OPENAI_API_KEY を確認する")]
        response.raise_for_status()
        ids = [m.get("id") for m in response.json().get("data", [])]
    except Exception as exc:
        return [
            Check(
                "LLM",
                False,
                f"{base} に届かない（{type(exc).__name__}）",
                ".env の OPENAI_BASE_URL と VPN を確認する",
            )
        ]
    checks = [Check("LLM", True, f"{base} に届いた")]
    if not model:
        checks.append(Check("LLM のモデル", False, "OPENAI_BASE_MODEL が未設定", f"選べるモデル: {', '.join(map(str, ids))}"))
    elif model not in ids:
        checks.append(
            Check("LLM のモデル", False, f"{model} が無い", f"選べるモデル: {', '.join(map(str, ids))}")
        )
    else:
        checks.append(Check("LLM のモデル", True, model))
    return checks


def check_agent(path: str | None) -> Check:
    if not path:
        return Check("Agent", True, "指定なし（確認を飛ばした）")
    from capx.bench import load_agent
    from capx.bench.errors import explain_load

    if not Path(path).exists():
        return Check("Agent", False, f"ファイルが無い: {Path(path).resolve()}")
    try:
        load_agent(path)
    except Exception as exc:
        return Check("Agent", False, explain_load(exc, path).message)
    return Check("Agent", True, f"{path} を読み込めた")


def run_checks(server_url: str | None, agent: str | None) -> list[Check]:
    steps: list[Callable[[], Check | list[Check]]] = [
        check_python,
        check_dependencies,
        lambda: check_server(server_url),
        check_llm,
        lambda: check_agent(agent),
    ]
    results: list[Check] = []
    for step in steps:
        out = step()
        results.extend(out if isinstance(out, list) else [out])
    return results


def render(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{'OK ' if c.ok else 'NG '} {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"      → {c.fix}")
    failed = sum(not c.ok for c in checks)
    lines.append("")
    lines.append("すべて OK。`trial.py` を実行できる" if not failed else f"{failed} 件、直す必要がある")
    return "\n".join(lines)
