"""失敗を、参加者が次に何をすればよいかが分かる言葉にする。

traceback だけを見せると、「自分の Agent のバグ」「LLM に繋がらない」「GPU マシン
側の問題」の区別がつかない。種類ごとに、原因と次の一手を 1 行ずつ出す。
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass

from capx.agent_api import (
    BudgetExceeded,
    CapacityFull,
    EnvStartFailed,
    EnvUnavailable,
)


@dataclass(frozen=True)
class Explained:
    kind: str
    """`agent_import` / `llm_auth` / `llm_timeout` / `llm_unreachable` /
    `capacity_full` / `env_startup` / `budget` / `infrastructure` / `agent_error`"""
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


def explain(exc: BaseException) -> Explained:
    """例外を種類に分ける。順序が大事: `CapacityFull` は `EnvUnavailable` でもある。"""
    if isinstance(exc, BudgetExceeded):
        return Explained("budget", f"予算を使い切った（{exc.reason}）。その時点の状態で採点する")
    if isinstance(exc, CapacityFull):
        return Explained(
            "capacity_full",
            f"GPU マシンの同時セッションが上限。少し待ってからもう一度実行する。{exc}",
        )
    if isinstance(exc, EnvStartFailed):
        return Explained("env_startup", f"環境を起動できなかった。{exc}")
    if isinstance(exc, EnvUnavailable):
        return Explained(
            "infrastructure",
            f"GPU マシンとの接続が切れた（あなたの Agent のせいではない）。もう一度実行する。{exc}",
        )

    llm = _explain_llm(exc)
    if llm is not None:
        return llm

    return Explained("agent_error", f"Agent でエラー: {_where(exc)}: {exc!r}")


def explain_load(exc: BaseException, path: str) -> Explained:
    """Agent ファイルを読み込めなかった。"""
    return Explained("agent_import", f"Agent を読み込めない（{path}）: {_where(exc)}: {exc!r}")


def _explain_llm(exc: BaseException) -> Explained | None:
    try:
        import openai
    except ImportError:
        return None

    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return Explained(
            "llm_auth",
            "LLM の認証に失敗した。.env の OPENAI_API_KEY と OPENAI_BASE_URL を確認する",
        )
    if isinstance(exc, openai.APITimeoutError):
        return Explained("llm_timeout", "LLM の応答が時間内に来なかった。もう一度実行する")
    if isinstance(exc, openai.APIConnectionError):
        return Explained(
            "llm_unreachable",
            "LLM のサーバに繋がらない。.env の OPENAI_BASE_URL と、ネットワーク（VPN）を確認する",
        )
    if isinstance(exc, openai.NotFoundError):
        return Explained("llm_auth", "LLM のモデル名が見つからない。OPENAI_BASE_MODEL を確認する")
    return None


def _where(exc: BaseException) -> str:
    """例外が起きた、参加者のコードの場所。ライブラリの中は飛ばす。"""
    frames = traceback.extract_tb(exc.__traceback__)
    for frame in reversed(frames):
        name = frame.filename.replace("\\", "/")
        if "site-packages" in name or "/capx/" in name:
            continue
        return f"{frame.filename}:{frame.lineno}"
    return "(場所は不明)"


_LINE = re.compile(r'File "<string>", line (\d+)')


def highlight_generated_error(code: str, stderr: str) -> str | None:
    """生成コードの stderr から、失敗した行を取り出す。

    `exec` したコードの traceback は `File "<string>", line N` と出る。
    最後の N が、例外を起こした行。
    """
    matches = _LINE.findall(stderr or "")
    if not matches:
        return None
    number = int(matches[-1])
    lines = code.splitlines()
    if not 1 <= number <= len(lines):
        return None
    last = [ln for ln in stderr.strip().splitlines() if ln.strip()][-1]
    return f"生成コードの {number} 行目:\n    {lines[number - 1].strip()}\n  {last}"
