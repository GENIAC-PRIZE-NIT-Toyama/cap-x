"""1 trial の結果。Agent が違っても同じ形で並べられるように固定する。

`result.json` として試行フォルダに書く。項目を足すときは `SCHEMA_VERSION` を上げる。

指標の呼び方:
- ``task_completed`` … 主指標。タスクを達成したか（worker が採点）。
- ``exec_ok`` … 最後のステップのコードがエラーなく走ったか。以前の "success"
  （`sandbox_rc == 0`）。タスクの成否ではない。
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

#: 成功後にも実行を続けると出る stderr。エピソードが終わっているだけで、コードの
#: 誤りではないので、`exec_ok` の判定では失敗にしない（現行の挙動を引き継ぐ）。
TERMINATED_EPISODE_MESSAGE = "executing action in terminated episode"


@dataclass
class StepRecord:
    index: int
    ok: bool
    execution_time_s: float


@dataclass
class LlmUsage:
    calls: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    #: ``bundled`` は同梱クライアント経由で Bench が数えた値。``self_reported`` は
    #: Agent の自己申告（参考値）。``none`` は分からない。
    source: str = "none"


@dataclass
class TrialResult:
    trial: int
    seed: int | None
    task_id: str
    task_completed: bool | None
    reward: float
    terminated: bool
    truncated: bool
    exec_ok: bool
    exec_ok_note: str
    steps_used: int
    execution_time_used_s: float
    wall_clock_s: float
    failure: dict[str, str] | None
    infrastructure_retries: int
    budget: dict[str, Any]
    agent: dict[str, Any]
    config: dict[str, Any]
    git: dict[str, Any]
    llm: LlmUsage = field(default_factory=LlmUsage)
    steps: list[StepRecord] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exec_ok(sandbox_rc: int, stderr: str) -> tuple[bool, str]:
    """最後のステップが「エラーなく走った」か。理由も返す。

    エピソードが終わったあとに実行して出る "terminated episode" は失敗にしない。
    上書きしたことは `exec_ok_note` に残す。黙って上書きすると、何を成功と
    数えたかが後から分からなくなる。
    """
    if sandbox_rc == 0:
        return True, ""
    if TERMINATED_EPISODE_MESSAGE in (stderr or ""):
        return True, "エピソード終了後の実行によるエラーのため、成功として数えた"
    return False, ""


def git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}
