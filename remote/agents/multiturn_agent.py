"""LLM とやり取りしながら、少しずつロボットを動かす Agent（Multi-turn）。

1 ターンの流れ:
  1. LLM にコードを書かせる
  2. `env.step(code)` で GPU マシンに送って実行する
  3. 実行結果（stdout / stderr）を LLM に返し、次のコードを書かせる

LLM が「完了」と答えるか、`budget.max_steps` 回 step したら終わる。
画像は使わない（テキストだけ）。

    uv run --env-file .env trial.py --agent agents/multiturn_agent.py

LLM を呼んで起きた例外は捕まえずにそのまま投げる（`example_agent.py` と同じ）。
"""

from __future__ import annotations

import os
import re

from capx.agent_api import AgentEnv, AgentResult, BaseAgent, Budget, StepResult, TaskSpec
from capx.llm.client import ModelQueryArgs, query_model

SYSTEM_PROMPT = """\
あなたはロボットアームを操作する Python コードを書くアシスタントです。
書いたコードはシミュレータで実行され、その結果（標準出力とエラー）があなたに返されます。
結果を見ながら、少しずつタスクを進めてください。

ルール:
- 1 回の返答には、```python で始まるコードブロックを 1 つだけ書く
- 使える関数は下の「使える関数」に書かれたものだけ。import は不要
- 状況を確かめたいときは print で出力する
- タスクが終わったと判断したら、コードを書かずに「完了」とだけ答える

使える関数:
{api_docs}
"""

FIRST_PROMPT = """\
タスク:
{instruction}

最初のコードを書いてください。
"""

FEEDBACK_PROMPT = """\
実行結果（{turn} 回目）: {status}
標準出力:
{stdout}
エラー出力:
{stderr}

残り {remaining} 回実行できます。続きのコードを書くか、終わったなら「完了」と答えてください。
"""

DONE_WORD = "完了"
#: LLM に返す出力の長さの上限（長すぎるとプロンプトがあふれる）
MAX_OUTPUT_CHARS = 2000

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class Agent(BaseAgent):
    def __init__(self, ctx=None) -> None:
        # `--model` などは ctx に入ってくる。無ければ環境変数を使う
        self.llm = ModelQueryArgs(
            model=getattr(ctx, "model", "") or os.environ.get("OPENAI_BASE_MODEL", ""),
            server_url=getattr(ctx, "server_url", None),
            api_key=getattr(ctx, "api_key", None),
            wire=getattr(ctx, "wire", None),
            temperature=getattr(ctx, "temperature", None),
            reasoning_effort=getattr(ctx, "reasoning_effort", None),
        )

    def run(self, env: AgentEnv, task: TaskSpec, budget: Budget) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(api_docs=task.api_docs)},
            {"role": "user", "content": FIRST_PROMPT.format(instruction=task.instruction)},
        ]

        for turn in range(1, budget.max_steps + 1):
            reply = query_model(self.llm, messages)["content"]
            messages.append({"role": "assistant", "content": reply})

            code = extract_code(reply)
            if code is None:
                if DONE_WORD in reply:
                    print(f"[{turn}] LLM が完了と判断した")
                    return AgentResult(notes={"turns": turn - 1, "finished": True})
                messages.append({"role": "user", "content": "コードブロックが見つかりません。```python で書いてください。"})
                continue

            result = env.step(code)
            print(f"[{turn}] 実行: {'成功' if result.ok else 'エラー'}")
            messages.append({"role": "user", "content": feedback(result, turn, budget.max_steps - turn)})

        return AgentResult(notes={"turns": budget.max_steps, "finished": False})


def extract_code(reply: str) -> str | None:
    """返答の中の最後のコードブロック。無ければ None。"""
    blocks = _CODE_BLOCK.findall(reply)
    return blocks[-1].strip() if blocks else None


def feedback(result: StepResult, turn: int, remaining: int) -> str:
    return FEEDBACK_PROMPT.format(
        turn=turn,
        status="成功" if result.ok else "エラー",
        stdout=_clip(result.stdout) or "（なし）",
        stderr=_clip(result.stderr) or "（なし）",
        remaining=remaining,
    )


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_OUTPUT_CHARS else "…（前略）\n" + text[-MAX_OUTPUT_CHARS:]
