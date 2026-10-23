"""過去の Multi-turn の trial の途中（ある Step の後）から、会話ごと再開して続ける例。

    cd remote
    uv run --env-file .env examples/resume_multiturn.py --snapshot snap_xxxxxxxx --task libero_object_3

    # 再開した直後に、人が直したコードを 1 つ流してから LLM に続きを任せる
    uv run --env-file .env examples/resume_multiturn.py --snapshot snap_xxxxxxxx --task libero_object_3 \\
        --first-code fixed_step3.py

やること:
  1. Snapshot から新しい worker を起こす（Gym と生成コードの変数・関数が戻る）
  2. その Step を記録した Langfuse の Trace から、LLM との会話（送ったメッセージと返事）を取り出す
     （Snapshot の記録に、元の Trace と env.step の id が入っている）
  3. その Step の実行結果と、戻した状態のカメラ画像を LLM に渡し、`agents/multiturn_agent.py` と
     同じやり方で続ける
  4. 新しい Trace（tags: fork）として Langfuse に記録する。元の Trace とは parent_trace_id でつながる

会話の履歴は Snapshot に入れていない（Langfuse にあるものを使う）。Agent が会話以外の状態を
持っているなら、`env.set_agent_state({...})` で Snapshot に添えておけば、再開後に
`env.resume_info["meta"]["agent_state"]` で受け取れる（docs/Snapshot.md）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

REMOTE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REMOTE))

from agents import multiturn_agent as mt  # noqa: E402
from forking import RunCode, load_conversation, open_fork, run_fork  # noqa: E402

from capx.agent_api import AgentResult  # noqa: E402


class ContinueMultiTurn(mt.Agent):
    """`multiturn_agent.Agent` を、途中の会話から続けられるようにしたもの。"""

    def __init__(self, messages: list[dict], ctx=None) -> None:
        super().__init__(ctx)
        self.messages = messages

    def run(self, env, task, budget):
        return self.continue_run(env, task, budget)

    def continue_run(self, env, task, budget) -> AgentResult:
        messages = self.messages
        # 最後に実行した Step の結果と、今（戻した状態）の画像を LLM に渡してから続ける
        last = (env.recorded_steps[-1] if env.recorded_steps else None) or _as_step(env.history[-1])
        remaining = budget.max_steps
        messages.append(mt.user_message(env, mt.feedback(last, env.agent_step, remaining), messages))

        for used in range(1, remaining + 1):
            reply = mt.query_model(self.llm, messages)["content"]
            messages.append({"role": "assistant", "content": reply})
            code = mt.extract_code(reply)
            if code is None:
                if mt.DONE_WORD in reply:
                    print(f"[Step {env.agent_step}] LLM が完了と判断した")
                    return AgentResult(notes={"finished": True})
                messages.append({"role": "user", "content": "コードブロックが見つかりません。```python で書いてください。"})
                continue
            result = env.step(code)
            print(f"[Step {env.agent_step}] {'成功' if result.ok else 'エラー'}  snapshot={result.snapshot_id}")
            if used == remaining:
                break
            messages.append(mt.user_message(env, mt.feedback(result, env.agent_step, remaining - used), messages))
        return AgentResult(notes={"finished": False})


def _as_step(history: dict):
    """Snapshot の記録（dict）を、`feedback()` が読める形にする。"""
    from types import SimpleNamespace

    return SimpleNamespace(ok=history["ok"], stdout=history["stdout"], stderr=history["stderr"])


def main(snapshot: str, task: str, first_code: str | None = None, save: bool = True) -> None:
    """
    Args:
        first_code: 再開した直後に実行する、人が直したコード（ファイル）。省略すると LLM がすぐ続ける。
    """
    env = open_fork(snapshot, task)
    try:
        meta = (env.resume_info or {}).get("meta") or {}
        if not meta.get("trace_id") or not meta.get("observation_id"):
            sys.exit("この Snapshot には Langfuse の Trace が記録されていない（トレースなしで実行した trial）")
        messages = load_conversation(meta["trace_id"], meta["observation_id"])
        print(f"会話を取り出した: {len(messages)} 件（Trace {meta['trace_id']}）")

        agent = ContinueMultiTurn(messages)
        if first_code:
            agent_or_code = RunCode([Path(first_code).read_text()], then=agent)
            summary = run_fork(env, agent_or_code, fork_type="human_edit", agent_path=first_code, save=save)
        else:
            summary = run_fork(env, agent, fork_type="resume", agent_path=mt.__file__, save=save)
        print(f"\ntask_completed: {summary.task_completed}  reward: {summary.reward:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
