"""Snapshot から再開して分岐する（Fork）ための小さな道具。`examples/` のスクリプトが使う。

    env = open_fork("snap_...", task="libero_object_3")     # 新しい worker に Step N の状態を戻す
    env.step("print(helper())")                              # Step N までの変数・関数がそのまま使える
    summary = run_fork(env, agent, fork_type="human_edit")   # Bench と Langfuse の記録つきで回す

再開そのものは backend と worker がやる（Langfuse には頼らない）。Langfuse は、どの Step の
Snapshot から分岐したかを残すのと、会話の履歴を取り出す（`load_conversation`）のに使う。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tracing  # noqa: E402


def open_fork(
    snapshot_id: str,
    task: str,
    *,
    snapshot_enabled: bool = True,
    record_video: bool = True,
):
    """`snapshot_id` の状態を戻した、新しい worker の env。`reset_for_trial()` は要らない。

    Args:
        task: Snapshot を取ったときのタスク名（違えば backend が断る）。
        snapshot_enabled: 分岐した先でも Snapshot を取る（さらに分岐できる）。
    """
    from capx.remote_env.client import RemoteAgentEnv

    server_url = os.environ.get("CAPX_ENV_SERVER_URL")
    if not server_url:
        sys.exit("CAPX_ENV_SERVER_URL が無い（`uv run --env-file .env ...` で実行する）")
    env = RemoteAgentEnv(
        server_url=server_url,
        task_id=task,
        resume_from=snapshot_id,
        snapshot_enabled=snapshot_enabled,
        record_video=record_video,
    )
    info = env.resume_info or {}
    print(f"再開: {snapshot_id}（{info.get('agent_step')} Step 目の後）→ 新しい worker")
    print(f"  使える変数・関数: {', '.join(info.get('namespace_names', []))}")
    return env


def remaining_budget(env):
    """再開した時点の残りの予算（Step 数）。worker が数えているものと同じ。"""
    from capx.agent_api import Budget

    info = env.resume_info or {}
    return Budget(max_steps=max(0, int(info.get("max_steps", 10)) - env.agent_step))


def run_fork(
    env,
    agent,
    *,
    fork_type: str,
    model: str = os.environ.get("OPENAI_BASE_MODEL", ""),
    agent_path: str | Path = __file__,
    save: bool = True,
    output_dir: str | None = None,
    trace: bool = True,
):
    """再開した env で Agent を走らせ、結果（コード・動画・result.json）と Langfuse の Trace を残す。

    通常の trial と同じ Bench（`capx.bench.run_trial`）を、reset せずに使う。Langfuse には
    新しい Trace として送り、元の Trace とは `parent_trace_id` などでつなぐ。

    Args:
        fork_type: 分岐の種類。`human_edit` / `best_of_n` / `retry` / `resume` など。
    """
    from capx.bench import run_trial

    task = env.task
    info = env.resume_info or {}
    label = f"fork_{env.resume_from}"
    run_dir = None
    if save:
        from trial import _make_run_dir

        run_dir = _make_run_dir(output_dir, model, f"{task.task_id}__{label}")
        print(f"保存先: {run_dir}")
    if tracing.setup(trace):
        print("トレース: Langfuse に Fork として送る")
    config = {"output_dir": str(run_dir) if run_dir else None, "record_video": save}
    meta = {"agent_spec": str(agent_path), "model": model, "endpoint_task": task.task_id, "fork_of": env.resume_from}
    trial = int(info.get("trial") or 1)
    try:
        with tracing.fork_trace(
            session=tracing.session_id(run_dir, model, f"{task.task_id}__{label}"),
            task=task.task_id,
            snapshot_id=env.resume_from,
            fork_type=fork_type,
            model=model,
            agent_path=agent_path,
            trial=trial,
            seed=task.seed,
        ) as fork:
            summary = run_trial(
                fork.wrap(env), agent, trial, config, remaining_budget(env),
                seed=task.seed, meta=meta, resumed=True,
            )
            fork.finish(summary)
            if fork.url:
                print(f"Fork の Trace: {fork.url}")
    finally:
        tracing.flush()
    return summary


class RunCode:
    """決まったコードを順に実行するだけの Agent。人が直したコード（human_edit）を流すのに使う。

    `then` に別の Agent の `continue_run(env, task, budget)` を持つものを渡すと、コードを
    実行したあとで、その Agent に続きを任せる（Multi-turn の継続）。
    """

    def __init__(self, codes: list[str], then: Any = None) -> None:
        self.codes = codes
        self.then = then

    def run(self, env, task, budget):
        from capx.agent_api import AgentResult

        for code in self.codes:
            result = env.step(code)
            print(f"[Step {env.agent_step}] {'成功' if result.ok else 'エラー'}  snapshot={result.snapshot_id}")
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip())
        if self.then is not None:
            from capx.agent_api import Budget

            left = Budget(max_steps=max(0, budget.max_steps - len(self.codes)))
            return self.then.continue_run(env, task, left)
        return AgentResult()


# -- Langfuse から会話を取り出す ------------------------------------------------


def load_conversation(trace_id: str, observation_id: str) -> list[dict]:
    """元の Trace から、その env.step の直前までの LLM との会話を取り出す。

    OpenAI SDK の記録（generation）のうち、その env.step より前で最後のものの input
    （送ったメッセージ）と output（LLM の返事）をつなぐ。画像は Langfuse の参照に置き換わって
    いるので外し、文字だけを残す（Multi-turn の Agent も、新しい画像 1 枚だけを LLM に渡す）。

    Langfuse のキー（LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL）が要る。
    """
    from langfuse import get_client

    observations = sorted(_observations(get_client(), trace_id), key=lambda o: o.start_time)
    step = next((o for o in observations if o.id == observation_id), None)
    if step is None:
        raise LookupError(f"Trace {trace_id} に observation {observation_id} が無い")
    generations = [o for o in observations if o.type == "GENERATION" and o.start_time <= step.start_time]
    if not generations:
        raise LookupError(f"Trace {trace_id} の、その Step より前に LLM の呼び出しが無い")
    last = generations[-1]
    messages = [_text_only(m) for m in _messages(_json(last.input))]
    reply = _json(last.output)
    if isinstance(reply, dict):
        reply = reply.get("content") or ""
    messages.append({"role": "assistant", "content": str(reply or "")})
    return messages


def _observations(client: Any, trace_id: str) -> list[Any]:
    """Trace の observation を全部。Langfuse v4（events_only）では trace の取得 API が無いので、
    observations の API を使う（input / output は JSON の文字列で返る）。"""
    out: list[Any] = []
    cursor = None
    while True:
        page = client.api.observations.get_many(
            trace_id=trace_id, limit=100, cursor=cursor, fields="core,basic,io,metadata"
        )
        out.extend(page.data)
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return out


def _json(value: Any) -> Any:
    import json

    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _messages(value: Any) -> list[dict]:
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, list):
        raise ValueError("generation の input がメッセージの list でない")
    return [m for m in value if isinstance(m, dict) and "role" in m]


#: 画像を外したところに入れる一言。
IMAGE_DROPPED = "（このときの画像は省略）"


def _text_only(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        has_image = any(isinstance(p, dict) and p.get("type") != "text" for p in content)
        text = "\n".join(texts)
        if has_image:
            text = text.removesuffix("（今の様子をカメラで撮った画像を添えます）") + IMAGE_DROPPED
        return {"role": message["role"], "content": text}
    return {"role": message["role"], "content": content if isinstance(content, str) else str(content)}
