"""LLM 呼び出しの記録。Bench が、同梱クライアント経由の入出力とトークンを集める。

Agent が `capx.llm.client.query_model` を使えば、Bench は自動で全部の呼び出しを
控える。Agent が自前で（OpenAI SDK などで）呼ぶ場合は見えないので、その分は
`AgentResult` の自己申告になる（参考値）。

`contextvars` で持つので、Agent の中のどこで呼ばれても、`recording()` の内側なら
同じ記録に入る。
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

_current: contextvars.ContextVar[LlmRecorder | None] = contextvars.ContextVar(
    "capx_llm_recorder", default=None
)


@dataclass
class LlmRecorder:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self, prompt: list[dict], output: dict[str, Any], usage: dict[str, int] | None, seconds: float
    ) -> None:
        self.entries.append(
            {
                "index": len(self.entries),
                "prompt": _strip_images(prompt),
                "content": output.get("content"),
                "reasoning": output.get("reasoning"),
                "usage": usage,
                "seconds": round(seconds, 3),
            }
        )

    def totals(self) -> dict[str, int]:
        tokens_in = sum((e["usage"] or {}).get("prompt_tokens", 0) for e in self.entries)
        tokens_out = sum((e["usage"] or {}).get("completion_tokens", 0) for e in self.entries)
        return {"calls": len(self.entries), "tokens_in": tokens_in, "tokens_out": tokens_out}


def current() -> LlmRecorder | None:
    return _current.get()


@contextlib.contextmanager
def recording() -> Iterator[LlmRecorder]:
    recorder = LlmRecorder()
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def _strip_images(prompt: Any) -> Any:
    """画像の base64 は記録に入れない（巨大で、JSON を読めなくする）。"""
    prompt = copy.deepcopy(prompt)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            url = node.get("image_url")
            if isinstance(url, dict) and str(url.get("url", "")).startswith("data:"):
                return {**node, "image_url": {"url": f"<image {len(url['url'])} chars>"}}
            if isinstance(url, str) and url.startswith("data:"):
                return {**node, "image_url": f"<image {len(url)} chars>"}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(prompt)
