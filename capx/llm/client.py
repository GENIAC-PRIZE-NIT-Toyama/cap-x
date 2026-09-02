"""LLM client utilities for querying language models.

All supported connections are OpenAI-API-compatible (OpenAI itself, vLLM's
OpenAI-compatible server, or any other OpenAI-compatible proxy). Requests are
issued via the ``openai`` SDK. A connection is described by three things,
each settable via CLI arg or environment variable:

- ``wire``: which OpenAI-compatible surface to call, ``"chat"`` (Chat
  Completions) or ``"responses"`` (Responses API). Env: ``CAPX_LLM_WIRE``.
- ``server_url``: the API base URL (no ``/chat/completions`` suffix — the
  SDK appends the right path for the chosen wire). Env: ``OPENAI_BASE_URL``.
- ``api_key``: bearer credential. Env: ``OPENAI_API_KEY``.

No model name is ever inspected to decide request shape; the caller is
responsible for choosing a wire/model combination the target server supports.
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import openai

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Connection defaults / env vars
# ---------------------------------------------------------------------------
DEFAULT_WIRE = "chat"
DEFAULT_BASE_URL = "http://127.0.0.1:8110"
_VALID_WIRES = ("chat", "responses")

_RETRYABLE_STATUS_CODES = {404, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str | None = None
    api_key: str | None = None
    wire: str | None = None
    temperature: float | None = None
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------


def _resolve_connection(args: "LaunchArgs | ModelQueryArgs") -> tuple[str, str | None, str]:
    """Resolve (base_url, api_key, wire) from args, falling back to env vars.

    Precedence for each field: explicit non-None value on ``args`` > env var
    > hardcoded default.
    """
    base_url = getattr(args, "server_url", None) or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    api_key = getattr(args, "api_key", None) or os.getenv("OPENAI_API_KEY")
    wire = getattr(args, "wire", None) or os.getenv("CAPX_LLM_WIRE") or DEFAULT_WIRE
    if wire not in _VALID_WIRES:
        raise ValueError(f"Invalid wire {wire!r}; expected one of {_VALID_WIRES}")
    return base_url, api_key, wire


def _get_client(base_url: str, api_key: str | None) -> openai.OpenAI:
    # The SDK refuses to construct a client with api_key=None (it requires a
    # non-empty string), but many local OpenAI-compatible servers (vLLM,
    # local proxies) don't require auth at all. Fall back to a placeholder.
    # max_retries=0: retry policy is handled explicitly by the callers below,
    # which retry indefinitely on transient server errors.
    return openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=0)


def _build_chat_kwargs(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": prompt,
        "max_tokens": args.max_tokens,
    }
    if getattr(args, "temperature", None) is not None:
        kwargs["temperature"] = args.temperature
    if getattr(args, "reasoning_effort", None) is not None:
        kwargs["reasoning_effort"] = args.reasoning_effort
    return kwargs


def _build_responses_kwargs(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "input": _completions_to_responses_convert_prompt(prompt),
        "max_output_tokens": args.max_tokens,
    }
    if getattr(args, "temperature", None) is not None:
        kwargs["temperature"] = args.temperature
    if getattr(args, "reasoning_effort", None) is not None:
        kwargs["reasoning"] = {"effort": args.reasoning_effort}
    return kwargs


def _extract_message_reasoning(message: Any) -> str | None:
    reasoning = getattr(message, "reasoning", None)
    if reasoning is not None:
        return reasoning
    extra = getattr(message, "model_extra", None) or {}
    return extra.get("reasoning")


def _extract_responses_content(response: Any) -> str:
    message_item = next(item for item in response.output if getattr(item, "type", None) == "message")
    return "".join(c.text for c in message_item.content if getattr(c, "type", None) == "output_text")


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> dict:
    """Query an OpenAI-API-compatible server for code generation.

    Args:
        args: Configuration with connection (server_url/api_key/wire) and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Dict with "content" and "reasoning" keys.
    """
    base_url, api_key, wire = _resolve_connection(args)
    client = _get_client(base_url, api_key)

    start_time = time.time()
    retry = 1
    while True:
        try:
            if wire == "chat":
                response = client.chat.completions.create(**_build_chat_kwargs(args, prompt))
            else:
                response = client.responses.create(**_build_responses_kwargs(args, prompt))
            break
        except openai.APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            sleep_time = 240 + random.uniform(-90, 90)
            print(
                f"Retry {retry}. Model query failed with status code {exc.status_code}. "
                f"Error: {exc.message}. Retrying in {sleep_time} seconds..."
            )
            time.sleep(sleep_time)
            retry += 1

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")

    if getattr(args, "debug", False):
        print(response.model_dump_json(indent=2))

    out: dict[str, Any] = {}
    try:
        if wire == "chat":
            message = response.choices[0].message
            out["content"] = message.content
            out["reasoning"] = _extract_message_reasoning(message)
        else:
            out["content"] = _extract_responses_content(response)
            out["reasoning"] = None
    except (AttributeError, IndexError, StopIteration) as exc:
        raise RuntimeError(f"Unexpected response format: {response!r}") from exc
    return out


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with connection (server_url/api_key/wire) and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    base_url, api_key, wire = _resolve_connection(args)
    client = _get_client(base_url, api_key)

    full_content = ""
    full_reasoning = ""

    start_time = time.time()

    if wire == "chat":
        stream = client.chat.completions.create(**_build_chat_kwargs(args, prompt), stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content_delta = delta.content or ""
            if content_delta:
                full_content += content_delta
                yield {"type": "content_delta", "content": content_delta}

            reasoning_delta = ((delta.model_extra or {}).get("reasoning") or "") if delta else ""
            if reasoning_delta:
                full_reasoning += reasoning_delta
                yield {"type": "reasoning_delta", "content": reasoning_delta}
    else:
        stream = client.responses.create(**_build_responses_kwargs(args, prompt), stream=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                content_delta = event.delta or ""
                if content_delta:
                    full_content += content_delta
                    yield {"type": "content_delta", "content": content_delta}
            elif event.type == "response.reasoning_summary_text.delta":
                reasoning_delta = event.delta or ""
                if reasoning_delta:
                    full_reasoning += reasoning_delta
                    yield {"type": "reasoning_delta", "content": reasoning_delta}

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str | None = None,
    is_multiturn=False,
) -> dict[str, Any]:
    """Query the configured model at several temperatures and synthesize final output."""

    temperatures = [0.1, 0.5, 0.9]

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=args.model,
            server_url=getattr(args, "server_url", None),
            api_key=getattr(args, "api_key", None),
            wire=getattr(args, "wire", None),
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": args.model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {args.model} temp={temp} FAILED: {error_msg}")
            return {"model": args.model, "temp": temp, "content": error_msg, "ok": False}

    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(temperatures)) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp["ok"]:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model or args.model,
        server_url=getattr(args, "server_url", None),
        api_key=getattr(args, "api_key", None),
        wire=getattr(args, "wire", None),
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synth_args.model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn=False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with connection (server_url/api_key/wire) and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=getattr(args, "server_url", None),
            api_key=getattr(args, "api_key", None),
            wire=getattr(args, "wire", None),
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp["ok"]:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:  # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=getattr(args, "server_url", None),
        api_key=getattr(args, "api_key", None),
        wire=getattr(args, "wire", None),
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
