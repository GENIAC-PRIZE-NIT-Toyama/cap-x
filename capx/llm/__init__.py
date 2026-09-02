"""LLM client module for querying language models.

All connections are OpenAI-API-compatible (OpenAI, vLLM, or any other
OpenAI-compatible proxy), reached via the ``openai`` SDK. Supports streaming
and ensemble queries, plus backward-compatible underscore-prefixed aliases.
"""

from capx.llm.client import (
    ModelQueryArgs,
    _completions_to_responses_convert_prompt,
    collapse_text_image_inputs,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
)

__all__ = [
    "ModelQueryArgs",
    "_completions_to_responses_convert_prompt",
    "collapse_text_image_inputs",
    "query_model",
    "query_model_ensemble",
    "query_model_streaming",
    "query_single_model_ensemble",
]
