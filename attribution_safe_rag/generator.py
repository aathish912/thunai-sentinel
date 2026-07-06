from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from context_budget import format_contexts_for_prompt, select_contexts_for_prompt
except ImportError:
    from .context_budget import format_contexts_for_prompt, select_contexts_for_prompt


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def build_generation_prompt(
    query: str,
    verified_answer: str,
    parent_contexts: list[dict[str, Any]],
) -> str:
    compact_contexts = select_contexts_for_prompt(parent_contexts)
    evidence_text = format_contexts_for_prompt(compact_contexts)
    return f"""You are the Sentinel Response Engine for an enterprise knowledge verification platform.

System contract:
- Sentinel Discovery Engine = relevance
- Sentinel Verification Engine = truth
- LLM = presentation

System rules:
- Use verified answer.
- Use only evidence.
- Cite evidence IDs.
- Do not infer.
- If insufficient, say insufficient evidence.
- Do not add outside knowledge.
- Rewrite for professional clarity without changing meaning.

User query:
{query}

Verified answer:
{verified_answer}

Evidence:
{evidence_text}
"""


def generate_grounded_answer(
    query: str,
    verified_answer: str,
    parent_contexts: list[dict[str, Any]],
) -> str:
    prompt = build_generation_prompt(query, verified_answer, parent_contexts)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except (URLError, TimeoutError, OSError, ValueError):
        return verified_answer

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return verified_answer

    generated_answer = (parsed.get("response") or "").strip()
    return generated_answer or verified_answer
