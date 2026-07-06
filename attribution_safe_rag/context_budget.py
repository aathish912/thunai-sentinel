from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    cutoff = stripped.rfind(" ", 0, max_chars)
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return stripped[:cutoff].rstrip() + "..."


def select_contexts_for_prompt(
    parent_contexts: list[dict[str, Any]],
    max_chars: int = 3500,
) -> list[dict[str, Any]]:
    compact_contexts: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    used_chars = 0

    sorted_contexts = sorted(
        parent_contexts,
        key=lambda item: item.get("final_score", 0.0),
        reverse=True,
    )

    for context in sorted_contexts:
        parent_id = str(context.get("parent_id") or context.get("source_evidence_id") or "")
        if not parent_id or parent_id in seen_parent_ids:
            continue

        remaining = max_chars - used_chars
        if remaining <= 120:
            break

        text_budget = min(900, max(180, remaining - 120))
        compact_text = truncate_text(str(context.get("text", "")), text_budget)
        if not compact_text:
            continue

        compact_context = {
            "parent_id": parent_id,
            "doc_id": context.get("doc_id"),
            "section": context.get("section"),
            "source_evidence_id": context.get("source_evidence_id"),
            "source_filename": context.get("source_filename"),
            "title": context.get("title"),
            "text": compact_text,
        }
        serialized = format_single_context(compact_context)
        if used_chars + len(serialized) > max_chars and compact_contexts:
            break

        compact_contexts.append(compact_context)
        seen_parent_ids.add(parent_id)
        used_chars += len(serialized) + 2

    return compact_contexts


def format_single_context(context: dict[str, Any]) -> str:
    section = context.get("section") or "Unknown Section"
    source = context.get("source_filename") or context.get("doc_id") or "unknown-source"
    evidence_id = context.get("source_evidence_id") or "unknown-evidence"
    text = context.get("text", "")
    return f"[Evidence {evidence_id} | {section} | {source}]\n{text}"


def format_contexts_for_prompt(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(format_single_context(context) for context in contexts)
