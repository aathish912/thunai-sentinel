from __future__ import annotations

import re
from typing import Any

try:
    from retrieve import expand_to_parent_context
except ImportError:
    from .retrieve import expand_to_parent_context


def _short_text(text: str, max_chars: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    cutoff = normalized.find(". ", 0, max_chars)
    if cutoff != -1:
        return normalized[: cutoff + 1]
    return normalized[:max_chars].rstrip() + "..."


def _normalize_target(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" ?.:")).strip()


def _detect_comparison_targets(query: str) -> list[str]:
    normalized = query.strip()
    patterns = (
        r"compare\s+(.+?)\s+and\s+(.+)$",
        r"difference between\s+(.+?)\s+and\s+(.+)$",
        r"(.+?)\s+vs\.?\s+(.+)$",
        r"(.+?)\s+versus\s+(.+)$",
    )
    lowered = normalized.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        start_a, end_a = match.span(1)
        start_b, end_b = match.span(2)
        target_a = _normalize_target(normalized[start_a:end_a])
        target_b = _normalize_target(normalized[start_b:end_b])
        if target_a and target_b:
            return [target_a, target_b]
    return []


def _build_comparison_contexts(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_contexts = expand_to_parent_context(evidence_items)
    by_parent_id: dict[str, dict[str, Any]] = {}
    metadata_by_parent: dict[str, dict[str, Any]] = {}

    for item in evidence_items:
        parent_id = item.get("parent_id") or item.get("evidence_id")
        if parent_id not in metadata_by_parent:
            metadata_by_parent[parent_id] = item.get("metadata", {})

    for context in parent_contexts:
        parent_id = context.get("parent_id") or context.get("source_evidence_id")
        enriched = dict(context)
        enriched["metadata"] = metadata_by_parent.get(parent_id, {})
        by_parent_id[parent_id] = enriched

    contexts = list(by_parent_id.values())
    contexts.sort(key=lambda ctx: float(ctx.get("final_score", 0.0)), reverse=True)
    return contexts


def _score_context_for_target(target: str, context: dict[str, Any]) -> tuple[int, float]:
    metadata = context.get("metadata", {})
    target_lower = target.lower()
    target_tokens = set(re.findall(r"[a-z0-9]+", target_lower))

    candidate_name = str(metadata.get("candidate_name", "")).lower()
    title = str(context.get("title", "")).lower()
    section = str(context.get("section", "")).lower()
    source_filename = str(context.get("source_filename", "")).lower()
    text = str(context.get("text", "")).lower()

    phrase_bonus = 0
    for field in (candidate_name, title, section, source_filename, text):
        if target_lower and target_lower in field:
            phrase_bonus += 4

    token_overlap = 0
    for field in (candidate_name, title, section, source_filename, text):
        field_tokens = set(re.findall(r"[a-z0-9]+", field))
        token_overlap = max(token_overlap, len(target_tokens & field_tokens))

    semantic_bonus = 0
    if "candidate" in target_lower and candidate_name == target_lower:
        semantic_bonus += 8
    if target_lower in title or target_lower in section:
        semantic_bonus += 5

    return phrase_bonus + semantic_bonus + token_overlap, float(context.get("final_score", 0.0))


def _target_requires_separate_source(targets: list[str]) -> bool:
    combined = " ".join(target.lower() for target in targets)
    return any(marker in combined for marker in ("candidate", "resume", ".txt", ".pdf"))


def _select_comparison_pair(query: str, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts = _build_comparison_contexts(evidence_items)
    if len(contexts) < 2:
        return contexts

    targets = _detect_comparison_targets(query)
    if len(targets) < 2:
        return contexts[:2]

    require_separate_source = _target_requires_separate_source(targets)
    ranked_per_target = [
        sorted(contexts, key=lambda ctx: _score_context_for_target(target, ctx), reverse=True)
        for target in targets
    ]

    first = ranked_per_target[0][0]
    first_doc = first.get("doc_id")
    first_source = first.get("source_filename")

    second = None
    for candidate in ranked_per_target[1]:
        same_doc = candidate.get("doc_id") == first_doc
        same_source = candidate.get("source_filename") == first_source
        same_section = candidate.get("section") == first.get("section")

        if require_separate_source:
            if same_doc or same_source:
                continue
        elif same_doc and same_source and same_section:
            continue

        second = candidate
        break

    if second is None:
        for candidate in ranked_per_target[1]:
            if candidate.get("source_evidence_id") != first.get("source_evidence_id"):
                second = candidate
                break

    if second is None:
        return contexts[:2]

    return [first, second]


def build_comparison_answer(query: str, evidence_items: list[dict[str, Any]]) -> str:
    selected_contexts = _select_comparison_pair(query, evidence_items)
    if len(selected_contexts) < 2:
        return "insufficient evidence"

    first, second = selected_contexts[:2]
    first_source = first.get("source_filename") or first.get("doc_id")
    second_source = second.get("source_filename") or second.get("doc_id")
    first_section = first.get("section") or "Unknown Section"
    second_section = second.get("section") or "Unknown Section"
    first_evidence = first.get("source_evidence_id")
    second_evidence = second.get("source_evidence_id")

    differences = [
        f"- Source A references {first_source} / {first_section} [{first_evidence}].",
        f"- Source B references {second_source} / {second_section} [{second_evidence}].",
        f"- Source A says: {_short_text(str(first.get('text', '')))}",
        f"- Source B says: {_short_text(str(second.get('text', '')))}",
    ]

    return (
        f"Source A:\n{first_source} / {first_section} [{first_evidence}]\n"
        f"{_short_text(str(first.get('text', '')))}\n\n"
        f"Source B:\n{second_source} / {second_section} [{second_evidence}]\n"
        f"{_short_text(str(second.get('text', '')))}\n\n"
        f"Differences:\n" + "\n".join(differences)
    )
