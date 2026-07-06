from __future__ import annotations

import re
from typing import Any


ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "built",
    "document",
    "doc",
    "explained",
    "for",
    "has",
    "have",
    "how",
    "in",
    "is",
    "of",
    "project",
    "resume",
    "section",
    "setup",
    "the",
    "to",
    "what",
    "where",
    "which",
    "who",
    "why",
    "when",
}


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def tokenize_text(value: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(value))


def check_metadata_consistency(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    doc_ids = {item.get("doc_id") for item in evidence_items if item.get("doc_id")}
    candidate_names = {
        item.get("metadata", {}).get("candidate_name")
        for item in evidence_items
        if item.get("metadata", {}).get("candidate_name")
    }
    source_filenames = {
        item.get("metadata", {}).get("source_filename")
        for item in evidence_items
        if item.get("metadata", {}).get("source_filename")
    }
    return {
        "is_consistent": len(doc_ids) <= 1 and len(candidate_names) <= 1 and len(source_filenames) <= 1,
        "doc_ids": sorted(doc_ids),
        "candidate_names": sorted(candidate_names),
        "source_filenames": sorted(source_filenames),
    }


def extract_candidate_entities(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entity_map: dict[str, dict[str, Any]] = {}

    for item in evidence_items:
        metadata = item.get("metadata", {})
        candidates = [
            item.get("section"),
            metadata.get("section_name"),
            metadata.get("title"),
            metadata.get("product"),
        ]

        text = item.get("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            candidates.append(lines[0])

        acronym_matches = re.findall(r"\b[A-Z]{2,}\b", text)
        candidates.extend(acronym_matches)

        for candidate in candidates:
            normalized = normalize_text(candidate)
            if not normalized:
                continue

            tokens = tokenize_text(candidate)
            if len(tokens) < 2 and not re.fullmatch(r"[A-Z]{2,}", str(candidate).strip()):
                continue
            if tokens and all(token in ENTITY_STOPWORDS for token in tokens):
                continue

            entity = entity_map.setdefault(
                normalized,
                {
                    "entity_name": str(candidate).strip(),
                    "supporting_evidence_ids": [],
                    "sections": set(),
                    "document_types": set(),
                },
            )
            entity["supporting_evidence_ids"].append(item.get("evidence_id"))
            if item.get("section"):
                entity["sections"].add(item.get("section"))
            if metadata.get("document_type"):
                entity["document_types"].add(metadata.get("document_type"))

    entities = []
    for entity in entity_map.values():
        entities.append(
            {
                "entity_name": entity["entity_name"],
                "supporting_evidence_ids": entity["supporting_evidence_ids"],
                "sections": sorted(entity["sections"]),
                "document_types": sorted(entity["document_types"]),
            }
        )

    entities.sort(key=lambda item: len(item["supporting_evidence_ids"]), reverse=True)
    return entities


def detect_query_entity(query: str, evidence_items: list[dict[str, Any]]) -> str | None:
    normalized_query = normalize_text(query)
    query_tokens = set(tokenize_text(query))
    best_match: tuple[float, str] | None = None

    for entity in extract_candidate_entities(evidence_items):
        entity_name = entity["entity_name"]
        normalized_entity = normalize_text(entity_name)
        entity_tokens = set(tokenize_text(entity_name))
        if not normalized_entity:
            continue

        score = 0.0
        if normalized_entity in normalized_query:
            score += 3.0
        if entity_tokens:
            score += len(query_tokens & entity_tokens) / max(len(entity_tokens), 1)
        if "Projects" in entity.get("sections", []):
            score += 0.25

        if best_match is None or score > best_match[0]:
            best_match = (score, entity_name)

    if best_match and best_match[0] >= 0.5:
        return best_match[1]
    return None


def verify_entity_ownership(
    evidence_items: list[dict[str, Any]],
    entity_name: str,
) -> dict[str, Any]:
    matches = []
    normalized_entity = normalize_text(entity_name)

    for item in evidence_items:
        metadata = item.get("metadata", {})
        haystacks = [
            item.get("text", ""),
            item.get("section", ""),
            metadata.get("section_name", ""),
            metadata.get("title", ""),
            metadata.get("product", ""),
        ]
        if any(normalized_entity in normalize_text(value) for value in haystacks):
            matches.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "doc_id": item.get("doc_id"),
                    "candidate_name": metadata.get("candidate_name"),
                    "owner": metadata.get("owner"),
                    "section": item.get("section"),
                    "source_filename": metadata.get("source_filename"),
                    "document_type": metadata.get("document_type"),
                }
            )

    candidate_names = {
        match["candidate_name"]
        for match in matches
        if match.get("candidate_name")
    }
    owners = {
        match["owner"]
        for match in matches
        if match.get("owner")
    }
    resolved_names = sorted(candidate_names or owners)

    return {
        "entity_name": entity_name,
        "match_count": len(matches),
        "candidate_names": resolved_names,
        "matches": matches,
        "ownership_is_unambiguous": len(resolved_names) == 1 and len(matches) > 0,
    }


def verify_project_ownership(
    evidence_items: list[dict[str, Any]],
    project_name: str,
) -> dict[str, Any]:
    result = verify_entity_ownership(evidence_items, project_name)
    result["project_name"] = project_name
    return result
