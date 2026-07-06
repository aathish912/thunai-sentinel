from __future__ import annotations

import argparse
import math
import re
from typing import Any

from sentence_transformers import SentenceTransformer

try:
    from config import ConfigError, load_settings
    from db import (
        DOCUMENTS_COLLECTION,
        EVIDENCE_UNITS_COLLECTION,
        QUERIES_COLLECTION,
        ensure_indexes,
        get_database,
    )
    from router import detect_query_type
    from schemas import QueryRecord
    from vector_search import atlas_vector_search
except ImportError:
    from .config import ConfigError, load_settings
    from .db import (
        DOCUMENTS_COLLECTION,
        EVIDENCE_UNITS_COLLECTION,
        QUERIES_COLLECTION,
        ensure_indexes,
        get_database,
    )
    from .router import detect_query_type
    from .schemas import QueryRecord
    from .vector_search import atlas_vector_search


METADATA_MATCH_FIELDS = (
    "title",
    "section_name",
    "candidate_name",
    "source_filename",
    "product",
    "department",
    "entities",
    "people",
    "organizations",
    "products",
    "technologies",
    "tools",
    "frameworks",
    "programming_languages",
    "ai_models",
    "projects",
    "skills",
    "topics",
    "keywords",
    "issue_type",
    "resolution_steps",
)

EXPLICIT_UPLOADED_QUERY_MARKERS = (
    "uploaded",
    "my resume",
    "my document",
    "current document",
    "current resume",
    "this resume",
    "this document",
    "uploaded resume",
    "uploaded document",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top evidence units for a query."
    )
    parser.add_argument("query", help="Question to search against the evidence units.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence units to return.",
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Workspace to search within.",
    )
    return parser.parse_args()


def build_scope_filter(
    scope: str,
    scope_filenames: list[str] | None = None,
) -> dict[str, Any]:
    filenames = [name for name in (scope_filenames or []) if name]
    if scope == "uploaded":
        return {"metadata.source_type": "uploaded_file"}
    if scope == "seeded":
        return {"metadata.source_type": {"$in": ["seeded", "local_file"]}}
    if scope == "specific" and filenames:
        return {"metadata.source_filename": filenames[0]}
    if scope == "compare" and filenames:
        return {"metadata.source_filename": {"$in": filenames}}
    return {}


def query_targets_uploaded_documents(query: str) -> bool:
    query_lower = query.lower()
    return any(marker in query_lower for marker in EXPLICIT_UPLOADED_QUERY_MARKERS)


def query_targets_projects(query: str) -> bool:
    query_lower = query.lower()
    return "project" in query_lower or "projects" in query_lower


def tokenize_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def compute_keyword_score(query: str, text: str, metadata: dict[str, Any]) -> float:
    query_lower = query.lower().strip()
    text_lower = text.lower()

    query_tokens = set(tokenize_text(query))
    text_tokens = set(tokenize_text(text))

    overlap_score = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
    phrase_score = 1.0 if query_lower and query_lower in text_lower else 0.0

    metadata_values: list[str] = []
    for field in METADATA_MATCH_FIELDS:
        value = metadata.get(field)
        if not value:
            continue
        if isinstance(value, list):
            metadata_values.extend(str(item).strip() for item in value if str(item).strip())
        else:
            metadata_values.append(str(value).strip())

    metadata_blob = " ".join(metadata_values).lower()
    metadata_tokens = set(tokenize_text(metadata_blob))

    metadata_overlap_score = len(query_tokens & metadata_tokens) / max(
        len(query_tokens),
        1,
    )

    metadata_phrase_hits = sum(
        1 for value in metadata_values if value.lower() in query_lower
    )

    metadata_phrase_score = min(
        metadata_phrase_hits / max(len(metadata_values), 1),
        1.0,
    )

    keyword_score = (
        0.4 * overlap_score
        + 0.3 * phrase_score
        + 0.2 * metadata_overlap_score
        + 0.1 * metadata_phrase_score
    )

    return min(keyword_score, 1.0)


def compute_intent_adjustment(
    query: str,
    query_type: str,
    text: str,
    metadata: dict[str, Any],
    document_info: dict[str, Any] | None = None,
) -> float:
    document_type = str(metadata.get("document_type", ""))
    section_name = str(metadata.get("section_name", "")).lower()
    title = str(metadata.get("title", "")).lower()
    text_lower = text.lower()
    query_lower = query.lower()
    adjustment = 0.0

    if query_type == "procedure":
        if document_type == "support_doc":
            adjustment += 0.12
        if any(marker in section_name for marker in ("setup", "troubleshooting", "overview", "login")):
            adjustment += 0.08
        if document_type == "resume":
            adjustment -= 0.08

    if query_type == "ownership":
        if document_type == "resume":
            adjustment += 0.12
        if document_type == "support_doc":
            adjustment -= 0.08
        if any(candidate.lower() in query_lower for candidate in (section_name, title, text_lower[:200])):
            adjustment += 0.08

    if document_info and document_info.get("filename"):
        filename_lower = str(document_info["filename"]).lower()
        if filename_lower and filename_lower in query_lower:
            adjustment += 0.12

    return adjustment


def retrieve_evidence(
    query: str,
    top_k: int,
    workspace_id: str,
    scope: str = "all",
    scope_filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    database = get_database(settings)
    ensure_indexes(database)
    query_type = detect_query_type(query)

    model = SentenceTransformer(settings.embedding_model_name)
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    scope_filter = build_scope_filter(scope, scope_filenames)
    if scope == "all" and query_targets_uploaded_documents(query):
        scope_filter = {"metadata.source_type": "uploaded_file"}
    strict_project_filter = (
        (query_targets_uploaded_documents(query) or scope in {"uploaded", "specific"})
        and query_targets_projects(query)
    )
    retrieval_filter = dict(scope_filter)
    if strict_project_filter:
        retrieval_filter["metadata.unit_type"] = "project"
    document_info_map = {
        record["doc_id"]: record
        for record in database[DOCUMENTS_COLLECTION].find(
            {"workspace_id": workspace_id},
            {"doc_id": 1, "filename": 1, "source_type": 1, "document_type": 1},
        )
    }

    records = atlas_vector_search(query_embedding, workspace_id, top_k, scope_filter=retrieval_filter)
    vector_backend = "atlas_vector_search"
    vector_candidates_only = bool(records)

    if not records:
        vector_backend = "python_cosine_fallback"
        records = list(
            database[EVIDENCE_UNITS_COLLECTION].find(
                {"workspace_id": workspace_id, **retrieval_filter}
            )
        )
    if not records and strict_project_filter:
        retrieval_filter = dict(scope_filter)
        records = atlas_vector_search(query_embedding, workspace_id, top_k, scope_filter=retrieval_filter)
        vector_backend = "atlas_vector_search_fallback_all_uploaded"
        vector_candidates_only = bool(records)
        if not records:
            vector_backend = "python_cosine_fallback_all_uploaded"
            records = list(
                database[EVIDENCE_UNITS_COLLECTION].find(
                    {"workspace_id": workspace_id, **retrieval_filter}
                )
            )

    if not records:
        raise RuntimeError(
            f"No evidence units found for workspace '{workspace_id}'. "
            "Run ingest.py first."
        )

    scored_results: list[dict[str, Any]] = []

    for record in records:
        embedding = record.get("embedding", [])
        if vector_candidates_only:
            vector_score = float(record.get("score", 0.0))
        else:
            vector_score = cosine_similarity(query_embedding, embedding)

        metadata = dict(record.get("metadata", {}))
        document_info = document_info_map.get(record.get("doc_id"), {})
        if "source_type" not in metadata and document_info.get("source_type"):
            metadata["source_type"] = document_info["source_type"]
        keyword_score = compute_keyword_score(
            query,
            record.get("text", ""),
            metadata,
        )
        final_score = 0.75 * vector_score + 0.25 * keyword_score
        adjustment = compute_intent_adjustment(
            query,
            query_type,
            record.get("text", ""),
            metadata,
            document_info,
        )
        final_score_v2 = final_score + adjustment

        scored_results.append(
            {
                "evidence_id": record["evidence_id"],
                "doc_id": record["doc_id"],

                # Parent-child retrieval fields
                "parent_id": record.get("parent_id"),
                "parent_text": record.get(
                    "parent_text",
                    record.get("text", ""),
                ),
                "parent_section_name": record.get(
                    "parent_section_name",
                    record.get("section"),
                ),

                # Scores
                "final_score": final_score,
                "adjusted_score": adjustment,
                "final_score_v2": final_score_v2,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "vector_backend": vector_backend,

                # Child evidence fields
                "text": record["text"],
                "section": record.get("section"),
                "metadata": metadata,
            }
        )

    scored_results.sort(key=lambda item: item["final_score_v2"], reverse=True)
    top_results = scored_results[:top_k]

    query_record = QueryRecord(
        query_text=query,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    database[QUERIES_COLLECTION].insert_one(query_record.to_mongo())

    return top_results


def expand_to_parent_context(
    evidence_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expanded_context = []
    seen_parent_ids = set()

    for item in evidence_items:
        parent_id = item.get("parent_id") or item.get("evidence_id")

        if parent_id in seen_parent_ids:
            continue

        seen_parent_ids.add(parent_id)

        expanded_context.append(
            {
                "parent_id": parent_id,
                "doc_id": item.get("doc_id"),
                "section": item.get("parent_section_name") or item.get("section"),
                "text": item.get("parent_text") or item.get("text", ""),
                "source_evidence_id": item.get("evidence_id"),
                "source_filename": item.get("metadata", {}).get("source_filename"),
                "title": item.get("metadata", {}).get("title"),
                "final_score": item.get("final_score_v2", item.get("final_score")),
            }
        )

    return expanded_context


def print_results(query: str, results: list[dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} evidence units:")

    for index, item in enumerate(results, start=1):
        print("-" * 80)
        print(
            f"{index}. evidence_id={item['evidence_id']} "
            f"final_score={item['final_score']:.4f} "
            f"adjusted_score={item['adjusted_score']:.4f} "
            f"final_score_v2={item['final_score_v2']:.4f} "
            f"vector_score={item['vector_score']:.4f} "
            f"keyword_score={item['keyword_score']:.4f}"
        )

        print(f"   doc_id={item['doc_id']} section={item['section']}")
        print(f"   vector_backend={item.get('vector_backend')}")
        print(f"   parent_section={item.get('parent_section_name')}")
        print(f"   parent_id={item.get('parent_id')}")
        print(f"   metadata={item['metadata']}")
        print(f"   text={item['text']}")


if __name__ == "__main__":
    args = parse_args()

    try:
        settings = load_settings()
        workspace_id = args.workspace_id or settings.default_workspace_id

        results = retrieve_evidence(
            args.query,
            args.top_k,
            workspace_id,
        )

        print_results(args.query, results)

    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
