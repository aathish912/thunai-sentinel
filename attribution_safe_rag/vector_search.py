from __future__ import annotations

from typing import Any

try:
    from config import load_settings
    from db import EVIDENCE_UNITS_COLLECTION, get_database
except ImportError:
    from .config import load_settings
    from .db import EVIDENCE_UNITS_COLLECTION, get_database


def atlas_vector_search(
    query_embedding: list[float],
    workspace_id: str,
    top_k: int,
    scope_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    database = get_database(settings)

    vector_filter: dict[str, Any] = {"workspace_id": workspace_id}
    if scope_filter:
        vector_filter.update(scope_filter)

    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.vector_search_index_name,
                "path": settings.vector_search_path,
                "queryVector": query_embedding,
                "numCandidates": max(top_k * 10, 50),
                "limit": max(top_k * 4, top_k),
                "filter": vector_filter,
            }
        },
        {
            "$project": {
                "_id": 1,
                "doc_id": 1,
                "evidence_id": 1,
                "workspace_id": 1,
                "text": 1,
                "section": 1,
                "parent_id": 1,
                "parent_text": 1,
                "parent_section_name": 1,
                "metadata": 1,
                "embedding": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(database[EVIDENCE_UNITS_COLLECTION].aggregate(pipeline))
    except Exception:
        return []

    return results
