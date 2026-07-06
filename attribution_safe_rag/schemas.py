from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DocumentRecord:
    filename: str
    source_type: str
    workspace_id: str
    document_type: str
    title: str
    owner: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "processed"
    doc_id: str = field(default_factory=lambda: f"doc_{uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=utc_now)

    def to_mongo(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceUnitRecord:
    doc_id: str
    workspace_id: str
    text: str
    section: str
    parent_id: str
    parent_text: str
    parent_section_name: str
    unit_type: str
    page: int | None
    metadata: dict[str, Any]
    embedding: list[float]
    evidence_id: str = field(default_factory=lambda: f"evi_{uuid4().hex[:12]}")

    def to_mongo(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryRecord:
    query_text: str
    workspace_id: str
    top_k: int
    query_id: str = field(default_factory=lambda: f"qry_{uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=utc_now)

    def to_mongo(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerRecord:
    query_id: str
    workspace_id: str
    prompt: str
    evidence_ids: list[str]
    answer_text: str | None = None
    answer_id: str = field(default_factory=lambda: f"ans_{uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=utc_now)

    def to_mongo(self) -> dict[str, Any]:
        return asdict(self)
