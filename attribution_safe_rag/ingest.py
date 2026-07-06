from __future__ import annotations

import argparse
from pathlib import Path
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from config import ConfigError, load_settings
    from db import DOCUMENTS_COLLECTION, EVIDENCE_UNITS_COLLECTION, ensure_indexes, get_database
    from metadata_enrichment import enrich_metadata
    from schemas import DocumentRecord, EvidenceUnitRecord
except ImportError:  # pragma: no cover - supports package execution
    from .config import ConfigError, load_settings
    from .db import DOCUMENTS_COLLECTION, EVIDENCE_UNITS_COLLECTION, ensure_indexes, get_database
    from .metadata_enrichment import enrich_metadata
    from .schemas import DocumentRecord, EvidenceUnitRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest .txt and .pdf documents into MongoDB.")
    parser.add_argument(
        "--docs-dir",
        default="sample_docs",
        help="Directory containing .txt and .pdf documents to ingest.",
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Workspace ID to associate with the ingested documents.",
    )
    return parser.parse_args()


def infer_document_type(filename: str, raw_text: str) -> str:
    name = filename.lower()
    text = raw_text.lower()
    if "resume" in name or "candidate_name:" in text:
        return "resume"
    if "support" in name or "troubleshooting" in text or "escalation" in text:
        return "support_doc"
    return "enterprise_doc"


def extract_header_value(raw_text: str, key: str, default: str) -> str:
    prefix = f"{key.strip().lower()}:"
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            value = stripped.split(":", 1)[1].strip()
            return value or default
    return default


RESUME_SECTION_HEADERS = (
    "PROFESSIONAL SUMMARY",
    "SUMMARY",
    "EDUCATION",
    "EXPERIENCE",
    "PROJECTS",
    "SKILLS",
)


def is_resume_section_header(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not stripped:
        return False
    upper = stripped.upper()
    return upper in RESUME_SECTION_HEADERS


def split_resume_sections(raw_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if is_resume_section_header(stripped):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped.rstrip(":").title()
            current_lines = []
            continue

        if not stripped:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        current_lines.append(stripped)

    if current_lines:
        sections.append((current_title, current_lines))

    return [
        (title, "\n".join(lines).strip())
        for title, lines in sections
        if "\n".join(lines).strip()
    ]


def extract_resume_section_text(raw_text: str, section_name: str) -> str:
    for title, text in split_resume_sections(raw_text):
        if title.lower() == section_name.lower():
            return text
    return ""


def line_looks_like_tech_stack(line: str) -> bool:
    stripped = line.strip(" -*•\t")
    if not stripped:
        return False
    if stripped.startswith("•"):
        return False
    if len(stripped) > 120:
        return False
    comma_count = stripped.count(",")
    tech_tokens = sum(1 for token in stripped.split(",") if token.strip())
    return comma_count >= 1 and tech_tokens >= 2


def is_date_only_line(line: str) -> bool:
    stripped = line.strip(" -*•\t")
    return bool(
        re.fullmatch(
            r"(?:[A-Za-z]{3,9}\s+)?20\d{2}\s*[–-]\s*(?:(?:[A-Za-z]{3,9}\s+)?20\d{2}|Present)",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def split_line_on_project_headers(line: str) -> list[str]:
    pattern = re.compile(
        r"(?=(?:[A-Z][A-Za-z0-9/&+().']{1,30}(?:\s+[A-Z][A-Za-z0-9/&+().']{1,30})*\s+[–-]\s+[A-Z]))"
    )
    segments: list[str] = []
    starts = [match.start() for match in pattern.finditer(line)]
    if not starts:
        return [line.strip()] if line.strip() else []

    if starts[0] != 0:
        starts = [0] + starts

    starts = sorted(set(starts))
    starts.append(len(line))
    for start, end in zip(starts, starts[1:]):
        segment = line[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def looks_like_dangling_project_prefix(line: str) -> bool:
    stripped = line.strip(" -*•\t")
    if not stripped or " " in stripped or "–" in stripped or "-" in stripped:
        return False
    return bool(re.fullmatch(r"[A-Z][a-zA-Z]{2,20}", stripped))


def extract_leading_project_header_token(line: str) -> str | None:
    stripped = line.strip(" -*•\t")
    match = re.match(r"^([A-Z][a-zA-Z]{1,20})(\s+[–-]\s+.+)$", stripped)
    if not match:
        return None
    return match.group(1)


def merge_broken_project_header_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []

    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else None

        if (
            next_line
            and looks_like_dangling_project_prefix(current)
            and extract_leading_project_header_token(next_line)
        ):
            next_match = re.match(r"^([A-Z][a-zA-Z]{1,20})(\s+[–-]\s+.+)$", next_line)
            if next_match:
                repaired = f"{current}{next_match.group(1)}{next_match.group(2)}"
                merged.append(repaired)
                index += 2
                continue

        merged.append(current)
        index += 1

    return merged


def normalize_project_section_lines(project_section_text: str) -> list[str]:
    normalized_text = re.sub(r"\s*([•])\s*", r"\n\1 ", project_section_text)
    raw_lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    normalized_lines: list[str] = []

    for raw_line in raw_lines:
        normalized_lines.extend(split_line_on_project_headers(raw_line))

    return merge_broken_project_header_lines(normalized_lines)


def extract_project_header_parts(line: str) -> tuple[str, str | None]:
    stripped = line.strip(" -*•\t")
    date_match = re.search(r"\b(20\d{2}\s*[–-]\s*(?:20\d{2}|Present))\b$", stripped)
    date_range = date_match.group(1) if date_match else None
    header = stripped[: date_match.start()].strip(" -–\t") if date_match else stripped
    return header, date_range


def is_project_header_line(line: str, next_line: str | None = None) -> bool:
    stripped = line.strip(" -*•\t")
    if not stripped or stripped.startswith("•"):
        return False
    if is_date_only_line(stripped) or line_looks_like_tech_stack(stripped):
        return False
    if len(stripped) > 140:
        return False
    has_project_delimiter = " – " in stripped or " - " in stripped
    has_date_range = bool(re.search(r"\b20\d{2}\s*[–-]\s*(?:20\d{2}|Present)\b", stripped))
    next_is_date = bool(next_line and is_date_only_line(next_line))
    next_is_tech_stack = bool(next_line and line_looks_like_tech_stack(next_line))
    next_is_bullet = bool(next_line and next_line.strip().startswith("•"))
    return has_project_delimiter and (has_date_range or next_is_date or next_is_tech_stack or next_is_bullet)


def parse_resume_projects(projects_text: str) -> list[dict[str, object]]:
    lines = normalize_project_section_lines(projects_text)
    if not lines:
        return []

    projects: list[dict[str, object]] = []
    current_name = ""
    current_dates: str | None = None
    current_stack: str | None = None
    current_bullets: list[str] = []
    current_body_lines: list[str] = []

    def flush_project() -> None:
        nonlocal current_name, current_dates, current_stack, current_bullets, current_body_lines
        if not current_name:
            current_dates = None
            current_stack = None
            current_bullets = []
            current_body_lines = []
            return

        text_parts = [current_name]
        if current_dates:
            text_parts.append(current_dates)
        if current_stack:
            text_parts.append(current_stack)
        text_parts.extend(current_bullets or current_body_lines)
        projects.append(
            {
                "project_name": current_name,
                "date_range": current_dates,
                "tech_stack": current_stack,
                "bullets": list(current_bullets),
                "text": "\n".join(part for part in text_parts if part).strip(),
            }
        )
        current_name = ""
        current_dates = None
        current_stack = None
        current_bullets = []
        current_body_lines = []

    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        if is_project_header_line(line, next_line):
            flush_project()
            current_name, inline_dates = extract_project_header_parts(line)
            current_dates = inline_dates
            current_stack = None
            current_bullets = []
            current_body_lines = []
            continue

        if not current_name:
            continue

        stripped = line.strip()
        if not current_dates and is_date_only_line(stripped):
            current_dates = stripped
            continue
        if not current_stack and line_looks_like_tech_stack(stripped):
            current_stack = stripped.strip(" -*•\t")
            continue
        if stripped.startswith("•"):
            current_bullets.append(stripped)
            continue
        current_body_lines.append(stripped)

    flush_project()
    return projects


def build_resume_project_units(projects_text: str) -> list[dict[str, str | int | None]]:
    parsed_projects = parse_resume_projects(projects_text)
    project_names = [str(project["project_name"]) for project in parsed_projects if project.get("project_name")]
    return [
        {
            "section_name": "Projects",
            "text": str(project["text"]),
            "unit_type": "project",
            "page": None,
            "metadata": {
                "project_name": str(project["project_name"]),
                "project_names": project_names,
                "project_date_range": project.get("date_range"),
                "project_tech_stack": project.get("tech_stack"),
            },
        }
        for project in parsed_projects
        if project.get("project_name") and project.get("text")
    ]


def split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if stripped.endswith(":") and len(stripped) < 80:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped[:-1]
            current_lines = []
            continue

        current_lines.append(stripped)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def split_into_paragraph_chunks(
    raw_text: str,
    *,
    target_chars: int = 450,
    max_chars: int = 700,
) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in raw_text.replace("\r", "\n").split("\n\n")
        if paragraph.strip()
    ]
    if len(paragraphs) <= 1:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) > 1:
            paragraphs = lines
    if not paragraphs:
        paragraphs = [raw_text.strip()] if raw_text.strip() else []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        projected_length = current_length + paragraph_length + (2 if current_parts else 0)

        if current_parts and projected_length > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_length = paragraph_length
            continue

        current_parts.append(paragraph)
        current_length = projected_length

        if current_length >= target_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    if len(chunks) == 1 and len(paragraphs) > 1:
        grouped_chunks: list[str] = []
        for index in range(0, len(paragraphs), 2):
            grouped_chunks.append("\n\n".join(paragraphs[index : index + 2]))
        return grouped_chunks

    return chunks


def extract_text_from_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    page_texts: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append((page_number, text))

    return page_texts


def read_document_content(file_path: Path) -> tuple[str, list[tuple[int, str]] | None]:
    if file_path.suffix.lower() == ".pdf":
        page_texts = extract_text_from_pdf(file_path)
        if not page_texts:
            print(f"Warning: {file_path.name}: No extractable text found. OCR is not supported yet.")
            return "", []
        raw_text = "\n\n".join(text for _, text in page_texts)
        return raw_text, page_texts

    raw_text = file_path.read_text(encoding="utf-8")
    return raw_text, None


def build_text_units(
    raw_text: str,
    page_texts: list[tuple[int, str]] | None,
    document_type: str,
) -> list[dict[str, str | int | None]]:
    if document_type == "resume":
        resume_sections = split_resume_sections(raw_text)
        if resume_sections:
            resume_units: list[dict[str, str | int | None]] = []
            projects_text = extract_resume_section_text(raw_text, "Projects")
            project_units = build_resume_project_units(projects_text) if projects_text else []
            project_section_added = False
            for section_name, section_text in resume_sections:
                if section_name.lower() == "projects":
                    if project_units:
                        resume_units.extend(project_units)
                        project_section_added = True
                        continue
                resume_units.append(
                    {
                        "section_name": section_name,
                        "text": section_text,
                        "unit_type": "section",
                        "page": None,
                        "metadata": {},
                    }
                )
            if project_units and not project_section_added:
                resume_units.extend(project_units)
            if resume_units:
                return resume_units

    section_texts = split_into_sections(raw_text)
    if section_texts and len(section_texts) > 1:
        return [
            {
                "section_name": section_name,
                "text": section_text,
                "unit_type": "section",
                "page": None,
                "metadata": {},
            }
            for section_name, section_text in section_texts
        ]

    if page_texts is not None:
        page_units: list[dict[str, str | int | None]] = []

        for page_number, page_text in page_texts:
            page_sections = split_into_sections(page_text)
            if len(page_sections) > 1:
                for section_name, section_text in page_sections:
                    page_units.append(
                        {
                            "section_name": f"Page {page_number} - {section_name}",
                            "text": section_text,
                            "unit_type": "section",
                            "page": page_number,
                            "metadata": {},
                        }
                    )
                continue

            paragraph_chunks = split_into_paragraph_chunks(page_text)
            if len(paragraph_chunks) > 1:
                for chunk_index, chunk_text in enumerate(paragraph_chunks, start=1):
                    page_units.append(
                        {
                            "section_name": f"Page {page_number} - Chunk {chunk_index}",
                            "text": chunk_text,
                            "unit_type": "page_chunk",
                            "page": page_number,
                            "metadata": {},
                        }
                    )
                continue

            if page_text.strip():
                page_units.append(
                    {
                        "section_name": f"Page {page_number}",
                        "text": page_text,
                        "unit_type": "page",
                        "page": page_number,
                        "metadata": {},
                    }
                )

        if page_units:
            return page_units

    if section_texts:
        return [
            {
                "section_name": section_texts[0][0],
                "text": section_texts[0][1],
                "unit_type": "section",
                "page": None,
                "metadata": {},
            }
        ]

    return [
        {
            "section_name": "Overview",
            "text": raw_text.strip(),
            "unit_type": "section",
            "page": None,
            "metadata": {},
        }
    ]


def build_evidence_units(
    doc_id: str,
    workspace_id: str,
    filename: str,
    source_type: str,
    document_type: str,
    owner: str,
    title: str,
    raw_text: str,
    enriched_metadata: dict[str, object],
    text_units: list[dict[str, str | int | None]],
    embeddings: list[list[float]],
) -> list[EvidenceUnitRecord]:
    evidence_units: list[EvidenceUnitRecord] = []

    for text_unit, embedding in zip(text_units, embeddings):
        section_name = str(text_unit["section_name"])
        section_text = str(text_unit["text"])
        unit_specific_metadata = dict(text_unit.get("metadata", {}))
        metadata = {
            "owner": owner,
            "candidate_name": owner if document_type == "resume" else None,
            "department": extract_header_value(raw_text, "Department", "Unknown")
            if document_type != "resume"
            else None,
            "product": extract_header_value(raw_text, "Product", "Unknown")
            if document_type == "support_doc"
            else None,
            "source_filename": filename,
            "source_type": source_type,
            "document_type": document_type,
            "section_name": section_name,
            "title": title,
            "unit_type": str(text_unit["unit_type"]),
        }
        for key, value in enriched_metadata.items():
            if key in {"owner", "candidate_name", "department", "product", "source_filename", "source_type", "document_type", "section_name", "title"}:
                continue
            if value not in (None, "", [], {}):
                metadata[key] = value
        for key, value in unit_specific_metadata.items():
            if value not in (None, "", [], {}):
                metadata[key] = value
        if unit_specific_metadata.get("project_name"):
            parent_id = f"{doc_id}_{str(unit_specific_metadata['project_name']).replace(' ', '_')}"
        else:
            parent_id = f"{doc_id}_{section_name.replace(' ', '_')}"

        evidence_units.append(
            EvidenceUnitRecord(
                doc_id=doc_id,
                workspace_id=workspace_id,
                text=section_text,
                section=section_name,
                parent_id=parent_id,
                parent_text=section_text,
                parent_section_name=section_name,
                unit_type=str(text_unit["unit_type"]),
                page=text_unit["page"],
                metadata=metadata,
                embedding=embedding,
    )
)

    return evidence_units


def get_cached_document_metadata(
    database,
    workspace_id: str,
    filename: str,
) -> dict[str, object] | None:
    existing = database[DOCUMENTS_COLLECTION].find_one(
        {"workspace_id": workspace_id, "filename": filename},
        {"_id": 0, "metadata": 1},
    )
    metadata = (existing or {}).get("metadata")
    if isinstance(metadata, dict) and metadata.get("enrichment_source"):
        return metadata
    return None


def ingest_directory(docs_dir: Path, workspace_id: str) -> None:
    settings = load_settings()
    database = get_database(settings)
    ensure_indexes(database)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    supported_files = sorted(
        [path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}]
    )
    if not supported_files:
        raise FileNotFoundError(f"No .txt or .pdf files found in {docs_dir}")

    model = SentenceTransformer(settings.embedding_model_name)
    ingested_count = 0

    for file_path in supported_files:
        raw_text, page_texts = read_document_content(file_path)
        if not raw_text.strip():
            continue

        document_type = infer_document_type(file_path.name, raw_text)
        enriched_metadata = get_cached_document_metadata(database, workspace_id, file_path.name) or enrich_metadata(
            raw_text,
            file_path.name,
            document_type,
            use_llm=settings.enable_llm_metadata_enrichment,
            model_name=settings.llm_metadata_model,
        )
        text_units = build_text_units(raw_text, page_texts, document_type)
        project_names = [
            str(unit.get("metadata", {}).get("project_name"))
            for unit in text_units
            if str(unit.get("unit_type")) == "project" and unit.get("metadata", {}).get("project_name")
        ]
        if project_names:
            enriched_metadata["projects"] = project_names
        title = str(enriched_metadata.get("title") or extract_header_value(raw_text, "Title", file_path.stem.replace("_", " ").title()))
        owner = str(enriched_metadata.get("candidate_name") or enriched_metadata.get("owner") or "Unknown")
        if document_type != "resume":
            owner = str(enriched_metadata.get("owner") or extract_header_value(raw_text, "Owner", owner))
        document_type = str(enriched_metadata.get("document_type") or document_type)

        doc_record = DocumentRecord(
            filename=file_path.name,
            source_type=settings.default_source_type,
            workspace_id=workspace_id,
            document_type=document_type,
            title=title,
            owner=owner,
            raw_text=raw_text,
            metadata=enriched_metadata,
        )

        section_embeddings = model.encode(
            [str(text_unit["text"]) for text_unit in text_units],
            normalize_embeddings=True,
        ).tolist()

        evidence_units = build_evidence_units(
            doc_id=doc_record.doc_id,
            workspace_id=workspace_id,
            filename=file_path.name,
            source_type=settings.default_source_type,
            document_type=document_type,
            owner=owner,
            title=title,
            raw_text=raw_text,
            enriched_metadata=enriched_metadata,
            text_units=text_units,
            embeddings=section_embeddings,
        )

        database[DOCUMENTS_COLLECTION].delete_many(
            {"workspace_id": workspace_id, "filename": file_path.name}
        )
        database[EVIDENCE_UNITS_COLLECTION].delete_many(
            {"workspace_id": workspace_id, "metadata.source_filename": file_path.name}
        )

        database[DOCUMENTS_COLLECTION].insert_one(doc_record.to_mongo())
        if evidence_units:
            database[EVIDENCE_UNITS_COLLECTION].insert_many(
                [unit.to_mongo() for unit in evidence_units]
            )

        ingested_count += 1
        print(
            f"Ingested {file_path.name} as {document_type} with {len(evidence_units)} evidence units "
            f"(doc_id={doc_record.doc_id})."
        )

    print(f"Finished ingestion for {ingested_count} documents into workspace '{workspace_id}'.")


if __name__ == "__main__":
    args = parse_args()
    try:
        ingest_directory(Path(args.docs_dir), args.workspace_id or load_settings().default_workspace_id)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


def ingest_document(file_path: Path, workspace_id: str) -> dict[str, str | int]:
    """
    Ingest a single document file into MongoDB.
    
    Args:
        file_path: Path to the document file (.txt or .pdf)
        workspace_id: Workspace ID to associate with the document
        
    Returns:
        Dictionary with keys:
        - doc_id: Generated document ID
        - filename: Original filename
        - evidence_count: Number of evidence units created
        - status: 'success' or 'error'
        - message: Status message
    """
    try:
        settings = load_settings()
        database = get_database(settings)
        ensure_indexes(database)
        
        if not file_path.exists():
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "doc_id": "",
                "filename": "",
                "evidence_count": 0,
            }
        
        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            return {
                "status": "error",
                "message": f"Unsupported file type: {file_path.suffix}",
                "doc_id": "",
                "filename": "",
                "evidence_count": 0,
            }
        
        raw_text, page_texts = read_document_content(file_path)
        if not raw_text.strip():
            return {
                "status": "error",
                "message": "No extractable text found in document",
                "doc_id": "",
                "filename": "",
                "evidence_count": 0,
            }
        
        document_type = infer_document_type(file_path.name, raw_text)
        enriched_metadata = get_cached_document_metadata(database, workspace_id, file_path.name) or enrich_metadata(
            raw_text,
            file_path.name,
            document_type,
            use_llm=settings.enable_llm_metadata_enrichment,
            model_name=settings.llm_metadata_model,
        )
        text_units = build_text_units(raw_text, page_texts, document_type)
        project_names = [
            str(unit.get("metadata", {}).get("project_name"))
            for unit in text_units
            if str(unit.get("unit_type")) == "project" and unit.get("metadata", {}).get("project_name")
        ]
        if project_names:
            enriched_metadata["projects"] = project_names
        title = str(enriched_metadata.get("title") or extract_header_value(raw_text, "Title", file_path.stem.replace("_", " ").title()))
        owner = str(enriched_metadata.get("candidate_name") or enriched_metadata.get("owner") or "Unknown")
        if document_type != "resume":
            owner = str(enriched_metadata.get("owner") or extract_header_value(raw_text, "Owner", owner))
        document_type = str(enriched_metadata.get("document_type") or document_type)

        source_type = "uploaded_file"

        doc_record = DocumentRecord(
            filename=file_path.name,
            source_type=source_type,
            workspace_id=workspace_id,
            document_type=document_type,
            title=title,
            owner=owner,
            raw_text=raw_text,
            metadata=enriched_metadata,
        )
        
        model = SentenceTransformer(settings.embedding_model_name)
        section_embeddings = model.encode(
            [str(text_unit["text"]) for text_unit in text_units],
            normalize_embeddings=True,
        ).tolist()
        
        evidence_units = build_evidence_units(
            doc_id=doc_record.doc_id,
            workspace_id=workspace_id,
            filename=file_path.name,
            source_type=source_type,
            document_type=document_type,
            owner=owner,
            title=title,
            raw_text=raw_text,
            enriched_metadata=enriched_metadata,
            text_units=text_units,
            embeddings=section_embeddings,
        )
        
        database[DOCUMENTS_COLLECTION].delete_many(
            {"workspace_id": workspace_id, "filename": file_path.name}
        )
        database[EVIDENCE_UNITS_COLLECTION].delete_many(
            {"workspace_id": workspace_id, "metadata.source_filename": file_path.name}
        )
        
        database[DOCUMENTS_COLLECTION].insert_one(doc_record.to_mongo())
        if evidence_units:
            database[EVIDENCE_UNITS_COLLECTION].insert_many(
                [unit.to_mongo() for unit in evidence_units]
            )
        
        return {
            "status": "success",
            "doc_id": doc_record.doc_id,
            "filename": file_path.name,
            "evidence_count": len(evidence_units),
            "message": f"Ingested {file_path.name} with {len(evidence_units)} evidence units",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ingestion failed: {str(e)}",
            "doc_id": "",
            "filename": "",
            "evidence_count": 0,
        }
