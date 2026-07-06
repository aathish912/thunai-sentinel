from __future__ import annotations

import argparse
from typing import Any
import re

try:
    from comparison import build_comparison_answer
    from config import ConfigError, load_settings
    from context_budget import format_contexts_for_prompt, select_contexts_for_prompt
    from db import ANSWERS_COLLECTION, ensure_indexes, get_database
    from generator import generate_grounded_answer
    from router import detect_query_type
    from retrieve import expand_to_parent_context, retrieve_evidence
    from schemas import AnswerRecord
    from verifier import (
        check_metadata_consistency,
        detect_query_entity,
        verify_entity_ownership,
    )
except ImportError:
    from .comparison import build_comparison_answer
    from .config import ConfigError, load_settings
    from .context_budget import format_contexts_for_prompt, select_contexts_for_prompt
    from .db import ANSWERS_COLLECTION, ensure_indexes, get_database
    from .generator import generate_grounded_answer
    from .router import detect_query_type
    from .retrieve import expand_to_parent_context, retrieve_evidence
    from .schemas import AnswerRecord
    from .verifier import (
        check_metadata_consistency,
        detect_query_entity,
        verify_entity_ownership,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an attribution-safe answer.")
    parser.add_argument("query", help="Question to answer with retrieved evidence.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workspace-id", default=None)
    return parser.parse_args()


def build_prompt(query: str, parent_contexts: list[dict[str, Any]]) -> str:
    compact_contexts = select_contexts_for_prompt(parent_contexts)
    evidence_block = format_contexts_for_prompt(compact_contexts)
    return f"""System rules:
- Use verified answer.
- Use only evidence.
- Cite evidence IDs.
- Do not infer.
- If insufficient, say insufficient evidence.

User query:
{query}

Evidence:
{evidence_block}
"""


def first_sentence(text: str, max_chars: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    cutoff = normalized.find(". ", 0, max_chars)
    if cutoff != -1:
        return normalized[: cutoff + 1]
    return normalized[:max_chars].rstrip() + "..."


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return " ".join(cleaned.split())


def is_useful_text(value: str) -> bool:
    cleaned = normalize_text(value)
    if not cleaned:
        return False
    if cleaned.lower() in {"unknown", "n/a", "none"}:
        return False
    return len(cleaned) >= 12


def detect_general_focus(query: str) -> str:
    query_lower = query.lower()
    if "project" in query_lower:
        return "project"
    if "technology" in query_lower or "technologies" in query_lower or "tools" in query_lower or "framework" in query_lower:
        return "technology"
    if (
        "ai model" in query_lower
        or "models are referenced" in query_lower
        or "model is referenced" in query_lower
        or "language model" in query_lower
        or "llm" in query_lower
        or ("model" in query_lower and "education" not in query_lower)
    ):
        return "ai_model"
    if any(term in query_lower for term in ("education", "degree", "university", "college", "graduation")):
        return "education"
    if any(
        term in query_lower
        for term in ("applicant", "candidate", "student name", "applicant name", "candidate name", "name of applicant", "name of candidate")
    ):
        return "person_name"
    if "skill" in query_lower or "programming language" in query_lower:
        return "skill"
    if "experience" in query_lower:
        return "experience"
    if "summarize" in query_lower or "summary" in query_lower:
        return "summary"
    return "general"


METADATA_INTENT_SPECS = [
    {
        "intent": "project",
        "heading": "Projects identified",
        "triggers": ("project", "projects", "portfolio", "built", "developed", "created", "implemented"),
        "metadata_keys": ("projects",),
    },
    {
        "intent": "skill",
        "heading": "Skills identified",
        "triggers": ("skill", "skills", "languages", "competencies", "expertise"),
        "metadata_keys": ("skills",),
    },
    {
        "intent": "technology",
        "heading": "Technologies identified",
        "triggers": ("technology", "technologies", "stack", "software", "libraries"),
        "metadata_keys": ("technologies",),
    },
    {
        "intent": "tool",
        "heading": "Tools identified",
        "triggers": ("tool", "tools"),
        "metadata_keys": ("tools",),
    },
    {
        "intent": "framework",
        "heading": "Frameworks identified",
        "triggers": ("framework", "frameworks"),
        "metadata_keys": ("frameworks",),
    },
    {
        "intent": "programming_language",
        "heading": "Programming Languages",
        "triggers": ("programming language", "programming languages", "python", "javascript", "typescript", "java", "c++", "language"),
        "metadata_keys": ("programming_languages",),
    },
    {
        "intent": "ai_model",
        "heading": "AI Models Identified",
        "triggers": ("ai model", "models", "llm", "language model", "foundation model"),
        "metadata_keys": ("ai_models",),
    },
    {
        "intent": "education",
        "heading": "Education",
        "triggers": ("education", "degree", "college", "university", "school", "coursework", "graduation"),
        "metadata_keys": ("education",),
    },
    {
        "intent": "person_name",
        "heading": "Applicant name",
        "triggers": ("name of applicant", "applicant name", "name of candidate", "candidate name", "student name", "who is the applicant", "who is the candidate"),
        "metadata_keys": ("candidate_name", "owner", "people"),
    },
    {
        "intent": "experience",
        "heading": "Experience highlights",
        "triggers": ("experience", "employment", "work history", "career", "roles", "job", "internship"),
        "metadata_keys": ("experience",),
    },
    {
        "intent": "organization",
        "heading": "Organizations mentioned",
        "triggers": ("company", "companies", "organization", "client", "customer", "vendor", "partner"),
        "metadata_keys": ("organizations", "customer_name", "account_name"),
    },
    {
        "intent": "product",
        "heading": "Products discussed",
        "triggers": ("product", "platform", "feature", "module", "service"),
        "metadata_keys": ("products", "product"),
    },
    {
        "intent": "api",
        "heading": "APIs identified",
        "triggers": ("api", "apis"),
        "metadata_keys": ("api_names",),
    },
    {
        "intent": "integration",
        "heading": "Integrations identified",
        "triggers": ("integration", "integrations"),
        "metadata_keys": ("integrations",),
    },
    {
        "intent": "database",
        "heading": "Databases identified",
        "triggers": ("database", "databases"),
        "metadata_keys": ("databases",),
    },
    {
        "intent": "cloud_service",
        "heading": "Cloud Services identified",
        "triggers": ("cloud", "cloud service", "cloud services", "aws", "azure", "gcp"),
        "metadata_keys": ("cloud_services",),
    },
    {
        "intent": "support",
        "heading": "Support details",
        "triggers": ("incident", "ticket", "issue", "problem", "bug", "resolution", "root cause", "sla", "escalation"),
        "metadata_keys": ("issue_type", "resolution_steps", "root_cause", "sla", "escalation", "ticket_id", "case_id", "incident_id", "priority", "severity", "status"),
    },
    {
        "intent": "contracts",
        "heading": "Policy and contract details",
        "triggers": ("policy", "contract", "agreement", "renewal", "compliance"),
        "metadata_keys": ("contract_terms", "compliance_terms", "renewal_date"),
    },
]

SECTION_FALLBACK_RULES: dict[str, dict[str, tuple[str, ...] | str]] = {
    "education": {
        "heading": "Education",
        "sections": ("education",),
        "forbidden": ("experience", "projects", "skills", "professional summary"),
    },
    "person_name": {
        "heading": "Applicant name",
        "sections": ("overview", "profile", "header", "contact"),
        "forbidden": (),
    },
    "experience": {
        "heading": "Experience highlights",
        "sections": ("experience", "employment", "work history"),
        "forbidden": ("projects", "skills", "education", "professional summary"),
    },
    "skill": {
        "heading": "Skills identified",
        "sections": ("skills",),
        "forbidden": ("experience", "projects", "education", "professional summary"),
    },
    "technology": {
        "heading": "Technologies identified",
        "sections": ("skills", "technology", "technologies", "tools", "frameworks", "stack"),
        "forbidden": ("experience", "projects", "education", "professional summary"),
    },
    "programming_language": {
        "heading": "Programming Languages",
        "sections": ("skills", "languages"),
        "forbidden": ("experience", "projects", "education", "professional summary"),
    },
    "ai_model": {
        "heading": "AI Models Identified",
        "sections": ("skills", "ai", "models"),
        "forbidden": ("experience", "projects", "education", "professional summary"),
    },
}


def detect_metadata_intent(query: str) -> dict[str, Any] | None:
    query_lower = query.lower()
    best_spec = None
    best_score = 0
    for spec in METADATA_INTENT_SPECS:
        score = sum(1 for trigger in spec["triggers"] if trigger in query_lower)
        if score > best_score:
            best_score = score
            best_spec = spec
    return best_spec


def is_summary_query(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in ("summarize", "summary", "overview"))


def query_targets_uploaded_resume(query: str) -> bool:
    query_lower = query.lower()
    return any(
        marker in query_lower
        for marker in (
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
    )


def select_preferred_evidence_items(query: str, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evidence_items:
        return []

    uploaded_items = [
        item
        for item in evidence_items
        if item.get("metadata", {}).get("source_type") == "uploaded_file"
    ]
    if query_targets_uploaded_resume(query) and uploaded_items:
        return uploaded_items
    return evidence_items


def collect_metadata_for_answer(query: str, evidence_items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    selected_items = select_preferred_evidence_items(query, evidence_items)
    if not selected_items:
        return {}, []

    merged: dict[str, Any] = {}
    evidence_ids: list[str] = []
    for item in selected_items:
        metadata = item.get("metadata", {})
        if item.get("evidence_id"):
            evidence_id = str(item["evidence_id"])
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        for key, value in metadata.items():
            if isinstance(value, list):
                existing = list(merged.get(key, []))
                for entry in value:
                    if entry not in existing:
                        existing.append(entry)
                merged[key] = existing
            elif value not in (None, "", "Unknown") and key not in merged:
                merged[key] = value
    return merged, evidence_ids[:3]


def format_metadata_list_answer(title: str, values: list[str], evidence_ids: list[str], limit: int = 8) -> str:
    cleaned_values = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_text(str(value)).strip(" -*•\t")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_values.append(cleaned)
    if not cleaned_values:
        return "insufficient evidence"
    items = "\n".join(f"• {value}" for value in cleaned_values[:limit])
    return f"{title}:\n{items}\n\nSupported by evidence:\n{', '.join(evidence_ids)}"


def _format_generic_item(item: Any) -> list[str]:
    if isinstance(item, str):
        cleaned = normalize_text(item).strip(" -*•\t")
        return [f"• {cleaned}"] if cleaned else []

    if not isinstance(item, dict):
        cleaned = normalize_text(str(item)).strip(" -*•\t")
        return [f"• {cleaned}"] if cleaned else []

    title_fields = ("title", "name", "institution", "issue", "product", "organization")
    title = ""
    for field in title_fields:
        value = normalize_text(str(item.get(field, "")))
        if value:
            title = value
            break
    if not title:
        for key, value in item.items():
            cleaned = normalize_text(str(value))
            if cleaned:
                title = cleaned
                break

    lines = [f"• {title}"] if title else []
    for key, value in item.items():
        if key in {"title", "name", "institution", "issue", "product", "organization"}:
            continue
        if isinstance(value, list):
            for sub_value in value:
                cleaned = normalize_text(str(sub_value))
                if cleaned:
                    lines.append(f"* {cleaned}")
        else:
            cleaned = normalize_text(str(value))
            if cleaned:
                lines.append(f"* {cleaned}")
    return lines


def format_generic_metadata_answer(heading: str, payload: Any, evidence_ids: list[str], limit: int = 8) -> str:
    lines: list[str] = []
    if isinstance(payload, str):
        cleaned = normalize_text(payload)
        if not cleaned:
            return "insufficient evidence"
        return f"{heading}:\n{cleaned}\n\nSupported by evidence:\n{', '.join(evidence_ids)}"

    if isinstance(payload, list):
        for item in payload[:limit]:
            lines.extend(_format_generic_item(item))

    if not lines:
        return "insufficient evidence"
    return f"{heading}:\n" + "\n".join(lines[: max(limit * 3, limit)]) + f"\n\nSupported by evidence:\n{', '.join(evidence_ids)}"


def extract_education_entries_from_evidence(evidence_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in evidence_items:
        metadata = item.get("metadata", {})
        section_name = str(item.get("section") or metadata.get("section_name") or "").lower()
        if "education" not in section_name:
            continue

        lines = [normalize_text(line) for line in str(item.get("text", "")).splitlines() if normalize_text(line)]
        if not lines:
            continue

        entry: dict[str, str] = {"institution": lines[0]}
        coursework_parts: list[str] = []
        for line in lines[1:]:
            lowered = line.lower()
            if any(marker in lowered for marker in ("experience", "projects", "skills", "professional summary")):
                break
            if "coursework" in lowered:
                coursework_parts.append(line)
                continue
            if re.search(r"\b(?:b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|bachelor|master)\b", line, re.IGNORECASE):
                entry["degree"] = line
                continue
            if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b", lowered) or re.fullmatch(r"20\d{2}", line):
                entry["graduation_date"] = line
                continue
            if re.search(r"\b[A-Z][a-z]+,\s*[A-Z]{2}\b", line):
                entry["location"] = line
                continue
            coursework_parts.append(line)

        if coursework_parts:
            entry["coursework"] = " ".join(coursework_parts)
        entries.append(entry)
    return entries


def extract_person_name_from_evidence(evidence_items: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add_name(value: str) -> None:
        cleaned = normalize_text(value).strip(" -*•\t|")
        if not cleaned or cleaned.lower() in {"unknown", "resume", "overview"}:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(cleaned)

    for item in evidence_items:
        metadata = item.get("metadata", {})
        for field in ("candidate_name", "owner"):
            value = normalize_text(str(metadata.get(field, "")))
            if value and value.lower() != "unknown":
                add_name(value)

    for item in evidence_items:
        metadata = item.get("metadata", {})
        section_name = str(item.get("section") or metadata.get("section_name") or "").lower()
        if "overview" not in section_name and "profile" not in section_name:
            continue
        lines = [normalize_text(line) for line in str(item.get("text", "")).splitlines() if normalize_text(line)]
        if not lines:
            continue
        first_line = lines[0]
        candidate = first_line.split("|", 1)[0] if "|" in first_line else first_line
        add_name(candidate)

    for item in evidence_items:
        title = normalize_text(str(item.get("metadata", {}).get("title", "")))
        match = re.search(r"resume\s*-\s*(.+)$", title, re.IGNORECASE)
        if match:
            add_name(match.group(1))

    return names


def extract_project_summary_from_text(text: str) -> str:
    lines = [line.strip(" -*•\t") for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    detail_lines = []
    for line in lines[1:]:
        lowered = line.lower()
        if line_looks_like_metadata_line(line):
            continue
        if lowered.startswith(("architected", "built", "developed", "designed", "implemented", "integrated", "deployed", "created")):
            detail_lines.append(line)
    if detail_lines:
        return normalize_text(first_sentence(" ".join(detail_lines), 180))
    for line in lines[1:]:
        if not line_looks_like_metadata_line(line):
            return normalize_text(first_sentence(line, 180))
    return ""


def line_looks_like_metadata_line(line: str) -> bool:
    lowered = line.lower()
    return bool(
        re.search(r"\b20\d{2}\s*[–-]\s*(?:20\d{2}|present)\b", lowered)
        or "," in line and len(line.split(",")) >= 2
    )


def build_project_units_answer(evidence_items: list[dict[str, Any]]) -> str | None:
    project_rows: list[tuple[int, str, str, str]] = []
    seen_names: set[str] = set()
    project_order: list[str] = []

    for item in evidence_items:
        for name in item.get("metadata", {}).get("project_names", []):
            normalized_name = normalize_text(str(name))
            if normalized_name and normalized_name not in project_order:
                project_order.append(normalized_name)

    for item in evidence_items:
        metadata = item.get("metadata", {})
        if metadata.get("unit_type") != "project":
            continue
        project_name = normalize_text(str(metadata.get("project_name", "")))
        if not project_name:
            continue
        key = project_name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        summary = extract_project_summary_from_text(str(item.get("text", "")))
        evidence_id = str(item.get("evidence_id", ""))
        sort_index = project_order.index(project_name) if project_name in project_order else len(project_order)
        project_rows.append((sort_index, project_name, summary, evidence_id))

    if not project_rows:
        return None

    project_rows.sort(key=lambda row: (row[0], row[1].lower()))

    lines: list[str] = ["Projects identified:"]
    evidence_ids: list[str] = []
    for _, project_name, summary, evidence_id in project_rows[:6]:
        if len(lines) > 1:
            lines.append("")
        lines.append(f"• {project_name}")
        if summary:
            lines.append(f"- {summary}")
        if evidence_id:
            evidence_ids.append(evidence_id)

    lines.append("")
    lines.append(f"Supported by evidence:\n{', '.join(evidence_ids)}")
    return "\n".join(lines)


def has_metadata_payload(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        return bool(normalize_text(payload))
    if isinstance(payload, list):
        return any(
            bool(normalize_text(str(item)))
            if not isinstance(item, dict)
            else any(bool(normalize_text(str(value))) for value in item.values())
            for item in payload
        )
    return True


def payload_contains_forbidden_terms(payload: Any, forbidden_terms: tuple[str, ...]) -> bool:
    if not forbidden_terms:
        return False
    if isinstance(payload, str):
        normalized = normalize_text(payload).lower()
        return any(term in normalized for term in forbidden_terms)
    if isinstance(payload, list):
        return any(payload_contains_forbidden_terms(item, forbidden_terms) for item in payload)
    if isinstance(payload, dict):
        return any(payload_contains_forbidden_terms(value, forbidden_terms) for value in payload.values())
    return False


def split_labeled_line_values(line: str) -> list[str]:
    text = normalize_text(line)
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return [normalize_text(part).strip(" -*•\t") for part in re.split(r",|;|\|", text) if normalize_text(part).strip(" -*•\t")]


def extract_section_bounded_items(
    evidence_items: list[dict[str, Any]],
    section_keywords: tuple[str, ...],
    forbidden_terms: tuple[str, ...],
) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for item in evidence_items:
        metadata = item.get("metadata", {})
        section_name = str(item.get("section") or metadata.get("section_name") or "").lower()
        if not any(keyword in section_name for keyword in section_keywords):
            continue
        for raw_line in str(item.get("text", "")).splitlines():
            cleaned = normalize_text(raw_line).strip(" -*•\t")
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if any(term in lowered for term in forbidden_terms):
                break
            if ":" in cleaned and any(
                label in lowered for label in ("languages:", "frameworks:", "tools:", "ai & ml:", "technologies:", "skills:")
            ):
                for value in split_labeled_line_values(cleaned):
                    key = value.lower()
                    if key not in seen:
                        seen.add(key)
                        items.append(value)
                continue
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                items.append(cleaned)
    return items


def build_section_fallback_answer(intent: str, evidence_items: list[dict[str, Any]], evidence_ids: list[str]) -> str | None:
    rule = SECTION_FALLBACK_RULES.get(intent)
    if not rule:
        return None
    heading = str(rule["heading"])
    sections = tuple(rule["sections"])
    forbidden = tuple(rule["forbidden"])

    if intent == "education":
        entries = extract_education_entries_from_evidence(evidence_items)
        if entries:
            return format_generic_metadata_answer(heading, entries, evidence_ids)
    if intent == "person_name":
        names = extract_person_name_from_evidence(evidence_items)
        if names:
            return format_metadata_list_answer(heading, names, evidence_ids, limit=3)

    items = extract_section_bounded_items(evidence_items, sections, forbidden)
    if not items:
        return None

    if intent in {"technology", "programming_language", "ai_model", "skill"}:
        filtered: list[str] = []
        for item in items:
            if len(item) > 80:
                continue
            filtered.append(item)
        items = filtered or items

    if intent == "experience":
        items = [first_sentence(item, 180) for item in items if is_useful_text(item)]

    return format_metadata_list_answer(heading, items, evidence_ids, limit=8)


def build_section_payload(intent: str, metadata: dict[str, Any]) -> Any:
    if intent == "summary":
        return normalize_text(str(metadata.get("summary", "")))
    if intent == "person_name":
        payload = []
        for key in ("candidate_name", "owner"):
            value = metadata.get(key)
            if value and str(value).strip().lower() != "unknown":
                payload.append(str(value))
        for person in metadata.get("people", []):
            payload.append(str(person))
        return payload
    if intent == "support":
        payload = []
        if metadata.get("issue_type"):
            support_entry = {"issue": metadata.get("issue_type")}
            for key in ("ticket_id", "case_id", "incident_id", "priority", "severity", "status", "escalation", "root_cause", "sla"):
                if metadata.get(key):
                    support_entry[key.replace("_", " ").title()] = metadata.get(key)
                if metadata.get("resolution_steps"):
                    support_entry["Resolution Steps"] = metadata.get("resolution_steps")
            payload.append(support_entry)
        elif metadata.get("resolution_steps"):
            payload.append({"issue": "Resolution", "Resolution Steps": metadata.get("resolution_steps")})
        return payload
    if intent == "contracts":
        payload = []
        if metadata.get("contract_terms"):
            payload.append({"title": "Contract Terms", "details": metadata.get("contract_terms")})
        if metadata.get("compliance_terms"):
            payload.append({"title": "Compliance Terms", "details": metadata.get("compliance_terms")})
        if metadata.get("renewal_date"):
            payload.append({"title": "Renewal Date", "details": [metadata.get("renewal_date")]})
        return payload
    if intent == "organization":
        payload = list(metadata.get("organizations", []))
        for key in ("customer_name", "account_name"):
            value = metadata.get(key)
            if value:
                payload.append(str(value))
        return payload
    if intent == "product":
        payload = list(metadata.get("products", []))
        if metadata.get("product"):
            payload.append(str(metadata.get("product")))
        return payload

    spec = next((spec for spec in METADATA_INTENT_SPECS if spec["intent"] == intent), None)
    if not spec:
        return None

    values: list[Any] = []
    for key in spec["metadata_keys"]:
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in (None, "", "Unknown"):
            values.append(value)
    return values


def build_section_aware_metadata_answer(query: str, evidence_items: list[dict[str, Any]]) -> str | None:
    metadata, evidence_ids = collect_metadata_for_answer(query, evidence_items)
    if not metadata or not evidence_ids:
        return None

    intent_spec = detect_metadata_intent(query)
    if intent_spec:
        intent = str(intent_spec["intent"])
        if intent_spec["intent"] == "project":
            project_units_answer = build_project_units_answer(evidence_items)
            if project_units_answer:
                return project_units_answer
        payload = build_section_payload(intent, metadata)
        fallback_rule = SECTION_FALLBACK_RULES.get(intent)
        forbidden_terms = tuple(fallback_rule["forbidden"]) if fallback_rule else ()
        if not has_metadata_payload(payload) or payload_contains_forbidden_terms(payload, forbidden_terms):
            fallback_answer = build_section_fallback_answer(intent, evidence_items, evidence_ids)
            if fallback_answer:
                return fallback_answer
        if has_metadata_payload(payload):
            return format_generic_metadata_answer(intent_spec["heading"], payload, evidence_ids)

    if is_summary_query(query):
        payload = build_section_payload("summary", metadata)
        if has_metadata_payload(payload):
            return format_generic_metadata_answer("Summary", payload, evidence_ids, limit=4)

    return None


PROJECT_POSITIVE_TERMS = (
    "project",
    "prototype",
    "system",
    "platform",
    "dashboard",
    "application",
    "tool",
    "built",
    "developed",
    "created",
    "implemented",
)

PROJECT_NEGATIVE_TERMS = (
    "@",
    "email",
    "phone",
    "contact",
    "address",
    "professional summary",
    "summary",
    "administrative",
    "front desk",
    "profile",
    "objective",
    "coordinated daily administrative tasks",
)

SKILL_NEGATIVE_TERMS = (
    "@",
    "email",
    "phone",
    "address",
    "professional summary",
    "contact",
)


def contains_indicator_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms)


def looks_like_contact_or_profile(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in PROJECT_NEGATIVE_TERMS):
        return True
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        return True
    if re.search(r"\b\S+@\S+\.\S+\b", text):
        return True
    return False


def is_likely_project_context(context: dict[str, Any]) -> bool:
    section = str(context.get("section", "")).lower()
    text = normalize_text(str(context.get("text", "")))
    lowered = text.lower()
    if not is_useful_text(text) or looks_like_contact_or_profile(text):
        return False
    if "project" in section:
        return True
    if contains_indicator_term(lowered, PROJECT_POSITIVE_TERMS) and not any(
        term in lowered for term in ("professional summary", "administrative", "contact information")
    ):
        return True
    return False


def extract_project_entries(context: dict[str, Any]) -> list[tuple[str, str]]:
    section = str(context.get("section", "")).lower()
    raw_lines = [line.strip(" -*•\t") for line in str(context.get("text", "")).splitlines() if line.strip()]
    if not raw_lines:
        return []

    found_projects_header = False
    projects_header_index = next(
        (
            index
            for index, line in enumerate(raw_lines)
            if line.strip().lower().rstrip(":") == "projects"
        ),
        None,
    )
    if projects_header_index is not None:
        found_projects_header = True
        raw_lines = raw_lines[projects_header_index + 1 :]
    if not raw_lines:
        return []

    filtered_lines = [line for line in raw_lines if not looks_like_contact_or_profile(line)]
    if not filtered_lines:
        return []

    entries: list[tuple[str, str]] = []
    current_title = ""
    current_description_parts: list[str] = []

    def flush_entry() -> None:
        nonlocal current_title, current_description_parts
        if not current_title:
            return
        if not found_projects_header and "project" not in section and not any(
            re.search(rf"\b{re.escape(term)}\b", current_title.lower())
            for term in ("project", "prototype", "system", "platform", "dashboard", "application", "tool")
        ):
            current_title = ""
            current_description_parts = []
            return
        description = first_sentence(" ".join(current_description_parts), 180) if current_description_parts else ""
        entries.append((current_title, description))
        current_title = ""
        current_description_parts = []

    for line in filtered_lines:
        lowered = line.lower()
        is_title_line = contains_indicator_term(
            lowered,
            ("project", "prototype", "system", "platform", "dashboard", "application", "tool"),
        ) or line.istitle() or (" - " in line and len(line.split()) <= 12)

        if is_title_line:
            flush_entry()
            current_title = line.rstrip(":")
            continue

        if current_title:
            current_description_parts.append(line)

    if not current_title and filtered_lines:
        first_line = filtered_lines[0]
        if "project" in section and contains_indicator_term(first_line.lower(), PROJECT_POSITIVE_TERMS):
            current_title = first_line.rstrip(":")
            current_description_parts = filtered_lines[1:]

    flush_entry()
    return entries


def score_context_for_query(query: str, context: dict[str, Any], focus: str) -> tuple[int, float]:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    section = str(context.get("section", "")).lower()
    text = str(context.get("text", ""))
    text_lower = text.lower()
    text_tokens = set(re.findall(r"[a-z0-9]+", text_lower))

    focus_bonus = 0
    if focus == "project" and "project" in section:
        focus_bonus += 6
    elif focus == "skill" and "skill" in section:
        focus_bonus += 6
    elif focus == "experience" and "experience" in section:
        focus_bonus += 6
    elif focus == "summary" and "summary" in section:
        focus_bonus += 6

    if focus != "general":
        focus_terms = {
            "project": ("project", "built", "designed", "dashboard", "system"),
            "skill": ("skill", "python", "sql", "machine", "language"),
            "experience": ("experience", "worked", "led", "built", "at "),
            "summary": ("summary", "experience", "projects", "skills"),
        }
        if any(term in text_lower for term in focus_terms.get(focus, ())):
            focus_bonus += 3

    return (focus_bonus, len(query_tokens & text_tokens) + float(context.get("final_score", 0.0)))


def select_general_contexts(
    query: str,
    parent_contexts: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    focus = detect_general_focus(query)
    candidates = [context for context in parent_contexts if is_useful_text(str(context.get("text", "")))]
    if focus == "project":
        focused_candidates = [context for context in candidates if is_likely_project_context(context)]
        if focused_candidates:
            candidates = focused_candidates
    elif focus == "summary":
        filtered_candidates = [
            context for context in candidates if not looks_like_contact_or_profile(str(context.get("text", "")))
        ]
        if filtered_candidates:
            candidates = filtered_candidates
    elif focus != "general":
        focused_candidates = [
            context
            for context in candidates
            if focus in str(context.get("section", "")).lower() or focus in str(context.get("text", "")).lower()
        ]
        if focused_candidates:
            candidates = focused_candidates
    ranked = sorted(
        candidates,
        key=lambda context: score_context_for_query(query, context, focus),
        reverse=True,
    )
    return ranked[:limit]


def build_general_response(query: str, parent_contexts: list[dict[str, Any]]) -> str:
    selected_contexts = select_general_contexts(query, parent_contexts, limit=3)
    if not selected_contexts:
        return "insufficient evidence"

    focus = detect_general_focus(query)
    evidence_ids = [
        str(context.get("source_evidence_id"))
        for context in selected_contexts
        if context.get("source_evidence_id")
    ]

    if focus == "project":
        project_items: list[str] = []
        filtered_evidence_ids: list[str] = []
        for context in selected_contexts:
            entries = extract_project_entries(context)
            if not entries:
                continue
            for title, description in entries:
                if description:
                    project_items.append(f"• {title}\n* {description}")
                else:
                    project_items.append(f"• {title}")
            if context.get("source_evidence_id") and entries:
                filtered_evidence_ids.append(str(context.get("source_evidence_id")))

        if project_items:
            return (
                "Projects identified:\n"
                + "\n".join(project_items[:3])
                + f"\n\nSupported by evidence:\n{', '.join(filtered_evidence_ids or evidence_ids)}"
            )
        return "insufficient evidence"

    if focus == "skill":
        skills: list[str] = []
        for context in selected_contexts:
            raw_text = str(context.get("text", ""))
            for item in re.split(r",|\n", raw_text):
                cleaned = item.strip(" -•\t")
                if (
                    cleaned
                    and len(cleaned) < 60
                    and not any(term in cleaned.lower() for term in SKILL_NEGATIVE_TERMS)
                    and cleaned.lower() not in {skill.lower() for skill in skills}
                ):
                    skills.append(cleaned)
        if skills:
            skill_items = "\n".join(f"• {skill}" for skill in skills[:8])
            return f"Skills identified:\n{skill_items}\n\nSupported by evidence:\n{', '.join(evidence_ids)}"
        return "insufficient evidence"

    if focus == "experience":
        experience_items: list[str] = []
        for context in selected_contexts:
            snippet = first_sentence(normalize_text(str(context.get("text", ""))), 180)
            if snippet and not looks_like_contact_or_profile(snippet):
                experience_items.append(f"• {snippet}")
        if experience_items:
            return (
                "Experience highlights:\n"
                + "\n".join(experience_items[:3])
                + f"\n\nSupported by evidence:\n{', '.join(evidence_ids)}"
            )
        return "insufficient evidence"

    if focus == "summary":
        summary_snippets = [
            first_sentence(normalize_text(str(context.get("text", ""))), 180)
            for context in selected_contexts
            if is_useful_text(str(context.get("text", ""))) and not looks_like_contact_or_profile(str(context.get("text", "")))
        ]
        if summary_snippets:
            return (
                "Resume Summary:\n"
                + " ".join(summary_snippets[:2])
                + f"\n\nSupported by evidence:\n{', '.join(evidence_ids)}"
            )
        return "insufficient evidence"

    general_items = [
        f"- {first_sentence(normalize_text(str(context.get('text', ''))), 180)}"
        for context in selected_contexts
        if is_useful_text(str(context.get("text", "")))
    ]
    if not general_items:
        return "insufficient evidence"
    return (
        "Relevant evidence:\n"
        + "\n".join(general_items[:3])
        + f"\n\nSupported by evidence: {', '.join(evidence_ids)}"
    )


def build_metadata_first_response(query: str, evidence_items: list[dict[str, Any]]) -> str | None:
    return build_section_aware_metadata_answer(query, evidence_items)


def build_grounded_answer(
    query: str,
    evidence_items: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], str | None, str]:
    query_type = detect_query_type(query)
    preferred_evidence_items = select_preferred_evidence_items(query, evidence_items)
    consistency = check_metadata_consistency(evidence_items)
    entity_name = detect_query_entity(query, evidence_items)
    parent_contexts = expand_to_parent_context(preferred_evidence_items)
    compact_contexts = select_contexts_for_prompt(parent_contexts)

    if query_type == "ownership" and entity_name:
        ownership = verify_entity_ownership(evidence_items, entity_name)
        if ownership.get("ownership_is_unambiguous"):
            candidate = ownership["candidate_names"][0]
            evidence_ids = ", ".join(match["evidence_id"] for match in ownership["matches"])
            return (
                f"The {entity_name} belongs to {candidate}. Supported by evidence {evidence_ids}.",
                ownership,
                entity_name,
                query_type,
            )
        return "insufficient evidence", ownership, entity_name, query_type

    if query_type == "procedure" and compact_contexts:
        top = compact_contexts[0]
        source = top.get("source_filename") or top.get("doc_id")
        section = top.get("section") or "Unknown Section"
        evidence_id = top.get("source_evidence_id")
        snippet = first_sentence(str(top.get("text", "")))
        if "where is" in query.lower() or "explained" in query.lower():
            answer = (
                f"The relevant guidance is in section {section} of {source}. "
                f"Supported by evidence {evidence_id}."
            )
        else:
            answer = (
                f"The relevant steps are in section {section} of {source}: {snippet} "
                f"Supported by evidence {evidence_id}."
            )
        return answer, consistency, entity_name, query_type

    if query_type == "summary" and compact_contexts:
        metadata_first_answer = build_metadata_first_response(query, preferred_evidence_items)
        if metadata_first_answer:
            return metadata_first_answer, consistency, entity_name, query_type
        summary_parts = []
        evidence_ids = []
        for context in compact_contexts[:2]:
            source = context.get("source_filename") or context.get("doc_id")
            summary_parts.append(
                f"{source}: {first_sentence(str(context.get('text', '')), 160)}"
            )
            evidence_ids.append(str(context.get("source_evidence_id")))
        answer = (
            "Overview: "
            + " ".join(summary_parts)
            + f" Supported by evidence {', '.join(evidence_ids)}."
        )
        return answer, consistency, entity_name, query_type

    if query_type == "comparison" and len(compact_contexts) >= 2:
        answer = build_comparison_answer(query, evidence_items)
        return answer, consistency, entity_name, query_type

    if compact_contexts:
        metadata_first_answer = build_metadata_first_response(query, preferred_evidence_items)
        if metadata_first_answer:
            return metadata_first_answer, consistency, None, query_type
        answer = build_general_response(query, compact_contexts)
        return answer, consistency, None, query_type

    return "insufficient evidence", consistency, entity_name, query_type

def run_query_pipeline(
    query: str,
    top_k: int,
    workspace_id: str,
    scope: str = "all",
    scope_filenames: list[str] | None = None,
) -> dict[str, Any]:
    query_type = detect_query_type(query)
    evidence_items = retrieve_evidence(
        query,
        top_k,
        workspace_id,
        scope=scope,
        scope_filenames=scope_filenames,
    )
    parent_contexts = expand_to_parent_context(evidence_items)
    prompt = build_prompt(query, parent_contexts)
    consistency = check_metadata_consistency(evidence_items)
    detected_entity = detect_query_entity(query, evidence_items)
    ownership_check = (
        verify_entity_ownership(evidence_items, detected_entity)
        if query_type == "ownership" and detected_entity
        else None
    )
    final_answer, verification_result, _, _ = build_grounded_answer(query, evidence_items)
    llm_answer = generate_grounded_answer(query, final_answer, parent_contexts)

    return {
        "query": query,
        "query_type": query_type,
        "evidence_items": evidence_items,
        "parent_contexts": parent_contexts,
        "prompt": prompt,
        "consistency": consistency,
        "detected_entity": detected_entity,
        "ownership_check": ownership_check,
        "verification_result": verification_result,
        "final_answer": final_answer,
        "llm_answer": llm_answer,
    }


if __name__ == "__main__":
    args = parse_args()

    try:
        settings = load_settings()
        workspace_id = args.workspace_id or settings.default_workspace_id
        result = run_query_pipeline(args.query, args.top_k, workspace_id)

        database = get_database(settings)
        ensure_indexes(database)

        answer_record = AnswerRecord(
            query_id="latest_cli_query",
            workspace_id=workspace_id,
            prompt=result["prompt"],
            evidence_ids=[item["evidence_id"] for item in result["evidence_items"]],
            answer_text=result["llm_answer"],
        )
        database[ANSWERS_COLLECTION].insert_one(answer_record.to_mongo())

        print("Final grounded answer:")
        print(result["final_answer"])

        print("\nQuery type:")
        print(result["query_type"])

        print("\nLLM grounded answer:")
        print(result["llm_answer"])

        print("\nVerification summary:")
        print(result["consistency"])
        print({"detected_entity": result["detected_entity"]})
        if result["ownership_check"]:
            print(result["ownership_check"])

        print("\nParent contexts used:")
        for context in result["parent_contexts"]:
            print("-" * 80)
            print(f"parent_id={context.get('parent_id')}")
            print(f"section={context.get('section')}")
            print(f"source_evidence_id={context.get('source_evidence_id')}")
            print(f"text={context.get('text')}")

        print("\nPrompt sent to answerer:")
        print(result["prompt"])

    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
