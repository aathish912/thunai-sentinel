from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_LLM_METADATA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
COMMON_TECHNOLOGIES = (
    "mongodb",
    "postgresql",
    "mysql",
    "redis",
    "snowflake",
    "airflow",
    "tableau",
    "power bi",
    "sso",
    "oauth",
    "saml",
    "dashboard",
    "kubernetes",
    "docker",
    "aws",
    "azure",
    "gcp",
    "terraform",
    "jira",
    "postman",
    "supabase",
    "fastapi",
    "next.js",
    "nextjs",
)
COMMON_PROGRAMMING_LANGUAGES = (
    "python",
    "sql",
    "java",
    "javascript",
    "typescript",
    "scala",
    "r",
    "c++",
    "go",
)
COMMON_FRAMEWORKS = (
    "django",
    "flask",
    "fastapi",
    "react",
    "next.js",
    "nextjs",
    "supabase",
    "spark",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
)
COMMON_AI_MODELS = (
    "gpt",
    "chatgpt",
    "nvidia nemotron",
    "nemotron",
    "llama 2",
    "qwen",
    "llama",
    "llama3",
    "llama 3",
    "mistral",
    "mixtral",
    "deepseek",
    "gpt-4",
    "gpt-4o",
    "claude",
    "gemini",
    "phi",
    "phi-3",
    "phi-4",
    "falcon",
    "command r",
    "cohere",
    "palm",
    "t5",
    "bert",
    "roberta",
)
COMMON_CLOUD_SERVICES = (
    "aws",
    "azure",
    "gcp",
    "s3",
    "ec2",
    "lambda",
    "bigquery",
    "cloud run",
)
SUPPORT_TERMS = (
    "sso",
    "sla",
    "escalation",
    "incident",
    "outage",
    "login",
    "dashboard",
    "ticket",
    "case",
    "tenant",
)
DISPLAY_TERM_MAP = {
    "python": "Python",
    "sql": "SQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "fastapi": "FastAPI",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "supabase": "Supabase",
    "postman": "Postman",
    "docker": "Docker",
    "mongodb": "MongoDB",
    "nvidia nemotron": "NVIDIA Nemotron",
    "gpt": "GPT",
    "chatgpt": "ChatGPT",
    "nemotron": "NVIDIA Nemotron",
    "qwen": "Qwen",
    "llama": "Llama",
    "llama 2": "Llama 2",
    "llama3": "Llama 3",
    "llama 3": "Llama 3",
    "mistral": "Mistral",
    "mixtral": "Mixtral",
    "deepseek": "DeepSeek",
    "claude": "Claude",
    "gemini": "Gemini",
    "phi": "Phi",
    "phi-3": "Phi-3",
    "phi-4": "Phi-4",
    "falcon": "Falcon",
    "command r": "Command R",
    "cohere": "Cohere",
    "palm": "PaLM",
    "t5": "T5",
    "bert": "BERT",
    "roberta": "RoBERTa",
}

RESUME_SECTION_HEADERS = {
    "PROFESSIONAL SUMMARY",
    "SUMMARY",
    "EDUCATION",
    "EXPERIENCE",
    "PROJECTS",
    "SKILLS",
}


def unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = " ".join(str(item).split()).strip(" -•\t")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def canonicalize_terms(items: list[str]) -> list[str]:
    return unique_preserve([DISPLAY_TERM_MAP.get(str(item).lower(), str(item)) for item in items])


def find_header_value(raw_text: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(raw_text)
    return match.group(1).strip() if match else ""


def is_project_like_line(line: str) -> bool:
    stripped = line.strip(" -*•\t")
    lowered = stripped.lower()
    if not stripped or stripped.endswith(":") or stripped.endswith("."):
        return False
    if len(stripped.split()) > 10:
        return False
    if re.match(r"^(built|developed|created|implemented|designed)\b", lowered):
        return False
    return any(term in lowered for term in ("project", "prototype", "system", "platform", "dashboard", "application", "tool"))


def extract_section_lines(raw_text: str, section_name: str) -> list[str]:
    lines = raw_text.splitlines()
    collected: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if capture and collected:
                break
            continue
        if stripped.lower().rstrip(":") == section_name.lower():
            capture = True
            continue
        if capture and (
            (stripped.endswith(":") and len(stripped) < 80)
            or stripped.rstrip(":").upper() in RESUME_SECTION_HEADERS
            or (stripped.isupper() and len(stripped) < 80 and len(stripped.split()) <= 6)
        ):
            break
        if capture:
            collected.append(stripped)
    return collected


def extract_education_entries(raw_text: str) -> list[dict[str, str]]:
    section_lines = extract_section_lines(raw_text, "Education")
    if not section_lines:
        return []

    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    institution_markers = ("university", "college", "institute", "school")
    degree_pattern = re.compile(
        r"\b(?:b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|bachelor|master)\b",
        re.IGNORECASE,
    )

    for line in section_lines:
        lowered = line.lower()
        if any(marker in lowered for marker in institution_markers):
            if current:
                entries.append(current)
            current = {"institution": line}
            continue
        if "graduation" in lowered or "expected" in lowered:
            current["graduation_date"] = line
            continue
        if degree_pattern.search(line):
            current["degree"] = line
            continue
        if "coursework" in lowered:
            current["coursework"] = line
            continue
        if "major" not in current and current and not current.get("degree"):
            current["major"] = line
        elif current:
            current["coursework"] = f"{current.get('coursework', '')} {line}".strip()

    if current:
        entries.append(current)

    return entries


def extract_rule_metadata(raw_text: str, filename: str, document_type: str) -> dict[str, Any]:
    title = find_header_value(raw_text, "Title") or filename.rsplit(".", 1)[0].replace("_", " ").title()
    owner = find_header_value(raw_text, "Owner") or find_header_value(raw_text, "Candidate_Name") or "Unknown"
    candidate_name = find_header_value(raw_text, "Candidate_Name")
    department = find_header_value(raw_text, "Department")
    product = find_header_value(raw_text, "Product")
    summary = " ".join(extract_section_lines(raw_text, "Summary")[:3]).strip()
    experience = unique_preserve(extract_section_lines(raw_text, "Experience"))
    projects = unique_preserve(extract_section_lines(raw_text, "Projects"))
    skills = unique_preserve(
        [
            item.strip()
            for line in extract_section_lines(raw_text, "Skills")
            for item in re.split(r",|;|\n", line)
        ]
    )
    technologies = canonicalize_terms(
        [item for item in skills if item.lower() in COMMON_TECHNOLOGIES or item.lower() in COMMON_FRAMEWORKS]
    )
    tools = canonicalize_terms(
        [item for item in skills if item.lower() in {"postman", "docker", "jira", "tableau", "power bi", "excel", "supabase"}]
    )
    frameworks = canonicalize_terms([item for item in skills if item.lower() in COMMON_FRAMEWORKS])
    programming_languages = canonicalize_terms([item for item in skills if item.lower() in COMMON_PROGRAMMING_LANGUAGES])
    ai_models = canonicalize_terms(
        [
            item.strip()
            for line in extract_section_lines(raw_text, "AI Models")
            for item in re.split(r",|;|\n", line)
        ]
    )
    if not ai_models:
        ai_models = canonicalize_terms([model for model in COMMON_AI_MODELS if model in raw_text.lower()])
    technologies = canonicalize_terms(technologies + frameworks + tools + programming_languages)
    certifications = unique_preserve(extract_section_lines(raw_text, "Certifications"))
    education = extract_education_entries(raw_text)
    resolution_steps = unique_preserve(extract_section_lines(raw_text, "Resolution Steps"))
    root_cause = " ".join(extract_section_lines(raw_text, "Root Cause")[:2]).strip()
    compliance_terms = unique_preserve(extract_section_lines(raw_text, "Compliance Terms"))
    contract_terms = unique_preserve(extract_section_lines(raw_text, "Contract Terms"))

    metadata: dict[str, Any] = {
        "document_type": document_type,
        "title": title,
        "owner": owner,
        "candidate_name": candidate_name or None,
        "department": department or None,
        "product": product or None,
        "skills": skills,
        "projects": projects,
        "technologies": technologies,
        "education": education,
        "experience": experience,
        "certifications": certifications,
        "customer_name": find_header_value(raw_text, "Customer_Name") or None,
        "account_name": find_header_value(raw_text, "Account_Name") or None,
        "ticket_id": find_header_value(raw_text, "Ticket_ID") or None,
        "case_id": find_header_value(raw_text, "Case_ID") or None,
        "incident_id": find_header_value(raw_text, "Incident_ID") or None,
        "issue_type": find_header_value(raw_text, "Issue_Type") or None,
        "priority": find_header_value(raw_text, "Priority") or None,
        "severity": find_header_value(raw_text, "Severity") or None,
        "status": find_header_value(raw_text, "Status") or None,
        "escalation": find_header_value(raw_text, "Escalation") or None,
        "resolution_steps": resolution_steps,
        "root_cause": root_cause or None,
        "sla": find_header_value(raw_text, "SLA") or None,
        "renewal_date": find_header_value(raw_text, "Renewal_Date") or None,
        "contract_terms": contract_terms,
        "compliance_terms": compliance_terms,
        "api_names": unique_preserve(extract_section_lines(raw_text, "APIs")),
        "integrations": unique_preserve(extract_section_lines(raw_text, "Integrations")),
        "databases": canonicalize_terms([item for item in skills if item.lower() in {"mongodb", "postgresql", "mysql", "redis", "snowflake", "supabase"}]),
        "cloud_services": canonicalize_terms([item for item in skills if item.lower() in COMMON_CLOUD_SERVICES]),
        "tools": tools,
        "frameworks": frameworks,
        "programming_languages": programming_languages,
        "ai_models": ai_models,
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "keywords": [],
        "summary": summary or None,
    }
    return metadata


def extract_heuristic_entities(raw_text: str) -> dict[str, Any]:
    raw_lower = raw_text.lower()
    emails = unique_preserve(re.findall(r"\b\S+@\S+\.\S+\b", raw_text))
    phones = unique_preserve(re.findall(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", raw_text))
    urls = unique_preserve(re.findall(r"https?://[^\s]+", raw_text))
    dates = unique_preserve(
        re.findall(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
            raw_text,
        )
    )
    ticket_ids = unique_preserve(re.findall(r"\b(?:INC|CASE|TICKET|REQ)[-_]?\d+\b", raw_text, flags=re.IGNORECASE))
    case_ids = [item for item in ticket_ids if item.lower().startswith("case")]
    incident_ids = [item for item in ticket_ids if item.lower().startswith("inc")]
    api_names = unique_preserve(re.findall(r"\b[A-Za-z0-9_-]+API\b", raw_text))
    integrations = unique_preserve(
        [term for term in ("okta", "salesforce", "jira", "slack", "servicenow") if term in raw_lower]
    )
    databases = unique_preserve([term for term in ("mongodb", "postgresql", "mysql", "redis", "snowflake") if term in raw_lower])
    cloud_services = unique_preserve([term for term in COMMON_CLOUD_SERVICES if term in raw_lower])
    frameworks = canonicalize_terms([term for term in COMMON_FRAMEWORKS if term in raw_lower])
    programming_languages = canonicalize_terms([term for term in COMMON_PROGRAMMING_LANGUAGES if re.search(rf"\b{re.escape(term)}\b", raw_lower)])
    technologies = canonicalize_terms([term for term in COMMON_TECHNOLOGIES if term in raw_lower] + frameworks + databases + cloud_services)
    tools = canonicalize_terms([term for term in ("jira", "tableau", "power bi", "excel", "docker", "kubernetes", "postman", "supabase") if term in raw_lower])
    ai_models = canonicalize_terms([model for model in COMMON_AI_MODELS if model in raw_lower])
    technologies = canonicalize_terms(technologies + tools + frameworks + programming_languages)
    support_keywords = unique_preserve([term.upper() if term == "sso" or term == "sla" else term for term in SUPPORT_TERMS if term in raw_lower])
    project_lines = unique_preserve(
        [
            line.strip(" -*•\t")
            for line in raw_text.splitlines()
            if is_project_like_line(line)
        ]
    )
    people = unique_preserve(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", raw_text))
    organizations = unique_preserve(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+(?:Labs|Analytics|Inc|LLC|Corp|Technologies|Systems))\b", raw_text))
    locations = unique_preserve(re.findall(r"\b(?:San Francisco|New York|London|Chennai|Bangalore|Remote)\b", raw_text))
    keywords = unique_preserve(
        support_keywords
        + project_lines
        + technologies
        + programming_languages
        + re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", raw_text)[:20]
    )

    return {
        "emails": emails,
        "phones": phones,
        "urls": urls,
        "dates": dates,
        "ticket_id": ticket_ids[0] if ticket_ids else None,
        "case_id": case_ids[0] if case_ids else None,
        "incident_id": incident_ids[0] if incident_ids else None,
        "technologies": technologies,
        "programming_languages": programming_languages,
        "cloud_services": cloud_services,
        "frameworks": frameworks,
        "tools": tools,
        "api_names": api_names,
        "integrations": integrations,
        "databases": databases,
        "projects": project_lines,
        "skills": unique_preserve(programming_languages + tools + technologies + frameworks + ai_models),
        "ai_models": ai_models,
        "issue_type": "authentication" if "sso" in raw_lower or "login" in raw_lower else None,
        "priority": "high" if "priority: high" in raw_lower else None,
        "severity": "high" if "severity: high" in raw_lower else None,
        "status": "open" if "status: open" in raw_lower else None,
        "escalation": "required" if "escalat" in raw_lower else None,
        "resolution_steps": unique_preserve(
            [line.strip() for line in raw_text.splitlines() if any(term in line.lower() for term in ("step", "review", "configure", "check", "open", "select"))][:8]
        ),
        "root_cause": " ".join(
            [line.strip() for line in raw_text.splitlines() if "root cause" in line.lower()][:1]
        )
        or None,
        "sla": "present" if "sla" in raw_lower else None,
        "customer_name": None,
        "account_name": None,
        "people": people[:10],
        "organizations": organizations[:10],
        "locations": locations[:10],
        "keywords": keywords[:25],
        "topics": support_keywords[:10],
        "summary": None,
    }


def metadata_is_sparse(metadata: dict[str, Any]) -> bool:
    meaningful_fields = 0
    for key in (
        "title",
        "owner",
        "candidate_name",
        "skills",
        "projects",
        "technologies",
        "tools",
        "frameworks",
        "programming_languages",
        "ai_models",
        "people",
        "organizations",
        "keywords",
        "issue_type",
        "resolution_steps",
    ):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            meaningful_fields += 1
        elif isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
            meaningful_fields += 1

    sparse_document_type = str(metadata.get("document_type", "")).strip().lower() in {"", "unknown", "enterprise_doc"}
    return meaningful_fields < 4 or sparse_document_type


def build_llm_metadata_prompt(raw_text: str, filename: str, document_type: str) -> str:
    excerpt = raw_text[:4000]
    return f"""Extract compact metadata as strict JSON only.

Filename: {filename}
Document type: {document_type}

Return this exact JSON shape:
{{
  "document_type_guess": "",
  "title": "",
  "summary": "",
  "entities": [],
  "people": [],
  "organizations": [],
  "products": [],
  "features": [],
  "technologies": [],
  "projects": [],
  "skills": [],
  "topics": [],
  "risks": [],
  "actions": [],
  "keywords": []
}}

Text:
{excerpt}
"""


def extract_llm_metadata(
    raw_text: str,
    filename: str,
    document_type: str,
    model_name: str = DEFAULT_LLM_METADATA_MODEL,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": build_llm_metadata_prompt(raw_text, filename, document_type),
        "stream": False,
        "format": "json",
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        llm_payload = json.loads((parsed.get("response") or "{}").strip())
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return {}

    if not isinstance(llm_payload, dict):
        return {}
    return llm_payload


def merge_metadata(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, list):
            merged[key] = unique_preserve(list(merged.get(key, [])) + value)
        elif value not in (None, "", [], {}):
            if not merged.get(key) or str(merged.get(key)).strip().lower() == "unknown":
                merged[key] = value
    return merged


def enrich_metadata(
    raw_text: str,
    filename: str,
    document_type: str,
    use_llm: bool = True,
    model_name: str = DEFAULT_LLM_METADATA_MODEL,
) -> dict[str, Any]:
    # Rule-based extraction remains the primary metadata source.
    metadata = extract_rule_metadata(raw_text, filename, document_type)
    metadata = merge_metadata(metadata, extract_heuristic_entities(raw_text))
    metadata["enrichment_source"] = "rule+heuristic"

    if not use_llm or not metadata_is_sparse(metadata) or metadata.get("enrichment_source") == "llm_fallback":
        return metadata

    llm_metadata = extract_llm_metadata(raw_text, filename, document_type, model_name=model_name)
    if not llm_metadata:
        return metadata

    normalized_llm = {
        "document_type": llm_metadata.get("document_type_guess") or metadata.get("document_type"),
        "title": llm_metadata.get("title"),
        "summary": llm_metadata.get("summary"),
        "entities": llm_metadata.get("entities", []),
        "people": llm_metadata.get("people", []),
        "organizations": llm_metadata.get("organizations", []),
        "products": llm_metadata.get("products", []),
        "technologies": llm_metadata.get("technologies", []),
        "projects": llm_metadata.get("projects", []),
        "skills": llm_metadata.get("skills", []),
        "ai_models": [
            item for item in llm_metadata.get("entities", [])
            if any(model in str(item).lower() for model in COMMON_AI_MODELS)
        ],
        "topics": llm_metadata.get("topics", []),
        "keywords": llm_metadata.get("keywords", []),
    }
    metadata = merge_metadata(metadata, normalized_llm)
    metadata["enrichment_source"] = "rule+heuristic+llm_fallback"
    return metadata
