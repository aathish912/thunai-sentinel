from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

from answer import run_query_pipeline
from config import load_settings
from db import DOCUMENTS_COLLECTION, EVIDENCE_UNITS_COLLECTION, ensure_indexes, get_database
from eval import run_benchmark
from ingest import ingest_document


APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
EXAMPLE_QUERIES = [
    "Which resume has the Fraud Detection System project?",
    "Who built the Credit Risk Dashboard?",
    "Where is SSO setup explained?",
    "How do I configure SSO?",
    "Compare Candidate A and Candidate B",
    "Summarize the SSO and Dashboard Access Guide",
]

CUSTOM_CSS = """
<style>
    :root {
        --sentinel-bg: #0b1220;
        --sentinel-panel: #111c2e;
        --sentinel-sidebar: #0d1728;
        --sentinel-border: #24344d;
        --sentinel-text: #f8fafc;
        --sentinel-muted: #94a3b8;
        --sentinel-accent: #38bdf8;
        --sentinel-accent-2: #2dd4bf;
        --sentinel-button: #13233a;
        --sentinel-button-hover: #1b3150;
    }
    html, body, [class*="css"]  {
        background-color: var(--sentinel-bg);
        color: var(--sentinel-text);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(45, 212, 191, 0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.10), transparent 24%),
            linear-gradient(180deg, #07111f 0%, #0b1220 100%);
        color: var(--sentinel-text);
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"] {
        background: var(--sentinel-bg) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1728 0%, #0b1220 100%) !important;
        border-right: 1px solid var(--sentinel-border);
    }
    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] * {
        color: var(--sentinel-text) !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    label,
    .stTextInput label,
    .stFileUploader label {
        color: var(--sentinel-muted) !important;
    }
    h1, h2, h3, h4, h5, h6, strong, div, span {
        color: var(--sentinel-text);
    }
    a {
        color: var(--sentinel-accent) !important;
    }
    [data-testid="stChatMessage"],
    [data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stFileUploader"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stForm"],
    [data-testid="stAlert"] {
        background: var(--sentinel-panel) !important;
        border: 1px solid var(--sentinel-border) !important;
        border-radius: 16px !important;
        color: var(--sentinel-text) !important;
        box-shadow: none !important;
    }
    [data-testid="stMetric"] {
        padding: 0.85rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: var(--sentinel-muted) !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: var(--sentinel-text) !important;
    }
    [data-testid="stExpander"] details {
        background: var(--sentinel-panel) !important;
        border-radius: 16px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--sentinel-text) !important;
    }
    [data-testid="stTextInputRootElement"] input,
    [data-testid="stFileUploader"] section,
    textarea,
    input {
        background: #0f1a2d !important;
        color: var(--sentinel-text) !important;
        border: 1px solid var(--sentinel-border) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #0f1a2d !important;
        border: 1px dashed var(--sentinel-border) !important;
    }
    [data-baseweb="input"] {
        background: #0f1a2d !important;
        border-color: var(--sentinel-border) !important;
    }
    [data-baseweb="base-input"] {
        background: #0f1a2d !important;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background: var(--sentinel-panel) !important;
        border: 1px solid var(--sentinel-border) !important;
        border-radius: 18px !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: transparent !important;
        color: var(--sentinel-text) !important;
    }
    button[kind],
    .stButton > button {
        background: var(--sentinel-button) !important;
        color: var(--sentinel-text) !important;
        border: 1px solid var(--sentinel-border) !important;
        border-radius: 12px !important;
    }
    button[kind]:hover,
    .stButton > button:hover {
        background: var(--sentinel-button-hover) !important;
        border-color: var(--sentinel-accent) !important;
        color: white !important;
    }
    [data-testid="stDataFrame"] * ,
    [data-testid="stTable"] * {
        color: var(--sentinel-text) !important;
        background-color: transparent !important;
    }
    [data-testid="stAlert"] * {
        color: var(--sentinel-text) !important;
    }
    .sentinel-hero {
        background: linear-gradient(135deg, #0f172a 0%, #123b5d 55%, #0f766e 100%);
        color: #f8fafc;
        padding: 1.6rem 1.8rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
        margin-bottom: 1rem;
    }
    .sentinel-hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .sentinel-hero p {
        margin: 0.45rem 0 0 0;
        max-width: 880px;
        color: rgba(248, 250, 252, 0.88);
        line-height: 1.55;
    }
    .sentinel-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }
    .sentinel-pill {
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        font-size: 0.85rem;
    }
    .sentinel-section-title {
        margin-top: 0.25rem;
        margin-bottom: 0.85rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--sentinel-text);
    }
    .sentinel-card {
        background: var(--sentinel-panel);
        border: 1px solid var(--sentinel-border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: none;
    }
    .sentinel-muted {
        color: var(--sentinel-muted);
        font-size: 0.95rem;
    }
    .stAlert {
        border-radius: 14px;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: transparent !important;
    }
</style>
"""


@st.cache_resource
def load_config():
    return load_settings()


@st.cache_data(ttl=3600)
def get_benchmark_results(workspace_id: str, top_k: int):
    return run_benchmark(workspace_id, top_k)


@st.cache_data(ttl=60)
def get_knowledge_sources(workspace_id: str) -> list[dict[str, Any]]:
    settings = load_config()
    database = get_database(settings)
    ensure_indexes(database)

    documents = list(
        database[DOCUMENTS_COLLECTION].find(
            {"workspace_id": workspace_id},
            {"_id": 0, "doc_id": 1, "filename": 1, "document_type": 1, "source_type": 1},
        )
    )
    evidence_counts = {
        row["_id"]: row["evidence_count"]
        for row in database[EVIDENCE_UNITS_COLLECTION].aggregate(
            [
                {"$match": {"workspace_id": workspace_id}},
                {"$group": {"_id": "$doc_id", "evidence_count": {"$sum": 1}}},
            ]
        )
    }

    return [
        {
            "filename": document.get("filename"),
            "document_type": document.get("document_type"),
            "source_type": document.get("source_type"),
            "evidence_count": evidence_counts.get(document.get("doc_id"), 0),
            "doc_id": document.get("doc_id"),
        }
        for document in documents
    ]


def get_source_overview(sources: list[dict[str, Any]]) -> dict[str, int]:
    uploaded = sum(1 for source in sources if source.get("source_type") == "uploaded_file")
    seeded = sum(1 for source in sources if source.get("source_type") != "uploaded_file")
    evidence_total = sum(int(source.get("evidence_count") or 0) for source in sources)
    return {
        "total_sources": len(sources),
        "uploaded_sources": uploaded,
        "seeded_sources": seeded,
        "evidence_total": evidence_total,
    }


def display_source_type(source_type: str | None) -> str:
    return "Uploaded" if source_type == "uploaded_file" else "Seeded"


def is_comparison_style_query(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in ("compare", "difference", "versus", " vs "))


def delete_uploaded_document(workspace_id: str, doc_id: str, filename: str, source_type: str) -> tuple[bool, str]:
    if source_type != "uploaded_file":
        return False, "Only uploaded documents can be deleted."

    settings = load_config()
    database = get_database(settings)
    ensure_indexes(database)

    file_path = UPLOADS_DIR / filename
    if file_path.exists():
        file_path.unlink()

    database[DOCUMENTS_COLLECTION].delete_many(
        {"workspace_id": workspace_id, "doc_id": doc_id}
    )
    database[EVIDENCE_UNITS_COLLECTION].delete_many(
        {"workspace_id": workspace_id, "doc_id": doc_id}
    )
    get_knowledge_sources.clear()
    return True, f"Deleted {filename}."


def render_metric_cards(summary: dict[str, float]) -> None:
    metrics = [
        ("Wrong Attribution Rate", summary.get("wrong_attribution_rate", 0.0)),
        ("Retrieval Hit Rate", summary.get("retrieval_hit_rate", 0.0)),
        ("Ownership Accuracy", summary.get("ownership_accuracy", 0.0)),
        ("Query Type Accuracy", summary.get("query_type_accuracy", 0.0)),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value:.2%}")


def render_page_header() -> None:
    st.markdown(
        """
        <div class="sentinel-hero">
            <h1>Thunai Sentinel</h1>
            <p><strong>Enterprise Knowledge Verification Platform</strong></p>
            <p>Every Answer. Every Source. Fully Verified. Thunai Sentinel combines retrieval, verification, attribution tracking, and evidence grounding to help organizations trust AI-generated answers.</p>
            <div class="sentinel-pill-row">
                <div class="sentinel-pill">Hybrid Retrieval</div>
                <div class="sentinel-pill">Ownership Verification</div>
                <div class="sentinel-pill">Attribution Tracking</div>
                <div class="sentinel-pill">Evidence Grounding</div>
                <div class="sentinel-pill">Enterprise Evaluation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benchmark_controls(workspace_id: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    st.markdown("### Metrics / Evaluation")

    if "benchmark_data" not in st.session_state:
        st.session_state["benchmark_data"] = None

    if st.button("Run Benchmark", use_container_width=False):
        with st.spinner("Running benchmark..."):
            st.session_state["benchmark_data"] = get_benchmark_results(workspace_id, top_k)

    benchmark_data = st.session_state.get("benchmark_data")
    if benchmark_data is None:
        st.info("Benchmark not yet executed.")
        placeholder_summary = {
            "wrong_attribution_rate": 0.0,
            "retrieval_hit_rate": 0.0,
            "ownership_accuracy": 0.0,
            "query_type_accuracy": 0.0,
        }
        render_metric_cards(placeholder_summary)
        return [], placeholder_summary

    benchmark_results, benchmark_summary = benchmark_data
    render_metric_cards(benchmark_summary)
    return benchmark_results, benchmark_summary


def render_example_buttons() -> None:
    st.markdown("### Example Queries")
    cols = st.columns(3)
    for idx, query in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 3]:
            if st.button(query, key=f"example_{idx}", use_container_width=True):
                st.session_state["pending_query"] = query


def render_search_scope_controls(workspace_id: str) -> tuple[str, list[str] | None]:
    sources = get_knowledge_sources(workspace_id)
    filenames = [str(source.get("filename")) for source in sources if source.get("filename")]
    uploaded_filenames = [
        str(source.get("filename"))
        for source in sources
        if source.get("filename") and source.get("source_type") == "uploaded_file"
    ]

    scope_options = [
        "All Knowledge Sources",
        "Uploaded Documents",
        "Seeded Demo Docs",
        "Specific Document",
        "Compare Two Documents",
    ]
    default_scope = st.session_state.get("search_scope_label", scope_options[0])
    if default_scope not in scope_options:
        default_scope = scope_options[0]

    st.markdown("### Search Scope")
    selected_scope = st.selectbox(
        "Search Scope",
        scope_options,
        index=scope_options.index(default_scope),
        key="search_scope_label",
    )

    scope = "all"
    scope_filenames: list[str] | None = None

    if selected_scope == "Uploaded Documents":
        scope = "uploaded"
    elif selected_scope == "Seeded Demo Docs":
        scope = "seeded"
    elif selected_scope == "Specific Document":
        scope = "specific"
        if not filenames:
            st.info("No documents available for specific-document search.")
            return scope, None
        specific_default = st.session_state.get("specific_scope_filename", filenames[0])
        if specific_default not in filenames:
            specific_default = filenames[0]
        selected_filename = st.selectbox(
            "Document",
            filenames,
            index=filenames.index(specific_default),
            key="specific_scope_filename",
        )
        scope_filenames = [selected_filename]
    elif selected_scope == "Compare Two Documents":
        scope = "compare"
        if len(filenames) < 2:
            st.info("At least two documents are required for comparison scope.")
            return scope, None
        doc_a_default = st.session_state.get("compare_doc_a", filenames[0])
        doc_b_fallback = filenames[1] if len(filenames) > 1 else filenames[0]
        doc_b_default = st.session_state.get("compare_doc_b", doc_b_fallback)
        if doc_a_default not in filenames:
            doc_a_default = filenames[0]
        if doc_b_default not in filenames:
            doc_b_default = doc_b_fallback
        compare_cols = st.columns(2)
        doc_a = compare_cols[0].selectbox(
            "Document A",
            filenames,
            index=filenames.index(doc_a_default),
            key="compare_doc_a",
        )
        doc_b = compare_cols[1].selectbox(
            "Document B",
            filenames,
            index=filenames.index(doc_b_default),
            key="compare_doc_b",
        )
        if doc_a == doc_b:
            st.error("Select two different documents for comparison.")
            return scope, None
        scope_filenames = [doc_a, doc_b]

    if selected_scope == "Uploaded Documents" and not uploaded_filenames:
        st.info("No uploaded documents are available in this workspace yet.")

    return scope, scope_filenames


def save_uploaded_file(uploaded_file) -> Path:
    UPLOADS_DIR.mkdir(exist_ok=True)
    target_path = UPLOADS_DIR / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def render_sidebar(settings) -> tuple[str, int]:
    with st.sidebar:
        st.markdown("## Thunai Sentinel")
        st.caption("Enterprise Knowledge Verification Platform")
        st.divider()
        st.markdown("### Workspace Settings")
        workspace_id = st.text_input("Workspace ID", settings.default_workspace_id)
        top_k = 5
        controls = st.columns(2)
        if controls[0].button("Refresh Sources", use_container_width=True):
            get_knowledge_sources.clear()
            st.rerun()
        if controls[1].button("Clear Chat", use_container_width=True):
            st.session_state["chat_results"] = []
            st.session_state["last_processed_query"] = None
            st.rerun()

        st.divider()
        st.markdown("### Upload PDF or TXT")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        upload_signature = None
        if uploaded_file is not None:
            upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if (
            uploaded_file is not None
            and st.session_state.get("last_uploaded_signature") != upload_signature
        ):
            saved_path = save_uploaded_file(uploaded_file)
            result = ingest_document(saved_path, workspace_id)
            st.session_state["last_uploaded_signature"] = upload_signature
            get_benchmark_results.clear()
            get_knowledge_sources.clear()
            if result.get("status") == "success":
                st.success(
                    f"Uploaded and ingested {result['filename']} "
                    f"({result['evidence_count']} evidence units)."
                )
                st.caption(f"Document ID: {result['doc_id']}")
                st.caption(f"Evidence Count: {result['evidence_count']}")
                st.caption(f"Workspace ID: {workspace_id}")
            else:
                st.error(str(result.get("message", "Upload failed.")))

        st.divider()
        st.markdown("### About")
        st.write(
            "Thunai Sentinel prevents attribution drift by verifying ownership, "
            "source lineage, and supporting evidence before generating answers."
        )

        st.divider()
        st.markdown("### Why This Matters")
        with st.expander("Why This Matters"):
            st.write(
                "Thunai Sentinel prevents attribution drift, ownership hallucinations, "
                "and source confusion by separating discovery, verification, and response generation."
            )

        with st.expander("How It Works"):
            st.write(
                "Documents -> Knowledge Evidence -> Sentinel Discovery Engine -> Query Router -> "
                "Sentinel Verification Engine -> Sentinel Response Engine -> Grounded Answer"
            )

    return workspace_id, top_k


def render_result(result: dict[str, Any]) -> None:
    st.chat_message("user").write(result["query"])
    with st.chat_message("assistant"):
        badge_cols = st.columns([1, 1, 4])
        badge_cols[0].metric("Query Type", result["query_type"].title())
        badge_cols[1].metric("Evidence", len(result.get("evidence_items", [])))
        render_answer_block(result["final_answer"])
        if result.get("llm_answer") and result["llm_answer"] != result["final_answer"]:
            with st.expander("Response Refinement"):
                render_answer_block(result["llm_answer"])

        with st.expander("Knowledge Evidence"):
            rows = []
            for item in result["evidence_items"]:
                rows.append(
                    {
                        "evidence_id": item.get("evidence_id"),
                        "score": round(item.get("final_score_v2", 0.0), 4),
                        "source_filename": item.get("metadata", {}).get("source_filename"),
                        "document_type": item.get("metadata", {}).get("document_type"),
                        "section": item.get("section"),
                        "source_type": item.get("metadata", {}).get("source_type"),
                    }
                )
            st.dataframe(rows, use_container_width=True)

        with st.expander("Source Context"):
            for context in result["parent_contexts"]:
                st.markdown(
                    f"**{context.get('source_filename') or context.get('doc_id')} / "
                    f"{context.get('section')}**"
                )
                st.caption(f"Evidence: {context.get('source_evidence_id')}")
                st.markdown(format_context_block(context.get("text", "")))

        with st.expander("Sentinel Verification Engine"):
            st.json(
                {
                    "metadata_consistency": result.get("consistency"),
                    "detected_entity": result.get("detected_entity"),
                    "ownership_check": result.get("ownership_check"),
                }
            )


def format_display_markdown(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", "\n", str(text))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def render_answer_block(text: str) -> None:
    cleaned = format_display_markdown(text)
    if not cleaned.startswith("Projects identified:"):
        st.markdown(cleaned)
        return

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    project_items: list[tuple[str, str]] = []
    evidence_line = ""
    current_title = ""
    current_summary = ""

    for line in lines[1:]:
        if line.startswith("Supported by evidence:"):
            evidence_line = line
            continue
        if line.startswith("• "):
            if current_title:
                project_items.append((current_title, current_summary))
            current_title = line[2:].strip()
            current_summary = ""
            continue
        if line.startswith("- "):
            current_summary = line[2:].strip()
            continue
        if current_title and not current_summary:
            current_summary = line
        elif evidence_line:
            evidence_line += f" {line}"

    if current_title:
        project_items.append((current_title, current_summary))

    html_parts = [
        '<div class="sentinel-card" style="padding:1rem 1.1rem;">',
        '<div class="sentinel-section-title" style="margin-top:0;">Projects identified</div>',
        '<div style="display:flex; flex-direction:column; gap:0.9rem;">',
    ]
    for title, summary in project_items:
        html_parts.append('<div style="padding:0.1rem 0;">')
        html_parts.append(f'<div style="font-weight:700; color:#f8fafc;">• {title}</div>')
        if summary:
            html_parts.append(
                f'<div style="color:#cbd5e1; margin-top:0.2rem; padding-left:1rem;">{summary}</div>'
            )
        html_parts.append("</div>")
    html_parts.append("</div>")
    if evidence_line:
        evidence_text = evidence_line.replace("Supported by evidence:", "").strip()
        html_parts.append(
            f'<div style="margin-top:1rem; color:#94a3b8;"><strong>Supported by evidence:</strong> {evidence_text}</div>'
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def format_context_block(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", "\n", str(text))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return f"```text\n{cleaned}\n```" if cleaned else ""


def render_knowledge_sources(workspace_id: str) -> None:
    st.markdown("### Knowledge Sources")
    if st.session_state.get("delete_message"):
        st.success(str(st.session_state.pop("delete_message")))

    sources = get_knowledge_sources(workspace_id)
    if not sources:
        st.info("No ingested knowledge sources found for this workspace.")
        return

    overview = get_source_overview(sources)
    stats = st.columns(4)
    stats[0].metric("Sources", overview["total_sources"])
    stats[1].metric("Uploaded", overview["uploaded_sources"])
    stats[2].metric("Seeded", overview["seeded_sources"])
    stats[3].metric("Knowledge Evidence", overview["evidence_total"])
    st.caption("Uploaded sources are prioritized for document-specific questions such as uploaded resumes and user-provided PDFs.")

    sorted_sources = sorted(
        sources,
        key=lambda item: (
            item.get("source_type") != "uploaded_file",
            item.get("source_type") != "local_file",
            -(item.get("evidence_count") or 0),
            str(item.get("filename") or "").lower(),
        ),
    )

    header = st.columns([3, 2, 2, 1, 1])
    header[0].markdown("**filename**")
    header[1].markdown("**document_type**")
    header[2].markdown("**Type**")
    header[3].markdown("**evidence_count**")
    header[4].markdown("**action**")

    for source in sorted_sources:
        cols = st.columns([3, 2, 2, 1, 1])
        cols[0].write(source.get("filename"))
        cols[1].write(source.get("document_type"))
        cols[2].write(display_source_type(source.get("source_type")))
        cols[3].write(source.get("evidence_count"))

        if source.get("source_type") == "uploaded_file":
            if cols[4].button("Delete", key=f"delete_{source.get('doc_id')}"):
                deleted, message = delete_uploaded_document(
                    workspace_id,
                    str(source.get("doc_id")),
                    str(source.get("filename")),
                    str(source.get("source_type")),
                )
                if deleted:
                    st.session_state["delete_message"] = message
                    st.rerun()
                cols[4].error(message)
        else:
            cols[4].caption("Seeded")


def render_benchmark_section(benchmark_results: list[dict[str, Any]], benchmark_summary: dict[str, float]) -> None:
    if not benchmark_results:
        return

    if px is not None:
        chart_rows = [
            {"metric": "Wrong Attribution Rate", "value": benchmark_summary.get("wrong_attribution_rate", 0.0)},
            {"metric": "Retrieval Hit Rate", "value": benchmark_summary.get("retrieval_hit_rate", 0.0)},
            {"metric": "Ownership Accuracy", "value": benchmark_summary.get("ownership_accuracy", 0.0)},
            {"metric": "Query Type Accuracy", "value": benchmark_summary.get("query_type_accuracy", 0.0)},
        ]
        chart = px.bar(
            chart_rows,
            x="metric",
            y="value",
            range_y=[0, 1],
            title="Current Evaluation Snapshot",
        )
        st.plotly_chart(chart, use_container_width=True)

    with st.expander("Benchmark Details"):
        st.dataframe(
            [
                {
                    "query": item["query"],
                    "query_type": item["query_type"],
                    "retrieval_hit": item["retrieval_hit"],
                    "wrong_attribution": item["wrong_attribution"],
                    "ownership_correct": item["ownership_correct"],
                    "comparison_correct": item["comparison_correct"],
                    "passed": item["passed"],
                }
                for item in benchmark_results
            ],
            use_container_width=True,
        )


def render_enterprise_readiness_panel() -> None:
    st.markdown("### Enterprise Readiness")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <div class="sentinel-card">
                <div class="sentinel-section-title">Operational Trust</div>
                <div class="sentinel-muted">Ownership checks, attribution-safe answers, and evidence citations remain enforced before final response presentation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="sentinel-card">
                <div class="sentinel-section-title">Offline-First Ingestion</div>
                <div class="sentinel-muted">Rule-based and heuristic metadata enrichment remain the default path, with optional LLM fallback only when coverage is sparse.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="sentinel-card">
                <div class="sentinel-section-title">Production Evaluation</div>
                <div class="sentinel-muted">Benchmarking remains available on demand so teams can validate retrieval accuracy and wrong-attribution safeguards without slowing the main workflow.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_query_input(
    workspace_id: str,
    top_k: int,
    scope: str,
    scope_filenames: list[str] | None,
) -> None:
    pending_query = st.session_state.get("pending_query")
    chat_query = st.chat_input("Ask a question about your documents…")
    query = pending_query or chat_query

    if not query:
        st.session_state["last_processed_query"] = None
        return

    scope_key = ",".join(scope_filenames or [])
    query_key = f"{workspace_id}:{scope}:{scope_key}:{query.strip()}"
    if st.session_state.get("last_processed_query") == query_key:
        if pending_query:
            st.session_state["pending_query"] = None
        return

    with st.spinner("Processing query..."):
        try:
            pipeline_query = query
            if scope == "compare" and scope_filenames and not is_comparison_style_query(query):
                pipeline_query = (
                    f"Compare {scope_filenames[0]} and {scope_filenames[1]}. {query}"
                )
            result = run_query_pipeline(
                pipeline_query,
                top_k,
                workspace_id,
                scope=scope,
                scope_filenames=scope_filenames,
            )
            result["query"] = query
        except Exception as exc:  # pragma: no cover - UI safety path
            st.error(f"Query failed: {exc}")
            return
    st.session_state.setdefault("chat_results", []).append(result)
    st.session_state["last_processed_query"] = query_key
    if pending_query:
        st.session_state["pending_query"] = None
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Thunai Sentinel",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "chat_results" not in st.session_state:
        st.session_state["chat_results"] = []
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    if "last_processed_query" not in st.session_state:
        st.session_state["last_processed_query"] = None

    settings = load_config()
    workspace_id, top_k = render_sidebar(settings)

    render_page_header()
    st.markdown("### Ask a Question")
    for result in st.session_state.get("chat_results", []):
        render_result(result)

    st.divider()
    render_example_buttons()
    st.divider()
    render_knowledge_sources(workspace_id)
    st.divider()
    render_enterprise_readiness_panel()
    st.divider()
    st.markdown("### About Thunai Sentinel")
    st.markdown(
        '<div class="sentinel-card"><div class="sentinel-muted">Thunai Sentinel is designed for enterprise knowledge verification with attribution-safe retrieval, source lineage tracking, grounded response generation, and production-style evaluation. The current UI keeps the operational benchmark tooling in the backend while presenting a cleaner executive-ready console.</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Continue the Conversation")
    scope, scope_filenames = render_search_scope_controls(workspace_id)
    if not st.session_state.get("chat_results"):
        st.info("Ask a question about enterprise documents, uploaded resumes, support guides, or policies to start a verified answer workflow.")
    run_query_input(workspace_id, top_k, scope, scope_filenames)
    st.divider()
    st.caption("Powered by Thunai Sentinel")
    st.caption("Enterprise Knowledge Verification Platform")
    st.caption(
        "Features: Hybrid Retrieval • Ownership Verification • Attribution Tracking • "
        "Evidence Grounding • Evaluation Framework"
    )


if __name__ == "__main__":
    main()
