from __future__ import annotations


def detect_query_type(query: str) -> str:
    normalized = query.lower()

    ownership_markers = (
        "who built",
        "who owns",
        "which resume",
        "which candidate",
        "belongs to",
    )
    procedure_markers = (
        "how to",
        "how do",
        "configure",
        "setup",
        "steps",
        "troubleshoot",
        "explain",
    )
    summary_markers = ("summarize", "summary", "overview")
    comparison_markers = (
        "compare",
        "difference between",
        "difference",
        " vs ",
        " vs.",
        " versus ",
        "versus ",
    )

    if any(marker in normalized for marker in ownership_markers):
        return "ownership"
    if any(marker in normalized for marker in comparison_markers):
        return "comparison"
    if any(marker in normalized for marker in procedure_markers):
        return "procedure"
    if "where is" in normalized:
        return "procedure"
    if any(marker in normalized for marker in summary_markers):
        return "summary"
    return "general"
