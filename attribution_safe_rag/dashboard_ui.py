from __future__ import annotations

import streamlit as st
from typing import Any


def render_metric_card(label: str, value: float | str, delta: float | None = None, help_text: str | None = None) -> None:
    col = st.columns(1)[0]
    with col:
        st.metric(label, value=value, delta=delta, help=help_text)


def render_metrics_grid(metrics: list[dict[str, Any]]) -> None:
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta"),
                help=metric.get("help"),
            )


def render_evidence_cards(evidence_items: list[dict[str, Any]]) -> None:
    for item in evidence_items:
        with st.container(border=True):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.write(f"**{item.get('evidence_id')}**")
                st.caption(f"{item.get('metadata', {}).get('source_filename')}")
                st.text(item.get('section', 'N/A')[:200])
            with cols[1]:
                score = item.get('final_score_v2', 0.0)
                st.metric("Score", f"{score:.3f}")
            with cols[2]:
                doc_type = item.get('metadata', {}).get('document_type', 'N/A')
                st.badge(doc_type)


def render_verification_panel(verification_data: dict[str, Any]) -> None:
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.write("**Metadata Consistency**")
            consistency = verification_data.get("metadata_consistency", {})
            if consistency.get("is_consistent"):
                st.success("✓ Consistent")
            else:
                st.error("✗ Inconsistent")
            if consistency.get("details"):
                st.caption(str(consistency.get("details")))

    with cols[1]:
        with st.container(border=True):
            st.write("**Detected Entity**")
            entity = verification_data.get("detected_entity")
            if entity:
                st.info(entity)
            else:
                st.warning("No entity detected")

    with cols[2]:
        with st.container(border=True):
            st.write("**Ownership Verification**")
            ownership = verification_data.get("ownership_check")
            if ownership:
                if ownership.get("ownership_is_unambiguous"):
                    st.success("✓ Verified")
                else:
                    st.warning("⚠ Ambiguous")
                if ownership.get("candidate_names"):
                    st.caption(", ".join(ownership.get("candidate_names", [])))
            else:
                st.info("N/A")


def render_parent_contexts(parent_contexts: list[dict[str, Any]]) -> None:
    for context in parent_contexts:
        with st.expander(
            f"📄 {context.get('source_filename') or context.get('doc_id')} / {context.get('section')}"
        ):
            cols = st.columns([1, 1, 2])
            with cols[0]:
                st.caption(f"Parent ID: {context.get('parent_id')}")
            with cols[1]:
                st.caption(f"Evidence: {context.get('source_evidence_id')}")
            with cols[2]:
                pass
            st.divider()
            st.write(context.get("text"))


def render_benchmark_charts(metric_rows: list[dict[str, Any]], px: Any) -> None:
    if px is None:
        for row in metric_rows:
            st.metric(row["metric"], f"{row['value']:.2%}")
    else:
        chart = px.bar(
            metric_rows,
            x="metric",
            y="value",
            title="Evaluation Metrics Summary",
            range_y=[0, 1],
            color="value",
            color_continuous_scale="RdYlGn",
            labels={"metric": "Metric", "value": "Score"},
        )
        chart.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(chart, use_container_width=True)
