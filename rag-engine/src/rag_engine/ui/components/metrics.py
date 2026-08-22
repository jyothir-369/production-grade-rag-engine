"""Query metrics component.

Displays quantitative details about a query response (latency,
confidence, source count, model, tokens) using ``st.metric``. Metrics
that are unavailable in the response are simply omitted rather than
shown as fake placeholders.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def _format_ms(value: Any) -> Optional[str]:
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return None


def _format_confidence(value: Any) -> Optional[str]:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return None


def render_metrics(response: Optional[dict]) -> None:
    """Render available query metrics as a metric row.

    Args:
        response: The parsed ``/api/v1/query`` response dict.
    """
    if not isinstance(response, dict):
        return

    entries: list[tuple[str, str]] = []

    query_time = _format_ms(response.get("query_time_ms"))
    if query_time:
        entries.append(("Query Time", query_time))

    confidence = _format_confidence(response.get("confidence"))
    if confidence:
        entries.append(("Confidence", confidence))

    sources = response.get("sources")
    if isinstance(sources, list):
        entries.append(("Sources", str(len(sources))))

    model = response.get("model")
    if model:
        entries.append(("Model", str(model)))

    tokens_used = response.get("tokens_used")
    if tokens_used is not None:
        entries.append(("Tokens", str(tokens_used)))

    if not entries:
        return

    st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
    columns = st.columns(len(entries))
    for col, (label, value) in zip(columns, entries):
        with col:
            st.metric(label=label, value=value)
    st.markdown("</div>", unsafe_allow_html=True)