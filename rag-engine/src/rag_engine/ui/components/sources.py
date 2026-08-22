"""Source/citation display component.

Renders the list of retrieved chunks returned alongside an answer,
defensively handling any missing fields from the backend response.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def _format_similarity(score: Any) -> str:
    """Format a similarity score as a percentage when possible."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value * 100:.0f}%"


def _truncate(text: str, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def render_sources(sources: Optional[list[dict]]) -> None:
    """Render a list of retrieved source chunks.

    Args:
        sources: List of source dicts from the query response. Each may
            contain ``filename``, ``page``, ``similarity_score``,
            ``chunk_id``, and ``content``. All fields are optional.
    """
    if not sources:
        return

    st.markdown('<div class="sources-heading">Sources</div>', unsafe_allow_html=True)

    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue

        filename = source.get("filename") or "Unknown document"
        page = source.get("page")
        similarity = source.get("similarity_score")
        chunk_id = source.get("chunk_id")
        content = source.get("content") or ""

        page_label = f"Page {page}" if page not in (None, "") else "Page N/A"
        similarity_label = _format_similarity(similarity)

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-card-header">
                    <span class="source-icon">📄</span>
                    <span class="source-filename">{filename}</span>
                    <span class="source-meta">{page_label} · Similarity {similarity_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview = _truncate(content) if content else "No preview available."
        expander_label = f"View source content — {filename}"
        with st.expander(expander_label, expanded=False):
            st.markdown(f'<div class="source-content">{content or preview}</div>', unsafe_allow_html=True)
            if chunk_id:
                st.caption(f"Chunk ID: {chunk_id}")