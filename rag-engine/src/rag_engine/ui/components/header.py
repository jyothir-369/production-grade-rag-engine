"""Application header component.

Renders the top banner with the app name, tagline, and a live
API/system status indicator. The header never fabricates status —
it only reflects whatever is passed in by the caller.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def render_header(api_status: bool = False, health: Optional[dict] = None) -> None:
    """Render the application header with a live status badge.

    Args:
        api_status: Whether the backend API is currently reachable.
        health: Optional parsed ``/health`` response used to enrich the
            badge with the indexed document count when available.
    """
    status_label = "API Connected" if api_status else "API Offline"
    status_class = "status-online" if api_status else "status-offline"

    docs_indexed = None
    if isinstance(health, dict):
        docs_indexed = health.get("documents_indexed")

    docs_html = ""
    if api_status and docs_indexed is not None:
        docs_html = f'<span class="header-subbadge">{docs_indexed} docs indexed</span>'

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-header-title">
                    <span class="app-header-icon">📄</span>
                    <span>RAG Engine</span>
                </div>
                <div class="app-header-subtitle">
                    Grounded document intelligence powered by Retrieval-Augmented Generation
                </div>
            </div>
            <div class="app-header-right">
                <div class="status-badge {status_class}">
                    <span class="status-dot"></span>
                    <span>{status_label}</span>
                </div>
                {docs_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )