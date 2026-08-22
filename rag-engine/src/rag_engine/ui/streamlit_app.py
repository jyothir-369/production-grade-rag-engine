"""RAG Engine — Streamlit application entry point.

Orchestrates the UI: page config, custom styling, session state, and
the header / sidebar / chat components. All API communication logic
lives in the individual components; this file only wires them together.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_engine.ui.components.chat import render_chat
from rag_engine.ui.components.header import render_header
from rag_engine.ui.components.sidebar import render_sidebar

_CSS_PATH = Path(__file__).parent / "styles" / "custom.css"


def _load_custom_css() -> None:
    """Inject the custom stylesheet if it exists."""
    if _CSS_PATH.exists():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="RAG Engine",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _load_custom_css()
    _init_session_state()

    settings = render_sidebar()

    render_header(api_status=settings["api_connected"], health=settings["health"])

    render_chat(
        top_k=settings["top_k"],
        similarity_threshold=settings["similarity_threshold"],
        api_connected=settings["api_connected"],
    )


if __name__ == "__main__":
    main()