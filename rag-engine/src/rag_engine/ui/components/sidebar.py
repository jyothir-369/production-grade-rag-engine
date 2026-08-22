"""Sidebar component.

Contains three sections:
    * System status (calls ``GET /health``)
    * Retrieval settings (Top K, similarity threshold)
    * PDF upload + ingestion (calls ``POST /api/v1/documents/ingest``)

Returns the current settings/status to the caller so the rest of the
app can use them without re-fetching.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
_HEALTH_TIMEOUT = 5
_INGEST_TIMEOUT = 120


def _check_health() -> tuple[Optional[dict], Optional[str]]:
    """Call the backend health endpoint.

    Returns:
        A tuple of ``(health_dict, error_message)``. Exactly one of
        the two will be ``None``.
    """
    try:
        response = requests.get(f"{API_BASE}/health", timeout=_HEALTH_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "Health check timed out."
    except requests.exceptions.HTTPError as exc:
        return None, f"Health check failed ({exc.response.status_code})."
    except ValueError:
        return None, "Health check returned an invalid response."
    except requests.exceptions.RequestException as exc:
        return None, f"Health check failed: {exc}"


def _ingest_pdf(file_path: str, source_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Call the document ingestion endpoint.

    Returns:
        A tuple of ``(result_dict, error_message)``.
    """
    payload = {"pdf_path": file_path, "source_id": source_id}
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/documents/ingest",
            json=payload,
            timeout=_INGEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "Ingestion timed out. Try a smaller file."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        status = exc.response.status_code if exc.response is not None else "?"
        return None, f"Ingestion failed ({status}). {detail}".strip()
    except ValueError:
        return None, "Ingestion returned an invalid response."
    except requests.exceptions.RequestException as exc:
        return None, f"Ingestion failed: {exc}"


def _render_status_section() -> tuple[Optional[dict], bool]:
    st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)

    health, error = _check_health()
    api_online = health is not None

    api_dot = "status-online" if api_online else "status-offline"
    api_label = "API Online" if api_online else "API Offline"
    st.markdown(
        f'<div class="status-row"><span class="status-dot {api_dot}"></span>{api_label}</div>',
        unsafe_allow_html=True,
    )

    if api_online:
        qdrant_connected = bool(health.get("qdrant_connected"))
        qdrant_dot = "status-online" if qdrant_connected else "status-offline"
        qdrant_label = "Qdrant Connected" if qdrant_connected else "Qdrant Disconnected"
        st.markdown(
            f'<div class="status-row"><span class="status-dot {qdrant_dot}"></span>{qdrant_label}</div>',
            unsafe_allow_html=True,
        )

        docs_indexed = health.get("documents_indexed")
        version = health.get("version")

        st.markdown('<div class="sidebar-subsection-title">Documents</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-stat">{docs_indexed if docs_indexed is not None else "—"} indexed</div>',
            unsafe_allow_html=True,
        )

        if version:
            st.caption(f"Backend version {version}")
    else:
        st.markdown(f'<div class="status-error">{error}</div>', unsafe_allow_html=True)

    if st.button("Refresh status", use_container_width=True):
        st.rerun()

    return health, api_online


def _render_retrieval_settings() -> tuple[int, float]:
    st.markdown('<div class="sidebar-section-title">Retrieval Settings</div>', unsafe_allow_html=True)

    top_k = st.slider("Top K", min_value=1, max_value=20, value=5, step=1)
    similarity_threshold = st.slider(
        "Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )

    return top_k, similarity_threshold


def _render_upload_section() -> None:
    st.markdown('<div class="sidebar-section-title">Upload PDF</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown(f'<div class="upload-filename">📎 {uploaded_file.name}</div>', unsafe_allow_html=True)

        if st.button("Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Ingesting document…"):
                try:
                    suffix = Path(uploaded_file.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name

                    result, error = _ingest_pdf(tmp_path, uploaded_file.name)
                finally:
                    Path(tmp_path).unlink(missing_ok=True) if "tmp_path" in locals() else None

            if error:
                st.markdown(f'<div class="status-error">{error}</div>', unsafe_allow_html=True)
            elif result:
                chunks = result.get("chunks_indexed", "—")
                proc_time = result.get("processing_time_ms")
                proc_time_label = f"{proc_time:.0f} ms" if isinstance(proc_time, (int, float)) else "—"
                st.success(f"Indexed {chunks} chunks in {proc_time_label}.")


def render_sidebar() -> dict[str, Any]:
    """Render the full sidebar and return current settings/status.

    Returns:
        A dict with keys ``top_k``, ``similarity_threshold``,
        ``health`` (the raw health dict or ``None``), and
        ``api_connected`` (bool).
    """
    with st.sidebar:
        health, api_connected = _render_status_section()
        st.divider()
        top_k, similarity_threshold = _render_retrieval_settings()
        st.divider()
        _render_upload_section()

    return {
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "health": health,
        "api_connected": api_connected,
    }