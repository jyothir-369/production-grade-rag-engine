"""Chat / Q&A component.

Renders the main conversational interface: persistent history via
``st.session_state``, a chat input, and calls to the query endpoint.
Each assistant turn renders its answer, metrics, and sources inline.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st

from rag_engine.ui.components.metrics import render_metrics
from rag_engine.ui.components.sources import render_sources

API_BASE = "http://127.0.0.1:8000"
_QUERY_TIMEOUT = 60


def _init_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _ask(question: str, top_k: int, similarity_threshold: float) -> tuple[Optional[dict], Optional[str]]:
    """Call the query endpoint.

    Returns:
        A tuple of ``(response_dict, error_message)``.
    """
    payload = {
        "question": question,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
    }
    try:
        response = requests.post(f"{API_BASE}/api/v1/query", json=payload, timeout=_QUERY_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is the backend running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "The request timed out. Try again or reduce Top K."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        status = exc.response.status_code if exc.response is not None else "?"
        return None, f"Query failed ({status}). {detail}".strip()
    except ValueError:
        return None, "The API returned an invalid response."
    except requests.exceptions.RequestException as exc:
        return None, f"Query failed: {exc}"


def _render_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        with st.chat_message(role):
            if role == "assistant":
                error = message.get("error")
                if error:
                    st.markdown(f'<div class="status-error">{error}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="answer-card">{content}</div>', unsafe_allow_html=True)
                    response = message.get("response")
                    if response:
                        render_metrics(response)
                        render_sources(response.get("sources"))
            else:
                st.markdown(content)


def render_chat(top_k: int = 5, similarity_threshold: float = 0.0, api_connected: bool = True) -> None:
    """Render the chat interface.

    Args:
        top_k: Number of chunks to retrieve, passed through to the query.
        similarity_threshold: Minimum similarity score, passed through
            to the query.
        api_connected: Whether the backend is currently reachable, used
            to disable input when offline.
    """
    _init_history()

    st.markdown('<div class="chat-heading">Chat / Q&amp;A</div>', unsafe_allow_html=True)

    _render_history()

    placeholder = "Ask a question about your documents…" if api_connected else "API is offline — check the sidebar."
    question = st.chat_input(placeholder, disabled=not api_connected)

    if question is None:
        return

    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response, error = _ask(question, top_k, similarity_threshold)

        if error:
            st.markdown(f'<div class="status-error">{error}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": "", "error": error})
        else:
            answer = response.get("answer") if isinstance(response, dict) else None
            answer_text = answer if answer else "No answer was returned."
            st.markdown(f'<div class="answer-card">{answer_text}</div>', unsafe_allow_html=True)
            render_metrics(response)
            render_sources(response.get("sources") if isinstance(response, dict) else None)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text, "response": response}
            )