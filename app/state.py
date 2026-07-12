"""
Centralized session state management for the HITL Questionnaire Assistant.

All st.session_state keys and their defaults live here.
Every Streamlit page calls init_session_state() at the top before reading
or writing any state key. This guarantees:
  - No KeyError or AttributeError when pages load in unexpected order.
  - A single place to add, rename, or remove a key.
  - Mutable defaults (lists / dicts) are always fresh copies, not shared references.
"""

from __future__ import annotations
import uuid
import streamlit as st


# ── Key registry ──────────────────────────────────────────────────────────────
# Callables are used for mutable defaults so each init gets a fresh object.

_DEFAULTS: dict[str, object] = {
    # ── KB pipeline (1_document_ingestion.py) ─────────────────────────────────
    "kb_df":              None,
    "kb_gaps":            None,

    # ── Questionnaire pipeline (app.py) ───────────────────────────────────────
    "draft_df":           None,
    "draft_df_id":        None,   # UUID written alongside every draft_df update
    "approved_df":        None,
    "is_excel_pipeline":  False,

    # ── LLM provider cache ────────────────────────────────────────────────────
    "gemini_provider":    None,

    # ── KB baseline interview — Tab A (2_ciso_interviewer.py) ─────────────────
    "answered_ids":       "list",   # sentinel for mutable
    "active_topic_id":    None,
    "chat_histories":     "dict",
    "pending_controls":   "dict",
    "question_cache":     "dict",

    # ── Questionnaire gap chat — Tab B (2_ciso_interviewer.py) ────────────────
    "q_gap_queue":        "list",
    "q_active_row_idx":   None,
    "q_chat_histories":   "dict",
    "q_draft_df_id":      None,   # mirrors draft_df_id at the time Tab B was built
}


def init_session_state() -> None:
    """
    Ensure every required session state key exists with its default value.
    Safe to call multiple times — only missing keys are initialised.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            if default == "list":
                st.session_state[key] = []
            elif default == "dict":
                st.session_state[key] = {}
            else:
                st.session_state[key] = default


# ── Mutation helpers ──────────────────────────────────────────────────────────

def set_draft_df(df, is_excel: bool = True) -> None:
    """
    Store a processed questionnaire DataFrame and stamp it with a fresh UUID.

    The UUID (draft_df_id) is the stable identity token used by Tab B to detect
    when a completely new questionnaire has been loaded, even if the new file has
    the same number of rows and the same integer indices as the old one.
    Clears approved_df so a stale approval is never exported for a new file.
    """
    st.session_state.draft_df           = df
    st.session_state.draft_df_id        = str(uuid.uuid4())
    st.session_state.is_excel_pipeline  = is_excel
    st.session_state.approved_df        = None


def reset_kb_interview_state() -> None:
    """Reset Tab A (KB baseline interview) without touching any other state."""
    st.session_state.answered_ids    = []
    st.session_state.active_topic_id = None
    st.session_state.chat_histories  = {}
    st.session_state.pending_controls = {}
    st.session_state.question_cache  = {}


def reset_q_chat_state() -> None:
    """Reset Tab B (questionnaire gap chat) without touching any other state."""
    st.session_state.q_gap_queue      = []
    st.session_state.q_active_row_idx = None
    st.session_state.q_chat_histories = {}
    st.session_state.q_draft_df_id    = None
