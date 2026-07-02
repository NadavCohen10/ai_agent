import sys
import os

# Walk up: pages/ → app/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from pathlib import Path

from core.llm_provider import OllamaProvider
from agents.interviewer_agent import CISOInterviewer, BASELINE_TOPICS, TOTAL_BASELINE

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CISO Interviewer",
    page_icon="🎙️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH      = PROJECT_ROOT / "data" / "Mock_Org_KB.xlsx"

# ── Session state init ────────────────────────────────────────────────────────

_DEFAULTS = {
    "kb_df":           None,   # loaded KB DataFrame
    "answered_ids":    [],     # topic IDs explicitly answered this session
    "active_topic_id": None,   # currently open topic
    "chat_histories":  {},     # {topic_id: [{role, content}]}
    "pending_controls":{},     # {topic_id: ctrl_dict awaiting approval}
    "question_cache":  {},     # {topic_id: conversational question string}
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    kb_path_input = st.text_input(
        "Knowledge Base path",
        value=str(KB_PATH),
        help="Path to Mock_Org_KB.xlsx produced by Document Ingestion.",
    )
    ollama_model = st.text_input(
        "Ollama model tag",
        value="gemma",
        placeholder="gemma, gemma2, llama3 …",
    )
    ollama_url = st.text_input(
        "Ollama server URL",
        value="http://localhost:11434",
    )

    st.divider()
    st.info(
        "🔒 **Privacy guarantee**: all LLM calls run locally via Ollama. "
        "No data is sent to external services."
    )

    if st.button("🔄 Reload KB & Reset", width="stretch"):
        for key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
        st.rerun()

# ── Load KB ───────────────────────────────────────────────────────────────────

kb_file = Path(kb_path_input)

if st.session_state.kb_df is None:
    if kb_file.exists():
        st.session_state.kb_df = pd.read_excel(kb_file)
    else:
        st.error(
            f"Knowledge Base not found at `{kb_file}`. "
            "Run **Document Ingestion** first to generate it."
        )
        st.stop()

kb_df: pd.DataFrame = st.session_state.kb_df

# ── Build agent + compute live coverage ──────────────────────────────────────

provider    = OllamaProvider(model=ollama_model, base_url=ollama_url)
interviewer = CISOInterviewer(provider=provider)

covered_count, covered_ids = interviewer.compute_coverage(
    kb_df, st.session_state.answered_ids
)
all_gaps = interviewer.find_baseline_gaps(kb_df, st.session_state.answered_ids)
gap_ids  = {g["id"] for g in all_gaps}

# ── Page header ───────────────────────────────────────────────────────────────

st.title("🎙️ CISO Interviewer")
st.caption(
    "Builds your **Company Baseline Profile** by identifying the 14 foundational "
    "security controls missing from your Knowledge Base and interviewing you to fill them in. "
    "All processing stays on your machine via Ollama."
)

# ── Progress bar ──────────────────────────────────────────────────────────────

progress_pct = covered_count / TOTAL_BASELINE
col_prog, col_metric = st.columns([4, 1])
with col_prog:
    st.progress(
        progress_pct,
        text=f"Company Baseline Profile: **{covered_count} / {TOTAL_BASELINE}** topics complete",
    )
with col_metric:
    st.metric("Remaining", TOTAL_BASELINE - covered_count)

# ── Optional notice ───────────────────────────────────────────────────────────

st.info(
    "💡 **This step is optional.** Completing the baseline dramatically improves "
    "the Assessor Agent's accuracy on external questionnaires. You can stop and navigate "
    "away at any time — your progress is saved automatically.",
    icon=None,
)

# ── All done? ─────────────────────────────────────────────────────────────────

if not all_gaps:
    st.success(
        "✅ **All 14 baseline topics are covered!** Your Knowledge Base is ready "
        "for high-quality questionnaire automation. Head to the main Assessor page."
    )
    st.stop()

st.divider()

# ── Determine active topic ────────────────────────────────────────────────────
# Default to first uncovered topic; respect user's manual selection when valid.

if st.session_state.active_topic_id not in gap_ids:
    st.session_state.active_topic_id = all_gaps[0]["id"]

# ── Two-column layout ─────────────────────────────────────────────────────────

col_list, col_chat = st.columns([1, 2], gap="large")

# ── Left column — topic checklist ────────────────────────────────────────────

with col_list:
    st.subheader("📋 Baseline Topics")

    for topic in BASELINE_TOPICS:
        tid = topic["id"]
        is_covered = tid in covered_ids
        is_active  = tid == st.session_state.active_topic_id
        has_pending = tid in st.session_state.pending_controls

        if is_covered:
            st.markdown(f"✅ ~~{topic['topic']}~~ &nbsp; `{topic['domain']}`")
        elif is_active:
            st.markdown(f"💬 **{topic['topic']}** &nbsp; `{topic['domain']}`")
        else:
            # Clickable button to jump to this topic
            if has_pending:
                label = f"📋 {topic['topic']}"
            else:
                label = f"⬜ {topic['topic']}"
            if st.button(label, key=f"nav_{tid}", use_container_width=True):
                st.session_state.active_topic_id = tid
                st.rerun()

# ── Right column — chat interface ─────────────────────────────────────────────

with col_chat:
    topic = next(t for t in BASELINE_TOPICS if t["id"] == st.session_state.active_topic_id)
    tid   = topic["id"]

    st.subheader(f"💬 {topic['domain']} — {topic['topic']}")
    st.caption(f"**Requirement:** {topic['vra_question']}")
    st.divider()

    # Generate (and cache) conversational opening question on first open
    if tid not in st.session_state.question_cache:
        with st.spinner("Preparing interview question…"):
            q = interviewer.generate_interview_question(topic)
            st.session_state.question_cache[tid] = q

        st.session_state.chat_histories.setdefault(tid, [])
        if not st.session_state.chat_histories[tid]:
            st.session_state.chat_histories[tid].append(
                {"role": "assistant", "content": q}
            )

    # Render conversation history
    for msg in st.session_state.chat_histories.get(tid, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Pending control approval ──────────────────────────────────────────────

    if tid in st.session_state.pending_controls:
        ctrl = st.session_state.pending_controls[tid]

        st.divider()
        st.markdown("**📋 Proposed control — review before adding to KB:**")

        col_d, col_s = st.columns([1, 2])
        col_d.markdown(f"**Domain:** `{ctrl['domain']}`")
        col_s.markdown(f"**Source:** `{ctrl['source_document']}`")
        st.info(f"> {ctrl['control_statement']}")

        col_approve, col_redo = st.columns(2)

        with col_approve:
            if st.button("✅ Approve & Save to KB", type="primary", width="stretch"):
                # Append control to KB DataFrame and persist to disk
                new_row = pd.DataFrame([ctrl])
                st.session_state.kb_df = pd.concat(
                    [st.session_state.kb_df, new_row], ignore_index=True
                )
                st.session_state.kb_df.to_excel(kb_file, index=False)

                # Mark topic as answered
                if tid not in st.session_state.answered_ids:
                    st.session_state.answered_ids.append(tid)

                del st.session_state.pending_controls[tid]

                st.session_state.chat_histories[tid].append({
                    "role": "assistant",
                    "content": (
                        f"✅ Control saved to your Knowledge Base under **{ctrl['domain']}**. "
                        "Great — let's move on to the next topic."
                    ),
                })

                # Advance to next uncovered topic
                remaining = interviewer.find_baseline_gaps(
                    st.session_state.kb_df, st.session_state.answered_ids
                )
                if remaining:
                    st.session_state.active_topic_id = remaining[0]["id"]

                st.rerun()

        with col_redo:
            if st.button("🔄 Give a different answer", width="stretch"):
                del st.session_state.pending_controls[tid]
                st.session_state.chat_histories[tid].append({
                    "role": "assistant",
                    "content": (
                        "No problem — please describe your organisation's approach again "
                        "with as much detail as possible."
                    ),
                })
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────

    else:
        col_input, col_skip = st.columns([5, 1])

        with col_input:
            user_input = st.chat_input(
                "Describe your organisation's approach…",
                key=f"input_{tid}",
            )

        with col_skip:
            # Vertical centering trick: empty label + button below
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Skip →", key=f"skip_{tid}", help="Move to the next topic without answering"):
                # Advance without saving
                remaining = [g for g in all_gaps if g["id"] != tid]
                if remaining:
                    st.session_state.active_topic_id = remaining[0]["id"]
                st.rerun()

        if user_input:
            st.session_state.chat_histories[tid].append(
                {"role": "user", "content": user_input}
            )

            history = st.session_state.chat_histories[tid]

            with st.spinner("Analysing your answer…"):
                result = interviewer.translate_to_formal_control(topic, history)

            if result["response_type"] == "follow_up":
                st.session_state.chat_histories[tid].append({
                    "role": "assistant",
                    "content": result["question"],
                })
                st.rerun()
            else:
                st.session_state.pending_controls[tid] = result
                st.session_state.chat_histories[tid].append({
                    "role": "assistant",
                    "content": (
                        "I have enough detail to draft a formal control. "
                        "Please review the proposed statement below before approving it."
                    ),
                })
                st.rerun()
