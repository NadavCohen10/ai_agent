import sys
import os
import io

# Walk up: pages/ → app/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from core.llm_provider import OllamaProvider
from agents.interviewer_agent import CISOInterviewer, BASELINE_TOPICS, TOTAL_BASELINE
from agents.assessor_agent import _detect_id_key
from app.exporter import generate_excel
from app.state import init_session_state, reset_kb_interview_state, reset_q_chat_state

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CISO Gap Resolution Hub",
    page_icon="🎙️",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────

init_session_state()

_Q_META_COLS = {"_AI_Status", "_AI_Reasoning", "Evidence/Notes"}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

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

    st.divider()

    if st.session_state.get("kb_df") is not None:
        st.markdown("**📥 Export Knowledge Base**")
        st.caption("Download the current enriched KB at any time.")
        buf = io.BytesIO()
        st.session_state.kb_df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Enriched KB (Excel)",
            data=buf,
            file_name="enriched_knowledge_base.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.divider()

    if st.button("🔄 Reset KB Interview Progress", use_container_width=True):
        reset_kb_interview_state()
        st.rerun()

    if st.button("🔄 Reset Questionnaire Chat", use_container_width=True):
        reset_q_chat_state()
        st.rerun()

# ── Provider + interviewer ────────────────────────────────────────────────────

provider    = OllamaProvider(model=ollama_model, base_url=ollama_url)
interviewer = CISOInterviewer(provider=provider)

# ── Page header ───────────────────────────────────────────────────────────────

st.title("🎙️ CISO Gap Resolution Hub")
st.caption("Resolve gaps in your Knowledge Base and Questionnaire — all in one place.")

tab_kb, tab_q = st.tabs(["🔍 KB Gap Resolution", "📋 Questionnaire Gap Resolution"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB A — KB Gap Resolution
# ══════════════════════════════════════════════════════════════════════════════

def _render_kb_tab():
    kb_df = st.session_state.get("kb_df")

    if kb_df is None:
        st.error(
            "**No Knowledge Base found in memory.** "
            "Please run the Document Ingestion step first to generate a KB."
        )
        if st.button("← Go to Document Ingestion", key="kb_goto_ingestion"):
            st.switch_page("pages/1_document_ingestion.py")
        return

    covered_count, covered_ids = interviewer.compute_coverage(
        kb_df, st.session_state.answered_ids
    )
    all_gaps = interviewer.find_baseline_gaps(kb_df, st.session_state.answered_ids)
    gap_ids  = {g["id"] for g in all_gaps}

    st.subheader("🎙️ KB Gap Resolution via CISO Interview")
    st.caption(
        "Builds your **Company Baseline Profile** by identifying the 14 foundational "
        "security controls missing from your Knowledge Base and interviewing you to fill them in. "
        "All processing stays on your machine via Ollama."
    )

    col_prog, col_metric = st.columns([4, 1])
    with col_prog:
        st.progress(
            covered_count / TOTAL_BASELINE,
            text=f"Company Baseline Profile: **{covered_count} / {TOTAL_BASELINE}** topics complete",
        )
    with col_metric:
        st.metric("Remaining", TOTAL_BASELINE - covered_count)

    st.info(
        "💡 **This step is optional.** Completing the baseline dramatically improves "
        "the Assessor Agent's accuracy on external questionnaires. You can stop at any time — "
        "progress is saved in memory and the KB can be downloaded from the sidebar.",
    )

    if not all_gaps:
        st.success(
            "✅ **All 14 baseline topics are covered!** Your Knowledge Base is ready. "
            "Download it from the sidebar and upload it to the Assessor Agent."
        )
        return

    st.divider()

    if st.session_state.active_topic_id not in gap_ids:
        st.session_state.active_topic_id = all_gaps[0]["id"]

    col_list, col_chat = st.columns([1, 2], gap="large")

    with col_list:
        st.subheader("📋 Baseline Topics")
        for topic in BASELINE_TOPICS:
            tid         = topic["id"]
            is_covered  = tid in covered_ids
            is_active   = tid == st.session_state.active_topic_id
            has_pending = tid in st.session_state.pending_controls

            if is_covered:
                st.markdown(f"✅ ~~{topic['topic']}~~ &nbsp; `{topic['domain']}`")
            elif is_active:
                st.markdown(f"💬 **{topic['topic']}** &nbsp; `{topic['domain']}`")
            else:
                label = f"📋 {topic['topic']}" if has_pending else f"⬜ {topic['topic']}"
                if st.button(label, key=f"nav_{tid}", use_container_width=True):
                    st.session_state.active_topic_id = tid
                    st.rerun()

    with col_chat:
        topic = next(t for t in BASELINE_TOPICS if t["id"] == st.session_state.active_topic_id)
        tid   = topic["id"]

        st.subheader(f"💬 {topic['domain']} — {topic['topic']}")
        st.caption(f"**Requirement:** {topic['vra_question']}")
        st.divider()

        if tid not in st.session_state.question_cache:
            with st.spinner("Preparing interview question…"):
                q = interviewer.generate_interview_question(topic)
                st.session_state.question_cache[tid] = q
            st.session_state.chat_histories.setdefault(tid, [])
            if not st.session_state.chat_histories[tid]:
                st.session_state.chat_histories[tid].append(
                    {"role": "assistant", "content": q}
                )

        for msg in st.session_state.chat_histories.get(tid, []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

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
                if st.button("✅ Approve & Add to KB", type="primary", width="stretch"):
                    new_row = pd.DataFrame([ctrl])
                    st.session_state.kb_df = pd.concat(
                        [st.session_state.kb_df, new_row], ignore_index=True
                    )
                    if tid not in st.session_state.answered_ids:
                        st.session_state.answered_ids.append(tid)
                    del st.session_state.pending_controls[tid]
                    st.session_state.chat_histories[tid].append({
                        "role": "assistant",
                        "content": (
                            f"✅ Control added to the Knowledge Base under **{ctrl['domain']}**. "
                            "Download the updated KB from the sidebar whenever you're ready."
                        ),
                    })
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

        else:
            col_input, col_skip = st.columns([5, 1])
            with col_input:
                user_input = st.chat_input(
                    "Describe your organisation's approach…",
                    key=f"input_{tid}",
                )
            with col_skip:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Skip →", key=f"skip_{tid}", help="Move to the next topic without answering"):
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
                        "role": "assistant", "content": result["question"],
                    })
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB B — Questionnaire Gap Resolution (chat interface)
# ══════════════════════════════════════════════════════════════════════════════

def _q_row_label(draft_df: pd.DataFrame, row_idx: int) -> str:
    """Short display label for a gap row (control ID or row number fallback)."""
    row    = draft_df.loc[row_idx]
    id_key = _detect_id_key(row.to_dict())
    if id_key and row.get(id_key):
        val = str(row[id_key]).strip()
        return val[:42] + "…" if len(val) > 42 else val
    return f"Row {row_idx}"


def _build_gap_agent_message(draft_df: pd.DataFrame, row_idx: int,
                              gap_num: int, total: int) -> str:
    """Build the agent's opening chat message for a single flagged control."""
    row      = draft_df.loc[row_idx]
    row_dict = row.to_dict()
    id_key   = _detect_id_key(row_dict)

    status    = str(row.get("_AI_Status", ""))
    reasoning = str(row.get("_AI_Reasoning", "") or "").strip()
    badge     = "⚠️ REVIEW" if status == "REVIEW" else "❌ NO_DATA"

    context_lines: list[str] = []
    for col, val in row_dict.items():
        if col in _Q_META_COLS or col == id_key:
            continue
        str_val = str(val).strip() if val is not None else ""
        if str_val and not (isinstance(val, float) and pd.isna(val)):
            context_lines.append(f"**{col}:** {str_val}")
        if len(context_lines) >= 4:
            break

    header = f"**Gap {gap_num} of {total}**"
    if id_key and row.get(id_key):
        header += f" — `{row[id_key]}`"

    parts = [header]
    if context_lines:
        parts.append("\n".join(context_lines))
    parts.append(f"**AI Assessment:** {badge}")
    if reasoning:
        parts.append(f"**Reason flagged:** {reasoning}")
    parts.append(
        "Please provide evidence or context for this control — "
        "for example, the name of a relevant policy, a brief process description, "
        "or any documentation that addresses this requirement."
    )
    return "\n\n".join(parts)


def _q_ensure_opening_message(draft_df: pd.DataFrame,
                               row_idx: int, gap_num: int, total: int) -> None:
    """Lazily generate and cache the opening agent message for a gap on first visit."""
    histories = st.session_state.q_chat_histories
    if row_idx not in histories:
        histories[row_idx] = [{
            "role": "assistant",
            "content": _build_gap_agent_message(draft_df, row_idx, gap_num, total),
        }]


def _q_find_next_pending(draft_df: pd.DataFrame, queue: list[int],
                          after_idx: int) -> int | None:
    """
    Return the next row_idx in queue that is still REVIEW/NO_DATA, searching
    forward from after_idx (wrapping around). Returns None if all are resolved.
    """
    n = len(queue)
    for offset in range(1, n + 1):
        candidate = queue[(after_idx + offset) % n]
        if str(draft_df.at[candidate, "_AI_Status"]) != "OK":
            return candidate
    return None  # every gap is resolved


def _q_init(draft_df: pd.DataFrame) -> None:
    """
    Build the gap queue on first call (or when draft_df changes).

    Identity is tracked via the UUID stored in draft_df_id (set by set_draft_df()
    in app.state whenever a new questionnaire is processed). Comparing it against
    q_draft_df_id (the UUID this chat was built for) is the only reliable way to
    detect a new file — integer row indices alone are insufficient because a new
    questionnaire almost always has the same 0-based indices as the old one.
    """
    current_id = st.session_state.get("draft_df_id")
    owned_id   = st.session_state.get("q_draft_df_id")

    if current_id != owned_id:
        # A new (or first) questionnaire was loaded — wipe stale chat state and
        # stamp the new ownership token so this branch only fires once per file.
        reset_q_chat_state()
        st.session_state.q_draft_df_id = current_id

    if st.session_state.q_gap_queue:
        # Queue already built for this df — just make sure active index is valid.
        if st.session_state.q_active_row_idx is None:
            pending = [r for r in st.session_state.q_gap_queue
                       if str(draft_df.at[r, "_AI_Status"]) != "OK"]
            if pending:
                st.session_state.q_active_row_idx = pending[0]
        return

    # First call for this df — build the queue from REVIEW/NO_DATA rows.
    if "_AI_Status" in draft_df.columns:
        queue = draft_df.index[
            draft_df["_AI_Status"].isin(["REVIEW", "NO_DATA"])
        ].tolist()
    else:
        queue = []

    st.session_state.q_gap_queue      = queue
    st.session_state.q_chat_histories = {}
    st.session_state.q_active_row_idx = queue[0] if queue else None


def _q_handle_response(user_text: str, row_idx: int,
                        draft_df: pd.DataFrame, queue: list[int]) -> None:
    """Record evidence, mark row OK, auto-advance to the next pending gap."""
    histories = st.session_state.q_chat_histories

    histories[row_idx].append({"role": "user", "content": user_text})

    if "Evidence/Notes" not in st.session_state.draft_df.columns:
        st.session_state.draft_df["Evidence/Notes"] = ""
    st.session_state.draft_df.at[row_idx, "Evidence/Notes"] = user_text.strip()
    st.session_state.draft_df.at[row_idx, "_AI_Status"]     = "OK"

    histories[row_idx].append({
        "role": "assistant",
        "content": "✅ Evidence recorded — this control is now marked as **OK**.",
    })

    # Advance to the next pending gap (circular search)
    current_pos = queue.index(row_idx)
    next_row    = _q_find_next_pending(st.session_state.draft_df, queue, current_pos)
    st.session_state.q_active_row_idx = next_row


def _q_handle_skip(row_idx: int, draft_df: pd.DataFrame, queue: list[int]) -> None:
    """Skip current gap and navigate to the next pending one."""
    histories = st.session_state.q_chat_histories
    histories.setdefault(row_idx, []).append({"role": "user", "content": "*(Skipped)*"})
    histories[row_idx].append({
        "role": "assistant",
        "content": (
            "No problem — this control will remain flagged. "
            "You can return to it at any time by clicking it in the list on the left."
        ),
    })

    current_pos = queue.index(row_idx)
    next_row    = _q_find_next_pending(draft_df, queue, current_pos)
    st.session_state.q_active_row_idx = next_row


def _render_questionnaire_tab():
    draft_df: pd.DataFrame | None = st.session_state.get("draft_df")

    if draft_df is None:
        st.info(
            "🗂️ **No questionnaire data found.** "
            "Process a questionnaire on the main Assessor page first, "
            "then return here to resolve any gaps."
        )
        return

    _q_init(draft_df)
    queue = st.session_state.q_gap_queue

    if not queue:
        st.success(
            "✅ **No gaps to resolve** — all questionnaire rows have OK status. "
            "Download the completed questionnaire from the main Assessor page."
        )
        return

    total    = len(queue)
    resolved = sum(
        1 for r in queue
        if str(draft_df.at[r, "_AI_Status"]) == "OK"
    )
    all_done = resolved == total

    # ── Header: mirrors Tab A exactly ────────────────────────────────────────

    st.subheader("📋 Questionnaire Gap Resolution")
    st.caption(
        "Click a control in the list on the left to open its chat session. "
        "Provide evidence in the input below — completed controls are marked with ✅."
    )

    col_prog, col_metric = st.columns([4, 1])
    with col_prog:
        st.progress(
            resolved / total,
            text=f"Questionnaire gaps: **{resolved} / {total}** resolved",
        )
    with col_metric:
        st.metric("Remaining", total - resolved)

    dl_bytes = generate_excel(draft_df)
    st.download_button(
        label="⬇️ Download questionnaire (current state)",
        data=dl_bytes,
        file_name="questionnaire_in_progress.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if all_done:
        st.success(
            "✅ All gaps have been resolved! "
            "Download the completed questionnaire above."
        )
        return

    st.divider()

    active_row = st.session_state.q_active_row_idx

    # ── Two-column layout (identical structure to Tab A) ──────────────────────

    col_list, col_chat = st.columns([1, 2], gap="large")

    # ── Left column: interactive gap navigation list ──────────────────────────
    with col_list:
        st.subheader("📋 Flagged Controls")

        for i, row_idx in enumerate(queue):
            label      = _q_row_label(draft_df, row_idx)
            is_ok      = str(draft_df.at[row_idx, "_AI_Status"]) == "OK"
            is_active  = row_idx == active_row
            was_opened = row_idx in st.session_state.q_chat_histories

            if is_ok:
                # Mirrors Tab A's covered topics: ✅ strikethrough, not clickable
                st.markdown(f"✅ ~~{label}~~")
            elif is_active:
                # Mirrors Tab A's active topic: 💬 bold, not a button
                st.markdown(f"💬 **{label}**")
            else:
                # Mirrors Tab A's uncovered topics: clickable button
                btn_label = f"📋 {label}" if was_opened else f"⬜ {label}"
                if st.button(btn_label, key=f"q_nav_{row_idx}", use_container_width=True):
                    st.session_state.q_active_row_idx = row_idx
                    st.rerun()

    # ── Right column: per-gap chat session ───────────────────────────────────
    with col_chat:
        if active_row is None:
            st.info("Select a control from the list on the left to begin.")
            return

        gap_num = queue.index(active_row) + 1
        row     = draft_df.loc[active_row]
        id_key  = _detect_id_key(row.to_dict())

        domain_label = str(row.get(id_key, "")) if id_key else f"Row {active_row}"
        st.subheader(f"💬 {domain_label}")
        st.caption(f"**AI Assessment:** {'⚠️ REVIEW' if row.get('_AI_Status') == 'REVIEW' else '❌ NO_DATA'}")
        st.divider()

        # Lazily initialise this gap's chat history on first visit
        _q_ensure_opening_message(draft_df, active_row, gap_num, total)

        for msg in st.session_state.q_chat_histories[active_row]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        is_resolved = str(draft_df.at[active_row, "_AI_Status"]) == "OK"

        if not is_resolved:
            col_input, col_skip = st.columns([5, 1])
            with col_input:
                user_input = st.chat_input(
                    "Describe your organisation's approach…",
                    key=f"q_input_{active_row}",
                )
            with col_skip:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button(
                    "Skip →",
                    key=f"q_skip_{active_row}",
                    help="Skip to the next gap — you can come back to this one later.",
                ):
                    _q_handle_skip(active_row, draft_df, queue)
                    st.rerun()

            if user_input:
                _q_handle_response(user_input, active_row, draft_df, queue)
                st.rerun()


# ── Render tabs ───────────────────────────────────────────────────────────────

with tab_kb:
    _render_kb_tab()

with tab_q:
    _render_questionnaire_tab()
