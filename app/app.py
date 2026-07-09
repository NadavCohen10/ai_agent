import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Streamlit UI for the HITL Security Questionnaire Assistant.
Responsibilities: upload → batch processing → results dashboard → download.
Gap resolution is handled by the CISO Agent page.
"""

import math
import pandas as pd
import streamlit as st

from agents.assessor_agent import (
    answer_excel_rows_batch,
    extract_questions,
    answer_questions_in_batches,
    EXCEL_BATCH_SIZE,
)
from core.llm_provider import GeminiProvider, OpenAIProvider, AnthropicProvider, OllamaProvider, BaseLLMProvider
from ingestion.document_parser import (
    extract_kb_text,
    parse_questionnaire_pdf,
    load_questionnaire_excel_as_dataframe,
)
from app.exporter import generate_excel


st.set_page_config(page_title="HITL Security Questionnaire Assistant", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────

if "draft_df" not in st.session_state:
    st.session_state.draft_df = None
if "approved_df" not in st.session_state:
    st.session_state.approved_df = None
if "is_excel_pipeline" not in st.session_state:
    st.session_state.is_excel_pipeline = False
if "gemini_provider" not in st.session_state:
    st.session_state.gemini_provider = None


def _get_gemini_provider() -> GeminiProvider:
    if st.session_state.gemini_provider is None:
        st.session_state.gemini_provider = GeminiProvider()
    return st.session_state.gemini_provider


def _build_provider(provider_name: str, api_key: str, ollama_model: str = "gemma") -> BaseLLMProvider:
    if provider_name == "Gemini":
        return _get_gemini_provider()
    if provider_name == "OpenAI":
        return OpenAIProvider(api_key=api_key)
    if provider_name == "Ollama (Local)":
        return OllamaProvider(model=ollama_model)
    if provider_name == "Anthropic":
        return AnthropicProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {provider_name}")


def _status_counts(df: pd.DataFrame) -> tuple[int, int, int, int]:
    """Return (total, ok, review, no_data) from _AI_Status column."""
    total   = len(df)
    ok      = int((df["_AI_Status"] == "OK").sum())      if "_AI_Status" in df.columns else 0
    review  = int((df["_AI_Status"] == "REVIEW").sum())  if "_AI_Status" in df.columns else 0
    no_data = int((df["_AI_Status"] == "NO_DATA").sum()) if "_AI_Status" in df.columns else 0
    return total, ok, review, no_data


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar_inputs():
    with st.sidebar:
        st.header("Inputs")
        st.markdown("Upload your Knowledge Base and the blank questionnaire, then run analysis.")

        kb_file = st.file_uploader(
            "Knowledge Base (TXT, PDF, CSV, XLSX)",
            type=["txt", "pdf", "csv", "xlsx", "xls"],
            key="kb_uploader",
        )
        questionnaire_file = st.file_uploader(
            "Questionnaire (Excel or PDF)",
            type=["xlsx", "xls", "pdf"],
            key="questionnaire_uploader",
        )
        analyze_clicked = st.button("🚀 Analyze & Process Files", type="primary", width="stretch")

        st.divider()
        st.subheader("Model Settings")

        provider_name = st.selectbox(
            "LLM Provider",
            ["Gemini", "OpenAI", "Ollama (Local)", "Anthropic"],
            index=0,
        )

        use_advanced_prompt = st.checkbox(
            "Enable Advanced System Prompt (Strict Mode)",
            value=True,
            help=(
                "ON: uses detailed rules with Chain-of-Thought, categorical matching, "
                "and zero-inference constraints.\n"
                "OFF: uses a simple 'helpful assistant' prompt for comparison."
            ),
        )

        ollama_model = "gemma"
        if provider_name == "Ollama (Local)":
            ollama_model = st.text_input(
                "Ollama Model Tag",
                value="gemma",
                placeholder="gemma2, llama3, mistral …",
            )
            st.caption("Make sure Ollama is running: `ollama serve`")

        api_key_input = ""
        if provider_name in ("OpenAI", "Anthropic"):
            import os
            from dotenv import load_dotenv
            load_dotenv()
            env_key_names = {
                "OpenAI": ("CHATGPT_API_KEY", "OPENAI_API_KEY"),
                "Anthropic": ("ANTHROPIC_API_KEY",),
            }
            has_env_key = any(os.getenv(k) for k in env_key_names.get(provider_name, ()))
            if has_env_key:
                st.info(f"{provider_name} API key loaded from .env ✓")
            else:
                api_key_input = st.text_input(
                    f"{provider_name} API Key",
                    type="password",
                    placeholder="sk-..." if provider_name == "OpenAI" else "sk-ant-...",
                )
                if not api_key_input:
                    st.warning(f"Enter your {provider_name} API key above.")

    return kb_file, questionnaire_file, analyze_clicked, provider_name, api_key_input, ollama_model, use_advanced_prompt


# ── Dashboard renderer (Excel pipeline) ──────────────────────────────────────

def _render_excel_dashboard():
    df = st.session_state.draft_df
    total, ok, review, no_data = _status_counts(df)

    st.subheader("📊 Processing Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",  total)
    c2.metric("✅ OK",        ok)
    c3.metric("⚠️ REVIEW",   review)
    c4.metric("❌ NO_DATA",   no_data)

    with st.expander("View full results table", expanded=False):
        _ai_cols = [c for c in ["_AI_Status", "_AI_Reasoning"] if c in df.columns]
        st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            height=450,
            column_config={
                "_AI_Status":    st.column_config.TextColumn("AI Status",    disabled=True),
                "_AI_Reasoning": st.column_config.TextColumn("AI Reasoning", disabled=True, width="large"),
            },
            disabled=_ai_cols,
        )

    st.divider()

    gap_count = review + no_data
    if gap_count > 0:
        st.warning(
            f"**{gap_count} row(s)** could not be fully filled — they are marked REVIEW or NO_DATA."
        )
        if st.button(
            f"💬 Complete {gap_count} gap(s) with CISO Agent →",
            type="primary",
        ):
            st.switch_page("pages/2_ciso_interviewer.py")
    else:
        st.success("✅ All rows processed with OK status — ready for download.")

    excel_bytes = generate_excel(df)
    st.download_button(
        label="⬇️ Download Completed Questionnaire (Excel)",
        data=excel_bytes,
        file_name="completed_questionnaire.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── PDF pipeline renderer (legacy tab UI) ─────────────────────────────────────

def _render_pdf_tabs():
    df = st.session_state.draft_df

    total  = len(df)
    review = (df["Status"] == "⚠️ REVIEW").sum()
    ready  = (df["Status"] == "✅ OK").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Questions",  total)
    c2.metric("Ready for Export", ready)
    c3.metric("Requires Review",  review)

    tab_review, tab_export = st.tabs(["1. Review & Edit", "2. Export Options"])

    with tab_review:
        st.subheader("AI Draft Responses")
        edited_df = st.data_editor(
            df,
            hide_index=True,
            width="stretch",
            height=600,
            column_order=[
                "question_id", "question_text", "proposed_yes_no",
                "proposed_comments", "Status", "confidence_level", "reasoning",
            ],
            column_config={
                "flag_for_human_review": None,
                "question_text": st.column_config.TextColumn("question_text", disabled=True, width="large"),
                "Status":        st.column_config.TextColumn("Status",        disabled=True),
                "reasoning":     st.column_config.TextColumn("reasoning",     disabled=True, width="large"),
                "proposed_yes_no":   st.column_config.TextColumn("proposed_yes_no",   help="Edit Yes/No/N/A"),
                "proposed_comments": st.column_config.TextColumn("proposed_comments", help="Edit the comments"),
            },
            disabled=["question_text", "Status", "reasoning", "confidence_level", "question_id"],
        )
        if st.button("Approve Final Answers", type="primary"):
            st.session_state.approved_df = edited_df
            st.rerun()

    with tab_export:
        if st.session_state.approved_df is not None:
            st.success("Answers approved and saved! Ready for export.")
            excel_bytes = generate_excel(st.session_state.approved_df)
            st.download_button(
                label="Download Completed Questionnaire (Excel)",
                data=excel_bytes,
                file_name="completed_questionnaire.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Approve the answers in the Review tab to enable export.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    (
        kb_file,
        questionnaire_file,
        analyze_clicked,
        provider_name,
        api_key_input,
        ollama_model,
        use_advanced_prompt,
    ) = sidebar_inputs()

    st.title("HITL Security Questionnaire Assistant")
    st.caption("Automate, review, and export vendor security questionnaire answers with confidence.")

    status_placeholder = st.empty()

    # ── Batch processing ──────────────────────────────────────────────────────
    if kb_file and questionnaire_file and analyze_clicked:
        try:
            st.session_state.approved_df = None   # clear previous approval on new run
            provider = _build_provider(provider_name, api_key_input, ollama_model)
            kb_text  = extract_kb_text(kb_file)
            is_excel = questionnaire_file.name.lower().endswith((".xlsx", ".xls"))

            # ── Excel pipeline ─────────────────────────────────────────────────
            if is_excel:
                status_placeholder.info("Loading Excel questionnaire…")
                progress = st.progress(0, text="Loading Excel questionnaire…")
                questionnaire_df = load_questionnaire_excel_as_dataframe(questionnaire_file)
                if questionnaire_df.empty:
                    st.error("No data found in the Excel file.")
                    status_placeholder.empty()
                    return

                original_columns = questionnaire_df.columns.tolist()
                rows_list     = questionnaire_df.to_dict(orient="records")
                total_rows    = len(rows_list)
                total_batches = math.ceil(total_rows / EXCEL_BATCH_SIZE)

                progress.progress(0.1, text=f"Processing {total_rows} rows in {total_batches} batch(es)…")
                master_results: list = []
                for b_idx in range(total_batches):
                    batch = rows_list[b_idx * EXCEL_BATCH_SIZE : (b_idx + 1) * EXCEL_BATCH_SIZE]
                    filled_batch = answer_excel_rows_batch(
                        kb_text, original_columns, batch, provider, use_advanced_prompt
                    )
                    master_results.extend(filled_batch)
                    progress.progress(
                        0.1 + 0.9 * (b_idx + 1) / total_batches,
                        text=f"Batch {b_idx + 1} of {total_batches} done.",
                    )

                result_df = pd.DataFrame(master_results)
                for col in original_columns:
                    if col not in result_df.columns:
                        result_df[col] = ""
                ai_cols   = [c for c in result_df.columns if c not in original_columns]
                result_df = result_df[original_columns + ai_cols]

                progress.progress(1.0, text="Done.")
                status_placeholder.empty()
                st.session_state.is_excel_pipeline = True
                st.session_state.draft_df          = result_df
                st.rerun()

            # ── PDF pipeline ───────────────────────────────────────────────────
            else:
                questionnaire_text = ""
                if questionnaire_file.name.lower().endswith(".pdf"):
                    questionnaire_text = parse_questionnaire_pdf(questionnaire_file)
                else:
                    st.warning("Unsupported questionnaire format. Use PDF or Excel.")

                if questionnaire_text:
                    status_placeholder.info("Extracting questions…")
                    progress = st.progress(0, text="Extracting questions…")

                    extracted     = extract_questions(questionnaire_text, provider)
                    total_q       = len(extracted)
                    if total_q == 0:
                        st.error("No questions extracted; cannot proceed.")
                        status_placeholder.empty()
                        return

                    batch_size    = 15
                    total_batches = math.ceil(total_q / batch_size)
                    master_answers = []
                    for i in range(total_batches):
                        batch = extracted[i * batch_size : (i + 1) * batch_size]
                        progress.progress(i / total_batches, text=f"Processing batch {i + 1} of {total_batches}…")
                        answers = answer_questions_in_batches(kb_text, batch, provider, use_advanced_prompt)
                        master_answers.extend(answers)

                    progress.progress(1.0, text="All batches processed.")
                    status_placeholder.empty()

                    df = pd.DataFrame(master_answers)
                    if not df.empty:
                        df["Status"] = df.apply(
                            lambda row: "⚠️ REVIEW"
                            if row.get("flag_for_human_review")
                            or str(row.get("confidence_level", "")).lower() == "low"
                            else "✅ OK",
                            axis=1,
                        )
                        st.session_state.is_excel_pipeline = False
                        st.session_state.draft_df          = df
                        st.rerun()
                    else:
                        st.info("No AI results to display.")
                else:
                    st.error("No questionnaire text extracted; unable to analyze.")
                    status_placeholder.empty()

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            status_placeholder.empty()

    # ── Render results ────────────────────────────────────────────────────────
    if st.session_state.is_excel_pipeline and st.session_state.draft_df is not None:
        _render_excel_dashboard()
    elif not st.session_state.is_excel_pipeline and st.session_state.draft_df is not None:
        _render_pdf_tabs()


if __name__ == "__main__":
    main()
