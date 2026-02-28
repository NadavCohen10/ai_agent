"""
Streamlit UI for the HITL Security Questionnaire Assistant.
Polished SaaS-style dashboard with sidebar inputs, metrics, tabs, and stateful flow.
"""

import math
import pandas as pd
import streamlit as st

from ai_agent import analyze_questionnaire
from document_parser import extract_kb_text, parse_questionnaire_pdf
from exporter import generate_excel


st.set_page_config(page_title="HITL Security Questionnaire Assistant", layout="wide")

# Initialize session state
if "draft_df" not in st.session_state:
    st.session_state.draft_df = None
if "approved_df" not in st.session_state:
    st.session_state.approved_df = None


def sidebar_inputs():
    with st.sidebar:
        st.header("📂 Inputs")
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
        analyze_clicked = st.button("🚀 Analyze & Process Files", type="primary", use_container_width=True)
            # ---------------------------------------------------------
        # כפתור זמני למפתחים - לבדיקת עיצוב הממשק בשנייה אחת!
        # ---------------------------------------------------------
        if st.sidebar.button("🧪 טען נתוני בדיקה (לעיצוב ה-UI)"):
            import pandas as pd
            dummy_data = [
                {"question_id": "Q1", "question_text": "האם המערכת תומכת ב-SSO?", "proposed_yes_no": "Yes", "proposed_comments": "תומכים ב-Okta ו-SAML.", "confidence_level": "High", "reasoning": "נמצא במאגר", "Status": "✅ OK"},
                {"question_id": "Q2", "question_text": "האם יש תרשים זרימה לנתונים?", "proposed_yes_no": "No", "proposed_comments": "אין מידע זמין במאגר.", "confidence_level": "Low", "reasoning": "לא נמצא אזכור", "Status": "⚠️ REVIEW"},
                {"question_id": "Q3", "question_text": "האם הנתונים מוצפנים במנוחה?", "proposed_yes_no": "Yes", "proposed_comments": "מוצפן ב-AES-256.", "confidence_level": "High", "reasoning": "כתוב במפורש במסמך האבטחה", "Status": "✅ OK"}
            ]
            st.session_state.draft_df = pd.DataFrame(dummy_data)
            st.rerun()
        # ---------------------------------------------------------
    return kb_file, questionnaire_file, analyze_clicked


def render_header():
    st.title("HITL Security Questionnaire Assistant")
    st.caption("Automate, review, and export vendor security questionnaire answers with confidence.")


def render_metrics(df: pd.DataFrame):
    total = len(df)
    review = (df["Status"] == "⚠️ REVIEW").sum()
    ready = (df["Status"] == "✅ OK").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Questions", f"{total}")
    c2.metric("Ready for Export", f"{ready}")
    c3.metric("Requires Review", f"{review}")


def main():
    kb_file, questionnaire_file, analyze_clicked = sidebar_inputs()
    render_header()

    status_placeholder = st.empty()

    # Step 1 - Analysis
    if kb_file and questionnaire_file and analyze_clicked:
        try:
            st.session_state.approved_df = None  # reset approvals on new run
            kb_text = extract_kb_text(kb_file)
            questionnaire_text = ""
            if questionnaire_file.name.lower().endswith(".pdf"):
                questionnaire_text = parse_questionnaire_pdf(questionnaire_file)
            else:
                st.warning("Questionnaire parsing is PDF-only in this step.")

            if questionnaire_text:
                status_placeholder.info("Extracting questions...")
                progress = st.progress(0, text="Extracting questions...")
                from ai_agent import extract_questions, answer_questions_in_batches

                extracted = extract_questions(questionnaire_text)
                total_questions = len(extracted)
                if total_questions == 0:
                    st.error("No questions extracted; cannot proceed.")
                    status_placeholder.empty()
                    return

                batch_size = 15
                total_batches = math.ceil(total_questions / batch_size)
                master_answers = []
                for idx in range(total_batches):
                    batch = extracted[idx * batch_size : (idx + 1) * batch_size]
                    progress.progress(
                        idx / total_batches,
                        text=f"Processing batch {idx + 1} of {total_batches}...",
                    )
                    answers = answer_questions_in_batches(kb_text, batch)
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
                    st.session_state.draft_df = df
                    st.rerun()
                else:
                    st.info("No AI results to display.")
            else:
                st.error("No questionnaire text extracted; unable to analyze.")
                status_placeholder.empty()
        except Exception as exc:  # broad for UI friendliness
            st.error(f"Processing failed: {exc}")
            status_placeholder.empty()

    # Step 2/3 - Review & Export (Tabs)
    if st.session_state.draft_df is not None:
        df = st.session_state.draft_df
        render_metrics(df)

        tab_review, tab_export = st.tabs(["1. Review & Edit", "2. Export Options"])

        with tab_review:
            st.subheader("AI Draft Responses")
            edited_df = st.data_editor(
                df,
                hide_index=True,
                use_container_width=True,
                height=600,
                column_order=[
                    "question_id",
                    "question_text",
                    "proposed_yes_no",
                    "proposed_comments",
                    "Status",
                    "confidence_level",
                    "reasoning",
                ],
                column_config={
                    "flag_for_human_review": None,  # hide column
                    "question_text": st.column_config.TextColumn(
                        "question_text", disabled=True, width="large"
                    ),
                    "Status": st.column_config.TextColumn("Status", disabled=True),
                    "reasoning": st.column_config.TextColumn(
                        "reasoning", disabled=True, width="large"
                    ),
                    "proposed_yes_no": st.column_config.TextColumn(
                        "proposed_yes_no", help="Edit Yes/No/N/A"
                    ),
                    "proposed_comments": st.column_config.TextColumn(
                        "proposed_comments", help="Edit the comments"
                    ),
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


if __name__ == "__main__":
    main()
