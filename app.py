"""
Streamlit UI for the HITL Security Questionnaire Assistant.
Stateful flow:
1) Analyze & Process -> saves draft_df then reruns.
2) Review editable table -> approve saves approved_df then reruns.
3) Export approved_df to Excel.
"""

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


def sidebar_uploads():
    """Render sidebar for uploading KB and target questionnaire."""
    st.sidebar.header("Uploads")
    kb_file = st.sidebar.file_uploader(
        "Knowledge Base document (TXT, PDF, CSV, XLSX)",
        type=["txt", "pdf", "csv", "xlsx", "xls"],
        key="kb_uploader",
    )
    questionnaire_file = st.sidebar.file_uploader(
        "Target Questionnaire (Excel or PDF)",
        type=["xlsx", "xls", "pdf"],
        key="questionnaire_uploader",
    )
    return kb_file, questionnaire_file


def main_area(kb_file, questionnaire_file):
    st.title("Human-in-the-Loop Security Questionnaire Assistant")
    st.markdown(
        "Upload your internal Knowledge Base and the blank client questionnaire on the left.\n"
        "We'll parse the documents, generate draft answers with Gemini, and let you review them."
    )
    status_box = st.empty()

    # Step 1 - Analysis
    if kb_file and questionnaire_file:
        if st.button("Analyze & Process Files", type="primary"):
            try:
                st.session_state.approved_df = None  # reset approvals on new run
                kb_text = extract_kb_text(kb_file)
                questionnaire_text = ""
                if questionnaire_file.name.lower().endswith(".pdf"):
                    questionnaire_text = parse_questionnaire_pdf(questionnaire_file)
                else:
                    st.warning("Questionnaire parsing is PDF-only in this step.")

                if questionnaire_text:
                    with st.spinner(
                        "Gemini is analyzing the questionnaire... this might take a minute or two"
                    ):
                        ai_results = analyze_questionnaire(
                            kb_text=kb_text, questionnaire_text=questionnaire_text
                        )
                    display_data = ai_results
                    if hasattr(ai_results, "model_dump"):
                        display_data = ai_results.model_dump()
                    elif hasattr(ai_results, "root"):
                        display_data = ai_results.root

                    df = pd.DataFrame(display_data)
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
            except Exception as exc:  # broad for UI friendliness
                st.error(f"Processing failed: {exc}")
        else:
            status_box.info("Click 'Analyze & Process Files' to begin parsing.")
    else:
        status_box.info("Waiting for both uploads…")

    # Step 2 - Review (outside button block)
    if st.session_state.draft_df is not None and st.session_state.approved_df is None:
        st.subheader("AI Draft Responses (editable)")
        edited_df = st.data_editor(
            st.session_state.draft_df,
            hide_index=True,
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

    # Step 3 - Export (outside button block)
    if st.session_state.approved_df is not None:
        st.success("Answers approved and saved! Ready for export.")
        excel_bytes = generate_excel(st.session_state.approved_df)
        st.download_button(
            label="Download Completed Questionnaire (Excel)",
            data=excel_bytes,
            file_name="completed_questionnaire.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main():
    kb_file, questionnaire_file = sidebar_uploads()
    main_area(kb_file, questionnaire_file)

    # Sidebar status echoes
    if kb_file:
        st.sidebar.success(f"Loaded KB: {kb_file.name}")
    if questionnaire_file:
        st.sidebar.success(f"Loaded Questionnaire: {questionnaire_file.name}")


if __name__ == "__main__":
    main()
