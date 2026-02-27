"""
Excel export utilities for the HITL Security Questionnaire Assistant.
"""

from io import BytesIO
import pandas as pd


FINAL_COLUMNS = ["question_id", "question_text", "proposed_yes_no", "proposed_comments"]


def generate_excel(df: pd.DataFrame) -> BytesIO:
    """
    Generate an in-memory Excel file with only the client-facing columns.
    Returns a BytesIO ready for download.
    """
    output = BytesIO()
    export_df = df.loc[:, [col for col in FINAL_COLUMNS if col in df.columns]]
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Questionnaire")
    output.seek(0)
    return output
