"""
Excel export utilities for the HITL Security Questionnaire Assistant.
"""

from io import BytesIO
import pandas as pd


# UI-only columns that should never appear in the exported file
_UI_COLUMNS = {"Status", "flag_for_human_review", "_AI_Status", "_AI_Reasoning"}


def generate_excel(df: pd.DataFrame) -> BytesIO:
    """
    Generate an in-memory Excel file from whatever DataFrame is passed in.
    Drops internal UI columns (Status, flag_for_human_review) if present,
    then writes all remaining columns dynamically.
    Returns a BytesIO ready for download.
    """
    output = BytesIO()
    export_df = df.drop(columns=[c for c in _UI_COLUMNS if c in df.columns])
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Questionnaire")
    output.seek(0)
    return output
