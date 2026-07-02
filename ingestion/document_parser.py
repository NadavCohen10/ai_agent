"""
Document parsing utilities for the HITL Security Questionnaire Assistant.

Step 2 focuses on extracting text from the Knowledge Base and Questionnaire.
"""

from io import BytesIO
from typing import Union

import pandas as pd
import pdfplumber


def extract_kb_text(uploaded_file) -> str:
    """
    Extract plain text from the Knowledge Base upload.
    Supports TXT, PDF, CSV, XLSX, and XLS.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        return _read_txt(uploaded_file)
    if filename.endswith(".pdf"):
        return _read_pdf(uploaded_file)
    if filename.endswith(".csv"):
        return _read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return _read_excel(uploaded_file)
    raise ValueError("Unsupported Knowledge Base format. Use TXT, PDF, CSV, XLSX, or XLS.")


def parse_questionnaire_pdf(uploaded_file) -> str:
    """Extract raw text from a Questionnaire PDF."""
    return _read_pdf(uploaded_file)


def load_questionnaire_excel_as_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Load a questionnaire Excel file directly into a DataFrame, preserving ALL original
    columns — including completely empty ones (they are the answer columns the AI fills).
    Combines all sheets into one DataFrame.
    """
    uploaded_file.seek(0)
    xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
    frames = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        # Drop fully-empty ROWS only — never drop columns
        df.dropna(how="all", inplace=True)
        if df.empty:
            continue
        if len(xl.sheet_names) > 1:
            df.insert(0, "_Sheet", sheet_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Replace NaN with empty string so JSON serialisation works cleanly
    combined = combined.fillna("")
    return combined


def parse_questionnaire_excel(uploaded_file) -> str:
    """
    Extract raw text from a Questionnaire Excel file (.xlsx / .xls).
    Reads ALL sheets with NO header row (header=None) so the first row is
    treated as data, not swallowed as column names.
    Drops fully-empty rows/columns, then converts each sheet to CSV-formatted text.
    """
    uploaded_file.seek(0)
    xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
    parts = []
    for sheet_name in xl.sheet_names:
        # header=None → every row is data; no row is silently consumed as a header
        df = xl.parse(sheet_name, header=None)
        df.dropna(how="all", inplace=True)        # drop fully-empty rows
        df.dropna(axis=1, how="all", inplace=True) # drop fully-empty columns
        if df.empty:
            continue
        csv_str = df.to_csv(index=False, header=False)
        parts.append(f"--- Sheet: {sheet_name} ---\n{csv_str}")

    final_text = "\n\n".join(parts)

    print("=== DEBUG: EXACT TEXT EXTRACTED FROM EXCEL ===")
    print(final_text)
    print("=== END DEBUG ===")

    return final_text


def _read_txt(uploaded_file) -> str:
    uploaded_file.seek(0)
    data = uploaded_file.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return str(data)


def _read_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    # Ensure pdfplumber gets a binary stream
    binary_stream = uploaded_file
    if not hasattr(uploaded_file, "read"):
        binary_stream = BytesIO(uploaded_file)
    text_chunks = []
    with pdfplumber.open(binary_stream) as pdf:
        for page in pdf.pages:
            text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def _read_csv(uploaded_file) -> str:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    table_str = df.to_string(index=False)
    return table_str


def _read_excel(uploaded_file) -> str:
    """Read all sheets of an Excel KB file and return as structured text."""
    uploaded_file.seek(0)
    xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
    parts = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        if df.empty:
            continue
        try:
            table_str = df.to_markdown(index=False)
        except ImportError:
            table_str = df.to_csv(index=False)
        parts.append(f"## Sheet: {sheet_name}\n\n{table_str}")
    return "\n\n".join(parts)
