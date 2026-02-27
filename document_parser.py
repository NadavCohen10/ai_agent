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
    Supports TXT, PDF, and CSV.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        return _read_txt(uploaded_file)
    if filename.endswith(".pdf"):
        return _read_pdf(uploaded_file)
    if filename.endswith(".csv"):
        return _read_csv(uploaded_file)
    raise ValueError("Unsupported Knowledge Base format. Use TXT, PDF, or CSV.")


def parse_questionnaire_pdf(uploaded_file) -> str:
    """Extract raw text from a Questionnaire PDF."""
    return _read_pdf(uploaded_file)


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
    # Represent as a readable table string
    table_str = df.to_string(index=False)
    return table_str
