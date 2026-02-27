"""
Gemini integration for analyzing security questionnaires.

Uses the new `google-genai` SDK with a strict JSON schema enforced via Pydantic.
"""

from __future__ import annotations

import json
import os
from typing import Any, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, RootModel


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure the new SDK sees the key (uses GOOGLE_API_KEY by default)
if API_KEY:
    os.environ.setdefault("GOOGLE_API_KEY", API_KEY)

client = genai.Client()


class Answer(BaseModel):
    question_id: str = Field(..., description="Identifier from the questionnaire")
    question_text: str = Field(..., description="Original question text")
    proposed_yes_no: str = Field(..., description="One of: Yes, No, N/A")
    proposed_comments: str = Field(..., description="Explanation grounded in KB")
    confidence_level: str = Field(..., description="High or Low")
    reasoning: str = Field(..., description="Internal reasoning; cite missing info if any")
    flag_for_human_review: bool = Field(..., description="True if confidence is Low or data missing")


class QuestionnaireResponse(RootModel[List[Answer]]):
    """Root model so the output is a JSON array."""

    root: List[Answer]


SYSTEM_INSTRUCTION = (
    "You are an Expert IT Security Officer. Complete the Vendor Security Questionnaire "
    "based ONLY on the provided Knowledge Base.\n"
    "Apply 'Semantic Bridging': e.g., if the KB says 'CrowdStrike', recognize it as 'Antivirus'. "
    "If it says 'AWS Security Groups', recognize it as a firewall.\n"
    "If information is missing, DO NOT guess. State 'No information available' and set confidence to Low.\n"
    "For this test run, please analyze and return ONLY the first 5 questions you find in the questionnaire. Ignore the rest.\n"
    "Your output MUST be a JSON array of objects with the following schema: "
    "question_id (string), question_text (string), proposed_yes_no (string: Yes/No/N/A), "
    "proposed_comments (string), confidence_level (string: High/Low), reasoning (string), "
    "flag_for_human_review (boolean)."
)


def clean_json_string(raw_string: str) -> str:
    """Remove markdown fences and trim whitespace for safer JSON parsing."""
    if raw_string is None:
        return ""
    cleaned = raw_string.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def _error_response(raw_string: str):
    """Return a safe error payload to avoid UI crashes."""
    snippet = (raw_string or "")[:200]
    return [
        {
            "question_id": "Error",
            "question_text": "Failed to parse AI response",
            "proposed_yes_no": "N/A",
            "proposed_comments": snippet,
            "confidence_level": "Low",
            "reasoning": "JSON Parse Error",
            "flag_for_human_review": True,
        }
    ]


def analyze_questionnaire(kb_text: str, questionnaire_text: str) -> Any:
    """
    Send KB and questionnaire text to Gemini and return a Python object (list of answers).
    Handles both parsed Pydantic responses and raw JSON fallbacks.
    """
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    user_prompt = (
        "Complete the questionnaire strictly from the Knowledge Base.\n"
        "Knowledge Base:\n"
        f"{kb_text}\n\n"
        "Questionnaire Text:\n"
        f"{questionnaire_text}\n\n"
        "Return only the JSON array."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=QuestionnaireResponse,
            max_output_tokens=8192,
            temperature=0.1,
        ),
    )

    raw_text = getattr(response, "text", "") or ""
    # Debug logging for troubleshooting truncated/markdown responses
    print("[Gemini raw response]\n", raw_text)

    # Try parsed object first
    try:
        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            # If RootModel, take .root; if BaseModel, dump to dict; otherwise return as-is.
            if hasattr(parsed, "root"):
                root = parsed.root
                return [
                    item.model_dump() if hasattr(item, "model_dump") else item for item in root
                ]
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
            return parsed
    except Exception:
        pass

    # Fallback: clean markdown fences then parse JSON text
    cleaned = clean_json_string(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        return _error_response(cleaned)
