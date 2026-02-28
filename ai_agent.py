"""
Gemini integration with two-stage batching:
1) Extract questions from the questionnaire text (chunked).
2) Answer questions in batches to avoid token limits.
Includes cleaning to handle noisy PDF extraction artifacts.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Iterable, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, RootModel
from key_provider import get_active_key, setup_and_get_key, rotate_key


load_dotenv()
current_api_key = setup_and_get_key()

API_KEY = get_active_key() #os.getenv("GEMINI_API_KEY")
if API_KEY:
    os.environ.setdefault("GOOGLE_API_KEY", API_KEY)

client = genai.Client()


# Schemas
class Answer(BaseModel):
    question_id: str = Field(..., description="Identifier from the questionnaire")
    question_text: str = Field(..., description="Original question text")
    proposed_yes_no: str = Field(..., description="One of: Yes, No, N/A")
    proposed_comments: str = Field(..., description="Explanation grounded in KB")
    confidence_level: str = Field(..., description="High or Low")
    reasoning: str = Field(..., description="Internal reasoning; cite missing info if any")
    flag_for_human_review: bool = Field(..., description="True if confidence is Low or data missing")


class QuestionnaireResponse(RootModel[List[Answer]]):
    root: List[Answer]


class ExtractedQuestion(BaseModel):
    question_id: str
    question_text: str


class ExtractionList(BaseModel):
    questions: List[ExtractedQuestion]


SYSTEM_INSTRUCTION_ANSWER = (
    "You are an Expert IT Security Officer. Complete the Vendor Security Questionnaire "
    "based ONLY on the provided Knowledge Base.\n"
    "Apply 'Semantic Bridging': e.g., if the KB says 'CrowdStrike', recognize it as 'Antivirus'. "
    "If it says 'AWS Security Groups', recognize it as a firewall.\n"
    "If information is missing, DO NOT guess. State 'No information available' and set confidence to Low.\n"
    "You must analyze the entire questionnaire and provide an answer for EVERY SINGLE question provided to you.\n"
    "Your output MUST be a JSON array of objects with the following schema: "
    "question_id (string), question_text (string), proposed_yes_no (string: Yes/No/N/A), "
    "proposed_comments (string), confidence_level (string: High/Low), reasoning (string), "
    "flag_for_human_review (boolean)."
)

SYSTEM_INSTRUCTION_EXTRACT = (
    "You are an expert document parser. Extract every single question from the provided questionnaire text chunk. "
    "Ignore general text, instructions, or headers. Return an exhaustive JSON list of questions with their IDs and text. "
    "If you do not find any specific questions in this exact text block (e.g., it is just a cover page, generic instructions, or a table of contents), DO NOT invent questions. Simply return an empty list [] for the questions array. "
    "Use fields: question_id (string), question_text (string). Preserve original order; if no IDs, generate sequential IDs like Q1, Q2, ..."
)


def clean_json_string(raw_string: str) -> str:
    """Remove markdown fences and trim whitespace for safer JSON parsing."""
    if raw_string is None:
        return ""
    cleaned = raw_string.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def clean_extracted_text(text: str) -> str:
    """Reduce noisy newlines and spaces from PDF extraction artifacts."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    return text.strip()


def get_smart_chunks(text, max_chars=4000):
    chunks = []
    while len(text) > 0:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        
        # Find the last question mark or period before the limit
        split_index_q = text.rfind('? ', 0, max_chars)
        split_index_p = text.rfind('. ', 0, max_chars)
        split_index = max(split_index_q, split_index_p)
        
        # If no punctuation found, fall back to the last space
        if split_index == -1:
            split_index = text.rfind(' ', 0, max_chars)
            
        # If still no space (huge block of text), force split
        if split_index == -1:
            split_index = max_chars
        else:
            # Include the space/punctuation in the current chunk
            split_index += 1
            
        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()
        
    return chunks


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


def _chunk_list(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_questions(questionnaire_text: str) -> List[dict]:
    """Stage 1: Extract all questions as a list of {question_id, question_text}."""
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    cleaned_text = clean_extracted_text(questionnaire_text)
    text_chunks = get_smart_chunks(cleaned_text, max_chars=4000)
    all_extracted_questions: List[dict] = []

    if not text_chunks:
        print("[Extraction] No text to process.")
        return []

    for idx, chunk in enumerate(text_chunks, start=1):
        chunk_prompt = (
            "Extract every single question from the provided questionnaire text chunk. "
            "Ignore general text, instructions, or headers. Return an exhaustive JSON list of questions with their IDs and text.\n"
            "If you do not find any specific questions in this exact text block (e.g., it is just a cover page, generic instructions, or a table of contents), DO NOT invent questions. Simply return an empty list [] for the questions array.\n"
            "Questionnaire Chunk:\n"
            f"{chunk}\n\n"
            "Return only the JSON array."
        )
        print(f"[Extraction] Processing chunk {idx}/{len(text_chunks)} with ~{len(chunk)} chars")
        chunk_success = False
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=chunk_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION_EXTRACT,
                        response_mime_type="application/json",
                        response_schema=ExtractionList,
                        max_output_tokens=8192,
                        temperature=0.1,
                    ),
                )

                raw_text = getattr(response, "text", "") or ""
                print(f"[Gemini extraction raw response chunk {idx} attempt {attempt+1}]\n", raw_text)

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    if hasattr(parsed, "questions"):
                        chunk_questions = [
                            q.model_dump() if hasattr(q, "model_dump") else q for q in parsed.questions
                        ]
                    elif hasattr(parsed, "model_dump"):
                        data = parsed.model_dump()
                        chunk_questions = data.get("questions", data)
                    else:
                        chunk_questions = parsed
                else:
                    cleaned = clean_json_string(raw_text)
                    chunk_questions = json.loads(cleaned)

                if isinstance(chunk_questions, dict):
                    chunk_questions = [chunk_questions]

                all_extracted_questions.extend(chunk_questions)
                chunk_success = True
                break  # exit retry loop on success
            except Exception as exc:
                print(f"[Extraction] Chunk {idx} failed on attempt {attempt+1}: {exc}")
                time.sleep(3)

        if not chunk_success:
            print(f"[Extraction] Skipping chunk {idx} after 3 failed attempts.")
            continue

        time.sleep(2)  # gentle pacing to avoid rate limits

    print(f"[Extraction] Total extracted questions: {len(all_extracted_questions)}")
    return all_extracted_questions


def answer_questions_in_batches(kb_text: str, extracted_questions: List[dict]) -> List[dict]:
    """Stage 2: Answer questions in batches to avoid token limits."""
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    if not extracted_questions:
        return _error_response("No questions extracted")

    # Normalize to list of dicts
    normalized_questions = []
    for q in extracted_questions:
        if hasattr(q, "model_dump"):
            normalized_questions.append(q.model_dump())
        else:
            normalized_questions.append(q)

    master_answers: List[dict] = []
    batch_size = 15
    total_batches = math.ceil(len(normalized_questions) / batch_size)

    for batch_index, batch in enumerate(_chunk_list(normalized_questions, batch_size), start=1):
        batch_json = json.dumps(batch, ensure_ascii=False)
        prompt = (
            "Answer ONLY the questions provided below using the Knowledge Base. "
            "Do not add new questions and do not omit any provided questions.\n"
            "Knowledge Base:\n"
            f"{kb_text}\n\n"
            "Questions (JSON):\n"
            f"{batch_json}\n\n"
            "Return only the JSON array of answers."
        )

        answers = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION_ANSWER,
                        response_mime_type="application/json",
                        response_schema=QuestionnaireResponse,
                        max_output_tokens=8192,
                        temperature=0.1,
                    ),
                )

                raw_text = getattr(response, "text", "") or ""
                print(f"[Gemini answer batch {batch_index}/{total_batches} raw response attempt {attempt+1}]\n", raw_text)

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    if hasattr(parsed, "root"):
                        answers = [
                            item.model_dump() if hasattr(item, "model_dump") else item for item in parsed.root
                        ]
                    elif hasattr(parsed, "model_dump"):
                        answers = parsed.model_dump()
                    else:
                        answers = parsed
                else:
                    cleaned = clean_json_string(raw_text)
                    answers = json.loads(cleaned)

                time.sleep(5)  # pacing between successful batches
                break  # success
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    rotate_key()
                    print("[Rate Limit] Hit 429/RESOURCE_EXHAUSTED. Waiting 60s before retrying...")
                    time.sleep(60)
                else:
                    print(f"[Answering] Error on attempt {attempt+1}: {e}")
                    time.sleep(5)
                answers = None

        if answers is None:
            answers = _error_response("Batch failed after retries")

        if isinstance(answers, dict):
            answers = [answers]
        master_answers.extend(answers)

    return master_answers


def analyze_questionnaire(kb_text: str, questionnaire_text: str) -> List[dict]:
    """Main orchestrator: extract questions, then answer them in batches."""
    extracted = extract_questions(questionnaire_text)
    return answer_questions_in_batches(kb_text, extracted)
