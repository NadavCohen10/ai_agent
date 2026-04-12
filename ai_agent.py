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

SYSTEM_INSTRUCTION_EXCEL = (
    "You are an expert AI Data Entry Assistant specializing in enterprise security and compliance questionnaires. "
    "Your task is to analyze a batch of rows from a spreadsheet and fill in the missing empty columns based strictly "
    "on the provided Knowledge Base (KB) text.\n\n"

    "You must rigorously adhere to the following operational constraints:\n\n"

    "1. DYNAMIC ANCHOR DETECTION:\n"
    "Do not rely on hardcoded column names to find the subject. Analyze all provided key-value pairs in a given row "
    "to dynamically deduce the core entity, system, or control being queried. Use this inferred subject to search the KB.\n\n"

    "2. STRICT CATEGORICAL MATCHING (DATA VALIDATION):\n"
    "Analyze the column headers and any placeholder text. If a cell or header implicitly represents a predefined list "
    "of options (e.g., statuses separated by slashes '/', commas, pipes '|', or instructions like \"Choose one\"), "
    "you MUST treat this as a strict dropdown menu. You are forbidden from generating free text for that column. "
    "Map the KB data to EXACTLY one of the explicitly allowed options.\n\n"

    "3. ZERO INFERENCE FOR MISSING DATA:\n"
    "If you are unsure, assume the data does not exist. Never invent data to be \"helpful\". Never write generic "
    "placeholder phrases such as \"No information available\", \"N/A\", or \"Not found\". If the answer is not "
    "explicitly in the KB, the generated value must be an empty string \"\".\n\n"

    "4. CONFIDENCE TRACKING:\n"
    "Add one special key named `_AI_Status` to your JSON output for each row. Set its value to \"OK\" if you "
    "successfully mapped KB data to the row. Set it to \"REVIEW\" ONLY if the anchor subject itself is completely "
    "missing from the KB.\n\n"

    "5. CONTROLLED INFERENCE & SYNONYM RESOLUTION:\n"
    "You are expected to make intelligent semantic deductions when terminology differs, BUT the underlying factual "
    "meaning must remain 100% identical. You may resolve established industry synonyms and alternative phrasings "
    "for the same concept. However, you are strictly forbidden from bridging conceptually distinct entities — "
    "different security controls, different compliance frameworks, or different operational domains must never be "
    "treated as interchangeable. If a deduction requires assuming any information not present in the KB, "
    "you must reject it and return an empty string.\n\n"

    "6. EVIDENCE-BASED REASONING (MANDATORY CoT):\n"
    "Before deciding on a value for any field, you must evaluate the conceptual bridge and record your reasoning "
    "in the `_AI_Reasoning` key using this exact format: "
    "\"Questionnaire asks for X. KB states Y. Is Y a direct factual equivalent of X? Yes/No.\" "
    "Only if the answer is a definitive 'Yes' may you map the data. If 'No', the output for that field must be "
    "an empty string \"\".\n\n"

    "7. OUTPUT FORMAT:\n"
    "You must return a valid JSON array of objects. Every single object must contain ALL the original keys provided "
    "in the input, plus the `_AI_Status` key and the `_AI_Reasoning` key. Do not drop any columns."
)

SYSTEM_INSTRUCTION_EXTRACT = (
    "You are a precise data extraction assistant processing compliance forms, questionnaires, and capability matrices. "
    "Your task is to extract every single item that requires an answer, evaluation, or response. "
    "CRITICAL: These items often DO NOT end with a question mark. They might be standalone terms, criteria, or table row items "
    "(e.g., 'Antivirus', 'Firewall', 'City of birth', 'Encryption', 'אנטי וירוס', 'חומת אש'). "
    "Treat every row item, criterion, or topic that expects a status or comment as a 'question_text'. "
    "Extract them exactly as they appear in the source text (in their original language, including Hebrew). "
    "Do not filter based on topic. If it is a line item in a form meant to be filled out, extract it as a question. "
    "Use fields: question_id (string), question_text (string). Generate sequential IDs Q1, Q2, ... if none exist. "
    "If the text is truly empty or contains no extractable items, return an empty list []."
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
            "The text below is from a compliance form or capability matrix. Items may be in Hebrew or English.\n"
            "Extract EVERY line item, criterion, or term that represents something requiring an answer or evaluation. "
            "IMPORTANT: Items will NOT have question marks — they are standalone terms or short phrases (e.g., 'אנטי וירוס', 'Firewall'). "
            "Do NOT filter by topic. Do NOT translate. Return every item exactly as it appears.\n\n"
            "Text:\n"
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


def answer_excel_rows_batch(
    kb_text: str,
    original_columns: List[str],
    rows_batch: List[dict],
) -> List[dict]:
    """
    Native Excel pipeline: send a batch of rows to the LLM and return filled dicts.
    Each returned dict contains the original keys plus '_AI_Status' (UI-only).
    Returns the original rows unchanged on failure so the caller's DataFrame is safe.
    """
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    safe_batch = json.loads(json.dumps(rows_batch, ensure_ascii=False, default=str))
    headers_str = json.dumps(original_columns, ensure_ascii=False)
    batch_json = json.dumps(safe_batch, ensure_ascii=False)
    row_count = len(safe_batch)

    prompt = (
        f"You are receiving a JSON array containing exactly {row_count} row object(s). "
        f"You MUST process EVERY row and return a JSON array of exactly {row_count} object(s) — no more, no less.\n\n"
        f"The form has these exact column headers: {headers_str}\n\n"
        "For each row:\n"
        "  a) Identify the populated anchor value (the non-empty subject, e.g. 'Antivirus').\n"
        "  b) Search the Knowledge Base for that subject using intelligent context mapping "
        "(e.g. if the KB says 'Exists: Yes', map that to the Implementation Stage column as 'Implemented').\n"
        "  c) Fill every empty string field in the row with the best answer from the Knowledge Base.\n"
        "  d) Keep already-populated fields exactly as they are.\n"
        "  e) If no specific info is found for a field, return an EMPTY STRING \"\". "
        "NEVER write 'No information available' or 'N/A'.\n"
        "  f) Add a key '_AI_Status' to each object: set it to 'OK' if the anchor subject was found "
        "in the KB, or 'REVIEW' only if the anchor subject is entirely missing from the KB.\n\n"
        "RULES: Return ONLY a JSON array. Each object MUST contain every key from the headers list "
        "plus '_AI_Status'. Do not add any other new keys.\n\n"
        "Knowledge Base:\n"
        f"{kb_text}\n\n"
        "Rows:\n"
        f"{batch_json}\n\n"
        "Return only the JSON array."
    )

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION_EXCEL,
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
            raw_text = getattr(response, "text", "") or ""
            print(f"[Excel batch] attempt {attempt + 1}: {raw_text[:300]}")
            result = json.loads(clean_json_string(raw_text))
            if isinstance(result, dict):
                result = [result]
            if isinstance(result, list):
                return result
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                rotate_key()
                print("[Rate Limit] Hit 429/RESOURCE_EXHAUSTED. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"[Excel batch] Error on attempt {attempt + 1}: {e}")
                time.sleep(5)

    return list(rows_batch)  # fallback: caller keeps original values


def analyze_questionnaire(kb_text: str, questionnaire_text: str) -> List[dict]:
    """Main orchestrator: extract questions, then answer them in batches."""
    extracted = extract_questions(questionnaire_text)
    return answer_questions_in_batches(kb_text, extracted)
