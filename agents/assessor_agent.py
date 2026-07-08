"""
LLM integration with two-stage batching:
1) Extract questions from the questionnaire text (chunked).
2) Answer questions in batches to avoid token limits.
Includes cleaning to handle noisy PDF extraction artifacts.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Iterable, List, Optional

from core.llm_provider import BaseLLMProvider


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION_ANSWER = (
    "You are an Expert IT Security Officer. Complete the Vendor Security Questionnaire "
    "based ONLY on the provided Knowledge Base.\n"
    "Apply 'Semantic Bridging': e.g., if the KB says 'CrowdStrike', recognize it as 'Antivirus'. "
    "If it says 'AWS Security Groups', recognize it as a firewall.\n"
    "If information is missing, DO NOT guess. State 'No information available' and set confidence to Low.\n"
    "You must analyze the entire questionnaire and provide an answer for EVERY SINGLE question provided to you.\n"
    "Return a JSON object with an 'answers' key containing an array of answer objects."
)

NAIVE_SYSTEM_PROMPT_ANSWER = (
    "You are a helpful assistant. Answer the provided security questionnaire questions "
    "based on the Knowledge Base. Return a JSON object with an 'answers' key containing "
    "an array of answer objects, each with: question_id, question_text, proposed_yes_no, "
    "proposed_comments, confidence_level, reasoning, flag_for_human_review."
)

SYSTEM_INSTRUCTION_EXCEL = (
    "You are a strict JSON-to-JSON data processor specializing in enterprise security and compliance questionnaires. "
    "You receive a JSON array of spreadsheet rows and must return that same array — every row fully preserved — "
    "with your assessment appended to each object. "
    "Your primary objective is to evaluate each row against the provided Knowledge Base (KB) while FLAWLESSLY preserving the original data structure.\n\n"

    "You must rigorously adhere to the following rules:\n\n"

    "1. DYNAMIC REPLICATION:\n"
    "You MUST return the ENTIRE original JSON object for EVERY row. "
    "Copy every single key-value pair exactly as it appears in the input — even keys you did not use, "
    "even keys whose values are empty strings. Do not alter, translate, or reformat any original value.\n\n"

    "2. NO ASSUMPTIONS ABOUT COLUMN NAMES:\n"
    "Do not hardcode expectations for any specific column name (e.g., do not assume 'הבקרה', 'ראיות נדרשות', "
    "or any other fixed key exists). The input schema varies between questionnaires. "
    "Dynamically analyze all key-value pairs in each row to deduce the core entity, control, or subject being queried, "
    "then search the KB for that subject.\n\n"

    "3. NO OMISSIONS:\n"
    "Do NOT omit, drop, summarize, or truncate ANY original key or its value under any circumstances. "
    "Returning a partial object corrupts the database and is a critical failure.\n\n"

    "4. APPEND ONLY — TWO MANDATORY ASSESSMENT FIELDS:\n"
    "After copying all original keys, append EXACTLY these two fields to each object:\n"
    "  a) \"_AI_Status\": Your confidence assessment — choose ONE of:\n"
    "       \"OK\"      — You found explicit KB evidence and produced a definitive answer for every empty field.\n"
    "       \"REVIEW\"  — The anchor subject was found but details are incomplete or uncertain.\n"
    "       \"NO_DATA\" — The anchor subject is entirely absent from the KB.\n"
    "  b) \"_AI_Reasoning\": Your brief chain-of-thought using this exact format:\n"
    "       'Questionnaire asks for [X]. KB states [Y]. Is Y a direct factual equivalent of X? Yes/No.'\n"
    "       Only if Yes may you set a definitive answer. If No, the empty fields stay empty strings.\n"
    "CRITICAL: Do NOT modify or overwrite any key that already had a NON-EMPTY value in the input row. "
    "Fields that were empty strings (\"\") MUST be filled according to Rule 5 below whenever KB evidence exists. "
    "Your assessment metadata belongs exclusively in \"_AI_Status\" and \"_AI_Reasoning\".\n\n"

    "5. DYNAMIC FIELD FILLING (CATEGORICAL vs. DESCRIPTIVE):\n"
    "For every field that was an empty string in the input, apply the appropriate logic based on the column header:\n"
    "  - CATEGORICAL: If the header lists allowed options (e.g., separated by '/', '|', commas, or parentheses), "
    "output EXACTLY one listed option.\n"
    "  - DESCRIPTIVE: If the header asks for free text (e.g., 'פירוט', 'הערות', 'Details'), write a concise, "
    "professional summary using ONLY KB facts. If no KB info exists, leave as empty string \"\".\n"
    "NEVER write 'N/A', 'No information available', 'Not found', or any placeholder phrase.\n\n"

    "5a. CONSERVATIVE CATEGORICAL FILLING — STATUS DETERMINES FIELD ACTION:\n"
    "The \"_AI_Status\" value you assign CONTROLS whether you fill categorical empty fields. These two rules are absolute:\n"
    "  RULE A — \"OK\" status: You found explicit, unambiguous KB evidence. "
    "You MUST fill every empty categorical field with exactly one of its listed options. "
    "An 'OK' row with any empty categorical field is a contradiction and is FORBIDDEN.\n"
    "  RULE B — \"REVIEW\" or \"NO_DATA\" status: Evidence is partial, ambiguous, or absent. "
    "You MUST leave every categorical field as an empty string \"\". "
    "Do NOT guess or pick the most likely option. Human review is required — pre-filling with a guess would corrupt the review process.\n"
    "DECISION TREE (apply in order for each row):\n"
    "  1. Is the anchor subject explicitly documented in the KB with sufficient detail? If No → set \"NO_DATA\", leave categorical fields empty.\n"
    "  2. Is the KB evidence a direct, unambiguous match for what the categorical field asks? If No → set \"REVIEW\", leave categorical fields empty.\n"
    "  3. Only if both answers are Yes: set \"OK\" and fill each empty categorical field with exactly one listed option.\n\n"

    "6. ZERO INFERENCE:\n"
    "If explicit evidence for a field is not in the KB, leave that field as an empty string \"\". "
    "Do not invent, assume, or extrapolate.\n\n"

    "7. OUTPUT FORMAT:\n"
    "Return ONLY a valid JSON array. "
    "The array length MUST exactly match the input array length — one object per input row. "
    "Each object must contain ALL original keys plus the two appended assessment fields. "
    "Do not add any other new keys. Do not wrap the array in another object."
)

NAIVE_SYSTEM_PROMPT_EXCEL = (
    "You are a helpful assistant. Fill in the missing columns in each row based on the provided "
    "Knowledge Base text. Return the completed rows as a JSON array. Add an '_AI_Status' key to "
    "each row: 'OK' if you found relevant info, 'REVIEW' if the subject was not in the KB."
)

SYSTEM_INSTRUCTION_EXTRACT = (
    "You are a precise data extraction assistant processing compliance forms, questionnaires, and capability matrices. "
    "Your task is to extract every single item that requires an answer, evaluation, or response. "
    "CRITICAL: These items often DO NOT end with a question mark. They might be standalone terms, criteria, or table row items. "
    "Treat every row item, criterion, or topic that expects a status or comment as a 'question_text'. "
    "Extract them exactly as they appear in the source text (in their original language, including Hebrew). "
    "Do not filter based on topic. If it is a line item in a form meant to be filled out, extract it as a question. "
    "Use fields: question_id (string), question_text (string). Generate sequential IDs Q1, Q2, ... if none exist. "
    "If the text is truly empty or contains no extractable items, return an empty list []."
)


# ── JSON Schema constants ─────────────────────────────────────────────────────

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question_text": {"type": "string"},
                },
                "required": ["question_id", "question_text"],
            },
        }
    },
    "required": ["questions"],
}

QA_ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question_text": {"type": "string"},
                    "proposed_yes_no": {"type": "string"},
                    "proposed_comments": {"type": "string"},
                    "confidence_level": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "flag_for_human_review": {"type": "boolean"},
                },
                "required": [
                    "question_id",
                    "question_text",
                    "proposed_yes_no",
                    "proposed_comments",
                    "confidence_level",
                    "reasoning",
                    "flag_for_human_review",
                ],
            },
        }
    },
    "required": ["answers"],
}


# ── Utilities ─────────────────────────────────────────────────────────────────

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
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text.strip()


def get_smart_chunks(text: str, max_chars: int = 4000) -> list[str]:
    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break

        split_index_q = text.rfind("? ", 0, max_chars)
        split_index_p = text.rfind(". ", 0, max_chars)
        split_index = max(split_index_q, split_index_p)

        if split_index == -1:
            split_index = text.rfind(" ", 0, max_chars)
        if split_index == -1:
            split_index = max_chars
        else:
            split_index += 1

        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()

    return chunks


def _error_response(raw_string: str) -> list[dict]:
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


# ── Core pipeline functions ───────────────────────────────────────────────────

def extract_questions(
    questionnaire_text: str,
    provider: BaseLLMProvider,
) -> List[dict]:
    """Stage 1 – Extract all line items as {question_id, question_text}."""
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
            "IMPORTANT: Items will NOT have question marks — they are standalone terms or short phrases. "
            "Do NOT filter by topic. Do NOT translate. Return every item exactly as it appears.\n\n"
            f"Text:\n{chunk}\n"
        )
        print(f"[Extraction] Processing chunk {idx}/{len(text_chunks)} (~{len(chunk)} chars)")
        chunk_success = False
        for attempt in range(3):
            try:
                result = provider.generate_response(
                    SYSTEM_INSTRUCTION_EXTRACT,
                    chunk_prompt,
                    EXTRACTION_SCHEMA,
                )
                chunk_questions: list = (
                    result.get("questions", []) if isinstance(result, dict) else result or []
                )
                all_extracted_questions.extend(chunk_questions)
                chunk_success = True
                break
            except Exception as exc:
                print(f"[Extraction] Chunk {idx} attempt {attempt + 1} failed: {exc}")
                time.sleep(3)

        if not chunk_success:
            print(f"[Extraction] Skipping chunk {idx} after 3 failed attempts.")

        time.sleep(2)

    print(f"[Extraction] Total extracted: {len(all_extracted_questions)}")
    return all_extracted_questions


def answer_questions_in_batches(
    kb_text: str,
    extracted_questions: List[dict],
    provider: BaseLLMProvider,
    use_advanced_prompt: bool = True,
) -> List[dict]:
    """Stage 2 – Answer questions in batches using the KB."""
    if not extracted_questions:
        return _error_response("No questions extracted")

    system_prompt = SYSTEM_INSTRUCTION_ANSWER if use_advanced_prompt else NAIVE_SYSTEM_PROMPT_ANSWER

    normalized = [
        q.model_dump() if hasattr(q, "model_dump") else q for q in extracted_questions
    ]

    master_answers: List[dict] = []
    batch_size = 15
    total_batches = math.ceil(len(normalized) / batch_size)

    for batch_index, batch in enumerate(_chunk_list(normalized, batch_size), start=1):
        prompt = (
            "Answer ONLY the questions provided below using the Knowledge Base. "
            "Do not add new questions and do not omit any provided questions.\n\n"
            f"Knowledge Base:\n{kb_text}\n\n"
            f"Questions (JSON):\n{json.dumps(batch, ensure_ascii=False)}\n"
        )

        answers: Optional[List[dict]] = None
        for attempt in range(5):
            try:
                result = provider.generate_response(system_prompt, prompt, QA_ANSWER_SCHEMA)
                print(f"[QA batch {batch_index}/{total_batches} attempt {attempt + 1}] type={type(result).__name__}")

                if isinstance(result, dict):
                    answers = result.get("answers", [])
                elif isinstance(result, list):
                    answers = result
                else:
                    answers = []

                time.sleep(5)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print("[Rate Limit] 429/RESOURCE_EXHAUSTED — waiting 60 s before retry...")
                    time.sleep(60)
                else:
                    print(f"[Answering] Error on attempt {attempt + 1}: {e}")
                    time.sleep(5)

        if answers is None:
            answers = _error_response("Batch failed after retries")
        if isinstance(answers, dict):
            answers = [answers]
        master_answers.extend(answers)

    return master_answers


_AI_ONLY_KEYS = {"_AI_Status", "_AI_Reasoning"}

# Batch size for the native Excel pipeline.
# Set to 5 rows: large enough to amortise the per-call overhead and benefit from
# prompt caching, small enough that dense Hebrew output stays well within token limits.
# If a batch truncates the LLM response, _process_batch_with_fallback automatically
# splits it in half and retries each half independently.
EXCEL_BATCH_SIZE: int = 5


def _detect_id_key(row: dict) -> str | None:
    """
    Dynamically find the column that most likely holds the unique row identifier.
    Prefers columns whose name contains common identifier keywords (case-insensitive),
    then falls back to the first column whose value looks like an ID (short scalar).
    Returns None if no candidate is found.
    """
    id_hints = ("id", "זיהוי", "מזהה", "identifier", "control", "number", "no.", "#")
    for key in row:
        if any(hint in str(key).lower() for hint in id_hints):
            return key
    # Fallback: first key with a short scalar value (likely an ID, not a long text field)
    for key, val in row.items():
        if isinstance(val, (int, float)):
            return key
        if isinstance(val, str) and 0 < len(val) <= 20 and "\n" not in val:
            return key
    return None


def _safe_merge(orig_row: dict, ai_row: dict) -> dict:
    """
    FIX 3: Merge AI-generated fields back into the original row safely.

    Rules:
    - AI-only metadata keys (_AI_Status, _AI_Reasoning) are always taken from ai_row.
    - Any field that was originally non-empty is NEVER overwritten by the LLM.
      This prevents wrong-row hallucinations from corrupting identifier columns.
    - Fields that were originally empty string "" are filled from ai_row.
    - Keys present in orig_row but absent in ai_row are kept from orig_row unchanged.
    """
    merged = orig_row.copy()
    for key, ai_value in ai_row.items():
        if key in _AI_ONLY_KEYS:
            merged[key] = ai_value
        elif key in orig_row:
            if orig_row[key] == "":
                merged[key] = ai_value   # fill blank with AI answer
            # else: original had content → do NOT overwrite
        else:
            merged[key] = ai_value       # new key from AI (e.g. future AI fields)
    return merged


def _validate_id_match(orig_row: dict, ai_row: dict, id_key: str | None) -> None:
    """
    FIX 1: Raise immediately if the LLM returned a row for a different identifier.
    This catches the "wrong-row hallucination" bug where the LLM copies a previous
    batch's content into the current batch's response.
    """
    if id_key is None:
        return  # can't validate without a known ID column
    orig_id = orig_row.get(id_key)
    ai_id   = ai_row.get(id_key)
    if ai_id is not None and orig_id != ai_id:
        raise ValueError(
            f"[ID MISMATCH] LLM returned wrong row. "
            f"Expected '{id_key}'={orig_id!r}, got {ai_id!r}. "
            "Discarding response and retrying."
        )


def _build_static_prefix(headers_str: str, kb_text: str) -> str:
    """
    The cacheable portion of the Excel prompt: column schema + full Knowledge Base.
    This is identical across every row in a questionnaire run, making it the ideal
    candidate for provider-native prompt caching (Gemini Context Cache / Anthropic
    cache_control). It is built once in answer_excel_rows_batch and reused by every
    recursive call in _process_batch_with_fallback.
    """
    return (
        f"Column headers present in this form: {headers_str}\n\n"
        f"Knowledge Base:\n{kb_text}"
    )


def _build_dynamic_suffix(rows_batch: List[dict]) -> str:
    """
    The non-cacheable portion of the Excel prompt: row count, step-by-step
    instructions, and the actual batch JSON. Rebuilt for each batch / sub-batch
    because the row count and data change at every recursion level.
    """
    row_count = len(rows_batch)
    batch_json = json.dumps(rows_batch, ensure_ascii=False, default=str)
    return (
        f"You are receiving a JSON array containing exactly {row_count} row object(s). "
        f"You MUST process EVERY row and return a JSON array of exactly {row_count} object(s) — no more, no less.\n\n"
        "For each row, follow these steps IN ORDER:\n"
        "  1. COPY every original key-value pair exactly as provided — do NOT omit or alter any key.\n"
        "  2. Dynamically identify the anchor (the non-empty subject/control being assessed).\n"
        "  3. Search the Knowledge Base for that anchor.\n"
        "  4. Fill empty string fields using KB evidence:\n"
        "       - Categorical headers (options listed): output exactly one listed option.\n"
        "       - Descriptive headers: output a concise KB-grounded summary, or \"\" if no info.\n"
        "       - Already-populated (non-empty) fields: copy them unchanged.\n"
        "  5. APPEND exactly two new fields to each object:\n"
        "       \"_AI_Status\"    : \"OK\" | \"REVIEW\" | \"NO_DATA\"\n"
        "       \"_AI_Reasoning\" : 'Questionnaire asks for X. KB states Y. Direct equivalent? Yes/No.'\n"
        "     Do NOT modify or overwrite any key that already had a NON-EMPTY value in the input row.\n\n"
        "CRITICAL RULES:\n"
        "  - Return ONLY a raw JSON array — no markdown, no wrapper object.\n"
        "  - Output array length MUST equal input array length.\n"
        "  - Every output object MUST contain ALL original keys plus the two appended fields.\n"
        "  - NEVER write 'N/A', 'No information available', or any placeholder. Use \"\" instead.\n"
        "  - DO NOT TRUNCATE. Return every row in the output array.\n"
        "  - CONSERVATIVE CATEGORICAL FILLING: '_AI_Status' controls whether you fill categorical fields.\n"
        "      'OK'               → MUST fill every empty categorical field with one valid listed option.\n"
        "      'REVIEW'/'NO_DATA' → MUST leave every categorical field as empty string \"\". "
        "Do NOT guess — human review is required and pre-filling would corrupt the review process.\n\n"
        f"Rows:\n{batch_json}\n"
    )


# Maximum flat retry attempts before escalating to split-fallback or giving up.
_MAX_BATCH_ATTEMPTS = 3


def _process_batch_with_fallback(
    rows_batch: List[dict],
    system_prompt: str,
    static_prefix: str,
    provider: BaseLLMProvider,
    id_key: str | None,
    depth: int = 0,
) -> List[dict]:
    """
    Send a batch of rows to the LLM with adaptive split-fallback on truncation.

    Strategy:
    - Attempt the batch up to _MAX_BATCH_ATTEMPTS times (flat retries).
    - ID mismatch (ValueError from _validate_id_match): retry the same batch — the
      LLM confused rows but the batch size isn't the problem.
    - JSON parse error or wrong output count: these signal output-token truncation —
      immediately split the batch in half and recurse on each half independently.
    - Rate limit: rotate key and wait before retrying.
    - Single-row batch that still fails: return the original row unchanged so the
      caller's output is never shorter than its input.

    The static_prefix (KB + headers) is passed through to every recursion level so
    providers can reuse their cached context object without rebuilding it.
    """
    row_count = len(rows_batch)
    if row_count == 0:
        return []

    label = f"[Batch depth={depth} rows={row_count}]"
    dynamic_suffix = _build_dynamic_suffix(rows_batch)
    should_split = False

    for attempt in range(_MAX_BATCH_ATTEMPTS):
        try:
            result = provider.generate_response_cached(
                system_prompt, static_prefix, dynamic_suffix, None
            )
            print(f"{label} attempt {attempt + 1}: type={type(result).__name__}")

            # Unwrap {"rows": [...]} or similar wrapper dicts the model sometimes emits.
            if isinstance(result, dict):
                for val in result.values():
                    if isinstance(val, list):
                        result = val
                        break
                else:
                    result = [result]

            if isinstance(result, list) and len(result) == row_count:
                # Validate that each returned row matches the expected identifier.
                for orig_row, ai_row in zip(rows_batch, result):
                    _validate_id_match(orig_row, ai_row, id_key)
                # Safe-merge: AI may only write to originally empty fields.
                return [_safe_merge(o, a) for o, a in zip(rows_batch, result)]

            # Wrong count — output was likely truncated.
            got = len(result) if isinstance(result, list) else type(result).__name__
            print(f"{label} attempt {attempt + 1}: got {got} items, expected {row_count}. Will split.")
            should_split = True
            break

        except json.JSONDecodeError as je:
            # Malformed / unterminated JSON → output truncation. Split immediately.
            # IMPORTANT: must come before `except ValueError` because JSONDecodeError
            # is a ValueError subclass — wrong order silently routes truncation errors
            # into the flat-retry branch instead of the split branch.
            print(f"{label} attempt {attempt + 1}: JSON parse error ({je}). Will split.")
            should_split = True
            break

        except ValueError as ve:
            # ID mismatch from _validate_id_match — retry the same batch flat
            # (wrong-row hallucination, not a size/truncation problem).
            print(f"{label} attempt {attempt + 1}: {ve}")
            time.sleep(2)

        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(f"{label} Rate limit — waiting 60 s...")
                time.sleep(60)
            else:
                print(f"{label} attempt {attempt + 1} error: {e}")
                time.sleep(5)

    # ── Split-fallback (truncation path) ──────────────────────────────────────
    if should_split and row_count > 1:
        mid = row_count // 2
        print(f"{label} Splitting into {mid} + {row_count - mid} rows.")
        left = _process_batch_with_fallback(
            rows_batch[:mid], system_prompt, static_prefix, provider, id_key, depth + 1
        )
        right = _process_batch_with_fallback(
            rows_batch[mid:], system_prompt, static_prefix, provider, id_key, depth + 1
        )
        return left + right

    # Single-row batch that still failed all attempts — return original unchanged.
    print(f"{label} All recovery attempts failed — returning original row(s) unchanged.")
    return list(rows_batch)


def answer_excel_rows_batch(
    kb_text: str,
    original_columns: List[str],
    rows_batch: List[dict],
    provider: BaseLLMProvider,
    use_advanced_prompt: bool = True,
) -> List[dict]:
    """
    Native Excel pipeline entry point.

    Splits the prompt into a cacheable static_prefix (headers + KB) and a
    per-batch dynamic_suffix (row count + instructions + JSON), then delegates
    to _process_batch_with_fallback which handles retries, split-on-truncation,
    ID validation, and safe merging.
    """
    system_prompt = SYSTEM_INSTRUCTION_EXCEL if use_advanced_prompt else NAIVE_SYSTEM_PROMPT_EXCEL

    safe_batch = json.loads(json.dumps(rows_batch, ensure_ascii=False, default=str))
    headers_str = json.dumps(original_columns, ensure_ascii=False)

    id_key = _detect_id_key(safe_batch[0]) if safe_batch else None
    if id_key:
        print(f"[Excel batch] Detected identifier key: {id_key!r}")

    static_prefix = _build_static_prefix(headers_str, kb_text)

    return _process_batch_with_fallback(
        safe_batch, system_prompt, static_prefix, provider, id_key
    )


def analyze_questionnaire(
    kb_text: str,
    questionnaire_text: str,
    provider: BaseLLMProvider,
    use_advanced_prompt: bool = True,
) -> List[dict]:
    """Main orchestrator: extract questions, then answer them in batches."""
    extracted = extract_questions(questionnaire_text, provider)
    return answer_questions_in_batches(kb_text, extracted, provider, use_advanced_prompt)
