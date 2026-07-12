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
from core.prompts import (
    SYSTEM_INSTRUCTION_EXCEL,
    NAIVE_SYSTEM_PROMPT_EXCEL,
    SYSTEM_INSTRUCTION_ANSWER,
    NAIVE_SYSTEM_PROMPT_ANSWER,
    SYSTEM_INSTRUCTION_EXTRACT,
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
    The non-cacheable, per-batch portion of the Excel prompt.

    Contains ONLY information that changes with each call: the row count and
    the actual batch JSON. All behavioural rules (zero-inference, conservative
    categorical filling, output format, etc.) are defined once in
    SYSTEM_INSTRUCTION_EXCEL (core/prompts.py) and must not be repeated here.
    Duplication between the system prompt and this suffix causes instruction
    drift whenever one is updated without the other.
    """
    row_count  = len(rows_batch)
    batch_json = json.dumps(rows_batch, ensure_ascii=False, default=str)
    return (
        f"Process the following batch of exactly {row_count} row(s). "
        f"Apply all rules from the system prompt and return a JSON array of exactly {row_count} objects.\n\n"
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
