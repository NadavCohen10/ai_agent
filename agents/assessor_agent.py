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
    "You are an expert AI Data Entry Assistant specializing in enterprise security and compliance questionnaires. "
    "Your task is to analyze a batch of rows from a spreadsheet and fill in the missing empty columns based strictly on the provided Knowledge Base (KB) text.\n\n"
    
    "You must rigorously adhere to the following operational constraints:\n\n"

    "1. DYNAMIC ANCHOR DETECTION:\n"
    "Do not rely on hardcoded column names to find the subject. Analyze all provided key-value pairs in a given row to dynamically deduce the core entity, system, or control being queried. Use this inferred subject to search the KB.\n\n"

    "2. DYNAMIC FIELD INSTRUCTIONS (CATEGORICAL VS. DESCRIPTIVE):\n"
    "Treat the column header as your formatting instruction. Read the exact text of the target column header:\n"
    "- CATEGORICAL FIELDS: If the header explicitly lists allowed options (e.g., '(כן/לא/לא רלוונטי)', '[Met/Not Met]', 'Status'), you MUST output EXACTLY one of those listed options. If the KB supports the control, output the positive value (e.g., 'כן'). If it explicitly contradicts, output the negative.\n"
    "- DESCRIPTIVE / OPEN-ENDED FIELDS: If the header asks for details, explanations, or does not provide a list of options (e.g., 'פירוט טכני', 'הערות', 'Implementation Details'), you must generate a professional, concise free-text summary based ONLY on the KB. \n"
    "CRITICAL FOR OPEN TEXT: Never invent, assume, or guess details to make the answer sound better. If there is no relevant information in the KB to write a description, leave the field as an empty string \"\".\n\n"

    "3. ZERO INFERENCE FOR MISSING DATA:\n"
    "If you are unsure, assume the data does not exist. Never write generic placeholder phrases such as 'No information available', 'N/A', or 'Not found'. If the answer is not explicitly in the KB, the generated value must be an empty string \"\".\n\n"

    "4. CONFIDENCE TRACKING (AI STATUS):\n"
    "Add a special key named `_AI_Status` to your JSON output for each row. \n"
    "Set its value to \"OK\" ONLY if you successfully generated a definitive answer (e.g., 'כן' or a valid summary) for the target field based on explicit KB evidence. \n"
    "Set its value to \"REVIEW\" if you left the target field empty as an empty string \"\" (e.g., because the subject is missing, details are lacking, or you are uncertain).\n\n"

    "5. CONTROLLED INFERENCE & SYNONYM RESOLUTION:\n"
    "You are expected to make intelligent semantic deductions when terminology differs, BUT the underlying factual meaning must remain 100% identical. You may resolve established industry synonyms.\n\n"

    "6. EVIDENCE-BASED REASONING (MANDATORY CoT):\n"
    "Before deciding on a value for any field, you must evaluate the conceptual bridge and record your reasoning in the `_AI_Reasoning` key using this exact format: 'Questionnaire asks for X. KB states Y. Is Y a direct factual equivalent of X? Yes/No.' Only if the answer is a definitive 'Yes' may you fill the field. If 'No', the output for that field must be an empty string \"\".\n\n"

    "7. OUTPUT FORMAT:\n"
    "You must return a valid JSON array of objects. CRITICAL: To prevent JSON corruption from maximum output limits, DO NOT return all original columns. Your output objects MUST contain ONLY the following keys:\n"
    "1. The exact key name from the input row that represents the unique row identifier (e.g., Control ID, Question ID, זיהוי בקרה).\n"
    "2. ANY AND ALL keys from the input row that originally had an empty string \"\" value. You must include these keys and provide your generated answer for them (or an empty string if no info).\n"
    "3. \"_AI_Status\"\n"
    "4. \"_AI_Reasoning\"\n"
    "Do not invent new key names. Do not drop any rows. You must process and return exactly the same number of rows as provided in the input.\n" )

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
                    provider.rotate_key()
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


def answer_excel_rows_batch(
    kb_text: str,
    original_columns: List[str],
    rows_batch: List[dict],
    provider: BaseLLMProvider,
    use_advanced_prompt: bool = True,
) -> List[dict]:
    """
    Native Excel pipeline – send a batch of rows to the LLM and return filled dicts.
    Each returned dict contains the original keys plus '_AI_Status'.
    Falls back to the original rows unchanged on total failure.
    """
    system_prompt = SYSTEM_INSTRUCTION_EXCEL if use_advanced_prompt else NAIVE_SYSTEM_PROMPT_EXCEL

    safe_batch = json.loads(json.dumps(rows_batch, ensure_ascii=False, default=str))
    headers_str = json.dumps(original_columns, ensure_ascii=False)
    batch_json = json.dumps(safe_batch, ensure_ascii=False)
    row_count = len(safe_batch)

    prompt = (
        f"You are receiving a JSON array containing exactly {row_count} row object(s). "
        f"You MUST process EVERY row and return a JSON array of exactly {row_count} object(s) — no more, no less.\n\n"
        f"The form has these exact column headers: {headers_str}\n\n"
        "For each row:\n"
        "  a) Identify the populated anchor value (the non-empty subject).\n"
        "  b) Search the Knowledge Base for that subject using intelligent context mapping.\n"
        "  c) Fill every empty string field with the best answer from the Knowledge Base.\n"
        "  d) Keep already-populated fields exactly as they are.\n"
        "  e) If no specific info is found for a field, return an EMPTY STRING \"\". "
        "NEVER write 'No information available' or 'N/A'.\n"
        "  f) Add '_AI_Status' to each object: 'OK' if the anchor was found in the KB, "
        "'REVIEW' if the anchor is entirely missing.\n\n"
        "RULES: Return ONLY a JSON array. Each object MUST contain every key from the headers "
        "list plus '_AI_Status'. Do not add any other new keys.\n"
        "CRITICAL: DO NOT TRUNCATE YOUR RESPONSE. You must process and return EVERY row provided. "
        "The output array length MUST EXACTLY MATCH the input array length.\n\n"
        f"Knowledge Base:\n{kb_text}\n\n"
        f"Rows:\n{batch_json}\n"
    )

    for attempt in range(5):
        try:
            result = provider.generate_response(system_prompt, prompt, None)
            print(f"[Excel batch] attempt {attempt + 1}: type={type(result).__name__}")
            if isinstance(result, dict):
                # Extract array if model wrongly wrapped it in an object (e.g. {"rows": [...]})
                for val in result.values():
                    if isinstance(val, list):
                        result = val
                        break
                else:
                    result = [result]
            if isinstance(result, list):
                if len(result) == row_count:
                    # --- NEW MERGE LOGIC ---
                    # Merge the AI's brief answers back into the original heavy rows
                    merged_batch = []
                    for orig_row, ai_row in zip(rows_batch, result):
                        updated_row = orig_row.copy()
                        # Update the original row with AI answers (only overriding existing keys or adding the AI keys)
                        for key, value in ai_row.items():
                            updated_row[key] = value
                        merged_batch.append(updated_row)
                    return merged_batch
                    # ------------------------
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                provider.rotate_key()
                print("[Rate Limit] 429/RESOURCE_EXHAUSTED — waiting 60 s...")
                time.sleep(60)
            else:
                print(f"[Excel batch] Error on attempt {attempt + 1}: {e}")
                time.sleep(5)

    return list(rows_batch)  # fallback: caller keeps original values


def analyze_questionnaire(
    kb_text: str,
    questionnaire_text: str,
    provider: BaseLLMProvider,
    use_advanced_prompt: bool = True,
) -> List[dict]:
    """Main orchestrator: extract questions, then answer them in batches."""
    extracted = extract_questions(questionnaire_text, provider)
    return answer_questions_in_batches(kb_text, extracted, provider, use_advanced_prompt)
