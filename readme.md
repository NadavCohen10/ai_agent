# HITL Security Questionnaire Assistant 🛡️🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)]()
[![Gemini](https://img.shields.io/badge/Gemini-2.5-orange)]()

**Tagline:** Automate, review, and export massive vendor security questionnaires using RAG, Gemini 2.5, and a Human-In-The-Loop UI. Now with Native Excel In-Place Filling — the AI fills your form without ever changing its structure.

---

## Why This Matters
Filling SOC 2 / HIPAA / PCI questionnaires is slow, error-prone, and painful. Raw PDFs are messy; naïvely sending the whole file to an LLM blows context windows and invites hallucinations. This project delivers a resilient, production-ready pipeline that cleans ugly PDFs, extracts questions without breaking sentences, answers them with your Knowledge Base, and routes low-confidence items to humans for quick approval and export.

![Dashboard UI](link-to-screenshot.png)

---

## Key Features

### PDF Pipeline
1) **Intelligent PDF Parsing & Cleaning** — Aggressive RegEx normalization removes invisible newlines and fixes broken words before any LLM call.
2) **Sentence-Aware Smart Chunking** — Custom splitter prefers `?` or `.` boundaries (≤4,000 chars) so questions are never cut in half.
3) **Structured Outputs (JSON)** — `google-genai` + Pydantic schemas enforce strict JSON arrays for both extraction and answering.
4) **Fault-Tolerant RAG Engine** — Batch answering (15 Qs/batch) with retries, 429/`RESOURCE_EXHAUSTED` handling, and backoff sleeps.

### Native Excel Mode (In-Place Filling)
5) **Structure-Preserving In-Place Filling** — When an Excel questionnaire is uploaded, the system fills it row by row without ever rebuilding or restructuring the file. The downloaded output looks exactly like the original upload — same columns, same order — just with the answers filled in.
6) **Dynamic Anchor Detection** — The AI does not rely on hardcoded column names like "Category" or "Column B". For each row it dynamically reads all key-value pairs and semantically identifies the core subject (e.g., "Antivirus", "Password Policy") to use as its Knowledge Base search term.
7) **Strict Categorical Matching** — The AI automatically detects multiple-choice constraints in headers or placeholder cells (options separated by `/`, `,`, `|`, parentheses, or phrases like "Choose one:"). It restricts its generated answer to exactly one of the allowed options — acting like built-in data validation.
8) **Clean Outputs** — If no matching data is found for a cell, the AI leaves it blank. It never writes "N/A" or "No information available" unless that is a valid allowed option.
9) **Smart Row-Based Batching** — Excel rows are chunked by row count (15 rows per API call), completely separate from the character-based chunking used for PDFs. This prevents JSON corruption and keeps the system within free-tier rate limits.

### Shared
10) **Human-In-The-Loop UI** — Streamlit dashboard with `st.data_editor` to review and edit answers. In Native Excel Mode, a temporary `AI Status` column (OK / REVIEW) is displayed so reviewers can instantly spot rows where the anchor subject was missing from the Knowledge Base.
11) **One-Click Excel Export** — Approved answers exported to `.xlsx` via `pandas` + `xlsxwriter`. The `AI Status` column and all other internal UI columns are automatically stripped from the download so the exported file is clean.

---

## Architecture (How It Works)

### PDF Pipeline
1. **Extraction**: Clean PDF text ➜ sentence-aware character-based chunking ➜ Gemini `ExtractionList` schema returns `{question_id, question_text}` JSON.
2. **Batching & RAG**: Questions are answered in batches of 15 using the Knowledge Base; Pydantic `Answer` schema enforces structure.
3. **Rate-Limit Resilience**: Automatic retries and cooldowns when 429/`RESOURCE_EXHAUSTED` occurs.
4. **HITL Review**: Streamlit UI shows metrics, confidence, and flags; humans edit only the relevant fields.
5. **Export**: Approved dataframe ➜ Excel (`completed_questionnaire.xlsx`).

### Native Excel Pipeline
1. **Loading**: Excel file is read with `openpyxl` via pandas. All columns are preserved — including completely empty ones, since those are the answer columns the AI needs to fill. `NaN` values are replaced with empty strings for clean JSON serialisation.
2. **Row-Based Batching**: The DataFrame is converted to a list of row dictionaries and chunked into sub-lists of 15 rows. Each sub-list becomes one API call. Row-count batching is used exclusively here — never character-count — to prevent JSON objects from being split mid-value.
3. **Dynamic Inference**: For each batch, the LLM receives the exact column headers and all row data. It dynamically detects the anchor subject per row, applies context mapping against the Knowledge Base, and respects categorical constraints derived from the column headers or placeholder values.
4. **Safe Reassembly**: After all batches return, the final DataFrame is reconstructed with `pd.DataFrame(results, columns=original_headers)` — enforcing that every original column exists regardless of what the LLM returned. An `_AI_Status` column (OK/REVIEW) is appended for HITL review.
5. **Export**: The `_AI_Status` column and all other UI-only columns are stripped before writing to `.xlsx`, producing a clean file that matches the original upload structure.

---

## Prerequisites
- Python 3.10+  
- Google Gemini API key (Free tier works; set rate limits accordingly)  

---

## Installation
```bash
git clone https://github.com/your-org/hitl-security-questionnaire-assistant.git
cd hitl-security-questionnaire-assistant
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` (or set env vars) with:
```
GOOGLE_API_KEY=your_gemini_key_here
```

---

## Usage
```bash
streamlit run app.py
```
1. Upload the Knowledge Base (txt/pdf/csv/xlsx) and the blank questionnaire (xlsx/pdf) in the sidebar.
2. Click **Analyze & Process Files**. The pipeline auto-detects the questionnaire format and runs the appropriate flow.
   - **PDF questionnaire**: Extracts questions ➜ answers them in batches ➜ shows progress per batch.
   - **Excel questionnaire**: Loads the file preserving all columns ➜ processes rows in batches of 15 ➜ fills cells in-place.
3. Review in the **Review & Edit** tab.
   - *PDF mode*: Edit `proposed_yes_no` / `proposed_comments`, check confidence flags and reasoning.
   - *Excel mode*: Edit any cell directly. Use the `AI Status` column (OK / REVIEW) to quickly spot rows that need attention.
4. Click **Approve Final Answers**.
5. In the **Export Options** tab, download `completed_questionnaire.xlsx`. The exported file matches the original structure exactly — no internal columns included.

---

## Tech Stack
- **LLM**: Google GenAI (Gemini 2.5 Flash) via `google-genai` (new SDK — not the deprecated `google-generativeai`)
- **Validation**: Pydantic models for strict JSON schemas
- **UI**: Streamlit (`st.data_editor`, metrics, tabs)
- **Data**: pandas, `openpyxl` (Excel read/write), pdfplumber / PyMuPDF (PDF parsing)
- **Export**: xlsxwriter

---

## Notes & Best Practices
- Free tier is 15 RPM — the pipeline includes retries and cooldowns; keep the app open during long runs.
- Provide a concise, high-quality Knowledge Base to minimize “Low confidence” flags and “REVIEW” statuses.
- PDFs vary widely; the cleaning + sentence-aware chunker is optimized for noisy questionnaires but can be tweaked via `clean_extracted_text` and `get_smart_chunks`.
- For Excel questionnaires, include clear column headers that describe the expected answer format (e.g., “Status (Yes/No/Partial)”). The AI uses these as data-validation rules and will restrict its output accordingly.
- Make sure you are using `google-genai` (not `google-generativeai`) to avoid deprecation warnings. If both are installed, run `pip uninstall google-generativeai`.

---

## Roadmap
- Add vector store retrieval to further tighten grounding.
- Support round-trip PDF writing for forms.
- Role-based access and audit logging for approvals.
- Per-column confidence scores in Native Excel Mode.
- Auto-detection of sheet tabs in multi-sheet Excel questionnaires.

---

## License
MIT License. Contributions welcome!  
