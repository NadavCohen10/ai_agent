# HITL Security Questionnaire Assistant 🛡️🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)]()
[![Gemini](https://img.shields.io/badge/Gemini-2.5-orange)]()

**Tagline:** Automate, review, and export massive vendor security questionnaires using RAG, Gemini 2.5, and a Human-In-The-Loop UI.

---

## Why This Matters
Filling SOC 2 / HIPAA / PCI questionnaires is slow, error-prone, and painful. Raw PDFs are messy; naïvely sending the whole file to an LLM blows context windows and invites hallucinations. This project delivers a resilient, production-ready pipeline that cleans ugly PDFs, extracts questions without breaking sentences, answers them with your Knowledge Base, and routes low-confidence items to humans for quick approval and export.

![Dashboard UI](link-to-screenshot.png)

---

## Key Features
1) **Intelligent PDF Parsing & Cleaning** — Aggressive RegEx normalization removes invisible newlines and fixes broken words before any LLM call.  
2) **Sentence-Aware Smart Chunking** — Custom splitter prefers `?` or `.` boundaries (≤4,000 chars) so questions are never cut in half.  
3) **Structured Outputs (JSON)** — `google-genai` + Pydantic schemas enforce strict JSON arrays for both extraction and answering.  
4) **Fault-Tolerant RAG Engine** — Batch answering (15 Qs/batch) with retries, 429/`RESOURCE_EXHAUSTED` handling, and backoff sleeps.  
5) **Human-In-The-Loop UI** — Streamlit dashboard with metrics, tabs, and `st.data_editor` to review/edit answers; flags low confidence or missing info.  
6) **One-Click Excel Export** — Approved answers exported to `.xlsx` via `pandas` + `xlsxwriter`.  

---

## Architecture (How It Works)
1. **Extraction**: Clean PDF text ➜ sentence-aware chunking ➜ Gemini `ExtractionList` schema returns `{question_id, question_text}` JSON.  
2. **Batching & RAG**: Questions are answered in batches of 15 using the Knowledge Base; Pydantic `Answer` schema enforces structure.  
3. **Rate-Limit Resilience**: Automatic retries and cooldowns when 429/`RESOURCE_EXHAUSTED` occurs.  
4. **HITL Review**: Streamlit UI shows metrics, confidence, and flags; humans edit only the relevant fields.  
5. **Export**: Approved dataframe ➜ Excel (`completed_questionnaire.xlsx`).  

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
2. Click **Analyze & Process Files**. The app extracts all questions, answers them in batches, and shows progress.  
3. Review in **Review & Edit** tab: edit `proposed_yes_no` / `proposed_comments`, check flags and reasoning.  
4. Click **Approve Final Answers**.  
5. In **Export Options** tab, download `completed_questionnaire.xlsx`.  

---

## Tech Stack
- **LLM**: Google GenAI (Gemini 2.5) via `google-genai`  
- **Validation**: Pydantic models for strict JSON schemas  
- **UI**: Streamlit (`st.data_editor`, metrics, tabs)  
- **Data**: pandas, openpyxl, pdfplumber / PyMuPDF  
- **Export**: xlsxwriter  

---

## Notes & Best Practices
- Free tier is 15 RPM — the pipeline includes retries and cooldowns; keep the app open during long runs.  
- Provide a concise, high-quality Knowledge Base to minimize “Low confidence” flags.  
- PDFs vary widely; the cleaning + sentence-aware chunker is optimized for noisy questionnaires but can be tweaked via `clean_extracted_text` and `get_smart_chunks`.  

---

## Roadmap
- Add vector store retrieval to further tighten grounding.  
- Support round-trip PDF writing for forms.  
- Role-based access and audit logging for approvals.  

---

## License
MIT License. Contributions welcome!  
