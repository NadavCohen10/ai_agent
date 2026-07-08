# AI Security Questionnaire Assistant 🛡️🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)]()
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)]()
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Haiku-yellow)]()
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-gray)]()

Automate Vendor Risk Assessment (VRA) and security compliance questionnaires using RAG, multiple LLM providers, and a Human-In-The-Loop Streamlit UI. Includes a privacy-safe Document Ingestion engine and an AI-driven CISO Interviewer that builds your organisation's security baseline — all without hallucinating facts.

---

## What This Does

Security teams spend hours answering the same SOC 2 / ISO 27001 / PCI-DSS questionnaires sent by customers and vendors. This tool:

1. **Ingests** your internal security policy documents locally (no data leaves your machine) and extracts a structured Knowledge Base.
2. **Interviews** your CISO / security lead to fill 14 baseline controls that are missing from the KB — with a strict Zero Inference guarantee.
3. **Answers** external questionnaires (Excel or PDF) automatically using the KB, with a HITL review step before export.

---

## Architecture

```
ai_agent/
├── app/
│   ├── app.py                      # Main Assessor page (KB upload + questionnaire)
│   ├── exporter.py                 # Excel export helper
│   └── pages/
│       ├── 1_document_ingestion.py # Local document → KB pipeline
│       └── 2_ciso_interviewer.py   # Baseline Profile wizard
│
├── agents/
│   ├── assessor_agent.py           # Questionnaire answering logic + prompts
│   └── interviewer_agent.py        # CISO interview logic + 14 baseline topics
│
├── core/
│   └── llm_provider.py             # Strategy Pattern — Gemini / OpenAI / Anthropic / Ollama
│
├── ingestion/
│   ├── kb_builder.py               # Ollama-powered local document scanner
│   └── document_parser.py          # PDF / Excel / TXT / CSV text extractor
│
├── data/
│   ├── mock_organization/          # Sample policy documents (generated)
│   └── Mock_Org_KB.xlsx            # Generated Knowledge Base (git-ignored)
│
└── generate_mock_org.py            # Script to create sample policy documents
```

---

## Three Workflows

### 1. Document Ingestion (`pages/1_document_ingestion.py`)
Scans a local folder of security policy documents (`.txt`, `.md`) and extracts structured controls using a **local Ollama model** — no document content is ever sent to an external API.

- Runs entirely on-premises via Ollama
- Extracts `domain` + `control_statement` per policy clause
- Assigns each control to one of 10 standard security domains
- Saves the result to `data/Mock_Org_KB.xlsx`

### 2. CISO Interviewer — Company Baseline Profile (`pages/2_ciso_interviewer.py`)
A guided wizard that checks whether 14 foundational security controls are present in the KB. For each missing control it interviews the user, collects the required specifics, and formalises the answer.

**14 Baseline Topics:**

| Domain | Topic |
|---|---|
| Governance | InfoSec Policy |
| Governance | CISO / Security Officer |
| Identity & Access Management | MFA / 2FA |
| Identity & Access Management | RBAC & Least Privilege |
| Identity & Access Management | Offboarding & Access Revocation |
| Data Protection | Encryption at Rest |
| Data Protection | Encryption Standards |
| Vulnerability Management | Penetration Testing |
| Vulnerability Management | Critical Patch SLA |
| Incident Response | Incident Response Plan |
| Incident Response | Breach Notification SLA |
| HR Security | Background Checks |
| HR Security | Security Awareness Training |
| Business Continuity | BCP / DRP |

**Zero Inference Guarantee:** The LLM is forbidden from inventing timeframes, SLAs, frequencies, scope statements, or technology names. If the user's answer is vague, the agent asks a targeted follow-up question. Only when all required specifics are provided does it produce a formal, regulatory-style control statement — always in professional English.

The wizard is **optional and non-blocking**. A visual progress bar (`X / 14 complete`) tracks completion. Users can skip topics or navigate away at any time.

### 3. Assessor Agent (`app/app.py`)
Answers external vendor questionnaires (Excel or PDF) using the Knowledge Base built in steps 1 and 2.

**Excel mode (in-place filling):**
- Preserves the original file structure column-for-column
- Dynamic Anchor Detection — identifies the subject per row without relying on hardcoded column names
- Strict Categorical Matching — respects dropdown constraints (`Yes/No`, `Compliant/Non-Compliant`, etc.)
- Zero Inference — leaves cells blank rather than writing "N/A" or placeholder text
- Adds `_AI_Status` (`OK` / `REVIEW`) for HITL review

**PDF mode:**
- Sentence-aware chunking keeps questions intact
- Batch answering (configurable batch size) with retry / backoff

---

## Multi-Provider LLM Support (Strategy Pattern)

All LLM calls go through a common `BaseLLMProvider` interface. Switch providers from the sidebar without changing any business logic.

| Provider | Best For | Key |
|---|---|---|
| **Gemini 2.5 Flash** | Assessor (speed + cost) | `GEMINI_API_KEY` in `.env` |
| **OpenAI GPT-4o** | Assessor (accuracy) | `CHATGPT_API_KEY` in `.env` |
| **Anthropic Claude Sonnet** | Assessor (reasoning) | `ANTHROPIC_API_KEY` in `.env` |
| **Ollama (local)** | Document Ingestion & CISO Interview (privacy) | No key — runs locally |

Structured output (JSON schema enforcement) is supported for all four providers.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed locally with at least one model pulled:
  ```bash
  ollama pull gemma
  ```
- At least one cloud LLM API key (for the Assessor Agent)

---

## Installation

```bash
git clone https://github.com/your-org/ai-security-questionnaire-assistant.git
cd ai-security-questionnaire-assistant/ai_agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with the keys you have:

```env
GEMINI_API_KEY=your_gemini_key
CHATGPT_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Usage

```bash
streamlit run app/app.py
```

### Recommended first-run order

1. **Generate sample documents** (optional, for testing):
   ```bash
   python generate_mock_org.py
   ```

2. **Document Ingestion** → sidebar page `1 Document Ingestion`
   - Point to your policy document folder
   - Select Ollama model and click **Scan Local Directory**
   - Save the extracted KB to `data/Mock_Org_KB.xlsx`

3. **CISO Interviewer** → sidebar page `2 CISO Interviewer`
   - Review the progress bar — see which of 14 baseline topics are already covered
   - Answer the interview questions; the agent will ask follow-ups until it has enough specifics
   - Approve each proposed control to add it to the KB automatically

4. **Assessor Agent** → main page
   - Upload the KB and a blank questionnaire (Excel or PDF)
   - Choose your LLM provider from the sidebar
   - Click **Analyze & Process Files**
   - Review answers in the **Review & Edit** tab
   - Export the completed questionnaire

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page, `st.chat_message`, `st.data_editor`) |
| LLM Abstraction | Custom Strategy Pattern (`BaseLLMProvider`) |
| Cloud LLMs | `google-genai`, `openai`, `anthropic` |
| Local LLM | Ollama REST API (`/api/chat`) |
| Data | pandas, openpyxl |
| PDF Parsing | pdfplumber |
| Export | xlsxwriter |
| Config | python-dotenv |

---

## Privacy Model

| Component | Data Destination |
|---|---|
| Document Ingestion | Local only (Ollama) |
| CISO Interviewer | Local only (Ollama) |
| Assessor Agent | Cloud LLM of your choice (or Ollama) |

Internal policy documents and interview answers never leave the machine when using Ollama for ingestion and the CISO wizard.

---

## Notes

- **Assessor accuracy improves significantly** after completing Document Ingestion and the CISO Baseline Profile — the KB is the single biggest factor in answer quality.
- **Zero Inference is non-negotiable** in the CISO Interviewer. If you get multiple follow-up questions, the agent genuinely needs those specifics to produce a defensible policy control.
- **Ollama model quality matters** — `gemma2` or `llama3` produce noticeably better extractions than the base `gemma` model for complex policy documents.
- For cloud LLMs, if no API key is found in `.env`, the sidebar shows a manual input field.

---

## Roadmap

- Vector store retrieval (FAISS / ChromaDB) for large KB files
- Round-trip PDF writing for fillable forms
- Per-cell confidence scores in Native Excel Mode
- Audit log for HITL approvals
- Multi-sheet Excel questionnaire support

---

## License

MIT License. Contributions welcome!
