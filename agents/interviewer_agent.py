"""
CISOInterviewer — builds a Company Baseline Profile by identifying which
of 14 foundational security controls are missing from the Knowledge Base,
interviewing the user to collect the missing details, and formalising the
answers into regulatory-style controls.

ZERO INFERENCE GUARANTEE:
translate_to_formal_control() will NEVER invent timeframes, frequencies,
SLAs, scope statements, or technology names that the user has not explicitly
stated. Vague answers trigger a follow-up question; only sufficient answers
produce a formal control.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from core.llm_provider import BaseLLMProvider


# ── 14 Baseline security topics ───────────────────────────────────────────────

BASELINE_TOPICS: list[dict] = [
    {
        "id": "GOV_POLICY",
        "domain": "Governance",
        "topic": "InfoSec Policy",
        "vra_question": (
            "Does the organization have a formally documented Information Security "
            "Policy that has been approved by senior management and is reviewed and "
            "updated at least annually?"
        ),
        "keywords": [
            "information security policy", "infosec policy", "security policy",
            "approved by management", "annual review", "annually reviewed",
        ],
    },
    {
        "id": "GOV_CISO",
        "domain": "Governance",
        "topic": "CISO / Security Officer",
        "vra_question": (
            "Has the organization formally appointed a Chief Information Security "
            "Officer (CISO) or equivalent Security Officer with defined responsibilities?"
        ),
        "keywords": [
            "ciso", "chief information security officer", "security officer",
            "head of security", "vp security", "security lead",
        ],
    },
    {
        "id": "IAM_MFA",
        "domain": "Identity & Access Management",
        "topic": "MFA / 2FA",
        "vra_question": (
            "Is Multi-Factor Authentication (MFA / 2FA) enforced for access to "
            "sensitive systems, administrative consoles, email, and VPN?"
        ),
        "keywords": [
            "mfa", "2fa", "multi-factor", "two-factor", "multifactor",
            "authenticator",
        ],
    },
    {
        "id": "IAM_RBAC",
        "domain": "Identity & Access Management",
        "topic": "RBAC & Least Privilege",
        "vra_question": (
            "Does the organization enforce Role-Based Access Control (RBAC) and "
            "the principle of Least Privilege for all system and data access?"
        ),
        "keywords": [
            "rbac", "role-based access", "least privilege", "role based",
            "need-to-know", "access control model",
        ],
    },
    {
        "id": "IAM_OFFBOARD",
        "domain": "Identity & Access Management",
        "topic": "Offboarding & Access Revocation",
        "vra_question": (
            "Is there a documented offboarding process with a defined SLA for "
            "revoking all access rights upon employee termination or role change?"
        ),
        "keywords": [
            "offboarding", "termination", "access revocation", "revoke access",
            "employee departure", "deprovisioning",
        ],
    },
    {
        "id": "DATA_REST",
        "domain": "Data Protection",
        "topic": "Encryption at Rest",
        "vra_question": (
            "Is all sensitive and confidential data encrypted at rest across "
            "all storage systems, including databases, file servers, and cloud storage?"
        ),
        "keywords": [
            "encrypted at rest", "encryption at rest", "data at rest",
            "at-rest encryption", "storage encryption",
        ],
    },
    {
        "id": "DATA_TRANSIT",
        "domain": "Data Protection",
        "topic": "Encryption Standards",
        "vra_question": (
            "What encryption standards does the organization use for data in transit "
            "and at rest (e.g., AES-256, TLS 1.2+)?"
        ),
        "keywords": [
            "aes-256", "aes 256", "tls 1.2", "tls 1.3", "tls1.2", "tls1.3",
            "encryption standard", "in transit", "data in transit",
        ],
    },
    {
        "id": "VULN_PENTEST",
        "domain": "Vulnerability Management",
        "topic": "Penetration Testing",
        "vra_question": (
            "How frequently does the organization conduct third-party penetration "
            "tests of its internet-facing systems and internal infrastructure?"
        ),
        "keywords": [
            "penetration test", "pentest", "pen test", "pen-test",
            "ethical hack", "red team",
        ],
    },
    {
        "id": "VULN_PATCH",
        "domain": "Vulnerability Management",
        "topic": "Critical Patch SLA",
        "vra_question": (
            "What is the organization's defined SLA (in days) for patching or "
            "remediating critical severity vulnerabilities?"
        ),
        "keywords": [
            "patch sla", "patching sla", "remediation sla", "critical vulnerability",
            "critical patch", "vulnerability remediation",
        ],
    },
    {
        "id": "IR_PLAN",
        "domain": "Incident Response",
        "topic": "Incident Response Plan",
        "vra_question": (
            "Does the organization have a documented Incident Response Plan (IRP) "
            "that is tested via tabletop exercises at least annually?"
        ),
        "keywords": [
            "incident response plan", "irp", "tabletop", "table-top",
            "incident response procedure",
        ],
    },
    {
        "id": "IR_BREACH",
        "domain": "Incident Response",
        "topic": "Breach Notification SLA",
        "vra_question": (
            "What is the organization's contractual or regulatory SLA for notifying "
            "affected parties and authorities following a material data breach?"
        ),
        "keywords": [
            "breach notification", "data breach", "breach report",
            "notify customers", "regulatory notification", "gdpr notification",
            "72 hour", "72-hour",
        ],
    },
    {
        "id": "HR_BACKGROUND",
        "domain": "HR Security",
        "topic": "Background Checks",
        "vra_question": (
            "Are pre-employment background checks conducted for all employees "
            "prior to granting access to sensitive systems or data?"
        ),
        "keywords": [
            "background check", "background screening", "pre-employment",
            "employee screening", "criminal check",
        ],
    },
    {
        "id": "HR_SAT",
        "domain": "HR Security",
        "topic": "Security Awareness Training",
        "vra_question": (
            "How frequently does the organization conduct security awareness training "
            "and phishing simulation exercises for all employees?"
        ),
        "keywords": [
            "security awareness", "awareness training", "phishing drill",
            "phishing simulation", "phishing test", "security training",
        ],
    },
    {
        "id": "BCDR",
        "domain": "Business Continuity",
        "topic": "BCP / DRP",
        "vra_question": (
            "Does the organization have a documented and tested Business Continuity "
            "Plan (BCP) and Disaster Recovery Plan (DRP) with defined RTO and RPO targets?"
        ),
        "keywords": [
            "business continuity", "disaster recovery", "bcp", "drp",
            "rto", "rpo", "continuity plan",
        ],
    },
]

TOTAL_BASELINE = len(BASELINE_TOPICS)
_TOPIC_MAP: dict = {t["id"]: t for t in BASELINE_TOPICS}

# Required detail hints per topic — used in the Zero Inference prompt
_REQUIRED_DETAILS: dict[str, str] = {
    "GOV_POLICY":    "review frequency, who approved it",
    "GOV_CISO":      "name or role title, reporting line",
    "IAM_MFA":       "systems in scope, approved second-factor methods",
    "IAM_RBAC":      "access request/approval process, review frequency",
    "IAM_OFFBOARD":  "SLA in hours/days for access revocation after termination",
    "DATA_REST":     "storage systems in scope (DB, file server, cloud)",
    "DATA_TRANSIT":  "exact standard(s) used (e.g. AES-256, TLS 1.2+)",
    "VULN_PENTEST":  "frequency (e.g. annually), provider type (CREST-accredited)",
    "VULN_PATCH":    "SLA in days for critical, high, medium severity",
    "IR_PLAN":       "tabletop test frequency, RCA deadline after incident",
    "IR_BREACH":     "notification SLA in hours, which authorities/parties notified",
    "HR_BACKGROUND": "scope (all employees / contractors), timing (pre-hire / within N days)",
    "HR_SAT":        "training frequency, phishing simulation frequency, remediation for failures",
    "BCDR":          "test frequency, RTO target, RPO target per system tier",
}


# ── LLM schemas ───────────────────────────────────────────────────────────────

_QUESTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
    },
    "required": ["question"],
}

_TRANSLATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "response_type":     {"type": "string"},
        "question":          {"type": "string"},
        "domain":            {"type": "string"},
        "control_statement": {"type": "string"},
    },
    "required": ["response_type", "question", "domain", "control_statement"],
}


# ── System prompts ────────────────────────────────────────────────────────────

_INTERVIEW_QUESTION_SYSTEM = (
    "You are a senior CISO conducting an internal security baseline interview. "
    "Rephrase the formal compliance question provided into a single, warm, "
    "conversational question (1-2 sentences). No jargon. No preamble. "
    "Return ONLY the JSON object."
)

_TRANSLATE_SYSTEM = """\
You are a strict cybersecurity compliance officer building an organization's \
security baseline profile. Evaluate the interview transcript and decide whether \
enough specific, verifiable detail has been provided to write a formal control.

══════════════════════════════════════════════════════
CRITICAL RULE — ZERO INFERENCE (NON-NEGOTIABLE)
══════════════════════════════════════════════════════
You are ABSOLUTELY FORBIDDEN from inventing, assuming, or extrapolating ANY \
specific value the user did not explicitly state. This includes:
  • Timeframes        — e.g. "annually", "quarterly", "within 30 days"
  • Frequencies       — e.g. "monthly", "weekly", "every 6 months"
  • Numerical values  — e.g. "7 days", "AES-256", "3 vendors"
  • Scope statements  — e.g. "all employees", "all systems" (unless user stated it)
  • Technology names  — unless the user named the product
  • Process steps     — unless the user described them
  • Regulatory labels — e.g. "GDPR", "ISO 27001" (unless user mentioned them)

Violations produce legally unreliable policy documents and MUST NOT occur.
══════════════════════════════════════════════════════

DECISION LOGIC:
Step 1 — Read the FULL interview transcript.
Step 2 — Check which REQUIRED DETAILS (provided below) are still MISSING.
Step 3 — If ANY required detail is missing:
           • response_type = "follow_up"
           • question      = ONE short, specific question for the single most
                             important missing detail. Do not ask multiple things.
           • domain            = ""
           • control_statement = ""
Step 4 — If ALL required details are present:
           • response_type     = "formal_control"
           • question          = ""
           • domain            = the relevant security domain
           • control_statement = a single, enforceable policy clause built ONLY
                                 from facts the user stated.
                                 ALWAYS write in professional English, even if
                                 the user answered in Hebrew or another language.
                                 Format: "<Scope> must/shall <action> <user-stated value>."

Return ONLY the JSON object. No explanation, no markdown.\
"""


# ── Module-level coverage helpers (no LLM required) ──────────────────────────
# Exported as standalone functions so the Document Ingestion page can run gap
# analysis without needing an Ollama connection or a CISOInterviewer instance.

def compute_coverage(
    kb_df: pd.DataFrame, answered_ids: list[str] | None = None
) -> tuple[int, set[str]]:
    """
    Return (covered_count, covered_id_set).

    Coverage = keyword match in KB  OR  explicitly answered this session.
    answered_ids is a list stored in st.session_state (must be JSON-safe).
    """
    all_text = (
        " ".join(kb_df["control_statement"].str.lower().tolist())
        if not kb_df.empty else ""
    )
    answered = set(answered_ids or [])
    covered = {
        t["id"] for t in BASELINE_TOPICS
        if t["id"] in answered
        or any(kw.lower() in all_text for kw in t["keywords"])
    }
    return len(covered), covered


def find_baseline_gaps(
    kb_df: pd.DataFrame, answered_ids: list[str] | None = None
) -> list[dict]:
    """Return BASELINE_TOPICS entries not yet covered by the KB or answered_ids."""
    _, covered = compute_coverage(kb_df, answered_ids)
    return [t for t in BASELINE_TOPICS if t["id"] not in covered]


class CISOInterviewer:
    """
    Drives the Company Baseline Profile wizard.

    Use OllamaProvider to keep all document data on-premises.
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    # ── Coverage helpers (delegate to module-level functions) ─────────────────

    def compute_coverage(
        self, kb_df: pd.DataFrame, answered_ids: list[str] | None = None
    ) -> tuple[int, set[str]]:
        return compute_coverage(kb_df, answered_ids)

    def find_baseline_gaps(
        self, kb_df: pd.DataFrame, answered_ids: list[str] | None = None
    ) -> list[dict]:
        return find_baseline_gaps(kb_df, answered_ids)

    # ── Interview helpers ─────────────────────────────────────────────────────

    def generate_interview_question(self, topic: dict) -> str:
        """
        Ask the LLM to rephrase the formal requirement as a friendly question.
        Falls back to the raw vra_question on any error.
        """
        user_prompt = (
            f"Formal baseline question:\n{topic['vra_question']}\n\n"
            "Rephrase this as a friendly, conversational interview question."
        )
        try:
            result = self.provider.generate_response(
                _INTERVIEW_QUESTION_SYSTEM, user_prompt, _QUESTION_SCHEMA
            )
            if isinstance(result, dict) and result.get("question"):
                return result["question"]
        except Exception as exc:
            print(f"[CISOInterviewer] Question generation failed: {exc}")
        return topic["vra_question"]

    def translate_to_formal_control(
        self, topic: dict, chat_history: list[dict]
    ) -> dict:
        """
        Evaluate the full conversation and return ONE of:

          {"response_type": "follow_up",     "question": "..."}
          {"response_type": "formal_control", "domain": "...",
           "control_statement": "...", "source_document": "CISO Interview"}

        Zero Inference is enforced: specific values must come from the user's
        own words. Output is always in professional English.
        """
        transcript = "\n".join(
            f"[{'Interviewer' if m['role'] == 'assistant' else 'User'}]: {m['content']}"
            for m in chat_history
        )

        required = _REQUIRED_DETAILS.get(topic["id"], "all relevant specifics")

        user_prompt = (
            f"Security Domain    : {topic['domain']}\n"
            f"Baseline Topic     : {topic['topic']}\n"
            f"VRA Requirement    : {topic['vra_question']}\n"
            f"Required Details   : {required}\n\n"
            f"Interview Transcript:\n{transcript}\n\n"
            "Evaluate the transcript and return the appropriate JSON response."
        )

        try:
            result = self.provider.generate_response(
                _TRANSLATE_SYSTEM, user_prompt, _TRANSLATE_SCHEMA
            )
            if isinstance(result, dict):
                rtype = result.get("response_type", "").strip().lower()

                if rtype == "follow_up" and result.get("question", "").strip():
                    return {
                        "response_type": "follow_up",
                        "question": result["question"].strip(),
                    }

                if rtype == "formal_control" and result.get("control_statement", "").strip():
                    return {
                        "response_type": "formal_control",
                        "domain":            result.get("domain", topic["domain"]).strip(),
                        "control_statement": result["control_statement"].strip(),
                        "source_document":   "CISO Interview",
                    }

        except Exception as exc:
            print(f"[CISOInterviewer] Translation failed: {exc}")

        # Safe fallback — never invent; ask for more detail
        return {
            "response_type": "follow_up",
            "question": (
                f"Could you be more specific? I still need: {required}."
            ),
        }
