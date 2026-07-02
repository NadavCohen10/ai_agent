"""
KBBuilder — scans a local directory of security policy documents,
extracts structured controls via the local Ollama LLM (privacy-safe,
no data leaves the machine), and returns a pandas DataFrame.
"""

import sys
import os

# Resolve project root so core/ imports work regardless of invocation path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from typing import Callable, Optional

from core.llm_provider import OllamaProvider


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a cybersecurity compliance analyst. Your task is to read a corporate \
security policy document and extract every specific, measurable security control \
it contains.

Rules:
1. Each extracted item must be a single, self-contained control statement.
2. Focus on controls with concrete, verifiable values: numbers, timeframes, \
technology names, thresholds, or named frameworks (e.g. "Passwords must be \
at least 14 characters", "MFA is mandatory for all accounts", \
"Backups are retained for 30 days").
3. Do NOT extract vague or aspirational statements (e.g. "security is important").
4. Assign each control to exactly one domain from this list:
   Access Control | Authentication | Encryption | Network Security |
   Incident Response | Backup & Recovery | Vulnerability Management |
   Endpoint Security | Logging & Monitoring | Compliance & Legal
5. Return ONLY the JSON object — no markdown, no explanation.
"""

_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "controls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain":            {"type": "string"},
                    "control_statement": {"type": "string"},
                },
                "required": ["domain", "control_statement"],
            },
        }
    },
    "required": ["controls"],
}

COLUMNS = ["domain", "control_statement", "source_document"]
SUPPORTED_EXTENSIONS = {".txt", ".md"}


class KBBuilder:
    """
    Scans a local directory, extracts security controls from each document
    using the local Ollama LLM, and aggregates them into a DataFrame.

    Args:
        directory: Path to the folder containing policy documents.
        model:     Ollama model tag (must be pulled locally, e.g. "gemma").
        base_url:  Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        directory: str,
        model: str = "gemma",
        base_url: str = "http://localhost:11434",
    ):
        self.directory = Path(directory)
        self.provider = OllamaProvider(model=model, base_url=base_url)

    def build(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Iterate over supported files, extract controls, return a DataFrame.
        progress_callback receives (current_index, total, filename).
        """
        files = sorted(
            p for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            print(f"[KBBuilder] No supported files found in {self.directory}")
            return pd.DataFrame(columns=COLUMNS)

        all_controls: list[dict] = []

        for idx, filepath in enumerate(files):
            if progress_callback:
                progress_callback(idx, len(files), filepath.name)

            print(f"[KBBuilder] Processing {filepath.name} ({idx + 1}/{len(files)})...")
            try:
                controls = self._extract_from_file(filepath)
                all_controls.extend(controls)
                print(f"[KBBuilder]   → {len(controls)} controls extracted")
            except Exception as exc:
                print(f"[KBBuilder]   ✗ Failed on {filepath.name}: {exc}")

        if progress_callback:
            progress_callback(len(files), len(files), "Done")

        df = pd.DataFrame(all_controls, columns=COLUMNS)
        if not df.empty:
            df["domain"] = df["domain"].str.strip().str.title()
        return df

    def _extract_from_file(self, filepath: Path) -> list[dict]:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            return []

        user_prompt = (
            f"Document name: {filepath.name}\n\n"
            f"Document content:\n{text}\n\n"
            "Extract all security controls and return the JSON."
        )

        result = self.provider.generate_response(
            _SYSTEM_PROMPT,
            user_prompt,
            _EXTRACTION_SCHEMA,
        )

        controls: list[dict] = []
        if isinstance(result, dict):
            controls = result.get("controls", [])
        elif isinstance(result, list):
            controls = result

        for ctrl in controls:
            ctrl["source_document"] = filepath.name

        return [
            c for c in controls
            if c.get("domain") and c.get("control_statement")
        ]
