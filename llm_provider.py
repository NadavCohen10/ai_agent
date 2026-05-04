"""
Strategy pattern for LLM providers.
Adding a new provider = subclassing BaseLLMProvider and implementing generate_response().
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMProvider(ABC):
    """Abstract strategy interface for LLM backends."""

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> Any:
        """
        Call the LLM and return parsed JSON (dict or list).
        Wraps the internal _generate_response_impl with global logging.
        """
        print("\n" + "=" * 50)
        print("--- RAW REQUEST (System Prompt) ---")
        print(system_prompt)
        print("\n--- RAW REQUEST (User Prompt) ---")
        print(user_prompt)
        print("=" * 50 + "\n")

        raw_response, parsed_json = self._generate_response_impl(system_prompt, user_prompt, json_schema)

        print("\n" + "=" * 50)
        print("--- RAW RESPONSE ---")
        print(raw_response)
        print("=" * 50 + "\n")

        return parsed_json

    @abstractmethod
    def _generate_response_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> tuple[str, Any]:
        """
        Internal implementation. Must return a tuple of (raw_text, parsed_json).

        Args:
            system_prompt: The system instruction.
            user_prompt:   The user message / data to process.
            json_schema:   Optional JSON Schema dict constraining the response.
                           When None, the provider returns free-form JSON.

        Returns:
            Parsed JSON — a dict when json_schema is an object type,
            a list when the top-level type is an array.

        Raises:
            Exception on API errors; the caller is responsible for retries.
        """

    def rotate_key(self) -> None:
        """Rotate the API credential. No-op by default; override when supported."""


class GeminiProvider(BaseLLMProvider):
    """Concrete strategy backed by Google Gemini (google-genai SDK)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        import os
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not key:
            raise ValueError("Set GEMINI_API_KEY in your .env file.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def _generate_response_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> tuple[str, Any]:
        from google.genai import types

        config_kwargs: dict = dict(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            max_output_tokens=8192,
            temperature=0.1,
        )
        if json_schema is not None:
            config_kwargs["response_schema"] = json_schema

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text, json.loads(response.text)


class OpenAIProvider(BaseLLMProvider):
    """Concrete strategy backed by OpenAI (openai SDK)."""

    def __init__(self, api_key: str = "", model: str = "gpt-5.4"):
        import openai
        import os
        from dotenv import load_dotenv

        load_dotenv()
        resolved_key = api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found. Set CHATGPT_API_KEY in your .env file or pass it explicitly."
            )
        self.client = openai.OpenAI(api_key=resolved_key)
        self.model = model

    def _generate_response_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> tuple[str, Any]:
        # Use Structured Outputs when a schema is provided, otherwise plain JSON mode.
        # Removing 'json_object' format when returning lists stops OpenAI from forcing object structures.
        if json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
            effective_system = system_prompt
        else:
            response_format = None
            effective_system = system_prompt + "\n\nReturn ONLY raw JSON. Do not include markdown formatting or explanations."

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_completion_tokens": 8192,
        }
        if response_format:
            kwargs["response_format"] = response_format

        completion = self.client.chat.completions.create(**kwargs)
        raw_text = completion.choices[0].message.content
        return raw_text, json.loads(_strip_markdown_fences(raw_text))


class OllamaProvider(BaseLLMProvider):
    """Concrete strategy backed by a local Ollama server (no API key required)."""

    def __init__(self, model: str = "gemma", base_url: str = "http://localhost:11434"):
        self.model = model
        self.chat_url = f"{base_url.rstrip('/')}/api/chat"

    def _generate_response_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> tuple[str, Any]:
        import requests

        # Embed the schema in the system prompt so the model knows the exact shape expected.
        effective_system = system_prompt
        if json_schema is not None:
            effective_system += (
                "\n\nYou MUST respond with valid JSON that strictly follows this schema:\n"
                + json.dumps(json_schema, indent=2)
                + "\nReturn ONLY the raw JSON — no markdown fences, no extra text."
            )
        else:
            effective_system += "\n\nReturn ONLY raw JSON — no markdown fences, no extra text."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",  # Ollama native JSON mode — suppresses most prose
            "stream": False,
            "options": {"temperature": 0.1},
        }

        response = requests.post(self.chat_url, json=payload, timeout=300)
        response.raise_for_status()

        raw = response.json()["message"]["content"]
        return raw, json.loads(_strip_markdown_fences(raw))


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown fences and conversational preamble to extract raw JSON."""
    text = text.strip()

    # 1. Try to extract from a markdown code block
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        
    # 2. Trim any remaining leading/trailing conversational garbage
    start_idx = text.find("{")
    start_array = text.find("[")
    if start_idx == -1:
        start_idx = start_array
    elif start_array != -1:
        start_idx = min(start_idx, start_array)
        
    end_idx = text.rfind("}")
    end_array = text.rfind("]")
    if end_idx == -1:
        end_idx = end_array
    elif end_array != -1:
        end_idx = max(end_idx, end_array)
        
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx : end_idx + 1]

    return text


class AnthropicProvider(BaseLLMProvider):
    """Concrete strategy backed by Anthropic Claude (anthropic SDK)."""

    def __init__(self, api_key: str = "", model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        import os
        from dotenv import load_dotenv

        load_dotenv()
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY") or ""
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in your .env file or pass it explicitly."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def _generate_response_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> tuple[str, Any]:
        # Embed schema in the system prompt — Anthropic's native JSON mode
        # enforces JSON output without needing tool use.
        effective_system = system_prompt
        if json_schema is not None:
            effective_system += (
                "\n\nYou MUST respond with valid JSON that strictly follows this schema:\n"
                + json.dumps(json_schema, indent=2)
                + "\nReturn ONLY the raw JSON — no markdown fences, no extra text."
            )
        else:
            effective_system += "\n\nReturn ONLY raw JSON — no markdown fences, no extra text."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=0.1,
            system=effective_system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        return raw, json.loads(_strip_markdown_fences(raw))
