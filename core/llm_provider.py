"""
Strategy pattern for LLM providers.
Adding a new provider = subclassing BaseLLMProvider and implementing _generate_response_impl().
"""

from __future__ import annotations

import hashlib
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

    def generate_response_cached(
        self,
        system_prompt: str,
        static_prefix: str,
        dynamic_suffix: str,
        json_schema: Optional[dict] = None,
    ) -> Any:
        """
        Optimised path: static_prefix (headers + KB) is cacheable; dynamic_suffix
        (row instructions + batch JSON) changes per call.

        Default implementation concatenates both and calls generate_response().
        Providers that support native caching override this method.
        """
        return self.generate_response(
            system_prompt,
            static_prefix + "\n\n" + dynamic_suffix,
            json_schema,
        )

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



class GeminiProvider(BaseLLMProvider):
    """
    Concrete strategy backed by Google Gemini (google-genai SDK).

    Supports Context Caching: the system prompt + KB prefix is uploaded once per
    session (per unique KB hash) and referenced by name in subsequent calls,
    avoiding redundant input-token charges across all rows of a questionnaire run.
    Falls back to uncached if the content is below Gemini's minimum token threshold
    or if the caching API is unavailable.
    """

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
        # Maps MD5(system_prompt + static_prefix) → cache_name (str) or None (uncacheable)
        self._cache_registry: dict[str, str | None] = {}

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

    def _get_or_create_cache(self, system_prompt: str, static_prefix: str) -> str | None:
        """
        Return an existing Gemini context-cache name for this (system_prompt, KB) pair,
        creating it on first use. Returns None if caching is unavailable.
        """
        from google.genai import types

        cache_key = hashlib.md5(f"{system_prompt}{static_prefix}".encode()).hexdigest()

        if cache_key in self._cache_registry:
            name = self._cache_registry[cache_key]
            if name:
                print(f"[Gemini Cache] Reusing cache: {name}")
            return name

        try:
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    contents=[static_prefix],
                    ttl="1800s",  # 30-minute TTL — well beyond any single questionnaire run
                ),
            )
            print(f"[Gemini Cache] Created new cache: {cache.name}")
            self._cache_registry[cache_key] = cache.name
            return cache.name
        except Exception as e:
            print(f"[Gemini Cache] Cache creation failed ({e}). Falling back to uncached.")
            self._cache_registry[cache_key] = None
            return None

    def generate_response_cached(
        self,
        system_prompt: str,
        static_prefix: str,
        dynamic_suffix: str,
        json_schema: Optional[dict] = None,
    ) -> Any:
        from google.genai import types

        # Attempt 1: use (or create) a context cache for the static prefix.
        # Attempt 2: if the cached call fails with NOT_FOUND (cache expired mid-run),
        #            invalidate the registry entry and recreate before giving up.
        cache_key = hashlib.md5(f"{system_prompt}{static_prefix}".encode()).hexdigest()

        for _pass in range(2):
            cache_name = self._get_or_create_cache(system_prompt, static_prefix)

            if cache_name is None:
                # Caching unavailable — fall back to the regular uncached path.
                combined = static_prefix + "\n\n" + dynamic_suffix
                return self.generate_response(system_prompt, combined, json_schema)

            config_kwargs: dict = dict(
                cached_content=cache_name,
                response_mime_type="application/json",
                max_output_tokens=8192,
                temperature=0.1,
            )
            if json_schema is not None:
                config_kwargs["response_schema"] = json_schema

            print("\n" + "=" * 50)
            print(f"[Gemini Cached Request] cache={cache_name}, dynamic={len(dynamic_suffix)} chars")
            print(dynamic_suffix[:400])
            print("=" * 50 + "\n")

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=dynamic_suffix,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                raw = response.text
                print(f"\n[Gemini Cached Response]\n{raw[:400]}\n")
                return json.loads(raw)

            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e).upper():
                    # Cache expired between creation and use — clear and let the loop recreate it.
                    print(f"[Gemini Cache] Cache expired mid-run. Recreating...")
                    self._cache_registry.pop(cache_key, None)
                    continue
                raise  # propagate all other errors to the caller's retry logic

        # Both passes failed — fall back to uncached.
        combined = static_prefix + "\n\n" + dynamic_suffix
        return self.generate_response(system_prompt, combined, json_schema)


class OpenAIProvider(BaseLLMProvider):
    """
    Concrete strategy backed by OpenAI (openai SDK).

    OpenAI automatically caches prompt prefixes > 1024 tokens at a discounted rate —
    no explicit API changes are needed. generate_response_cached() inherits the default
    (concatenate + call generate_response), which places the large static KB prefix
    first so OpenAI's automatic caching applies to it.
    """

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
    """
    Concrete strategy backed by Anthropic Claude (anthropic SDK).

    Supports Prompt Caching: the system prompt and the static KB prefix are marked
    with cache_control={"type":"ephemeral"}, causing Anthropic to cache them for
    up to 5 minutes at ~10% of the normal input-token cost. Only the dynamic row
    batch is sent fresh on each call, giving a significant cost reduction across
    all rows of a questionnaire run.
    """

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

    def generate_response_cached(
        self,
        system_prompt: str,
        static_prefix: str,
        dynamic_suffix: str,
        json_schema: Optional[dict] = None,
    ) -> Any:
        """
        Send the system prompt and KB prefix with cache_control=ephemeral so Anthropic
        caches them for 5 minutes. Only the dynamic row batch is transmitted fresh,
        cutting input-token cost by ~90% across a typical questionnaire run.

        Falls back to the regular uncached path on any API error.
        """
        effective_system_text = system_prompt
        if json_schema is not None:
            effective_system_text += (
                "\n\nYou MUST respond with valid JSON that strictly follows this schema:\n"
                + json.dumps(json_schema, indent=2)
                + "\nReturn ONLY the raw JSON — no markdown fences, no extra text."
            )
        else:
            effective_system_text += "\n\nReturn ONLY raw JSON — no markdown fences, no extra text."

        print("\n" + "=" * 50)
        print(f"[Anthropic Cached Request] static={len(static_prefix)} chars, dynamic={len(dynamic_suffix)} chars")
        print(dynamic_suffix[:400])
        print("=" * 50 + "\n")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=0.1,
                # Cache the system prompt — reduces charge to ~10% on cache hits.
                system=[
                    {
                        "type": "text",
                        "text": effective_system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            # Cache the large static KB prefix.
                            {
                                "type": "text",
                                "text": static_prefix,
                                "cache_control": {"type": "ephemeral"},
                            },
                            # Dynamic suffix is NOT cached — it changes every call.
                            {
                                "type": "text",
                                "text": dynamic_suffix,
                            },
                        ],
                    }
                ],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            raw = response.content[0].text
            print(f"\n[Anthropic Cached Response]\n{raw[:400]}\n")
        except Exception as e:
            print(f"[Anthropic Cache] Cached call failed ({e}). Falling back to uncached.")
            combined = static_prefix + "\n\n" + dynamic_suffix
            return self.generate_response(system_prompt, combined, json_schema)

        # JSON parse is outside the API error handler so JSONDecodeError propagates
        # to _process_batch_with_fallback's split-retry logic instead of silently
        # falling back to uncached with a malformed response.
        return json.loads(_strip_markdown_fences(raw))
