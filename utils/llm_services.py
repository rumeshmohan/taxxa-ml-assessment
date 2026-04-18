import json
import re
import time
from typing import Any, Optional
from openai import OpenAI


class LLMService:
    """Universal wrapper for any OpenAI-compatible API endpoint."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client     = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.is_groq    = "groq" in base_url.lower()

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                if self.is_groq:
                    time.sleep(2.1)   # proactive pacing: ~30 RPM free tier
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e).lower()
                if ("429" in msg or "rate limit" in msg) and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"[RATE LIMIT] retrying in {wait}s …")
                    time.sleep(wait)
                    continue
                print(f"[LLM ERROR] {self.model_name}: {e}")
                return ""
        return ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        directive = "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no prose."
        raw = self.generate_text(prompt + directive, system_prompt, temperature)
        if not raw:
            return {}

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"[LLM] No JSON found:\n{raw[:300]}")
        return {}
