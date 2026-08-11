from __future__ import annotations

from dataclasses import dataclass

import httpx


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant inside a communication prototype. "
    "Answer clearly and concisely."
)


class OllamaServiceError(Exception):
    """Raised when the Ollama backend cannot satisfy a request."""


@dataclass(frozen=True)
class LLMResponse:
    model: str
    output_text: str


class OllamaService:
    def __init__(
        self,
        *,
        base_url: str = OLLAMA_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.system_prompt = system_prompt
        self.timeout = timeout

    async def generate_reply(
        self,
        *,
        message_text: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        selected_model = model or self.default_model
        prompt = system_prompt or self.system_prompt

        payload = {
            "model": selected_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message_text},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.RequestError as exc:
            raise OllamaServiceError("Ollama is unavailable.") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaServiceError(
                f"Ollama request failed with status {exc.response.status_code}."
            ) from exc

        data = response.json()
        message = data.get("message") or {}
        output_text = (message.get("content") or "").strip()
        if not output_text:
            raise OllamaServiceError("Ollama returned an empty response.")

        return LLMResponse(model=selected_model, output_text=output_text)
