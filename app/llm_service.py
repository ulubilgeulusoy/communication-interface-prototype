from __future__ import annotations

from dataclasses import dataclass

import httpx


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gpt-oss:20b"
MAX_SESSION_HISTORY_CHARS = 1200
MAX_LLM_THREAD_HISTORY_CHARS = 1400
MAX_MESSAGE_TEXT_CHARS = 5000
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant inside a communication prototype. "
    "You will receive recent user-to-user chat history for the current session, "
    "recent LLM-thread history for the same user, and the user's latest drafting "
    "request. Use that conversation context to maintain continuity, and answer "
    "clearly and concisely."
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
        conversation_history: str = "",
        llm_thread_history: str = "",
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        selected_model = model or self.default_model
        prompt = system_prompt or self.system_prompt
        user_messages = [
            self._build_user_message(
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                message_text=message_text,
            ),
            self._build_user_message(
                conversation_history="",
                llm_thread_history=self._trim_text(
                    llm_thread_history,
                    MAX_LLM_THREAD_HISTORY_CHARS // 2,
                ),
                message_text=message_text,
            ),
        ]

        last_error: OllamaServiceError | None = None
        for user_message in user_messages:
            try:
                output_text = await self._request_completion(
                    prompt=prompt,
                    user_message=user_message,
                    model=selected_model,
                )
                return LLMResponse(model=selected_model, output_text=output_text)
            except OllamaServiceError as exc:
                last_error = exc
                if str(exc) != "Ollama returned an empty response.":
                    raise

        raise last_error or OllamaServiceError("Ollama returned an empty response.")

    async def _request_completion(
        self,
        *,
        prompt: str,
        user_message: str,
        model: str,
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
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

        return output_text

    def _build_user_message(
        self,
        *,
        conversation_history: str,
        llm_thread_history: str,
        message_text: str,
    ) -> str:
        sections: list[str] = []

        if conversation_history:
            sections.append(
                "Recent session conversation:\n"
                f"{self._trim_text(conversation_history, MAX_SESSION_HISTORY_CHARS)}"
            )

        if llm_thread_history:
            sections.append(
                "Recent LLM thread:\n"
                f"{self._trim_text(llm_thread_history, MAX_LLM_THREAD_HISTORY_CHARS)}"
            )

        sections.append(
            "User request:\n"
            f"{self._trim_text(message_text, MAX_MESSAGE_TEXT_CHARS)}"
        )

        if len(sections) == 1:
            return f"User request:\n{message_text}"

        return "\n\n".join(sections)

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return f"...\n{cleaned[-limit:]}"
