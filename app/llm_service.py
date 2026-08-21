from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import httpx


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gpt-oss:20b"
VISION_MODEL = "llama3.2-vision:11b"
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
VISION_FINDINGS_SYSTEM_PROMPT = (
    "You are a visual-analysis component. Analyze the attached image(s) only and "
    "return valid JSON with exactly these keys: objects, visible_text, observations, "
    "measurements_or_details, uncertainties. Each value must be an array of concise "
    "strings. Do not answer the user's question, offer advice, or use markdown. "
    "Do not infer facts that are not visible. Record uncertainty explicitly."
)


class OllamaServiceError(Exception):
    """Raised when the Ollama backend cannot satisfy a request."""


@dataclass(frozen=True)
class LLMResponse:
    model: str
    output_text: str


@dataclass(frozen=True)
class LLMImageInput:
    name: str
    content_type: str
    base64_data: str


@dataclass(frozen=True)
class VisionAnalysis:
    model: str
    raw_findings: str
    findings: dict[str, list[str]]


class VisionAnalysisService:
    """Isolated image-analysis stage so vision backends can be swapped independently."""

    def __init__(
        self,
        llm_service: "OllamaService",
        *,
        model: str = VISION_MODEL,
        system_prompt: str = VISION_FINDINGS_SYSTEM_PROMPT,
    ) -> None:
        self.llm_service = llm_service
        self.model = model
        self.system_prompt = system_prompt

    async def analyze(
        self,
        *,
        original_question: str,
        images: list[LLMImageInput],
    ) -> VisionAnalysis:
        response = await self.llm_service.generate_reply(
            message_text=(
                "Original user question (context only; do not answer it):\n"
                f"{original_question}"
            ),
            images=images,
            model=self.model,
            system_prompt=self.system_prompt,
        )
        return VisionAnalysis(
            model=response.model,
            raw_findings=response.output_text,
            findings=self._parse_findings(response.output_text),
        )

    @staticmethod
    def _parse_findings(raw_findings: str) -> dict[str, list[str]]:
        keys = (
            "objects",
            "visible_text",
            "observations",
            "measurements_or_details",
            "uncertainties",
        )
        try:
            parsed = json.loads(raw_findings)
        except (TypeError, ValueError):
            return {"observations": [raw_findings.strip()]} if raw_findings.strip() else {}
        if not isinstance(parsed, dict):
            return {"observations": [raw_findings.strip()]} if raw_findings.strip() else {}

        findings: dict[str, list[str]] = {}
        for key in keys:
            value = parsed.get(key, [])
            if isinstance(value, list):
                findings[key] = [str(item).strip() for item in value if str(item).strip()]
            elif str(value).strip():
                findings[key] = [str(value).strip()]
        return findings


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
        attachment_context: str = "",
        images: list[LLMImageInput] | None = None,
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
                attachment_context=attachment_context,
            ),
            self._build_user_message(
                conversation_history="",
                llm_thread_history=self._trim_text(
                    llm_thread_history,
                    MAX_LLM_THREAD_HISTORY_CHARS // 2,
                ),
                message_text=message_text,
                attachment_context=attachment_context,
            ),
        ]

        last_error: OllamaServiceError | None = None
        for user_message in user_messages:
            try:
                output_text = await self._request_completion(
                    prompt=prompt,
                    user_message=user_message,
                    images=images or [],
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
        images: list[LLMImageInput],
        model: str,
    ) -> str:
        user_payload: dict[str, object] = {
            "role": "user",
            "content": user_message,
        }
        if images:
            user_payload["images"] = [image.base64_data for image in images]

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                user_payload,
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
        attachment_context: str,
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

        if attachment_context:
            sections.append(
                "Attached file context:\n"
                f"{self._trim_text(attachment_context, MAX_MESSAGE_TEXT_CHARS)}"
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

    @staticmethod
    def encode_image_bytes(raw_bytes: bytes) -> str:
        return base64.b64encode(raw_bytes).decode("ascii")
