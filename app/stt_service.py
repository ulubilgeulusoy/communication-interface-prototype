from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


DEFAULT_WHISPER_MODEL = "base"
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "faster-whisper"
DEFAULT_REPO_LOCAL_MODEL_PATH = DEFAULT_LOCAL_MODEL_DIR / DEFAULT_WHISPER_MODEL
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"


class SpeechToTextServiceError(Exception):
    """Raised when local speech transcription cannot be completed."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str


class FasterWhisperService:
    def __init__(
        self,
        *,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_WHISPER_DEVICE,
        compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
    ) -> None:
        self.model_size = os.getenv("FASTER_WHISPER_MODEL", model_size)
        configured_model_path = os.getenv("FASTER_WHISPER_MODEL_PATH", "").strip()
        default_model_path = (
            str(DEFAULT_REPO_LOCAL_MODEL_PATH)
            if DEFAULT_REPO_LOCAL_MODEL_PATH.exists()
            else ""
        )
        self.model_path = configured_model_path or default_model_path
        self.device = os.getenv("FASTER_WHISPER_DEVICE", device).strip() or device
        self.compute_type = (
            os.getenv("FASTER_WHISPER_COMPUTE_TYPE", compute_type).strip() or compute_type
        )

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        model = _load_model(
            self.model_path or self.model_size,
            self.device,
            self.compute_type,
        )
        try:
            segments, info = model.transcribe(
                str(audio_path),
                vad_filter=True,
                beam_size=5,
            )
        except Exception as exc:  # pragma: no cover - library raises implementation-specific errors
            raise SpeechToTextServiceError(
                f"Speech transcription failed: {type(exc).__name__}: {exc}"
            ) from exc

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not text:
            raise SpeechToTextServiceError("No speech was detected in the recording.")

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", "") or "",
        )


@lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise SpeechToTextServiceError(
            "faster-whisper is not installed. Add the dependency and reinstall requirements."
        ) from exc

    try:
        download_root = str(DEFAULT_LOCAL_MODEL_DIR)
        Path(download_root).mkdir(parents=True, exist_ok=True)
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover
        raise SpeechToTextServiceError(
            "Unable to load a local faster-whisper model. "
            "Place the model in the local cache first or set FASTER_WHISPER_MODEL_PATH "
            "to a pre-downloaded model directory."
        ) from exc
