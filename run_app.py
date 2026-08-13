from __future__ import annotations

import os
from pathlib import Path

import uvicorn


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_WHISPER_MODEL_PATH = REPO_ROOT / "models" / "faster-whisper" / "base"


def configure_environment() -> None:
    if DEFAULT_WHISPER_MODEL_PATH.exists():
        os.environ.setdefault("FASTER_WHISPER_MODEL_PATH", str(DEFAULT_WHISPER_MODEL_PATH))

    os.environ.setdefault("FASTER_WHISPER_DEVICE", "cpu")
    os.environ.setdefault("FASTER_WHISPER_COMPUTE_TYPE", "int8")


def main() -> None:
    configure_environment()
    uvicorn.run(
        "app.main:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=True,
        reload_dirs=[str(REPO_ROOT)],
    )


if __name__ == "__main__":
    main()
