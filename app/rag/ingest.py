"""Knowledge-base document ingestion and chunking helpers.

This module is intentionally independent from embeddings and vector storage. It
only knows how to:

- discover supported files under ``knowledge_base/``
- extract raw text from those files
- normalize text for chunking
- split text into overlapping chunks with metadata preserved
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceDocument:
    """A normalized document loaded from the knowledge base."""

    source_path: str
    filename: str
    category: str
    extension: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """A chunk of document text ready for embedding."""

    chunk_id: str
    text: str
    source_path: str
    filename: str
    category: str
    chunk_index: int
    start_char: int
    end_char: int


def load_documents(base_path: Path) -> list[SourceDocument]:
    """Load all supported knowledge-base documents under ``base_path``."""

    documents: list[SourceDocument] = []
    for path in sorted(base_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _normalize_text(_read_document_text(path))
        if not text:
            continue

        category = _infer_category(base_path, path)
        documents.append(
            SourceDocument(
                source_path=str(path),
                filename=path.name,
                category=category,
                extension=path.suffix.lower(),
                text=text,
            )
        )

    return documents


def chunk_documents(
    documents: list[SourceDocument],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[KnowledgeChunk]:
    """Split loaded documents into overlapping text chunks.

    Args:
        documents: Normalized source documents.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(
            _chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def ingest_knowledge_base(
    base_path: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[KnowledgeChunk]:
    """Load and chunk the full knowledge base in one step."""

    documents = load_documents(base_path)
    return chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _chunk_document(
    document: SourceDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    text = document.text
    chunks: list[KnowledgeChunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        adjusted_end = _find_chunk_boundary(text, start, end)
        chunk_text = text[start:adjusted_end].strip()
        if chunk_text:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.category}:{document.filename}:{chunk_index}",
                    text=chunk_text,
                    source_path=document.source_path,
                    filename=document.filename,
                    category=document.category,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=adjusted_end,
                )
            )
            chunk_index += 1

        if adjusted_end >= len(text):
            break

        start = max(adjusted_end - chunk_overlap, start + step)

    return chunks


def _find_chunk_boundary(text: str, start: int, end: int) -> int:
    """Prefer to end a chunk at whitespace when possible."""

    if end >= len(text):
        return len(text)

    window = text[start:end]
    split_at = window.rfind(" ")
    if split_at <= 0:
        split_at = window.rfind("\n")
    if split_at <= 0:
        return end
    return start + split_at


def _read_document_text(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if extension == ".pdf":
        return _read_pdf_text(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _normalize_text(text: str) -> str:
    normalized = text.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _infer_category(base_path: Path, path: Path) -> str:
    relative_path = path.relative_to(base_path)
    if len(relative_path.parts) <= 1:
        return "root"
    return relative_path.parts[0]
