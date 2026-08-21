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
STEP_HEADER_PATTERN = re.compile(r"(?mi)(?:^|\n)\s*(step\s+\d+[\s:.-]|(?:\d+[\).:-]\s+))")
HEADING_LINE_PATTERN = re.compile(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Z0-9 /&()_-]{3,}|.+:\s*)$")
PAGE_NUMBER_PATTERN = re.compile(r"(?i)^page\s+\d+\s+of\s+\d+$")
GUIDE_METADATA_PATTERN = re.compile(
    r"(?i)^(draft:\s*|guide id:\s*|this document was generated on|www\.ifixit\.com|©\s*ifixit)"
)
STEP_LINE_PATTERN = re.compile(r"(?im)^step\s+\d+\s*$")
STEP_TITLE_SEPARATOR_PATTERN = re.compile(r"(?m)^\s*[—-]\s*$")
LINE_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
THREE_PLUS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
NON_ALNUM_LINE_PATTERN = re.compile(r"^[^\w]+$")


@dataclass(frozen=True)
class SourceDocument:
    """A normalized document loaded from the knowledge base."""

    document_id: str
    source_path: str
    filename: str
    category: str
    extension: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """A chunk of document text ready for embedding."""

    chunk_id: str
    document_id: str
    text: str
    embedding_text: str
    source_path: str
    filename: str
    category: str
    chunk_index: int
    start_char: int
    end_char: int
    task_summary: str = ""


@dataclass(frozen=True)
class DocumentBlock:
    """A structure-aware unit used to assemble retrieval chunks."""

    start_char: int
    end_char: int
    text: str
    kind: str = "section"


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
                document_id=_build_document_id(base_path, path),
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
    blocks = _extract_document_blocks(document.text)
    if not blocks:
        return []

    block_groups = _group_blocks_into_chunks(
        blocks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[KnowledgeChunk] = []
    task_summary = _extract_task_summary(document.text, document.filename)
    summary_chunk = _build_summary_chunk(
        document,
        task_summary=task_summary,
    )
    if summary_chunk is not None:
        chunks.append(summary_chunk)

    chunk_index = len(chunks)
    for group in block_groups:
        raw_start = group[0].start_char
        raw_end = group[-1].end_char
        raw_chunk_text = document.text[raw_start:raw_end]
        chunk_text = raw_chunk_text.strip()
        if chunk_text:
            leading_offset = raw_chunk_text.find(chunk_text)
            start = raw_start + max(0, leading_offset)
            end = start + len(chunk_text)
            chunk_id = f"{document.category}:{document.filename}:{chunk_index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    embedding_text=_build_embedding_text(
                        document=document,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    ),
                    source_path=document.source_path,
                    filename=document.filename,
                    category=document.category,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    task_summary=task_summary,
                )
            )
            chunk_index += 1

    return chunks


def _extract_document_blocks(text: str) -> list[DocumentBlock]:
    raw_sections = _split_structured_sections(text)
    blocks: list[DocumentBlock] = []
    for start, end in raw_sections:
        section_text = text[start:end].strip()
        if not section_text:
            continue
        blocks.append(
            DocumentBlock(
                start_char=start,
                end_char=end,
                text=section_text,
                kind="step" if _is_step_block(section_text) else "section",
            )
        )
    return blocks


def _split_structured_sections(text: str) -> list[tuple[int, int]]:
    anchors = sorted(set(_find_structure_anchors(text)))
    if not anchors:
        return []
    sections: list[tuple[int, int]] = []
    for index, start in enumerate(anchors):
        end = anchors[index + 1] if index + 1 < len(anchors) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        sections.append((start, end))
    return sections


def _find_structure_anchors(text: str) -> list[int]:
    anchors: set[int] = {0}

    for match in STEP_LINE_PATTERN.finditer(text):
        anchors.add(match.start())

    for match in HEADING_LINE_PATTERN.finditer(text):
        line = match.group(0).strip()
        if _is_structural_heading(line):
            anchors.add(match.start())

    for match in re.finditer(r"\n{2,}", text):
        next_start = match.end()
        if next_start < len(text):
            next_line = text[next_start:].splitlines()[0].strip() if text[next_start:].splitlines() else ""
            if _starts_new_section(next_line):
                anchors.add(next_start)

    return [anchor for anchor in sorted(anchors) if anchor < len(text)]


def _starts_new_section(first_line: str) -> bool:
    cleaned = first_line.strip()
    if not cleaned:
        return False
    if cleaned.startswith("#"):
        return True
    if STEP_HEADER_PATTERN.match(cleaned):
        return True
    if _is_structural_heading(cleaned):
        return True
    return False


def _is_structural_heading(line: str) -> bool:
    if not line:
        return False
    if line.startswith("#"):
        return True
    if line.upper() == line and 3 <= len(line) <= 80:
        return True
    if line.endswith(":") and len(line) <= 80:
        return True
    return False


def _group_blocks_into_chunks(
    blocks: list[DocumentBlock],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[DocumentBlock]]:
    prepared_blocks: list[DocumentBlock] = []
    for block in blocks:
        if len(block.text) <= chunk_size:
            prepared_blocks.append(block)
            continue
        prepared_blocks.extend(
            _split_oversized_block(
                block,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    grouped: list[list[DocumentBlock]] = []
    current_group: list[DocumentBlock] = []
    current_size = 0

    for block in prepared_blocks:
        if block.kind == "step":
            if current_group:
                grouped.append(current_group)
                current_group = []
                current_size = 0
            grouped.append([block])
            continue

        block_size = len(block.text)
        separator_size = 2 if current_group else 0
        if current_group and current_size + separator_size + block_size > chunk_size:
            grouped.append(current_group)
            current_group = []
            current_size = 0

        if current_group:
            current_size += 2
        current_group.append(block)
        current_size += block_size

    if current_group:
        grouped.append(current_group)

    return grouped


def _split_oversized_block(
    block: DocumentBlock,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentBlock]:
    sub_blocks: list[DocumentBlock] = []
    text = block.text
    relative_start = 0
    step = max(1, chunk_size - chunk_overlap)

    while relative_start < len(text):
        relative_end = min(relative_start + chunk_size, len(text))
        adjusted_end = _find_split_boundary(text, relative_start, relative_end)
        if adjusted_end <= relative_start:
            adjusted_end = relative_end

        chunk_text = text[relative_start:adjusted_end].strip()
        if chunk_text:
            leading_offset = text[relative_start:adjusted_end].find(chunk_text)
            absolute_start = block.start_char + relative_start + max(0, leading_offset)
            absolute_end = absolute_start + len(chunk_text)
            sub_blocks.append(
                DocumentBlock(
                    start_char=absolute_start,
                    end_char=absolute_end,
                    text=chunk_text,
                    kind=block.kind,
                )
            )

        if adjusted_end >= len(text):
            break

        relative_start = max(adjusted_end - chunk_overlap, relative_start + step)

    return sub_blocks


def _find_split_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)

    window = text[start:end]
    candidate_positions: list[int] = []

    for marker in ("\n\n", "\n", ". ", "; ", ", ", " "):
        position = window.rfind(marker)
        if position > 0:
            candidate_positions.append(position + (0 if marker.startswith("\n") else 1))

    if not candidate_positions:
        return end

    return start + max(candidate_positions)


def _read_document_text(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if extension == ".pdf":
        return _read_pdf_text(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    cleaned_pages = _clean_pdf_pages(raw_pages)
    return "\n\n".join(page for page in cleaned_pages if page.strip())


def _clean_pdf_pages(pages: list[str]) -> list[str]:
    normalized_pages = [_normalize_pdf_page(page) for page in pages]
    repeated_lines = _find_repeated_pdf_lines(normalized_pages)
    cleaned_pages: list[str] = []

    for page in normalized_pages:
        kept_lines: list[str] = []
        lines = page.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                if kept_lines and kept_lines[-1] != "":
                    kept_lines.append("")
                index += 1
                continue
            if _looks_like_split_page_counter(lines, index):
                index += min(4, len(lines) - index)
                continue
            if _should_drop_pdf_line(line, repeated_lines):
                index += 1
                continue
            kept_lines.append(line)
            index += 1
        cleaned_pages.append(_compact_blank_lines("\n".join(kept_lines)).strip())

    return cleaned_pages


def _normalize_pdf_page(page_text: str) -> str:
    normalized = page_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _find_repeated_pdf_lines(pages: list[str]) -> set[str]:
    line_pages: dict[str, set[int]] = {}
    for page_index, page in enumerate(pages):
        unique_lines = {line.strip() for line in page.splitlines() if line.strip()}
        for line in unique_lines:
            line_pages.setdefault(line, set()).add(page_index)

    repeated: set[str] = set()
    for line, page_indexes in line_pages.items():
        if len(page_indexes) < 2:
            continue
        if GUIDE_METADATA_PATTERN.search(line) or PAGE_NUMBER_PATTERN.match(line):
            repeated.add(line)
            continue
        if len(line) <= 80 and (line.upper() == line or "Honda EU2200IT" in line or "iFixit" in line):
            repeated.add(line)
    return repeated


def _should_drop_pdf_line(line: str, repeated_lines: set[str]) -> bool:
    if line in repeated_lines:
        return True
    if PAGE_NUMBER_PATTERN.match(line):
        return True
    if GUIDE_METADATA_PATTERN.search(line):
        return True
    if STEP_TITLE_SEPARATOR_PATTERN.match(line):
        return True
    if NON_ALNUM_LINE_PATTERN.match(line) and len(line) <= 3:
        return True
    if line in {"…", "•", "", ""}:
        return True
    return False


def _looks_like_split_page_counter(lines: list[str], index: int) -> bool:
    window = [line.strip() for line in lines[index:index + 4]]
    if len(window) < 3:
        return False
    if window[0].lower() != "page":
        return False
    if len(window) >= 3 and window[1].isdigit() and window[2].lower() == "of":
        return True
    if len(window) >= 4 and window[1].isdigit() and window[2].lower() == "of" and window[3].isdigit():
        return True
    return False


def _compact_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_step_block(text: str) -> bool:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    return bool(STEP_HEADER_PATTERN.match(first_line))


def _normalize_text(text: str) -> str:
    normalized = text.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = LINE_WHITESPACE_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"(?m)^ +", "", normalized)
    normalized = THREE_PLUS_NEWLINES_PATTERN.sub("\n\n", normalized)
    return normalized.strip()


def _infer_category(base_path: Path, path: Path) -> str:
    relative_path = path.relative_to(base_path)
    if len(relative_path.parts) <= 1:
        return "root"
    return relative_path.parts[0]


def _build_embedding_text(
    *,
    document: SourceDocument,
    chunk_id: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    filename_stem = Path(document.filename).stem.replace("_", " ").replace("-", " ")
    step_match = re.search(r"(?im)^step\s+\d+.*$", chunk_text)
    step_label = step_match.group(0).strip() if step_match else ""
    task_summary = _extract_task_summary(document.text, document.filename)
    return (
        f"Document title: {filename_stem}\n"
        f"Task summary: {task_summary}\n"
        f"Document id: {document.document_id}\n"
        f"Source file: {document.filename}\n"
        f"Category: {document.category}\n"
        f"Chunk id: {chunk_id}\n"
        f"Chunk number: {chunk_index}\n"
        f"Step label: {step_label or 'n/a'}\n"
        "Document content:\n"
        f"{chunk_text}"
    )


def _build_document_id(base_path: Path, path: Path) -> str:
    relative_path = path.relative_to(base_path)
    stem = relative_path.with_suffix("")
    return "::".join(stem.parts).lower()


def _extract_task_summary(text: str, filename: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines:
        title = lines[0]
        lowered = title.lower()
        if "guide id:" in lowered and len(lines) > 1:
            return lines[1]
        return title
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()


def _build_summary_chunk(
    document: SourceDocument,
    *,
    task_summary: str,
) -> KnowledgeChunk | None:
    intro_excerpt = document.text[:700].strip()
    if not intro_excerpt:
        return None

    summary_text = (
        f"{task_summary}\n"
        f"Document type: {document.category}\n"
        f"Source file: {document.filename}\n"
        "Summary:\n"
        f"{intro_excerpt}"
    ).strip()
    return KnowledgeChunk(
        chunk_id=f"{document.category}:{document.filename}:summary",
        document_id=document.document_id,
        text=summary_text,
        embedding_text=_build_embedding_text(
            document=document,
            chunk_id=f"{document.category}:{document.filename}:summary",
            chunk_index=0,
            chunk_text=summary_text,
        ),
        source_path=document.source_path,
        filename=document.filename,
        category=document.category,
        chunk_index=0,
        start_char=0,
        end_char=min(len(document.text), len(intro_excerpt)),
        task_summary=task_summary,
    )
