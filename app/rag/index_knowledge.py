"""CLI entry point for indexing the local knowledge base into ChromaDB."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .embeddings import EmbeddingServiceError, OllamaEmbeddingService
from .ingest import ingest_knowledge_base, load_documents
from .retrieval import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_STORE_PATH,
    upsert_document_chunks,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base"


async def run_indexing(
    *,
    knowledge_base_path: Path,
    vector_store_path: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    documents = load_documents(knowledge_base_path)
    if not documents:
        print("No supported knowledge-base documents found. Nothing indexed.")
        return 0

    chunks = ingest_knowledge_base(
        knowledge_base_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        print("Documents were found, but no non-empty chunks were produced.")
        return 0

    embedding_service = OllamaEmbeddingService(model=embedding_model)
    embeddings = await embedding_service.embed_texts([chunk.text for chunk in chunks])
    upsert_document_chunks(
        chunks,
        embeddings,
        store_path=vector_store_path,
        collection_name=collection_name,
    )

    print(
        "Indexed "
        f"{len(documents)} file(s) into '{collection_name}' "
        f"with {len(chunks)} chunk(s) using model '{embedding_model}'."
    )
    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index knowledge-base files into ChromaDB.")
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=KNOWLEDGE_BASE_PATH,
        help="Path to the local knowledge_base directory.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=DEFAULT_VECTOR_STORE_PATH,
        help="Path to the local ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name to index into.",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Local Ollama embedding model to use.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of overlapping characters between chunks.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(
            run_indexing(
                knowledge_base_path=args.knowledge_base,
                vector_store_path=args.vector_store,
                collection_name=args.collection,
                embedding_model=args.embedding_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
    except EmbeddingServiceError as exc:
        message = str(exc).strip()
        print(f"Indexing failed: {message}")
        if "not found" in message and args.embedding_model in message:
            print(
                "Pull the Ollama embedding model first, then retry:\n"
                f"  ollama pull {args.embedding_model}\n"
                "  python -m app.rag.index_knowledge"
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
