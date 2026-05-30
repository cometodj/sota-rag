from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml

from embeddings import EmbeddingModel
from table_aware import classify_chunk_text


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
CHUNKS_FILENAME = "chunks.jsonl"
DOCLING_CHUNKS_FILENAME = "chunks_docling.jsonl"
BATCH_SIZE = 128
TABLE_METADATA_KEYS = [
    "chunk_type",
    "table_id",
    "parent_table_id",
    "table_group_index",
    "table_fragment_index",
    "table_markdown",
    "full_table_markdown",
    "parent_table_text",
    "parent_table_title",
    "field_name",
    "field_aliases",
    "table_value_codes",
    "nearby_context",
    "caption",
    "source_parser",
]
SUPPORTED_SOURCES = {"pymupdf", "docling"}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def read_chunks(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {input_path}")

    chunks: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            chunk = json.loads(line)
            if not isinstance(chunk, dict):
                raise ValueError(f"JSONL line must be an object at {input_path}:{line_number}")
            chunks.append(chunk)

    return chunks


def batch_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def get_collection(
    vector_db_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    source: str | None = None,
) -> Any:
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    collection_metadata = {"embedding_model": embedding_model_name}
    if source:
        collection_metadata["source"] = source

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=collection_metadata,
    )

    metadata = collection.metadata or {}
    existing_model_name = metadata.get("embedding_model")
    existing_source = metadata.get("source")

    if existing_model_name is None:
        metadata["embedding_model"] = embedding_model_name
        collection.modify(metadata=metadata)
    elif existing_model_name != embedding_model_name:
        raise ValueError(
            "Chroma collection embedding model mismatch: "
            f"collection={collection_name}, existing={existing_model_name}, "
            f"configured={embedding_model_name}"
        )

    if source:
        if existing_source is None:
            metadata["source"] = source
            collection.modify(metadata=metadata)
        elif existing_source != source:
            raise ValueError(
                "Chroma collection source mismatch: "
                f"collection={collection_name}, existing={existing_source}, configured={source}"
            )

    return collection


def pymupdf_chunk_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "chunk_id": str(chunk["chunk_id"]),
        "document_name": str(chunk["document_name"]),
        "chunk_index": int(chunk["chunk_index"]),
        "char_count": int(chunk["char_count"]),
        "chunk_type": str(chunk.get("chunk_type") or classify_chunk_text(str(chunk.get("text", "")))),
        "source_parser": str(chunk.get("source_parser") or "pymupdf"),
    }
    if chunk.get("page_number") not in (None, ""):
        metadata["page_number"] = int(chunk["page_number"])
    for key in TABLE_METADATA_KEYS:
        value = chunk.get(key)
        if key in metadata or value in (None, ""):
            continue
        metadata[key] = str(value)
    return metadata


def docling_chunk_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "chunk_id": str(chunk["chunk_id"]),
        "document_name": str(chunk["document_name"]),
        "chunk_index": int(chunk["chunk_index"]),
        "char_count": int(chunk["char_count"]),
        "source": "docling",
        "chunk_type": str(chunk.get("chunk_type") or classify_chunk_text(str(chunk.get("text", "")))),
        "source_parser": str(chunk.get("source_parser") or "docling"),
    }

    section_title = chunk.get("section_title")
    if section_title:
        metadata["section_title"] = str(section_title)
    for key in TABLE_METADATA_KEYS:
        value = chunk.get(key)
        if key in metadata or value in (None, ""):
            continue
        metadata[key] = str(value)

    return metadata


def chunk_metadata(chunk: dict[str, Any], source: str) -> dict[str, str | int]:
    if source == "pymupdf":
        return pymupdf_chunk_metadata(chunk)
    if source == "docling":
        return docling_chunk_metadata(chunk)

    raise ValueError(f"Unsupported source: {source}")


def index_chunks(
    chunks: list[dict[str, Any]],
    collection: Any,
    embedding_model: EmbeddingModel,
    source: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    for batch in batch_records(chunks, batch_size):
        ids = [str(chunk["chunk_id"]) for chunk in batch]
        documents = [str(chunk.get("text_for_embedding") or chunk["text"]) for chunk in batch]
        metadatas = [chunk_metadata(chunk, source) for chunk in batch]
        embeddings = embedding_model.embed_texts(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "document"


def docling_collection_name(config: dict[str, Any]) -> str:
    configured_name = config["embedding"].get("docling_collection_name")
    if configured_name:
        return str(configured_name)

    base_collection_name = str(config["embedding"]["collection_name"])
    document_prefix = f"{slugify(Path(config['document']['pdf_path']).stem)}_doc"
    prefix = f"{document_prefix}_"

    if base_collection_name.startswith(prefix):
        suffix = base_collection_name[len(prefix) :]
        return f"{document_prefix}_docling_{suffix}"

    return f"{base_collection_name}_docling"


def source_settings(config: dict[str, Any], source: str) -> tuple[Path, str]:
    output_dir = Path(config["paths"]["output_dir"])

    if source == "pymupdf":
        return output_dir / CHUNKS_FILENAME, str(config["embedding"]["collection_name"])
    if source == "docling":
        return output_dir / DOCLING_CHUNKS_FILENAME, docling_collection_name(config)

    raise ValueError(f"Unsupported source: {source}")


def run(config_path: Path = DEFAULT_CONFIG_PATH, source: str = "pymupdf") -> Any:
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    config = load_config(config_path)

    vector_db_dir = Path(config["paths"]["vector_db_dir"])
    model_name = str(config["embedding"]["model_name"])
    chunks_path, collection_name = source_settings(config, source)

    chunks = read_chunks(chunks_path)
    embedding_model = EmbeddingModel(model_name)
    collection = get_collection(
        vector_db_dir=vector_db_dir,
        collection_name=collection_name,
        embedding_model_name=model_name,
        source="docling" if source == "docling" else None,
    )
    index_chunks(chunks, collection, embedding_model, source=source)

    indexed_count = collection.count()
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    print(f"Source: {source}")
    print(f"Indexed {indexed_count} chunks into Chroma collection '{collection_name}'")
    print(f"Embedding model: {model_name}")
    print(f"Chroma directory: {vector_db_dir}")

    return collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index chunks into Chroma.")
    parser.add_argument(
        "--source",
        choices=sorted(SUPPORTED_SOURCES),
        default="pymupdf",
        help="Chunk source to index.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(config_path=args.config, source=args.source)
