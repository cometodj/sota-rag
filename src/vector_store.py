from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import yaml

from embeddings import EmbeddingModel


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
CHUNKS_FILENAME = "chunks.jsonl"
BATCH_SIZE = 128


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
) -> Any:
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": embedding_model_name},
    )

    metadata = collection.metadata or {}
    existing_model_name = metadata.get("embedding_model")

    if existing_model_name is None:
        metadata["embedding_model"] = embedding_model_name
        collection.modify(metadata=metadata)
    elif existing_model_name != embedding_model_name:
        raise ValueError(
            "Chroma collection embedding model mismatch: "
            f"collection={collection_name}, existing={existing_model_name}, "
            f"configured={embedding_model_name}"
        )

    return collection


def chunk_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "document_name": str(chunk["document_name"]),
        "page_number": int(chunk["page_number"]),
        "chunk_index": int(chunk["chunk_index"]),
        "char_count": int(chunk["char_count"]),
    }


def index_chunks(
    chunks: list[dict[str, Any]],
    collection: Any,
    embedding_model: EmbeddingModel,
    batch_size: int = BATCH_SIZE,
) -> None:
    for batch in batch_records(chunks, batch_size):
        ids = [str(chunk["chunk_id"]) for chunk in batch]
        documents = [str(chunk["text"]) for chunk in batch]
        metadatas = [chunk_metadata(chunk) for chunk in batch]
        embeddings = embedding_model.embed_texts(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Any:
    config = load_config(config_path)

    output_dir = Path(config["paths"]["output_dir"])
    chunks_path = output_dir / CHUNKS_FILENAME
    vector_db_dir = Path(config["paths"]["vector_db_dir"])
    model_name = str(config["embedding"]["model_name"])
    collection_name = str(config["embedding"]["collection_name"])

    chunks = read_chunks(chunks_path)
    embedding_model = EmbeddingModel(model_name)
    collection = get_collection(vector_db_dir, collection_name, model_name)
    index_chunks(chunks, collection, embedding_model)

    indexed_count = collection.count()
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    print(f"Indexed {indexed_count} chunks into Chroma collection '{collection_name}'")
    print(f"Embedding model: {model_name}")
    print(f"Chroma directory: {vector_db_dir}")

    return collection


if __name__ == "__main__":
    run()
