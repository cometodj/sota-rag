from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import yaml

from embeddings import EmbeddingModel


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_FILENAME = "original_retrieval_results.jsonl"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def read_benchmark_questions(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark questions JSONL not found: {input_path}")

    questions: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line must be an object at {input_path}:{line_number}")
            if "id" not in record or "question" not in record:
                raise ValueError(f"Missing id or question at {input_path}:{line_number}")

            questions.append(
                {
                    "id": str(record["id"]),
                    "question": str(record["question"]),
                }
            )

    return questions


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_existing_collection(
    vector_db_dir: Path,
    collection_name: str,
    embedding_model_name: str,
) -> Any:
    if not vector_db_dir.exists():
        raise FileNotFoundError(f"Chroma directory not found: {vector_db_dir}")

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = client.get_collection(name=collection_name)

    metadata = collection.metadata or {}
    existing_model_name = metadata.get("embedding_model")
    if existing_model_name != embedding_model_name:
        raise ValueError(
            "Chroma collection embedding model mismatch: "
            f"collection={collection_name}, existing={existing_model_name}, "
            f"configured={embedding_model_name}"
        )

    return collection


def format_retrieved_chunks(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    ids = query_result.get("ids", [[]])[0]
    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    retrieved_chunks: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        chunk: dict[str, Any] = {
            "rank": index + 1,
            "chunk_id": str(metadata.get("chunk_id", chunk_id)),
            "document_name": str(metadata["document_name"]),
            "page_number": int(metadata["page_number"]),
            "chunk_index": int(metadata["chunk_index"]),
            "text": str(documents[index]),
        }

        if distances:
            chunk["distance"] = distances[index]

        retrieved_chunks.append(chunk)

    return retrieved_chunks


def retrieve_original_questions(
    questions: list[dict[str, str]],
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    records: list[dict[str, Any]] = []
    for question in questions:
        query_text = question["question"]
        query_embedding = embedding_model.embed_texts([query_text])[0]
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        records.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "query_type": "original",
                "query_text": query_text,
                "retrieved_chunks": format_retrieved_chunks(query_result),
            }
        )

    return records


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)

    output_dir = Path(config["paths"]["output_dir"])
    benchmark_path = Path(config["paths"]["benchmark_questions"])
    output_path = output_dir / OUTPUT_FILENAME
    vector_db_dir = Path(config["paths"]["vector_db_dir"])
    collection_name = str(config["embedding"]["collection_name"])
    model_name = str(config["embedding"]["model_name"])
    top_k = int(config["retrieval"]["top_k"])

    questions = read_benchmark_questions(benchmark_path)
    embedding_model = EmbeddingModel(model_name)
    collection = get_existing_collection(vector_db_dir, collection_name, model_name)
    records = retrieve_original_questions(questions, collection, embedding_model, top_k)
    write_jsonl(records, output_path)

    print(f"Loaded {len(questions)} benchmark questions from {benchmark_path}")
    print(f"Retrieved top-{top_k} chunks for {len(records)} original queries")
    print(f"Saved original retrieval results to {output_path}")

    return output_path


if __name__ == "__main__":
    run()
