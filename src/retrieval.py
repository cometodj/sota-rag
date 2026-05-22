from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
import yaml

from embeddings import EmbeddingModel


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
ORIGINAL_OUTPUT_FILENAME = "original_retrieval_results.jsonl"
EXPANDED_OUTPUT_FILENAME = "expanded_retrieval_results.jsonl"
QUERY_EXPANSIONS_FILENAME = "query_expansions.jsonl"


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


def read_query_expansions(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Query expansions JSONL not found: {input_path}")

    expansion_records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line must be an object at {input_path}:{line_number}")

            required_fields = {"question_id", "original_question", "expanded_queries"}
            missing_fields = required_fields - set(record)
            if missing_fields:
                raise ValueError(
                    f"Missing fields at {input_path}:{line_number}: {sorted(missing_fields)}"
                )
            if not isinstance(record["expanded_queries"], list):
                raise ValueError(f"expanded_queries must be a list at {input_path}:{line_number}")

            expansion_records.append(record)

    return expansion_records


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


def retrieve_chunks(
    query_text: str,
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = embedding_model.embed_texts([query_text])[0]
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return format_retrieved_chunks(query_result)


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

        records.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "query_type": "original",
                "query_text": query_text,
                "retrieved_chunks": retrieve_chunks(
                    query_text=query_text,
                    collection=collection,
                    embedding_model=embedding_model,
                    top_k=top_k,
                ),
            }
        )

    return records


def retrieve_expanded_queries(
    expansion_records: list[dict[str, Any]],
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    records: list[dict[str, Any]] = []
    for expansion_record in expansion_records:
        expanded_queries = [
            str(query) for query in expansion_record["expanded_queries"] if str(query).strip()
        ]

        for query_index, query_text in enumerate(expanded_queries):
            records.append(
                {
                    "question_id": str(expansion_record["question_id"]),
                    "original_question": str(expansion_record["original_question"]),
                    "query_type": "expanded",
                    "expanded_query_index": query_index,
                    "query_text": query_text,
                    "retrieved_chunks": retrieve_chunks(
                        query_text=query_text,
                        collection=collection,
                        embedding_model=embedding_model,
                        top_k=top_k,
                    ),
                }
            )

    return records


def run_original(config: dict[str, Any]) -> Path:
    output_dir = Path(config["paths"]["output_dir"])
    benchmark_path = Path(config["paths"]["benchmark_questions"])
    output_path = output_dir / ORIGINAL_OUTPUT_FILENAME
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


def run_expanded(config: dict[str, Any]) -> Path:
    output_dir = Path(config["paths"]["output_dir"])
    expansions_path = output_dir / QUERY_EXPANSIONS_FILENAME
    output_path = output_dir / EXPANDED_OUTPUT_FILENAME
    vector_db_dir = Path(config["paths"]["vector_db_dir"])
    collection_name = str(config["embedding"]["collection_name"])
    model_name = str(config["embedding"]["model_name"])
    top_k = int(config["retrieval"]["top_k"])

    expansion_records = read_query_expansions(expansions_path)
    embedding_model = EmbeddingModel(model_name)
    collection = get_existing_collection(vector_db_dir, collection_name, model_name)
    records = retrieve_expanded_queries(expansion_records, collection, embedding_model, top_k)
    write_jsonl(records, output_path)

    print(f"Loaded {len(expansion_records)} query expansion records from {expansions_path}")
    print(f"Retrieved top-{top_k} chunks for {len(records)} expanded queries")
    print(f"Saved expanded retrieval results to {output_path}")

    return output_path


def run(config_path: Path = DEFAULT_CONFIG_PATH, mode: str = "original") -> Path:
    config = load_config(config_path)

    if mode == "original":
        return run_original(config)
    if mode == "expanded":
        return run_expanded(config)

    raise ValueError(f"Unsupported retrieval mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOTA RAG retrieval.")
    parser.add_argument(
        "--mode",
        choices=["original", "expanded"],
        default="original",
        help="Retrieval mode to run.",
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
    run(config_path=args.config, mode=args.mode)
