from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from time import perf_counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from table_aware import (
    add_table_metadata,
    assign_table_group_ids,
    classify_chunk_text,
    is_possible_table_text,
)


BATCH_SIZE = 128
SUPPORTED_PARSERS = {"pymupdf", "docling"}
SUPPORTED_CHUNKING_STRATEGIES = {
    "fixed-size",
    "page-based",
    "section-aware",
    "table-aware",
    "parent-child table context",
}
ProgressCallback = Callable[[str, int, int], None]
TABLE_METADATA_KEYS = [
    "chunk_type",
    "table_id",
    "parent_table_id",
    "table_group_index",
    "table_fragment_index",
    "table_markdown",
    "full_table_markdown",
    "parent_table_text",
    "table_value_codes",
    "nearby_context",
    "caption",
    "source_parser",
]


def default_table_handling() -> dict[str, bool]:
    return {
        "detect_tables": True,
        "preserve_table_markdown": True,
        "include_full_table_context": True,
        "include_nearby_context": True,
        "include_table_json": False,
        "use_parent_child_table_context": False,
    }


def normalized_table_handling(table_handling: dict[str, Any] | None = None) -> dict[str, bool]:
    config = default_table_handling()
    if isinstance(table_handling, dict):
        for key in config:
            if key in table_handling:
                config[key] = bool(table_handling[key])
    return config


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Run config must contain a YAML mapping: {path}")

    return config


def read_benchmark_questions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions not found: {path}")

    questions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line must be an object at {path}:{line_number}")

            question_id = record.get("id") or record.get("question_id")
            question = record.get("question")
            if question_id is None or question is None:
                raise ValueError(f"Missing id/question at {path}:{line_number}")

            questions.append({"id": str(question_id), "question": str(question)})

    if not questions:
        raise ValueError(f"Benchmark question file is empty: {path}")

    return questions


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "value"


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def preview_text(value: str, limit: int = 280) -> str:
    normalized = normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def text_identity(value: str) -> str:
    normalized = normalize_text(value).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def make_collection_name(parser_id: str, embedding_model_name: str, run_id: str) -> str:
    digest = hashlib.sha1(f"{parser_id}|{embedding_model_name}|{run_id}".encode("utf-8")).hexdigest()
    return f"{slugify(parser_id)[:16]}_{digest[:16]}"


def make_strategy_collection_name(
    run_id: str,
    parser_id: str,
    chunking_strategy: str,
    embedding_model_name: str,
) -> str:
    embedding_slug = slugify(embedding_model_name.split("/")[-1])
    strategy_slug = slugify(chunking_strategy)
    base = f"{slugify(run_id)[:18]}__{slugify(parser_id)}__{strategy_slug}__{embedding_slug}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:54].strip('_')}_{digest}"


def safe_embedding_model_name(model_name: str) -> str:
    return slugify(model_name.split("/")[-1])


def batch_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


class BenchmarkEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.sentence_transformer: Any | None = None
        if "/" in model_name or model_name.startswith("sentence-transformers"):
            from embeddings import EmbeddingModel

            self.sentence_transformer = EmbeddingModel(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.sentence_transformer is not None:
            return self.sentence_transformer.embed_texts(texts)

        try:
            import ollama

            response = ollama.embed(model=self.model_name, input=texts)
            embeddings = response.get("embeddings")
            if isinstance(embeddings, list):
                return [[float(value) for value in embedding] for embedding in embeddings]
        except AttributeError:
            pass

        embeddings: list[list[float]] = []
        for text in texts:
            import ollama

            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Ollama did not return an embedding for {self.model_name}")
            embeddings.append([float(value) for value in embedding])
        return embeddings


def chunk_metadata(chunk: dict[str, Any], parser_id: str) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "chunk_id": str(chunk["chunk_id"]),
        "document_name": str(chunk["document_name"]),
        "chunk_index": int(chunk["chunk_index"]),
        "char_count": int(chunk["char_count"]),
        "parser": parser_id,
        "chunk_type": str(chunk.get("chunk_type") or classify_chunk_text(str(chunk.get("text", "")))),
        "source_parser": str(chunk.get("source_parser") or parser_id),
    }

    if "page_number" in chunk:
        metadata["page_number"] = int(chunk["page_number"])
    if chunk.get("section_title"):
        metadata["section_title"] = str(chunk["section_title"])
    if chunk.get("source"):
        metadata["source"] = str(chunk["source"])
    for key in TABLE_METADATA_KEYS:
        value = chunk.get(key)
        if key in metadata or value in (None, ""):
            continue
        metadata[key] = str(value)

    return metadata


def chunk_group_key(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return f"object:{id(chunk)}"
    chunk_id = chunk.get("chunk_id")
    if chunk_id not in (None, ""):
        return f"chunk:{chunk_id}"
    return f"object:{id(chunk)}"


def table_group_id(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    return str(chunk.get("parent_table_id") or chunk.get("table_id") or "")


def int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def adjacent_context_chunks(
    target: dict[str, Any],
    chunks: list[dict[str, Any]],
    window: int = 1,
) -> list[dict[str, Any]]:
    if not isinstance(target, dict):
        return []
    target_index = int_or_none(target.get("chunk_index"))
    if target_index is None:
        return []

    target_document = str(target.get("document_name") or "")
    target_source = str(
        target.get("source_parser")
        or target.get("parser")
        or target.get("source")
        or ""
    )
    target_page = target.get("page_number")

    adjacent: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk is target:
            continue
        if table_group_id(chunk):
            continue
        chunk_index = int_or_none(chunk.get("chunk_index"))
        if chunk_index is None or abs(chunk_index - target_index) > window:
            continue
        if target_document and str(chunk.get("document_name") or "") != target_document:
            continue

        chunk_source = str(
            chunk.get("source_parser")
            or chunk.get("parser")
            or chunk.get("source")
            or ""
        )
        if target_source and chunk_source and chunk_source != target_source:
            continue
        if target_page not in (None, "") and chunk.get("page_number") not in (None, ""):
            if str(chunk.get("page_number")) != str(target_page):
                continue
        adjacent.append(chunk)

    return sorted(adjacent, key=lambda item: int_or_none(item.get("chunk_index")) or 0)


def index_chunks(
    chunks: list[dict[str, Any]],
    collection: Any,
    embedding_model: BenchmarkEmbeddingModel,
    parser_id: str,
) -> None:
    for batch in batch_records(chunks, BATCH_SIZE):
        collection.upsert(
            ids=[str(chunk["chunk_id"]) for chunk in batch],
            documents=[str(chunk["text"]) for chunk in batch],
            metadatas=[chunk_metadata(chunk, parser_id) for chunk in batch],
            embeddings=embedding_model.embed_texts([str(chunk["text"]) for chunk in batch]),
        )


def format_retrieved_chunks(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    ids = query_result.get("ids", [[]])[0]
    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        chunk: dict[str, Any] = {
            "rank": index + 1,
            "chunk_id": str(metadata.get("chunk_id", chunk_id)),
            "document_name": str(metadata.get("document_name", "")),
            "chunk_index": metadata.get("chunk_index", ""),
            "text": str(documents[index]),
        }

        if "page_number" in metadata:
            chunk["page_number"] = metadata["page_number"]
        if "section_title" in metadata:
            chunk["section_title"] = metadata["section_title"]
        for key in TABLE_METADATA_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                chunk[key] = value
        if distances:
            chunk["distance"] = distances[index]

        chunks.append(chunk)

    return chunks


def retrieve_questions(
    questions: list[dict[str, str]],
    collection: Any,
    embedding_model: BenchmarkEmbeddingModel,
    run_id: str,
    experiment_type: str,
    parser_id: str,
    chunking_strategy: str,
    embedding_model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in questions:
        query_embedding = embedding_model.embed_texts([question["question"]])[0]
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        records.append(
            {
                "run_id": run_id,
                "experiment_type": experiment_type,
                "parser": parser_id,
                "chunking_strategy": chunking_strategy,
                "embedding_model": embedding_model_name,
                "question_id": question["id"],
                "question": question["question"],
                "top_k": top_k,
                "retrieved_chunks": format_retrieved_chunks(query_result),
            }
        )
    return records


def empty_retrieval_records(
    questions: list[dict[str, str]],
    run_id: str,
    experiment_type: str,
    parser_id: str,
    chunking_strategy: str,
    embedding_model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "experiment_type": experiment_type,
            "parser": parser_id,
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model_name,
            "question_id": question["id"],
            "question": question["question"],
            "top_k": top_k,
            "retrieved_chunks": [],
        }
        for question in questions
    ]


def evidence_label(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return "invalid_chunk"
    parts = [
        f"chunk_id={chunk.get('chunk_id', '')}",
        f"rank={chunk.get('rank', '')}",
    ]
    if chunk.get("page_number") not in (None, ""):
        parts.append(f"page_number={chunk.get('page_number')}")
    if chunk.get("section_title"):
        parts.append(f"section_title={chunk.get('section_title')}")
    if chunk.get("chunk_type"):
        parts.append(f"chunk_type={chunk.get('chunk_type')}")
    if chunk.get("caption"):
        parts.append(f"caption={chunk.get('caption')}")
    return ", ".join(parts)


def source_parser_for_chunk(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    for key in ["source_parser", "parser", "source"]:
        value = chunk.get(key)
        if value not in (None, ""):
            return str(value)
    parser_sources = chunk.get("parser_sources") or chunk.get("retrieved_by_parsers")
    if isinstance(parser_sources, list) and parser_sources:
        return ", ".join(str(value) for value in parser_sources)
    return ""


def chunk_rank(chunk: dict[str, Any]) -> Any:
    if not isinstance(chunk, dict):
        return ""
    return chunk.get("final_rank", chunk.get("rank", ""))


def effective_chunk_type(chunk: dict[str, Any], table_handling: dict[str, bool]) -> str:
    if not isinstance(chunk, dict):
        return "text"
    chunk_type = str(chunk.get("chunk_type") or chunk.get("content_type") or "").casefold()
    if chunk_type in {"table", "table_fragment", "possible_table"}:
        return chunk_type
    if table_handling["detect_tables"] and is_possible_table_text(str(chunk.get("text", ""))):
        return "possible_table"
    return "text"


def chunk_map_by_id(chunks: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id:
            mapped.setdefault(chunk_id, chunk)
    return mapped


def merge_chunk_with_corpus(
    chunk: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}
    chunk_id = str(chunk.get("chunk_id") or "")
    merged = dict(corpus_by_id.get(chunk_id, {}))
    merged.update(chunk)
    return merged


def sorted_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dict_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    return sorted(
        dict_chunks,
        key=lambda item: (
            str(item.get("document_name") or ""),
            str(item.get("source_parser") or item.get("parser") or item.get("source") or ""),
            int_or_none(item.get("page_number")) or 0,
            int_or_none(item.get("chunk_index")) or 0,
            str(item.get("chunk_id") or ""),
        ),
    )


def parent_table_context_text(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in sorted_chunks(chunks):
        text = str(
            chunk.get("full_table_markdown")
            or chunk.get("table_markdown")
            or chunk.get("parent_table_text")
            or chunk.get("text")
            or ""
        ).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def related_table_chunks(
    chunk: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(chunk, dict):
        return [], ""
    dict_chunks = [candidate for candidate in chunks if isinstance(candidate, dict)]
    parent_id = str(chunk.get("parent_table_id") or "")
    table_id = str(chunk.get("table_id") or "")
    if parent_id:
        return (
            sorted_chunks(
                [
                    candidate
                    for candidate in dict_chunks
                    if str(candidate.get("parent_table_id") or candidate.get("table_id") or "") == parent_id
                ]
            ),
            "same_parent_table_id",
        )
    if table_id:
        return (
            sorted_chunks(
                [candidate for candidate in dict_chunks if str(candidate.get("table_id") or "") == table_id]
            ),
            "same_table_id",
        )
    return [], ""


def expanded_context_record(
    chunk: dict[str, Any],
    expanded_from_chunk_id: str,
    reason: str,
) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        chunk = {}
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "expanded_from_chunk_id": expanded_from_chunk_id,
        "context_expansion_reason": reason,
        "table_id": chunk.get("table_id", ""),
        "parent_table_id": chunk.get("parent_table_id", ""),
        "page_number": chunk.get("page_number", ""),
        "section_title": chunk.get("section_title", ""),
        "source_parser": source_parser_for_chunk(chunk),
        "text_preview": preview_text(
            str(chunk.get("table_markdown") or chunk.get("text") or chunk.get("parent_table_text") or ""),
            limit=360,
        ),
    }


def prepare_answer_context_chunks(
    chunks: list[dict[str, Any]],
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    handling = normalized_table_handling(table_handling)
    corpus_by_id = chunk_map_by_id(all_chunks)
    corpus_chunks = list(corpus_by_id.values())
    input_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    retrieved_chunk_ids = {
        str(chunk.get("chunk_id") or "") for chunk in input_chunks if chunk.get("chunk_id")
    }
    context_chunks: list[dict[str, Any]] = []
    expansion_records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_context_chunk(chunk: dict[str, Any]) -> None:
        key = chunk_group_key(chunk)
        if key in seen:
            return
        seen.add(key)
        context_chunks.append(chunk)

    for retrieved_chunk in input_chunks:
        chunk = merge_chunk_with_corpus(retrieved_chunk, corpus_by_id)
        chunk_type = effective_chunk_type(chunk, handling)
        search_pool = corpus_chunks or [merge_chunk_with_corpus(item, corpus_by_id) for item in input_chunks]
        related_chunks, reason = related_table_chunks(chunk, search_pool)

        has_complete_group_context = len(related_chunks) > 1 or any(
            item.get("full_table_markdown") or item.get("table_markdown") for item in related_chunks
        )
        if related_chunks and has_complete_group_context:
            combined_context = parent_table_context_text(related_chunks)
            context_chunk = dict(chunk)
            context_chunk["has_full_parent_table_context"] = len(related_chunks) > 1 or bool(
                context_chunk.get("full_table_markdown") or context_chunk.get("table_markdown")
            )
            context_chunk["table_context_chunk_ids"] = [
                str(item.get("chunk_id")) for item in related_chunks if item.get("chunk_id") not in (None, "")
            ]
            if combined_context:
                if any(item.get("table_markdown") or item.get("full_table_markdown") for item in related_chunks):
                    context_chunk["full_table_markdown"] = combined_context
                else:
                    context_chunk["parent_table_text"] = combined_context
            add_context_chunk(context_chunk)

            expanded_from = str(chunk.get("chunk_id") or "")
            for related_chunk in related_chunks:
                related_id = str(related_chunk.get("chunk_id") or "")
                if related_id and related_id not in retrieved_chunk_ids:
                    expansion_records.append(expanded_context_record(related_chunk, expanded_from, reason))
                add_context_chunk(related_chunk)
            continue

        if chunk_type in {"table", "table_fragment", "possible_table"}:
            chunk = dict(chunk)
            chunk["table_context_incomplete"] = True
            add_context_chunk(chunk)
            for adjacent_chunk in adjacent_context_chunks(chunk, search_pool):
                adjacent_id = str(adjacent_chunk.get("chunk_id") or "")
                if adjacent_id and adjacent_id not in retrieved_chunk_ids:
                    expansion_records.append(
                        expanded_context_record(
                            adjacent_chunk,
                            str(chunk.get("chunk_id") or ""),
                            "adjacent_table_chunk_fallback",
                        )
                    )
                add_context_chunk(adjacent_chunk)
        else:
            add_context_chunk(chunk)

    return context_chunks, expansion_records


def answer_context_text(chunk: dict[str, Any], table_handling: dict[str, bool]) -> str:
    chunk_type = effective_chunk_type(chunk, table_handling)
    group_id = table_group_id(chunk)
    parts: list[str] = []
    header = [
        f"Rank: {chunk_rank(chunk)}" if chunk_rank(chunk) not in (None, "") else "",
        f"Chunk ID: {chunk.get('chunk_id', '')}",
        f"Table group ID: {group_id}" if group_id else "",
        f"Table ID: {chunk.get('table_id')}" if chunk.get("table_id") else "",
        f"Parent table ID: {chunk.get('parent_table_id')}" if chunk.get("parent_table_id") else "",
        "Table context incomplete: true" if chunk.get("table_context_incomplete") else "",
        f"Source parser: {source_parser_for_chunk(chunk)}" if source_parser_for_chunk(chunk) else "",
        f"Page: {chunk.get('page_number')}" if chunk.get("page_number") not in (None, "") else "",
    ]
    parts.append("\n".join(item for item in header if item))
    if chunk.get("section_title"):
        parts.append(f"Section: {chunk['section_title']}")
    if chunk.get("caption"):
        parts.append(f"Caption: {chunk['caption']}")

    if chunk_type in {"table", "table_fragment"}:
        if table_handling["include_nearby_context"] and chunk.get("nearby_context"):
            parts.append(f"Nearby context:\n{chunk['nearby_context']}")
        if table_handling["include_table_json"] and chunk.get("table_json"):
            parts.append(f"Raw table JSON:\n{json.dumps(chunk['table_json'], ensure_ascii=False)}")
        table_text = (
            chunk.get("full_table_markdown")
            or chunk.get("table_markdown")
            or chunk.get("parent_table_text")
            if table_handling["include_full_table_context"]
            else None
        ) or chunk.get("text", "")
        label = "Full parent table context" if chunk.get("parent_table_text") or chunk.get("full_table_markdown") else "Table evidence"
        parts.append(f"{label}:\n{table_text}")
    elif chunk_type == "possible_table":
        if table_handling["include_nearby_context"] and chunk.get("nearby_context"):
            parts.append(f"Nearby context:\n{chunk['nearby_context']}")
        parts.append(str(chunk.get("text", "")))
    else:
        parts.append(str(chunk.get("text", "")))
    return "\n\n".join(part for part in parts if part).strip()


def build_answer_context(
    chunks: list[dict[str, Any]],
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> str:
    handling = normalized_table_handling(table_handling)
    expanded_chunks, _expansion_records = prepare_answer_context_chunks(
        chunks,
        handling,
        all_chunks=all_chunks,
    )

    return "\n\n---\n\n".join(
        f"[{evidence_label(chunk)}]\n{answer_context_text(chunk, handling)}"
        for chunk in expanded_chunks
    )


def table_evidence_used(
    chunks: list[dict[str, Any]],
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    handling = normalized_table_handling(table_handling)
    context_chunks, _expansion_records = prepare_answer_context_chunks(
        chunks,
        handling,
        all_chunks=all_chunks,
    )
    context_by_id = chunk_map_by_id(context_chunks)
    evidence: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id and chunk_id in context_by_id:
            chunk = context_by_id[chunk_id]
        chunk_type = effective_chunk_type(chunk, handling)
        if chunk_type not in {"table", "table_fragment", "possible_table"}:
            continue
        has_table_markdown = chunk.get("table_markdown") not in (None, "")
        if has_table_markdown:
            preview_source = chunk.get("full_table_markdown") or chunk.get("table_markdown")
        else:
            preview_source = chunk.get("parent_table_text") or chunk.get("text", "")
        item = {
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_type": chunk_type,
            "table_id": chunk.get("table_id", ""),
            "parent_table_id": chunk.get("parent_table_id", ""),
            "page_number": chunk.get("page_number", ""),
            "section_title": chunk.get("section_title", ""),
            "source_parser": source_parser_for_chunk(chunk),
            "has_full_parent_table_context": bool(chunk.get("has_full_parent_table_context")),
            "table_context_incomplete": not bool(chunk.get("has_full_parent_table_context"))
            and chunk_type in {"table_fragment", "possible_table"},
        }
        if not table_group_id(chunk):
            adjacent_chunk_ids = sorted(
                {
                    str(record.get("chunk_id"))
                    for record in _expansion_records
                    if record.get("expanded_from_chunk_id") == chunk.get("chunk_id")
                    and record.get("context_expansion_reason") == "adjacent_table_chunk_fallback"
                    and record.get("chunk_id") not in (None, "")
                }
            )
            if adjacent_chunk_ids:
                item["adjacent_context_chunk_ids"] = adjacent_chunk_ids
        if chunk.get("final_rank") not in (None, ""):
            item["final_rank"] = chunk.get("final_rank")
        elif chunk.get("rank") not in (None, ""):
            item["rank"] = chunk.get("rank")
        if has_table_markdown:
            item["table_markdown_preview"] = preview_text(str(preview_source), limit=360)
        else:
            item["text_preview"] = preview_text(str(preview_source), limit=360)
        if chunk.get("caption"):
            item["caption"] = chunk["caption"]
        evidence.append(item)
    return evidence


COMPARE_KEYWORDS = [
    "compare",
    "comparison",
    "difference",
    "differences",
    "versus",
    "contrast",
    "비교",
    "차이",
    "다른 점",
    "공통점",
    "대비",
]


def is_compare_question(question: str) -> bool:
    normalized = str(question or "").casefold()
    if re.search(r"\bvs\.?\b", normalized):
        return True
    return any(keyword in normalized for keyword in COMPARE_KEYWORDS)


def answer_intent(question: str) -> str:
    return "comparison" if is_compare_question(question) else "normal"


def build_answer_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> str:
    context = build_answer_context(chunks, table_handling, all_chunks=all_chunks)
    if is_compare_question(question):
        return f"""You answer technical-document comparison questions using only retrieved and expanded context.

Rules:
- Use only the retrieved and expanded context below.
- Do not use outside knowledge.
- Do not invent unsupported differences or similarities.
- If one side lacks evidence, clearly say the evidence is missing.
- Preserve exact technical terms, field names, bit values, and table values.
- Pay special attention to table evidence.
- Some table evidence may be split across multiple chunks.
- If table fragments are grouped by table_id or parent_table_id, treat them as one table.
- If comparing table values, include all relevant values found in context.
- If the context contains a value table, list all relevant value-description pairs.
- Preserve exact value codes such as 000b, 001b, and 010b.
- Do not stop after the first table row.
- If a table lists values, preserve the value-to-description relationship.
- Do not invent table values.
- Do not infer values that are not in the table.
- If only partial table evidence is available, state: "The retrieved table evidence appears incomplete."
- In Evidence Used, cite chunk_id, page number, section title, and parser/source when available.

Question:
{question}

Retrieved and expanded context:
{context}

Return exactly these sections:
## Comparison Summary
## Side-by-side Comparison Table
| Aspect | Item A | Item B | Evidence |
|---|---|---|---|
## Key Differences
## Similarities
## Evidence Used
## Missing or Uncertain Information
"""

    return f"""You answer technical-document questions using only retrieved and expanded context.

Rules:
- Use only the retrieved and expanded context below.
- Do not use outside knowledge.
- Do not hallucinate fields, values, sections, or requirements.
- Pay special attention to table evidence.
- Some table evidence may be split across multiple chunks.
- If table fragments are grouped by table_id or parent_table_id, treat them as one table.
- If the context contains a value table, list all relevant value-description pairs.
- Preserve exact value codes such as 000b, 001b, and 010b.
- Do not stop after the first table row.
- If a table lists values, preserve the value-to-description relationship.
- Do not invent table values.
- Do not infer values that are not in the table.
- If only partial table evidence is available, state that the table evidence appears incomplete.
- If table evidence is insufficient or unclear, say so clearly.
- If the retrieved chunks are insufficient, say so clearly.
- Keep the answer concise and technical.
- In Evidence Used, mention table chunk IDs, page numbers, section titles, and parser/source when available.

Question:
{question}

Retrieved context:
{context}

Return exactly these sections:
## Answer Summary
## Table Values / Field Values Used
## Evidence Used
## Missing or Uncertain Information
"""


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    answer_model: str,
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> str:
    if not chunks:
        if is_compare_question(question):
            return (
                "## Comparison Summary\n"
                "The retrieved context is insufficient because no chunks were retrieved.\n\n"
                "## Side-by-side Comparison Table\n"
                "| Aspect | Item A | Item B | Evidence |\n"
                "|---|---|---|---|\n"
                "| Retrieved evidence | Missing | Missing | No retrieved chunks were available. |\n\n"
                "## Key Differences\n"
                "- Cannot determine differences from the retrieved context.\n\n"
                "## Similarities\n"
                "- Cannot determine similarities from the retrieved context.\n\n"
                "## Evidence Used\n"
                "- None\n\n"
                "## Missing or Uncertain Information\n"
                "- No retrieved chunks were available.\n"
            )
        return (
            "## Answer Summary\n"
            "The retrieved context is insufficient because no chunks were retrieved.\n\n"
            "## Table Values / Field Values Used\n"
            "- None\n\n"
            "## Evidence Used\n"
            "- None\n\n"
            "## Missing or Uncertain Information\n"
            "- No retrieved chunks were available.\n"
        )

    import ollama

    response = ollama.generate(
        model=answer_model,
        prompt=build_answer_prompt(question, chunks, table_handling, all_chunks=all_chunks),
        options={"temperature": 0.1},
    )
    return str(response["response"]).strip()


def generate_answers(
    retrieval_records: list[dict[str, Any]],
    answer_model: str,
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for record in retrieval_records:
        chunks = record["retrieved_chunks"]
        intent = answer_intent(str(record["question"]))
        _context_chunks, expansion_records = prepare_answer_context_chunks(
            chunks,
            table_handling,
            all_chunks=all_chunks,
        )
        answers.append(
            {
                "run_id": record["run_id"],
                "experiment_type": record["experiment_type"],
                "parser": record["parser"],
                "chunking_strategy": record["chunking_strategy"],
                "embedding_model": record["embedding_model"],
                "answer_model": answer_model,
                "answer_intent": intent,
                "question_id": record["question_id"],
                "question": record["question"],
                "retrieved_chunks": chunks,
                "table_evidence_used": table_evidence_used(
                    chunks,
                    table_handling,
                    all_chunks=all_chunks,
                ),
                "expanded_context_chunks": expansion_records,
                "generated_answer": generate_answer(
                    question=str(record["question"]),
                    chunks=chunks,
                    answer_model=answer_model,
                    table_handling=table_handling,
                    all_chunks=all_chunks,
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return answers


def extract_document(
    parser_id: str,
    pdf_path: Path,
    extraction_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    extraction_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if parser_id == "pymupdf":
        from ingest import extract_pdf_pages

        extracted = extract_pdf_pages(pdf_path)
        write_jsonl(extracted, extraction_dir / "extracted.jsonl")
        return extracted, warnings

    if parser_id == "docling":
        from ingest_docling import convert_pdf_to_markdown, markdown_sections

        markdown = convert_pdf_to_markdown(pdf_path)
        extracted = markdown_sections(markdown, document_name=pdf_path.name)
        write_jsonl(extracted, extraction_dir / "extracted.jsonl")
        (extraction_dir / "document.md").write_text(markdown, encoding="utf-8")
        if not any(record.get("section_title") for record in extracted):
            warnings.append("Docling extraction did not include section titles.")
        return extracted, warnings

    raise ValueError(f"Unsupported parser for runner: {parser_id}")


def chunk_id_base(document_name: str, chunking_strategy: str, chunk_index: int) -> str:
    return f"{document_name}:{slugify(chunking_strategy)}:c{chunk_index:04d}"


def chunk_record(
    source_record: dict[str, Any],
    text: str,
    chunking_strategy: str,
    chunk_index: int,
    offset: int = 0,
) -> dict[str, Any]:
    document_name = str(source_record["document_name"])
    chunk: dict[str, Any] = {
        "chunk_id": f"{chunk_id_base(document_name, chunking_strategy, chunk_index)}:s{offset:04d}",
        "document_name": document_name,
        "chunk_index": chunk_index,
        "text": text,
        "char_count": len(text),
        "chunking_strategy": chunking_strategy,
    }
    if source_record.get("page_number") not in (None, ""):
        chunk["page_number"] = int(source_record["page_number"])
    if source_record.get("section_title"):
        chunk["section_title"] = str(source_record["section_title"])
    if source_record.get("source"):
        chunk["source"] = str(source_record["source"])
    source_parser = str(
        source_record.get("source")
        or source_record.get("source_parser")
        or ("pymupdf" if source_record.get("page_number") not in (None, "") else "unknown")
    )
    add_table_metadata(
        chunk,
        source_parser=source_parser,
        table_id=str(source_record["table_id"]) if source_record.get("table_id") else None,
        parent_table_id=(
            str(source_record["parent_table_id"]) if source_record.get("parent_table_id") else None
        ),
        table_markdown=str(source_record["table_markdown"]) if source_record.get("table_markdown") else None,
        nearby_context=str(source_record["nearby_context"]) if source_record.get("nearby_context") else None,
        caption=str(source_record["caption"]) if source_record.get("caption") else None,
    )
    for key in TABLE_METADATA_KEYS:
        if key not in chunk and source_record.get(key) not in (None, ""):
            chunk[key] = source_record[key]
    return chunk


def create_page_based_chunks(
    extracted_records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    from chunking import split_text

    chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not any(record.get("page_number") for record in extracted_records):
        warnings.append("Page numbers were unavailable; page-based chunking used extracted records as page-like units.")

    for record in extracted_records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        text_chunks = [text] if len(text) <= chunk_size else split_text(text, chunk_size, chunk_overlap)
        for offset, chunk_text in enumerate(text_chunks):
            chunks.append(
                chunk_record(
                    record,
                    chunk_text,
                    chunking_strategy="page-based",
                    chunk_index=len(chunks),
                    offset=offset,
                )
            )
    return chunks, warnings


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if len(text) < 3 or len(text) > 120:
        return False
    if text.endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\s+\S+", text):
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", text):
        return True
    letters = [character for character in text if character.isalpha()]
    if len(letters) >= 3 and sum(character.isupper() for character in letters) / len(letters) > 0.65:
        return True
    words = text.split()
    return 1 <= len(words) <= 10 and text.istitle()


def pymupdf_section_records(extracted_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_records: list[dict[str, Any]] = []
    for page in extracted_records:
        current_title: str | None = None
        current_lines: list[str] = []

        def append_current() -> None:
            text = "\n".join(current_lines).strip()
            if not text or current_title is None:
                return
            section_records.append(
                {
                    "document_name": page["document_name"],
                    "page_number": page.get("page_number"),
                    "section_title": current_title,
                    "text": text,
                    "char_count": len(text),
                    "source": "pymupdf",
                }
            )

        for line in str(page.get("text", "")).splitlines():
            if looks_like_heading(line):
                append_current()
                current_title = line.strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        append_current()
    return section_records


def create_section_aware_chunks(
    parser_id: str,
    extracted_records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    from chunking import split_text

    warnings: list[str] = []
    section_records = extracted_records
    if parser_id == "pymupdf":
        section_records = pymupdf_section_records(extracted_records)
        if not section_records:
            warnings.append("Section-aware chunking fell back to fixed-size because PyMuPDF headings were unavailable.")
            chunks, fallback_warnings = create_chunks_for_strategy(
                parser_id,
                extracted_records,
                "fixed-size",
                chunk_size,
                chunk_overlap,
                force_strategy_name="section-aware",
            )
            return chunks, warnings + fallback_warnings
    elif not any(record.get("section_title") for record in extracted_records):
        warnings.append("Section-aware chunking fell back to fixed-size because section titles were unavailable.")
        chunks, fallback_warnings = create_chunks_for_strategy(
            parser_id,
            extracted_records,
            "fixed-size",
            chunk_size,
            chunk_overlap,
            force_strategy_name="section-aware",
        )
        return chunks, warnings + fallback_warnings

    chunks: list[dict[str, Any]] = []
    for record in section_records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        for offset, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                chunk_record(
                    record,
                    chunk_text,
                    chunking_strategy="section-aware",
                    chunk_index=len(chunks),
                    offset=offset,
                )
            )
    return chunks, warnings


def create_chunks_for_strategy(
    parser_id: str,
    extracted_records: list[dict[str, Any]],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    force_strategy_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    output_strategy = force_strategy_name or chunking_strategy
    warnings: list[str] = []

    if chunking_strategy in {"fixed-size", "table-aware", "parent-child table context"}:
        if parser_id == "pymupdf":
            from chunking import create_chunks as create_pymupdf_chunks

            chunks = create_pymupdf_chunks(extracted_records, chunk_size, chunk_overlap)
        else:
            from chunking_docling import create_chunks as create_docling_chunks

            chunks = create_docling_chunks(extracted_records, chunk_size, chunk_overlap)
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{chunk['document_name']}:{slugify(output_strategy)}:c{index:04d}"
            chunk["chunk_index"] = index
            chunk["chunking_strategy"] = output_strategy
            chunk["source_parser"] = parser_id
            if "chunk_type" not in chunk:
                add_table_metadata(chunk, source_parser=parser_id)
        assign_table_group_ids(chunks)
        if chunking_strategy == "parent-child table context":
            warnings.append(
                "parent-child table context currently stores table nearby_context metadata when available; "
                "hierarchical child-to-parent retrieval is not implemented yet."
            )
        return chunks, warnings

    if chunking_strategy == "page-based":
        chunks, warnings = create_page_based_chunks(extracted_records, chunk_size, chunk_overlap)
        return assign_table_group_ids(chunks), warnings

    if chunking_strategy == "section-aware":
        chunks, warnings = create_section_aware_chunks(parser_id, extracted_records, chunk_size, chunk_overlap)
        return assign_table_group_ids(chunks), warnings

    raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")


def extract_and_chunk(
    parser_id: str,
    pdf_path: Path,
    parser_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    if parser_id == "pymupdf":
        from chunking import create_chunks as create_pymupdf_chunks
        from ingest import extract_pdf_pages

        extracted = extract_pdf_pages(pdf_path)
        write_jsonl(extracted, parser_dir / "extracted.jsonl")
        chunks = create_pymupdf_chunks(extracted, chunk_size, chunk_overlap)
        write_jsonl(chunks, parser_dir / "chunks.jsonl")
        return chunks

    if parser_id == "docling":
        from chunking_docling import create_chunks as create_docling_chunks
        from ingest_docling import convert_pdf_to_markdown, markdown_sections

        markdown = convert_pdf_to_markdown(pdf_path)
        extracted = markdown_sections(markdown, document_name=pdf_path.name)
        write_jsonl(extracted, parser_dir / "extracted.jsonl")
        (parser_dir / "document.md").write_text(markdown, encoding="utf-8")
        chunks = create_docling_chunks(extracted, chunk_size, chunk_overlap)
        write_jsonl(chunks, parser_dir / "chunks.jsonl")
        return chunks

    raise ValueError(f"Unsupported parser for runner: {parser_id}")


def write_comparison_report(
    run_dir: Path,
    questions: list[dict[str, str]],
    parser_results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "parser_comparison.csv"
    report_path = reports_dir / "parser_comparison_report.md"
    parser_ids = list(parser_results)

    retrieval_by_parser = {
        parser_id: {
            str(record["question_id"]): record
            for record in result["retrieval_records"]
        }
        for parser_id, result in parser_results.items()
    }
    answers_by_parser = {
        parser_id: {
            str(record["question_id"]): record
            for record in result["answer_records"]
        }
        for parser_id, result in parser_results.items()
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["id"]
        evidence_sets: dict[str, set[str]] = {}
        retrieved_counts: dict[str, int] = {}
        answer_previews: dict[str, str] = {}

        for parser_id in parser_ids:
            retrieval_record = retrieval_by_parser[parser_id].get(question_id, {})
            chunks = retrieval_record.get("retrieved_chunks", [])
            evidence_sets[parser_id] = {text_identity(str(chunk.get("text", ""))) for chunk in chunks}
            retrieved_counts[parser_id] = len(chunks)
            answer_record = answers_by_parser[parser_id].get(question_id, {})
            answer_previews[parser_id] = preview_text(str(answer_record.get("generated_answer", "")))

        overlap_count = 0
        if evidence_sets:
            overlap = set.intersection(*evidence_sets.values()) if evidence_sets.values() else set()
            overlap_count = len(overlap)

        rows.append(
            {
                "question_id": question_id,
                "question": question["question"],
                "parsers": "|".join(parser_ids),
                "retrieved_counts": json.dumps(retrieved_counts, ensure_ascii=False),
                "unique_chunk_counts": json.dumps(
                    {parser_id: len(evidence_sets[parser_id]) for parser_id in parser_ids},
                    ensure_ascii=False,
                ),
                "overlap_count": overlap_count,
                "answer_previews": json.dumps(answer_previews, ensure_ascii=False),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "question_id",
            "question",
            "parsers",
            "retrieved_counts",
            "unique_chunk_counts",
            "overlap_count",
            "answer_previews",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Parser Comparison Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {len(questions)}",
        f"- Parsers compared: {', '.join(parser_ids)}",
        "",
        "This report is neutral and does not automatically claim one parser is better.",
        "",
        "## Per-Question Comparison",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"- Retrieved chunks: `{row['retrieved_counts']}`",
                f"- Unique evidence chunks by normalized text: `{row['unique_chunk_counts']}`",
                f"- Overlapping or similar evidence count: {row['overlap_count']}",
                f"- Answer previews: `{row['answer_previews']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Review whether each answer is grounded in its retrieved chunks.",
            "- Check whether parser-specific chunks preserve important table and field details.",
            "- Treat overlap counts as approximate because parser chunk boundaries may differ.",
            "- Do not use this report as an automatic parser recommendation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def average_chunk_length(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return sum(int(chunk.get("char_count", len(str(chunk.get("text", ""))))) for chunk in chunks) / len(chunks)


def retrieved_locations(chunks: list[dict[str, Any]]) -> str:
    pages = sorted(
        {
            str(chunk.get("page_number"))
            for chunk in chunks
            if chunk.get("page_number") not in (None, "")
        }
    )
    sections = sorted(
        {
            str(chunk.get("section_title"))
            for chunk in chunks
            if chunk.get("section_title")
        }
    )
    parts: list[str] = []
    if pages:
        parts.append(f"pages={', '.join(pages)}")
    if sections:
        parts.append(f"sections={', '.join(sections[:8])}")
    return "; ".join(parts)


def write_chunking_comparison_report(
    run_dir: Path,
    questions: list[dict[str, str]],
    strategy_results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "chunking_comparison.csv"
    report_path = reports_dir / "chunking_comparison_report.md"
    strategies = list(strategy_results)

    retrieval_by_strategy = {
        strategy: {
            str(record["question_id"]): record
            for record in result["retrieval_records"]
        }
        for strategy, result in strategy_results.items()
    }
    answers_by_strategy = {
        strategy: {
            str(record["question_id"]): record
            for record in result["answer_records"]
        }
        for strategy, result in strategy_results.items()
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["id"]
        for strategy in strategies:
            result = strategy_results[strategy]
            retrieval_record = retrieval_by_strategy[strategy].get(question_id, {})
            answer_record = answers_by_strategy[strategy].get(question_id, {})
            retrieved_chunks = retrieval_record.get("retrieved_chunks", [])
            rows.append(
                {
                    "question_id": question_id,
                    "question": question["question"],
                    "chunking_strategy": strategy,
                    "chunk_count": result["chunk_count"],
                    "average_chunk_length": f"{result['average_chunk_length']:.2f}",
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "retrieved_locations": retrieved_locations(retrieved_chunks),
                    "answer_preview": preview_text(str(answer_record.get("generated_answer", ""))),
                    "warnings": " | ".join(result.get("warnings", [])),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "question_id",
            "question",
            "chunking_strategy",
            "chunk_count",
            "average_chunk_length",
            "retrieved_chunk_count",
            "retrieved_locations",
            "answer_preview",
            "warnings",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Chunking Comparison Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {len(questions)}",
        f"- Chunking strategies compared: {', '.join(strategies)}",
        "",
        "This report is neutral and does not automatically claim one chunking strategy is better.",
        "",
        "## Strategy Warnings",
    ]
    for strategy in strategies:
        warnings = strategy_results[strategy].get("warnings", [])
        if warnings:
            lines.append(f"- {strategy}: {' | '.join(warnings)}")
        else:
            lines.append(f"- {strategy}: none")

    lines.extend(["", "## Per-Question Comparison"])
    for question in questions:
        lines.extend(["", f"### {question['id']}", "", f"Question: {question['question']}"])
        for row in [item for item in rows if item["question_id"] == question["id"]]:
            lines.extend(
                [
                    "",
                    f"#### {row['chunking_strategy']}",
                    "",
                    f"- Chunks generated: {row['chunk_count']}",
                    f"- Average chunk length: {row['average_chunk_length']}",
                    f"- Retrieved chunks: {row['retrieved_chunk_count']}",
                    f"- Retrieved pages or sections: {row['retrieved_locations'] or 'none'}",
                    f"- Answer preview: {row['answer_preview']}",
                    f"- Warnings: {row['warnings'] or 'none'}",
                ]
            )

    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Review whether each answer is grounded in the retrieved chunks.",
            "- Check whether chunk boundaries preserve table rows, field names, and section context.",
            "- Compare retrieved pages or sections without treating overlap as an automatic quality score.",
            "- Do not use this report as an automatic chunking strategy recommendation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def write_embedding_comparison_report(
    run_dir: Path,
    questions: list[dict[str, str]],
    model_results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "embedding_comparison.csv"
    report_path = reports_dir / "embedding_comparison_report.md"
    model_names = list(model_results)

    retrieval_by_model = {
        model_name: {
            str(record["question_id"]): record
            for record in result["retrieval_records"]
        }
        for model_name, result in model_results.items()
    }
    answers_by_model = {
        model_name: {
            str(record["question_id"]): record
            for record in result["answer_records"]
        }
        for model_name, result in model_results.items()
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["id"]
        for model_name in model_names:
            result = model_results[model_name]
            retrieval_record = retrieval_by_model[model_name].get(question_id, {})
            answer_record = answers_by_model[model_name].get(question_id, {})
            retrieved_chunks = retrieval_record.get("retrieved_chunks", [])
            rows.append(
                {
                    "question_id": question_id,
                    "question": question["question"],
                    "embedding_model": model_name,
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "retrieved_locations": retrieved_locations(retrieved_chunks),
                    "answer_preview": preview_text(str(answer_record.get("generated_answer", ""))),
                    "embedding_runtime_seconds": f"{float(result.get('embedding_runtime_seconds', 0.0)):.3f}",
                    "warnings": " | ".join(result.get("warnings", [])),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "question_id",
            "question",
            "embedding_model",
            "retrieved_chunk_count",
            "retrieved_locations",
            "answer_preview",
            "embedding_runtime_seconds",
            "warnings",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Embedding Comparison Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {len(questions)}",
        f"- Embedding models compared: {', '.join(model_names)}",
        "",
        "This report is neutral and does not automatically claim one embedding model is better.",
        "",
        "## Model Warnings",
    ]
    for model_name in model_names:
        warnings = model_results[model_name].get("warnings", [])
        if warnings:
            lines.append(f"- {model_name}: {' | '.join(warnings)}")
        else:
            lines.append(f"- {model_name}: none")

    lines.extend(["", "## Per-Question Comparison"])
    for question in questions:
        lines.extend(["", f"### {question['id']}", "", f"Question: {question['question']}"])
        for row in [item for item in rows if item["question_id"] == question["id"]]:
            lines.extend(
                [
                    "",
                    f"#### {row['embedding_model']}",
                    "",
                    f"- Retrieved chunks: {row['retrieved_chunk_count']}",
                    f"- Retrieved pages or sections: {row['retrieved_locations'] or 'none'}",
                    f"- Embedding runtime seconds: {row['embedding_runtime_seconds']}",
                    f"- Answer preview: {row['answer_preview']}",
                    f"- Warnings: {row['warnings'] or 'none'}",
                ]
            )

    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Review whether each answer is grounded in the retrieved chunks.",
            "- Compare retrieved pages or sections without treating overlap as an automatic quality score.",
            "- Check whether model-specific retrieved chunks preserve important table and field details.",
            "- Do not use this report as an automatic embedding model recommendation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def fused_chunk_from_candidate(
    chunk: dict[str, Any],
    final_rank: int,
    fusion_score: float,
    retrieved_by_models: list[str],
    ranks_by_model: dict[str, int],
) -> dict[str, Any]:
    fused: dict[str, Any] = {
        "final_rank": final_rank,
        "rank": final_rank,
        "chunk_id": str(chunk.get("chunk_id", "")),
        "document_name": str(chunk.get("document_name", "")),
        "chunk_index": chunk.get("chunk_index", ""),
        "text": str(chunk.get("text", "")),
        "fusion_score": fusion_score,
        "retrieved_by_embedding_models": retrieved_by_models,
        "original_ranks_by_model": ranks_by_model,
    }
    if chunk.get("page_number") not in (None, ""):
        fused["page_number"] = chunk.get("page_number")
    if chunk.get("section_title"):
        fused["section_title"] = chunk.get("section_title")
    for key in TABLE_METADATA_KEYS:
        value = chunk.get(key)
        if value not in (None, ""):
            fused[key] = value
    return fused


def fuse_retrieved_chunks(
    retrieval_by_model: dict[str, list[dict[str, Any]]],
    selected_models: list[str],
    fusion_method: str,
    final_top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, tuple[int, int]] = {}

    for model_index, model_name in enumerate(selected_models):
        for chunk in retrieval_by_model.get(model_name, []):
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue
            rank = int(chunk.get("rank", 10_000))
            candidate = candidates.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "retrieved_by": [],
                    "ranks_by_model": {},
                    "rrf_score": 0.0,
                },
            )
            if model_name not in candidate["retrieved_by"]:
                candidate["retrieved_by"].append(model_name)
            candidate["ranks_by_model"][model_name] = rank
            candidate["rrf_score"] += 1 / (rrf_k + rank)
            first_seen.setdefault(chunk_id, (model_index, rank))

    if fusion_method == "union_dedup":
        sorted_items = sorted(
            candidates.items(),
            key=lambda item: (
                first_seen[item[0]][0],
                first_seen[item[0]][1],
                item[0],
            ),
        )
    elif fusion_method == "rrf":
        sorted_items = sorted(
            candidates.items(),
            key=lambda item: (
                -float(item[1]["rrf_score"]),
                min(item[1]["ranks_by_model"].values()),
                item[0],
            ),
        )
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    fused_chunks: list[dict[str, Any]] = []
    for final_rank, (_chunk_id, candidate) in enumerate(sorted_items[:final_top_k], start=1):
        fusion_score = (
            float(len(candidate["retrieved_by"]))
            if fusion_method == "union_dedup"
            else float(candidate["rrf_score"])
        )
        fused_chunks.append(
            fused_chunk_from_candidate(
                chunk=candidate["chunk"],
                final_rank=final_rank,
                fusion_score=fusion_score,
                retrieved_by_models=list(candidate["retrieved_by"]),
                ranks_by_model=dict(candidate["ranks_by_model"]),
            )
        )
    return fused_chunks


def generate_fusion_answers(
    fused_records: list[dict[str, Any]],
    answer_model: str,
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for record in fused_records:
        chunks = record["fused_chunks"]
        intent = answer_intent(str(record["question"]))
        _context_chunks, expansion_records = prepare_answer_context_chunks(
            chunks,
            table_handling,
            all_chunks=all_chunks,
        )
        answers.append(
            {
                "run_id": record["run_id"],
                "experiment_type": record["experiment_type"],
                "parser": record["parser"],
                "chunking_strategy": record["chunking_strategy"],
                "selected_embedding_models": record["selected_embedding_models"],
                "fusion_method": record["fusion_method"],
                "answer_model": answer_model,
                "answer_intent": intent,
                "question_id": record["question_id"],
                "question": record["question"],
                "fused_chunks": chunks,
                "table_evidence_used": table_evidence_used(
                    chunks,
                    table_handling,
                    all_chunks=all_chunks,
                ),
                "expanded_context_chunks": expansion_records,
                "generated_answer": generate_answer(
                    question=str(record["question"]),
                    chunks=chunks,
                    answer_model=answer_model,
                    table_handling=table_handling,
                    all_chunks=all_chunks,
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return answers


def fused_parser_chunk_from_candidate(
    chunk: dict[str, Any],
    final_rank: int,
    fusion_score: float,
    retrieved_by_parsers: list[str],
    ranks_by_parser: dict[str, int],
) -> dict[str, Any]:
    fused: dict[str, Any] = {
        "final_rank": final_rank,
        "rank": final_rank,
        "chunk_id": str(chunk.get("chunk_id", "")),
        "document_name": str(chunk.get("document_name", "")),
        "chunk_index": chunk.get("chunk_index", ""),
        "text": str(chunk.get("text", "")),
        "fusion_score": fusion_score,
        "retrieved_by_parsers": retrieved_by_parsers,
        "original_ranks_by_parser": ranks_by_parser,
        "parser_sources": retrieved_by_parsers,
    }
    if chunk.get("page_number") not in (None, ""):
        fused["page_number"] = chunk.get("page_number")
    if chunk.get("section_title"):
        fused["section_title"] = chunk.get("section_title")
    for key in TABLE_METADATA_KEYS:
        value = chunk.get(key)
        if value not in (None, ""):
            fused[key] = value
    return fused


def fuse_parser_chunks(
    retrieval_by_parser: dict[str, list[dict[str, Any]]],
    selected_parsers: list[str],
    fusion_method: str,
    final_top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, tuple[int, int]] = {}

    for parser_index, parser_id in enumerate(selected_parsers):
        for chunk in retrieval_by_parser.get(parser_id, []):
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue
            rank = int(chunk.get("rank", 10_000))
            candidate = candidates.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "retrieved_by": [],
                    "ranks_by_parser": {},
                    "rrf_score": 0.0,
                },
            )
            if parser_id not in candidate["retrieved_by"]:
                candidate["retrieved_by"].append(parser_id)
            candidate["ranks_by_parser"][parser_id] = rank
            candidate["rrf_score"] += 1 / (rrf_k + rank)
            first_seen.setdefault(chunk_id, (parser_index, rank))

    if fusion_method == "union_dedup":
        sorted_items = sorted(
            candidates.items(),
            key=lambda item: (
                first_seen[item[0]][0],
                first_seen[item[0]][1],
                item[0],
            ),
        )
    elif fusion_method == "rrf":
        sorted_items = sorted(
            candidates.items(),
            key=lambda item: (
                -float(item[1]["rrf_score"]),
                min(item[1]["ranks_by_parser"].values()),
                item[0],
            ),
        )
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    fused_chunks: list[dict[str, Any]] = []
    for final_rank, (_chunk_id, candidate) in enumerate(sorted_items[:final_top_k], start=1):
        fusion_score = (
            float(len(candidate["retrieved_by"]))
            if fusion_method == "union_dedup"
            else float(candidate["rrf_score"])
        )
        fused_chunks.append(
            fused_parser_chunk_from_candidate(
                chunk=candidate["chunk"],
                final_rank=final_rank,
                fusion_score=fusion_score,
                retrieved_by_parsers=list(candidate["retrieved_by"]),
                ranks_by_parser=dict(candidate["ranks_by_parser"]),
            )
        )
    return fused_chunks


def generate_parser_fusion_answers(
    fused_records: list[dict[str, Any]],
    answer_model: str,
    table_handling: dict[str, Any] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for record in fused_records:
        chunks = record["fused_chunks"]
        intent = answer_intent(str(record["question"]))
        _context_chunks, expansion_records = prepare_answer_context_chunks(
            chunks,
            table_handling,
            all_chunks=all_chunks,
        )
        answers.append(
            {
                "run_id": record["run_id"],
                "experiment_type": record["experiment_type"],
                "selected_parsers": record["selected_parsers"],
                "chunking_strategy": record["chunking_strategy"],
                "embedding_model": record["embedding_model"],
                "fusion_method": record["fusion_method"],
                "answer_model": answer_model,
                "answer_intent": intent,
                "question_id": record["question_id"],
                "question": record["question"],
                "fused_chunks": chunks,
                "table_evidence_used": table_evidence_used(
                    chunks,
                    table_handling,
                    all_chunks=all_chunks,
                ),
                "expanded_context_chunks": expansion_records,
                "generated_answer": generate_answer(
                    question=str(record["question"]),
                    chunks=chunks,
                    answer_model=answer_model,
                    table_handling=table_handling,
                    all_chunks=all_chunks,
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return answers


def write_parser_fusion_comparison_report(
    run_dir: Path,
    selected_parsers: list[str],
    parser_retrieval_records: dict[str, list[dict[str, Any]]],
    fused_records: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "parser_fusion_comparison.csv"
    report_path = reports_dir / "parser_fusion_comparison_report.md"
    answers_by_question = {
        str(record["question_id"]): record for record in answer_records
    }
    retrieval_by_parser = {
        parser_id: {
            str(record["question_id"]): record
            for record in records
        }
        for parser_id, records in parser_retrieval_records.items()
    }

    rows: list[dict[str, Any]] = []
    for record in fused_records:
        question_id = str(record["question_id"])
        fused_chunks = record.get("fused_chunks", [])
        parser_counts = {
            parser_id: len(
                retrieval_by_parser.get(parser_id, {})
                .get(question_id, {})
                .get("retrieved_chunks", [])
            )
            for parser_id in selected_parsers
        }
        answer_record = answers_by_question.get(question_id, {})
        rows.append(
            {
                "question_id": question_id,
                "question": record["question"],
                "fusion_method": record["fusion_method"],
                "parser_retrieved_counts": json.dumps(parser_counts, ensure_ascii=False),
                "fused_chunk_count": len(fused_chunks),
                "parser_sources": json.dumps(
                    {
                        chunk["chunk_id"]: chunk.get("parser_sources", [])
                        for chunk in fused_chunks
                    },
                    ensure_ascii=False,
                ),
                "answer_preview": preview_text(str(answer_record.get("generated_answer", ""))),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "question_id",
            "question",
            "fusion_method",
            "parser_retrieved_counts",
            "fused_chunk_count",
            "parser_sources",
            "answer_preview",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Parser Fusion Comparison Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {len(fused_records)}",
        f"- Parsers compared: {', '.join(selected_parsers)}",
        f"- Fusion method: {fused_records[0]['fusion_method'] if fused_records else ''}",
        "",
        "This report is neutral and does not automatically claim parser fusion is better.",
        "",
        "## Per-Question Results",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"- Parser-only retrieved chunks: `{row['parser_retrieved_counts']}`",
                f"- Fused chunks: {row['fused_chunk_count']}",
                f"- Parser sources by fused chunk: `{row['parser_sources']}`",
                f"- Answer preview: {row['answer_preview']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Review whether fused chunks provide more stable or complete context.",
            "- Compare parser-only retrieved chunks against fused chunks manually.",
            "- Check whether the generated answer is grounded only in fused chunks.",
            "- Do not use this report as an automatic parser fusion recommendation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def write_retrieval_fusion_comparison_report(
    run_dir: Path,
    fused_records: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
    model_warnings: dict[str, list[str]],
) -> tuple[Path, Path]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "retrieval_fusion_comparison.csv"
    report_path = reports_dir / "retrieval_fusion_comparison_report.md"
    answers_by_question = {
        str(record["question_id"]): record for record in answer_records
    }

    rows: list[dict[str, Any]] = []
    for record in fused_records:
        fused_chunks = record.get("fused_chunks", [])
        answer_record = answers_by_question.get(str(record["question_id"]), {})
        rows.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "fusion_method": record["fusion_method"],
                "fused_chunk_count": len(fused_chunks),
                "retrieved_locations": retrieved_locations(fused_chunks),
                "retrieved_by_embedding_models": json.dumps(
                    {
                        chunk["chunk_id"]: chunk.get("retrieved_by_embedding_models", [])
                        for chunk in fused_chunks
                    },
                    ensure_ascii=False,
                ),
                "answer_preview": preview_text(str(answer_record.get("generated_answer", ""))),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "question_id",
            "question",
            "fusion_method",
            "fused_chunk_count",
            "retrieved_locations",
            "retrieved_by_embedding_models",
            "answer_preview",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Retrieval Fusion Comparison Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {len(fused_records)}",
        f"- Fusion method: {fused_records[0]['fusion_method'] if fused_records else ''}",
        "",
        "This report is neutral and does not automatically claim fusion is better.",
        "",
        "## Embedding Model Warnings",
    ]
    for model_name, warnings in model_warnings.items():
        lines.append(f"- {model_name}: {' | '.join(warnings) if warnings else 'none'}")

    lines.extend(["", "## Per-Question Results"])
    for row in rows:
        lines.extend(
            [
                "",
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"- Fused chunks: {row['fused_chunk_count']}",
                f"- Retrieved pages or sections: {row['retrieved_locations'] or 'none'}",
                f"- Retrieved by embedding models: `{row['retrieved_by_embedding_models']}`",
                f"- Answer preview: {row['answer_preview']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Review whether the fused chunks improve evidence coverage for each question.",
            "- Check whether the generated answer is grounded only in fused chunks.",
            "- Treat fusion as a retrieval candidate set, not an automatic quality claim.",
            "- Do not use this report as an automatic recommendation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def run_parser_compare(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config_path = Path(run_config_path)
    run_config = read_yaml(config_path)
    run_dir = config_path.parent
    pdf_path = Path(run_config.get("uploaded_pdf_path") or run_config["pdf"]["saved_path"])
    questions_config = run_config.get("benchmark_questions", {})
    questions_path = Path(
        run_config.get("benchmark_questions_path")
        or questions_config.get("saved_path")
        or questions_config["path"]
    )
    parser_ids = [str(parser_id) for parser_id in run_config["selected_parsers"]]
    unsupported = sorted(set(parser_ids) - SUPPORTED_PARSERS)
    if unsupported:
        raise ValueError(f"Unsupported parser(s) for runner: {unsupported}")

    experiment_type = str(run_config.get("experiment_type", "parser_compare"))
    if experiment_type != "parser_compare":
        raise ValueError(f"run_parser_compare only supports experiment_type=parser_compare: {experiment_type}")

    chunking_strategy = str(run_config.get("chunking_strategy", "fixed-size"))
    if chunking_strategy != "fixed-size":
        raise ValueError(
            "Parser Compare runner currently supports only chunking_strategy='fixed-size'. "
            f"Received: {chunking_strategy}"
        )

    retrieval_strategy = str(run_config.get("retrieval_strategy", "dense_vector"))
    if retrieval_strategy != "dense_vector":
        raise ValueError(
            "Parser Compare runner currently supports only retrieval_strategy='dense_vector'. "
            f"Received: {retrieval_strategy}"
        )

    run_id = str(run_config["run_id"])
    embedding_model_name = str(run_config["embedding_model"])
    answer_model = str(run_config["answer_model"])
    top_k = int(run_config["top_k"])
    chunk_size = int(run_config.get("chunk_size", 800))
    chunk_overlap = int(run_config.get("chunk_overlap", 150))
    table_handling = normalized_table_handling(run_config.get("table_handling"))
    questions = read_benchmark_questions(questions_path)
    chroma_dir = run_dir / "chroma"
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_model = BenchmarkEmbeddingModel(embedding_model_name)
    parser_results: dict[str, dict[str, Any]] = {}

    total_steps = 7
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")

    progress("Preparing run folder")
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("Extracting documents")
    chunks_by_parser: dict[str, list[dict[str, Any]]] = {}
    for parser_id in parser_ids:
        parser_dir = run_dir / parser_id
        parser_dir.mkdir(parents=True, exist_ok=True)
        chunks_by_parser[parser_id] = extract_and_chunk(
            parser_id,
            pdf_path,
            parser_dir,
            chunk_size,
            chunk_overlap,
        )

    progress("Chunking")
    for parser_id, chunks in chunks_by_parser.items():
        write_jsonl(chunks, run_dir / parser_id / "chunks.jsonl")

    progress("Building vector DB")
    collections_by_parser: dict[str, Any] = {}
    collection_names_by_parser: dict[str, str] = {}
    for parser_id, chunks in chunks_by_parser.items():
        collection_name = make_collection_name(
            parser_id=parser_id,
            embedding_model_name=embedding_model_name,
            run_id=run_id,
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "parser": parser_id,
                "embedding_model": embedding_model_name,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": retrieval_strategy,
            },
        )
        index_chunks(chunks, collection, embedding_model, parser_id)
        collections_by_parser[parser_id] = collection
        collection_names_by_parser[parser_id] = collection_name

    progress("Running retrieval")
    for parser_id, collection in collections_by_parser.items():
        parser_dir = run_dir / parser_id
        if not chunks_by_parser[parser_id]:
            retrieval_records = empty_retrieval_records(
                questions,
                run_id,
                experiment_type,
                parser_id,
                chunking_strategy,
                embedding_model_name,
                top_k,
            )
        else:
            retrieval_records = retrieve_questions(
                questions=questions,
                collection=collection,
                embedding_model=embedding_model,
                run_id=run_id,
                experiment_type=experiment_type,
                parser_id=parser_id,
                chunking_strategy=chunking_strategy,
                embedding_model_name=embedding_model_name,
                top_k=top_k,
            )
        write_jsonl(retrieval_records, parser_dir / "retrieval_results.jsonl")
        parser_results[parser_id] = {
            "collection_name": collection_names_by_parser[parser_id],
            "chunk_count": len(chunks_by_parser[parser_id]),
            "retrieval_records": retrieval_records,
            "answer_records": [],
        }

    progress("Generating answers")
    for parser_id, result in parser_results.items():
        parser_dir = run_dir / parser_id
        answer_records = generate_answers(
            result["retrieval_records"],
            answer_model,
            table_handling=table_handling,
            all_chunks=chunks_by_parser.get(parser_id, []),
        )
        write_jsonl(answer_records, parser_dir / "answer_results.jsonl")
        result["answer_records"] = answer_records

    progress("Saving report")
    csv_path, report_path = write_comparison_report(run_dir, questions, parser_results)

    return {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "run_dir": str(run_dir),
        "chroma_dir": str(chroma_dir),
        "parsers": parser_ids,
        "reports": {
            "parser_comparison_csv": str(csv_path),
            "parser_comparison_report": str(report_path),
        },
        "parser_results": {
            parser_id: {
                "collection_name": result["collection_name"],
                "chunk_count": result["chunk_count"],
                "chunks": str(run_dir / parser_id / "chunks.jsonl"),
                "retrieval_results": str(run_dir / parser_id / "retrieval_results.jsonl"),
                "answer_results": str(run_dir / parser_id / "answer_results.jsonl"),
            }
            for parser_id, result in parser_results.items()
        },
    }


def run_parser_comparison(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    return run_parser_compare(run_config_path, progress_callback=progress_callback)


def run_chunking_compare(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config_path = Path(run_config_path)
    run_config = read_yaml(config_path)
    run_dir = config_path.parent
    pdf_path = Path(run_config.get("uploaded_pdf_path") or run_config["pdf"]["saved_path"])
    questions_config = run_config.get("benchmark_questions", {})
    questions_path = Path(
        run_config.get("benchmark_questions_path")
        or questions_config.get("saved_path")
        or questions_config["path"]
    )

    experiment_type = str(run_config.get("experiment_type", "chunking_compare"))
    if experiment_type != "chunking_compare":
        raise ValueError(f"run_chunking_compare only supports experiment_type=chunking_compare: {experiment_type}")

    parser_id = str(run_config["parser"])
    if parser_id not in SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported parser for runner: {parser_id}")

    selected_strategies = [
        str(strategy)
        for strategy in run_config.get("selected_chunking_strategies", [])
    ]
    unsupported = sorted(set(selected_strategies) - SUPPORTED_CHUNKING_STRATEGIES)
    if unsupported:
        raise ValueError(f"Unsupported chunking strategy/strategies: {unsupported}")
    if len(selected_strategies) < 2:
        raise ValueError("Chunking Compare requires at least two selected chunking strategies.")

    retrieval_strategy = str(run_config.get("retrieval_strategy", "dense_vector"))
    if retrieval_strategy != "dense_vector":
        raise ValueError(
            "Chunking Compare runner currently supports only retrieval_strategy='dense_vector'. "
            f"Received: {retrieval_strategy}"
        )

    run_id = str(run_config["run_id"])
    embedding_model_name = str(run_config["embedding_model"])
    answer_model = str(run_config["answer_model"])
    top_k = int(run_config["top_k"])
    chunk_size = int(run_config.get("chunk_size", 800))
    chunk_overlap = int(run_config.get("chunk_overlap", 150))
    table_handling = normalized_table_handling(run_config.get("table_handling"))
    questions = read_benchmark_questions(questions_path)
    chroma_dir = run_dir / "chroma"

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_model = BenchmarkEmbeddingModel(embedding_model_name)
    strategy_results: dict[str, dict[str, Any]] = {}

    total_steps = 7
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")

    progress("Preparing run")
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("Extracting document")
    extracted_records, extraction_warnings = extract_document(parser_id, pdf_path, run_dir / "extraction")

    progress("Running chunking strategies")
    chunks_by_strategy: dict[str, list[dict[str, Any]]] = {}
    warnings_by_strategy: dict[str, list[str]] = {}
    for strategy in selected_strategies:
        strategy_dir = run_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        chunks, warnings = create_chunks_for_strategy(
            parser_id,
            extracted_records,
            strategy,
            chunk_size,
            chunk_overlap,
        )
        warnings = extraction_warnings + warnings
        chunks_by_strategy[strategy] = chunks
        warnings_by_strategy[strategy] = warnings
        write_jsonl(chunks, strategy_dir / "chunks.jsonl")

    progress("Building vector DBs")
    collections_by_strategy: dict[str, Any] = {}
    collection_names_by_strategy: dict[str, str] = {}
    for strategy, chunks in chunks_by_strategy.items():
        collection_name = make_strategy_collection_name(
            run_id=run_id,
            parser_id=parser_id,
            chunking_strategy=strategy,
            embedding_model_name=embedding_model_name,
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "parser": parser_id,
                "embedding_model": embedding_model_name,
                "chunking_strategy": strategy,
                "retrieval_strategy": retrieval_strategy,
            },
        )
        if chunks:
            index_chunks(chunks, collection, embedding_model, parser_id)
        collections_by_strategy[strategy] = collection
        collection_names_by_strategy[strategy] = collection_name

    progress("Running retrieval")
    for strategy, collection in collections_by_strategy.items():
        strategy_dir = run_dir / strategy
        if not chunks_by_strategy[strategy]:
            retrieval_records = empty_retrieval_records(
                questions,
                run_id,
                experiment_type,
                parser_id,
                strategy,
                embedding_model_name,
                top_k,
            )
        else:
            retrieval_records = retrieve_questions(
                questions=questions,
                collection=collection,
                embedding_model=embedding_model,
                run_id=run_id,
                experiment_type=experiment_type,
                parser_id=parser_id,
                chunking_strategy=strategy,
                embedding_model_name=embedding_model_name,
                top_k=top_k,
            )
        write_jsonl(retrieval_records, strategy_dir / "retrieval_results.jsonl")
        chunks = chunks_by_strategy[strategy]
        strategy_results[strategy] = {
            "collection_name": collection_names_by_strategy[strategy],
            "chunk_count": len(chunks),
            "average_chunk_length": average_chunk_length(chunks),
            "warnings": warnings_by_strategy[strategy],
            "retrieval_records": retrieval_records,
            "answer_records": [],
        }

    progress("Generating answers")
    for strategy, result in strategy_results.items():
        strategy_dir = run_dir / strategy
        answer_records = generate_answers(
            result["retrieval_records"],
            answer_model,
            table_handling=table_handling,
            all_chunks=chunks_by_strategy.get(strategy, []),
        )
        write_jsonl(answer_records, strategy_dir / "answer_results.jsonl")
        result["answer_records"] = answer_records

    progress("Saving comparison report")
    csv_path, report_path = write_chunking_comparison_report(run_dir, questions, strategy_results)

    return {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "run_dir": str(run_dir),
        "chroma_dir": str(chroma_dir),
        "parser": parser_id,
        "chunking_strategies": selected_strategies,
        "reports": {
            "chunking_comparison_csv": str(csv_path),
            "chunking_comparison_report": str(report_path),
        },
        "strategy_results": {
            strategy: {
                "collection_name": result["collection_name"],
                "chunk_count": result["chunk_count"],
                "average_chunk_length": result["average_chunk_length"],
                "warnings": result["warnings"],
                "chunks": str(run_dir / strategy / "chunks.jsonl"),
                "retrieval_results": str(run_dir / strategy / "retrieval_results.jsonl"),
                "answer_results": str(run_dir / strategy / "answer_results.jsonl"),
            }
            for strategy, result in strategy_results.items()
        },
    }


def run_embedding_compare(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config_path = Path(run_config_path)
    run_config = read_yaml(config_path)
    run_dir = config_path.parent
    pdf_path = Path(run_config.get("uploaded_pdf_path") or run_config["pdf"]["saved_path"])
    questions_config = run_config.get("benchmark_questions", {})
    questions_path = Path(
        run_config.get("benchmark_questions_path")
        or questions_config.get("saved_path")
        or questions_config["path"]
    )

    experiment_type = str(run_config.get("experiment_type", "embedding_compare"))
    if experiment_type != "embedding_compare":
        raise ValueError(f"run_embedding_compare only supports experiment_type=embedding_compare: {experiment_type}")

    parser_id = str(run_config["parser"])
    if parser_id not in SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported parser for runner: {parser_id}")

    chunking_strategy = str(run_config["chunking_strategy"])
    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    selected_models = [
        str(model_name)
        for model_name in run_config.get("selected_embedding_models", [])
    ]
    if len(selected_models) < 2:
        raise ValueError("Embedding Compare requires at least two selected embedding models.")

    retrieval_strategy = str(run_config.get("retrieval_strategy", "dense_vector"))
    if retrieval_strategy != "dense_vector":
        raise ValueError(
            "Embedding Compare runner currently supports only retrieval_strategy='dense_vector'. "
            f"Received: {retrieval_strategy}"
        )

    run_id = str(run_config["run_id"])
    answer_model = str(run_config["answer_model"])
    top_k = int(run_config["top_k"])
    chunk_size = int(run_config.get("chunk_size", 800))
    chunk_overlap = int(run_config.get("chunk_overlap", 150))
    table_handling = normalized_table_handling(run_config.get("table_handling"))
    questions = read_benchmark_questions(questions_path)
    chroma_dir = run_dir / "chroma"

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    model_results: dict[str, dict[str, Any]] = {}

    total_steps = 8
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")

    progress("Preparing run")
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("Extracting document")
    extracted_records, extraction_warnings = extract_document(parser_id, pdf_path, run_dir / "extraction")

    progress("Chunking document")
    chunks, chunk_warnings = create_chunks_for_strategy(
        parser_id,
        extracted_records,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
    )
    chunk_warnings = extraction_warnings + chunk_warnings
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, chunks_dir / "chunks.jsonl")

    progress("Running embedding models")
    embedding_models: dict[str, BenchmarkEmbeddingModel | None] = {}
    runtime_by_model: dict[str, float] = {}
    warnings_by_model: dict[str, list[str]] = {}
    for model_name in selected_models:
        start = perf_counter()
        try:
            embedding_models[model_name] = BenchmarkEmbeddingModel(model_name)
            warnings_by_model[model_name] = list(chunk_warnings)
        except Exception as exc:
            embedding_models[model_name] = None
            warnings_by_model[model_name] = list(chunk_warnings) + [
                f"Embedding model failed to load: {exc}"
            ]
        runtime_by_model[model_name] = perf_counter() - start

    progress("Building vector DBs")
    collections_by_model: dict[str, Any] = {}
    collection_names_by_model: dict[str, str] = {}
    for model_name, embedding_model in embedding_models.items():
        safe_model_name = safe_embedding_model_name(model_name)
        model_dir = run_dir / "embeddings" / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        collection_name = make_strategy_collection_name(
            run_id=run_id,
            parser_id=parser_id,
            chunking_strategy=chunking_strategy,
            embedding_model_name=model_name,
        )
        collection_names_by_model[model_name] = collection_name
        if embedding_model is None:
            continue

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "parser": parser_id,
                "embedding_model": model_name,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": retrieval_strategy,
            },
        )
        start = perf_counter()
        try:
            if chunks:
                index_chunks(chunks, collection, embedding_model, parser_id)
            collections_by_model[model_name] = collection
        except Exception as exc:
            warnings_by_model[model_name].append(f"Embedding/indexing failed: {exc}")
            embedding_models[model_name] = None
        runtime_by_model[model_name] += perf_counter() - start

    progress("Running retrieval")
    for model_name in selected_models:
        safe_model_name = safe_embedding_model_name(model_name)
        model_dir = run_dir / "embeddings" / safe_model_name
        embedding_model = embedding_models.get(model_name)
        collection = collections_by_model.get(model_name)
        if embedding_model is None or collection is None or not chunks:
            retrieval_records = empty_retrieval_records(
                questions,
                run_id,
                experiment_type,
                parser_id,
                chunking_strategy,
                model_name,
                top_k,
            )
        else:
            retrieval_records = retrieve_questions(
                questions=questions,
                collection=collection,
                embedding_model=embedding_model,
                run_id=run_id,
                experiment_type=experiment_type,
                parser_id=parser_id,
                chunking_strategy=chunking_strategy,
                embedding_model_name=model_name,
                top_k=top_k,
            )
        write_jsonl(retrieval_records, model_dir / "retrieval_results.jsonl")
        model_results[model_name] = {
            "safe_model_name": safe_model_name,
            "collection_name": collection_names_by_model.get(model_name, ""),
            "embedding_runtime_seconds": runtime_by_model.get(model_name, 0.0),
            "warnings": warnings_by_model.get(model_name, []),
            "retrieval_records": retrieval_records,
            "answer_records": [],
        }

    progress("Generating answers")
    for model_name, result in model_results.items():
        model_dir = run_dir / "embeddings" / str(result["safe_model_name"])
        answer_records = generate_answers(
            result["retrieval_records"],
            answer_model,
            table_handling=table_handling,
            all_chunks=chunks,
        )
        write_jsonl(answer_records, model_dir / "answer_results.jsonl")
        result["answer_records"] = answer_records

    progress("Saving comparison report")
    csv_path, report_path = write_embedding_comparison_report(run_dir, questions, model_results)

    return {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "run_dir": str(run_dir),
        "chroma_dir": str(chroma_dir),
        "parser": parser_id,
        "chunking_strategy": chunking_strategy,
        "embedding_models": selected_models,
        "chunks": str(chunks_dir / "chunks.jsonl"),
        "reports": {
            "embedding_comparison_csv": str(csv_path),
            "embedding_comparison_report": str(report_path),
        },
        "model_results": {
            model_name: {
                "safe_model_name": result["safe_model_name"],
                "collection_name": result["collection_name"],
                "embedding_runtime_seconds": result["embedding_runtime_seconds"],
                "warnings": result["warnings"],
                "retrieval_results": str(
                    run_dir / "embeddings" / str(result["safe_model_name"]) / "retrieval_results.jsonl"
                ),
                "answer_results": str(
                    run_dir / "embeddings" / str(result["safe_model_name"]) / "answer_results.jsonl"
                ),
            }
            for model_name, result in model_results.items()
        },
    }


def run_retrieval_fusion_compare(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config_path = Path(run_config_path)
    run_config = read_yaml(config_path)
    run_dir = config_path.parent
    pdf_path = Path(run_config.get("uploaded_pdf_path") or run_config["pdf"]["saved_path"])
    questions_config = run_config.get("benchmark_questions", {})
    questions_path = Path(
        run_config.get("benchmark_questions_path")
        or questions_config.get("saved_path")
        or questions_config["path"]
    )

    experiment_type = str(run_config.get("experiment_type", "retrieval_fusion_compare"))
    if experiment_type != "retrieval_fusion_compare":
        raise ValueError(
            "run_retrieval_fusion_compare only supports "
            f"experiment_type=retrieval_fusion_compare: {experiment_type}"
        )

    parser_id = str(run_config["parser"])
    if parser_id not in SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported parser for runner: {parser_id}")

    chunking_strategy = str(run_config["chunking_strategy"])
    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    selected_models = [
        str(model_name)
        for model_name in run_config.get("selected_embedding_models", [])
    ]
    if len(selected_models) < 2:
        raise ValueError("Retrieval Fusion Compare requires at least two selected embedding models.")

    fusion_method = str(run_config["fusion_method"])
    if fusion_method not in {"union_dedup", "rrf"}:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    retrieval_strategy = str(run_config.get("retrieval_strategy", "dense_vector"))
    if retrieval_strategy != "dense_vector":
        raise ValueError(
            "Retrieval Fusion Compare currently supports only retrieval_strategy='dense_vector'. "
            f"Received: {retrieval_strategy}"
        )

    run_id = str(run_config["run_id"])
    answer_model = str(run_config["answer_model"])
    per_model_top_k = int(run_config["per_model_top_k"])
    final_top_k = int(run_config["final_top_k"])
    rrf_k = int(run_config.get("rrf_k", 60))
    chunk_size = int(run_config.get("chunk_size", 800))
    chunk_overlap = int(run_config.get("chunk_overlap", 150))
    table_handling = normalized_table_handling(run_config.get("table_handling"))
    questions = read_benchmark_questions(questions_path)
    chroma_dir = run_dir / "chroma"

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    model_retrieval_records: dict[str, list[dict[str, Any]]] = {}
    model_warnings: dict[str, list[str]] = {}

    total_steps = 9
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")

    progress("Preparing run")
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("Extracting document")
    extracted_records, extraction_warnings = extract_document(parser_id, pdf_path, run_dir / "extraction")

    progress("Chunking document")
    chunks, chunk_warnings = create_chunks_for_strategy(
        parser_id,
        extracted_records,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
    )
    shared_warnings = extraction_warnings + chunk_warnings
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, chunks_dir / "chunks.jsonl")

    progress("Running embedding models")
    embedding_models: dict[str, BenchmarkEmbeddingModel | None] = {}
    for model_name in selected_models:
        try:
            embedding_models[model_name] = BenchmarkEmbeddingModel(model_name)
            model_warnings[model_name] = list(shared_warnings)
        except Exception as exc:
            embedding_models[model_name] = None
            model_warnings[model_name] = list(shared_warnings) + [
                f"Embedding model failed to load: {exc}"
            ]

    progress("Building vector DBs")
    collections_by_model: dict[str, Any] = {}
    for model_name, embedding_model in embedding_models.items():
        safe_model_name = safe_embedding_model_name(model_name)
        model_dir = run_dir / "embeddings" / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        if embedding_model is None:
            continue
        collection_name = make_strategy_collection_name(
            run_id=run_id,
            parser_id=parser_id,
            chunking_strategy=chunking_strategy,
            embedding_model_name=model_name,
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "parser": parser_id,
                "embedding_model": model_name,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": retrieval_strategy,
                "fusion_method": fusion_method,
            },
        )
        try:
            if chunks:
                index_chunks(chunks, collection, embedding_model, parser_id)
            collections_by_model[model_name] = collection
        except Exception as exc:
            model_warnings[model_name].append(f"Embedding/indexing failed: {exc}")
            embedding_models[model_name] = None

    progress("Running per-model retrieval")
    for model_name in selected_models:
        safe_model_name = safe_embedding_model_name(model_name)
        model_dir = run_dir / "embeddings" / safe_model_name
        embedding_model = embedding_models.get(model_name)
        collection = collections_by_model.get(model_name)
        if embedding_model is None or collection is None or not chunks:
            retrieval_records = empty_retrieval_records(
                questions,
                run_id,
                experiment_type,
                parser_id,
                chunking_strategy,
                model_name,
                per_model_top_k,
            )
        else:
            retrieval_records = retrieve_questions(
                questions=questions,
                collection=collection,
                embedding_model=embedding_model,
                run_id=run_id,
                experiment_type=experiment_type,
                parser_id=parser_id,
                chunking_strategy=chunking_strategy,
                embedding_model_name=model_name,
                top_k=per_model_top_k,
            )
        write_jsonl(retrieval_records, model_dir / "retrieval_results.jsonl")
        model_retrieval_records[model_name] = retrieval_records

    progress("Fusing retrieval results")
    retrieval_by_question_and_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, records in model_retrieval_records.items():
        for record in records:
            question_id = str(record["question_id"])
            retrieval_by_question_and_model.setdefault(question_id, {})[model_name] = record.get(
                "retrieved_chunks",
                [],
            )

    fused_records: list[dict[str, Any]] = []
    for question in questions:
        fused_chunks = fuse_retrieved_chunks(
            retrieval_by_model=retrieval_by_question_and_model.get(question["id"], {}),
            selected_models=selected_models,
            fusion_method=fusion_method,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )
        fused_records.append(
            {
                "run_id": run_id,
                "experiment_type": experiment_type,
                "parser": parser_id,
                "chunking_strategy": chunking_strategy,
                "selected_embedding_models": selected_models,
                "fusion_method": fusion_method,
                "question_id": question["id"],
                "question": question["question"],
                "per_model_top_k": per_model_top_k,
                "final_top_k": final_top_k,
                "fused_chunks": fused_chunks,
            }
        )

    fusion_dir = run_dir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(fused_records, fusion_dir / "fused_retrieval_results.jsonl")

    progress("Generating answers")
    answer_records = generate_fusion_answers(
        fused_records,
        answer_model,
        table_handling=table_handling,
        all_chunks=chunks,
    )
    write_jsonl(answer_records, fusion_dir / "answer_results.jsonl")

    progress("Saving comparison report")
    csv_path, report_path = write_retrieval_fusion_comparison_report(
        run_dir,
        fused_records,
        answer_records,
        model_warnings,
    )

    return {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "run_dir": str(run_dir),
        "chroma_dir": str(chroma_dir),
        "parser": parser_id,
        "chunking_strategy": chunking_strategy,
        "embedding_models": selected_models,
        "fusion_method": fusion_method,
        "chunks": str(chunks_dir / "chunks.jsonl"),
        "reports": {
            "retrieval_fusion_comparison_csv": str(csv_path),
            "retrieval_fusion_comparison_report": str(report_path),
        },
        "model_warnings": model_warnings,
        "per_model_results": {
            model_name: {
                "safe_model_name": safe_embedding_model_name(model_name),
                "retrieval_results": str(
                    run_dir / "embeddings" / safe_embedding_model_name(model_name) / "retrieval_results.jsonl"
                ),
            }
            for model_name in selected_models
        },
        "fusion_results": {
            "fused_retrieval_results": str(fusion_dir / "fused_retrieval_results.jsonl"),
            "answer_results": str(fusion_dir / "answer_results.jsonl"),
        },
    }


def run_parser_fusion_compare(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config_path = Path(run_config_path)
    run_config = read_yaml(config_path)
    run_dir = config_path.parent
    pdf_path = Path(run_config.get("uploaded_pdf_path") or run_config["pdf"]["saved_path"])
    questions_config = run_config.get("benchmark_questions", {})
    questions_path = Path(
        run_config.get("benchmark_questions_path")
        or questions_config.get("saved_path")
        or questions_config["path"]
    )

    experiment_type = str(run_config.get("experiment_type", "parser_fusion_compare"))
    if experiment_type != "parser_fusion_compare":
        raise ValueError(
            "run_parser_fusion_compare only supports "
            f"experiment_type=parser_fusion_compare: {experiment_type}"
        )

    selected_parsers = [str(parser_id) for parser_id in run_config.get("selected_parsers", [])]
    unsupported = sorted(set(selected_parsers) - SUPPORTED_PARSERS)
    if unsupported:
        raise ValueError(f"Unsupported parser(s) for runner: {unsupported}")
    if len(selected_parsers) < 2:
        raise ValueError("Parser Fusion Compare requires at least two selected parsers.")

    chunking_strategy = str(run_config["chunking_strategy"])
    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

    fusion_method = str(run_config["fusion_method"])
    if fusion_method not in {"union_dedup", "rrf"}:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    run_id = str(run_config["run_id"])
    embedding_model_name = str(run_config["embedding_model"])
    answer_model = str(run_config["answer_model"])
    per_parser_top_k = int(run_config["per_parser_top_k"])
    final_top_k = int(run_config["final_top_k"])
    rrf_k = int(run_config.get("rrf_k", 60))
    chunk_size = int(run_config.get("chunk_size", 800))
    chunk_overlap = int(run_config.get("chunk_overlap", 150))
    table_handling = normalized_table_handling(run_config.get("table_handling"))
    questions = read_benchmark_questions(questions_path)
    chroma_dir = run_dir / "chroma"

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_model = BenchmarkEmbeddingModel(embedding_model_name)
    parser_retrieval_records: dict[str, list[dict[str, Any]]] = {}

    total_steps = 9
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")

    progress("Preparing run")
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("Extracting documents")
    extracted_by_parser: dict[str, list[dict[str, Any]]] = {}
    extraction_warnings_by_parser: dict[str, list[str]] = {}
    for parser_id in selected_parsers:
        parser_dir = run_dir / parser_id
        extracted, warnings = extract_document(parser_id, pdf_path, parser_dir)
        extracted_by_parser[parser_id] = extracted
        extraction_warnings_by_parser[parser_id] = warnings

    progress("Chunking parser outputs")
    chunks_by_parser: dict[str, list[dict[str, Any]]] = {}
    warnings_by_parser: dict[str, list[str]] = {}
    for parser_id in selected_parsers:
        chunks, warnings = create_chunks_for_strategy(
            parser_id,
            extracted_by_parser[parser_id],
            chunking_strategy,
            chunk_size,
            chunk_overlap,
        )
        for chunk in chunks:
            chunk["chunk_id"] = f"{parser_id}:{chunk['chunk_id']}"
            chunk["parser"] = parser_id
        warnings_by_parser[parser_id] = extraction_warnings_by_parser[parser_id] + warnings
        chunks_by_parser[parser_id] = chunks
        write_jsonl(chunks, run_dir / parser_id / "chunks.jsonl")

    progress("Building parser vector DBs")
    collections_by_parser: dict[str, Any] = {}
    collection_names_by_parser: dict[str, str] = {}
    for parser_id, chunks in chunks_by_parser.items():
        collection_name = make_strategy_collection_name(
            run_id=run_id,
            parser_id=parser_id,
            chunking_strategy=chunking_strategy,
            embedding_model_name=embedding_model_name,
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "parser": parser_id,
                "embedding_model": embedding_model_name,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": "parser_fusion",
                "fusion_method": fusion_method,
            },
        )
        if chunks:
            index_chunks(chunks, collection, embedding_model, parser_id)
        collections_by_parser[parser_id] = collection
        collection_names_by_parser[parser_id] = collection_name

    progress("Running parser retrieval")
    for parser_id, collection in collections_by_parser.items():
        if not chunks_by_parser[parser_id]:
            retrieval_records = empty_retrieval_records(
                questions,
                run_id,
                experiment_type,
                parser_id,
                chunking_strategy,
                embedding_model_name,
                per_parser_top_k,
            )
        else:
            retrieval_records = retrieve_questions(
                questions=questions,
                collection=collection,
                embedding_model=embedding_model,
                run_id=run_id,
                experiment_type=experiment_type,
                parser_id=parser_id,
                chunking_strategy=chunking_strategy,
                embedding_model_name=embedding_model_name,
                top_k=per_parser_top_k,
            )
        write_jsonl(retrieval_records, run_dir / parser_id / "retrieval_results.jsonl")
        parser_retrieval_records[parser_id] = retrieval_records

    progress("Fusing parser retrieval results")
    retrieval_by_question_and_parser: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for parser_id, records in parser_retrieval_records.items():
        for record in records:
            question_id = str(record["question_id"])
            retrieval_by_question_and_parser.setdefault(question_id, {})[parser_id] = record.get(
                "retrieved_chunks",
                [],
            )

    fused_records: list[dict[str, Any]] = []
    for question in questions:
        fused_chunks = fuse_parser_chunks(
            retrieval_by_parser=retrieval_by_question_and_parser.get(question["id"], {}),
            selected_parsers=selected_parsers,
            fusion_method=fusion_method,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )
        fused_records.append(
            {
                "run_id": run_id,
                "experiment_type": experiment_type,
                "selected_parsers": selected_parsers,
                "chunking_strategy": chunking_strategy,
                "embedding_model": embedding_model_name,
                "fusion_method": fusion_method,
                "question_id": question["id"],
                "question": question["question"],
                "per_parser_top_k": per_parser_top_k,
                "final_top_k": final_top_k,
                "fused_chunks": fused_chunks,
            }
        )

    fusion_dir = run_dir / "parser_fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(fused_records, fusion_dir / "fused_retrieval_results.jsonl")

    progress("Generating answers")
    answer_records = generate_parser_fusion_answers(
        fused_records,
        answer_model,
        table_handling=table_handling,
        all_chunks=[chunk for parser_chunks in chunks_by_parser.values() for chunk in parser_chunks],
    )
    write_jsonl(answer_records, fusion_dir / "answer_results.jsonl")

    progress("Saving comparison report")
    csv_path, report_path = write_parser_fusion_comparison_report(
        run_dir,
        selected_parsers,
        parser_retrieval_records,
        fused_records,
        answer_records,
    )

    return {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "run_dir": str(run_dir),
        "chroma_dir": str(chroma_dir),
        "selected_parsers": selected_parsers,
        "chunking_strategy": chunking_strategy,
        "embedding_model": embedding_model_name,
        "fusion_method": fusion_method,
        "parser_warnings": warnings_by_parser,
        "reports": {
            "parser_fusion_comparison_csv": str(csv_path),
            "parser_fusion_comparison_report": str(report_path),
        },
        "parser_results": {
            parser_id: {
                "collection_name": collection_names_by_parser[parser_id],
                "chunk_count": len(chunks_by_parser[parser_id]),
                "warnings": warnings_by_parser[parser_id],
                "chunks": str(run_dir / parser_id / "chunks.jsonl"),
                "retrieval_results": str(run_dir / parser_id / "retrieval_results.jsonl"),
            }
            for parser_id in selected_parsers
        },
        "fusion_results": {
            "fused_retrieval_results": str(fusion_dir / "fused_retrieval_results.jsonl"),
            "answer_results": str(fusion_dir / "answer_results.jsonl"),
        },
    }


def run_benchmark_from_config(
    run_config_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = read_yaml(Path(run_config_path))
    experiment_type = str(config.get("experiment_type", "parser_compare"))
    if experiment_type == "parser_compare":
        return run_parser_compare(run_config_path, progress_callback=progress_callback)
    if experiment_type == "chunking_compare":
        return run_chunking_compare(run_config_path, progress_callback=progress_callback)
    if experiment_type == "embedding_compare":
        return run_embedding_compare(run_config_path, progress_callback=progress_callback)
    if experiment_type == "retrieval_fusion_compare":
        return run_retrieval_fusion_compare(run_config_path, progress_callback=progress_callback)
    if experiment_type == "parser_fusion_compare":
        return run_parser_fusion_compare(run_config_path, progress_callback=progress_callback)
    raise ValueError(f"Unsupported experiment_type for benchmark runner: {experiment_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    parser.add_argument(
        "--run-config",
        required=True,
        type=Path,
        help="Path to outputs/runs/<run_id>/run_config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_benchmark_from_config(args.run_config)
    print(json.dumps(result, indent=2))
