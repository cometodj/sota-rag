from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


BATCH_SIZE = 128
SUPPORTED_PARSERS = {"pymupdf", "docling"}
SUPPORTED_CHUNKING_STRATEGIES = {"fixed-size", "page-based", "section-aware"}
ProgressCallback = Callable[[str, int, int], None]


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
    }

    if "page_number" in chunk:
        metadata["page_number"] = int(chunk["page_number"])
    if chunk.get("section_title"):
        metadata["section_title"] = str(chunk["section_title"])
    if chunk.get("source"):
        metadata["source"] = str(chunk["source"])

    return metadata


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
    parts = [
        f"chunk_id={chunk.get('chunk_id', '')}",
        f"rank={chunk.get('rank', '')}",
    ]
    if chunk.get("page_number") not in (None, ""):
        parts.append(f"page_number={chunk.get('page_number')}")
    if chunk.get("section_title"):
        parts.append(f"section_title={chunk.get('section_title')}")
    return ", ".join(parts)


def build_answer_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    context = "\n\n---\n\n".join(
        f"[{evidence_label(chunk)}]\n{chunk.get('text', '')}" for chunk in chunks
    )
    return f"""You answer technical-document questions using only retrieved chunks.

Rules:
- Use only the retrieved context below.
- Do not use outside knowledge.
- Do not hallucinate fields, values, sections, or requirements.
- If the retrieved chunks are insufficient, say so clearly.
- Keep the answer concise and technical.

Question:
{question}

Retrieved context:
{context}

Return exactly these sections:
## Answer Summary
## Evidence Used
## Missing or Uncertain Information
"""


def generate_answer(question: str, chunks: list[dict[str, Any]], answer_model: str) -> str:
    if not chunks:
        return (
            "## Answer Summary\n"
            "The retrieved context is insufficient because no chunks were retrieved.\n\n"
            "## Evidence Used\n"
            "- None\n\n"
            "## Missing or Uncertain Information\n"
            "- No retrieved chunks were available.\n"
        )

    import ollama

    response = ollama.generate(
        model=answer_model,
        prompt=build_answer_prompt(question, chunks),
        options={"temperature": 0.1},
    )
    return str(response["response"]).strip()


def generate_answers(
    retrieval_records: list[dict[str, Any]],
    answer_model: str,
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for record in retrieval_records:
        answers.append(
            {
                "run_id": record["run_id"],
                "experiment_type": record["experiment_type"],
                "parser": record["parser"],
                "chunking_strategy": record["chunking_strategy"],
                "embedding_model": record["embedding_model"],
                "answer_model": answer_model,
                "question_id": record["question_id"],
                "question": record["question"],
                "retrieved_chunks": record["retrieved_chunks"],
                "generated_answer": generate_answer(
                    question=str(record["question"]),
                    chunks=record["retrieved_chunks"],
                    answer_model=answer_model,
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

    if chunking_strategy == "fixed-size":
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
        return chunks, warnings

    if chunking_strategy == "page-based":
        return create_page_based_chunks(extracted_records, chunk_size, chunk_overlap)

    if chunking_strategy == "section-aware":
        return create_section_aware_chunks(parser_id, extracted_records, chunk_size, chunk_overlap)

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
        answer_records = generate_answers(result["retrieval_records"], answer_model)
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
        answer_records = generate_answers(result["retrieval_records"], answer_model)
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
