from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st
import yaml


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


APP_TITLE = "SOTA RAG - Benchmark Setup"
CONFIG_PATH = Path("configs/config.yaml")
BENCHMARK_QUESTIONS_PATH = Path("benchmark/benchmark_questions.jsonl")
OUTPUT_DIR = Path("outputs")
ORIGINAL_RESULTS_PATH = OUTPUT_DIR / "original_retrieval_results.jsonl"
QUERY_EXPANSIONS_PATH = OUTPUT_DIR / "query_expansions.jsonl"
EXPANDED_RESULTS_PATH = OUTPUT_DIR / "expanded_retrieval_results.jsonl"
DOCLING_ORIGINAL_RESULTS_PATH = OUTPUT_DIR / "original_retrieval_results_docling.jsonl"
DOCLING_EXPANDED_RESULTS_PATH = OUTPUT_DIR / "expanded_retrieval_results_docling.jsonl"
COMPARISON_CSV_PATH = OUTPUT_DIR / "retrieval_comparison.csv"
COMPARISON_REPORT_PATH = OUTPUT_DIR / "retrieval_comparison_report.md"
PARSER_COMPARISON_CSV_PATH = OUTPUT_DIR / "parser_comparison.csv"
PARSER_COMPARISON_REPORT_PATH = OUTPUT_DIR / "parser_comparison_report.md"
EXPECTED_FILES = [
    BENCHMARK_QUESTIONS_PATH,
    ORIGINAL_RESULTS_PATH,
    QUERY_EXPANSIONS_PATH,
    EXPANDED_RESULTS_PATH,
    COMPARISON_CSV_PATH,
    COMPARISON_REPORT_PATH,
    DOCLING_ORIGINAL_RESULTS_PATH,
    DOCLING_EXPANDED_RESULTS_PATH,
    PARSER_COMPARISON_CSV_PATH,
    PARSER_COMPARISON_REPORT_PATH,
]
PREVIEW_LIMIT = 420
RUNS_DIR = OUTPUT_DIR / "runs"
EMBEDDING_MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-m3",
]
ANSWER_MODEL_OPTIONS = [
    "qwen2.5:14b",
    "llama3.1:8b",
    "gemma3:12b",
]
PARSER_OPTIONS = ["PyMuPDF", "Docling"]
PARSER_ID_BY_LABEL = {
    "PyMuPDF": "pymupdf",
    "Docling": "docling",
}
CHUNKING_STRATEGY_OPTIONS = ["fixed-size", "page-based", "section-aware"]
DEFAULT_RETRIEVAL_STRATEGY = "dense_vector"


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def preview_text(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def is_ollama_embedding_model(model_name: str) -> bool:
    normalized = model_name.casefold()
    embedding_markers = [
        "embed",
        "embedding",
        "bge",
        "minilm",
        "mxbai",
        "nomic",
        "snowflake-arctic",
    ]
    return any(marker in normalized for marker in embedding_markers)


def is_ollama_answer_model(model_name: str) -> bool:
    normalized = model_name.casefold()
    excluded_markers = [
        "embed",
        "embedding",
        "bge",
        "minilm",
        "mxbai",
        "nomic",
        "snowflake-arctic",
        "whisper",
        "ocr",
    ]
    return not any(marker in normalized for marker in excluded_markers)


@st.cache_data(show_spinner=False)
def ollama_model_names() -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return [], f"Could not read local Ollama model list: {exc}"

    model_names: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        model_names.append(line.split()[0])

    return model_names, None


def benchmark_model_options() -> tuple[list[str], list[str], str | None]:
    ollama_models, ollama_error = ollama_model_names()
    ollama_embedding_models = [
        model_name for model_name in ollama_models if is_ollama_embedding_model(model_name)
    ]
    ollama_answer_models = [
        model_name for model_name in ollama_models if is_ollama_answer_model(model_name)
    ]

    embedding_options = unique_preserve_order(
        EMBEDDING_MODEL_OPTIONS + sorted(ollama_embedding_models, key=str.casefold)
    )
    answer_options = unique_preserve_order(
        ANSWER_MODEL_OPTIONS + sorted(ollama_answer_models, key=str.casefold)
    )
    return embedding_options, answer_options, ollama_error


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def save_benchmark_run_config(
    run_config: dict[str, Any],
    uploaded_pdf: Any,
    uploaded_questions: Any | None = None,
) -> tuple[str, Path, Path]:
    run_id = str(run_config["run_id"])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    uploaded_pdf_path = run_dir / "uploaded.pdf"
    uploaded_pdf_path.write_bytes(uploaded_pdf.getvalue())
    run_config["pdf"] = {
        "mode": "uploaded",
        "filename": uploaded_pdf.name,
        "saved_path": str(uploaded_pdf_path),
    }

    if uploaded_questions is not None:
        questions_path = run_dir / "benchmark_questions.jsonl"
        questions_path.write_bytes(uploaded_questions.getvalue())
        run_config["benchmark_questions"] = {
            "mode": "uploaded",
            "filename": uploaded_questions.name,
            "saved_path": str(questions_path),
        }

    config_path = run_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return run_id, run_dir, config_path


def save_setup_only_run_config(
    run_config: dict[str, Any],
    uploaded_pdf: Any,
) -> tuple[str, Path, Path]:
    run_id = str(run_config["run_id"])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    uploaded_pdf_path = run_dir / "uploaded.pdf"
    uploaded_pdf_path.write_bytes(uploaded_pdf.getvalue())

    questions_path = run_dir / "benchmark_questions.jsonl"
    questions_path.write_bytes(BENCHMARK_QUESTIONS_PATH.read_bytes())

    run_config["uploaded_pdf_path"] = str(uploaded_pdf_path)
    run_config["benchmark_questions_source_path"] = str(BENCHMARK_QUESTIONS_PATH)
    run_config["benchmark_questions_path"] = str(questions_path)

    config_path = run_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return run_id, run_dir, config_path


def render_saved_run(run_id: str, run_dir: Path, config_path: Path, run_config: dict[str, Any]) -> None:
    st.success(f"Saved benchmark run configuration: {run_id}")
    st.write({"run_id": run_id, "run_dir": str(run_dir), "config_path": str(config_path)})
    st.code(yaml.safe_dump(run_config, sort_keys=False), language="yaml")


def benchmark_question_file_control(key_prefix: str) -> tuple[str, Any | None]:
    mode = st.radio(
        "Benchmark questions",
        options=[
            "Use existing benchmark/benchmark_questions.jsonl",
            "Upload benchmark questions JSONL",
        ],
        key=f"{key_prefix}_question_mode",
    )
    if mode.startswith("Upload"):
        uploaded_questions = st.file_uploader(
            "Benchmark questions JSONL",
            type=["jsonl"],
            key=f"{key_prefix}_questions_upload",
        )
        return mode, uploaded_questions
    return mode, None


def base_run_config(
    experiment_type: str,
    top_k: int,
    question_mode: str,
) -> dict[str, Any]:
    return {
        "run_id": generate_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        "top_k": int(top_k),
        "benchmark_questions": {
            "mode": "existing" if question_mode.startswith("Use existing") else "uploaded",
            "path": str(BENCHMARK_QUESTIONS_PATH) if question_mode.startswith("Use existing") else None,
        },
        "query_expansion": {
            "included": False,
            "note": "Query expansion is intentionally excluded from this phase and will be added later.",
        },
        "execution": {
            "status": "configured_only",
            "note": "The full benchmark execution engine is not run from this UI yet.",
        },
    }


def validate_benchmark_inputs(uploaded_pdf: Any, uploaded_questions: Any | None, question_mode: str) -> list[str]:
    errors: list[str] = []
    if uploaded_pdf is None:
        errors.append("Upload a PDF before saving a benchmark run.")
    if question_mode.startswith("Upload") and uploaded_questions is None:
        errors.append("Upload a benchmark questions JSONL file or use the existing benchmark file.")
    return errors


def render_placeholder_status() -> None:
    progress_bar = st.progress(0)
    with st.status("Benchmark setup saved", expanded=True) as status:
        st.write("Validated controlled experiment settings.")
        progress_bar.progress(33)
        st.write("Saved uploaded PDF.")
        progress_bar.progress(66)
        st.write("Saved run_config.yaml.")
        progress_bar.progress(100)
        st.write("Benchmark execution is intentionally not started yet.")
        status.update(label="Ready for future benchmark execution", state="complete")


def parser_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    parsers = config.get("parsers", {}).get("available", [])
    if isinstance(parsers, list) and parsers:
        return [parser for parser in parsers if isinstance(parser, dict)]

    return [
        {
            "id": "pymupdf",
            "name": "PyMuPDF Baseline",
            "enabled": True,
            "status": "ready",
        },
        {
            "id": "docling",
            "name": "Docling Structured Parser",
            "enabled": True,
            "status": "ready",
        },
    ]


def ready_parser_options(config: dict[str, Any]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for parser in parser_registry(config):
        if parser.get("enabled") is True and parser.get("status") == "ready":
            options.append((str(parser["id"]), str(parser["name"])))
    return options


def render_parser_checkboxes(config: dict[str, Any]) -> list[str]:
    selected_parser_ids: list[str] = []
    st.markdown("Parser candidates")

    for parser in parser_registry(config):
        parser_id = str(parser.get("id", ""))
        parser_name = str(parser.get("name", parser_id))
        enabled = parser.get("enabled") is True
        status = str(parser.get("status", "unknown"))
        selectable = enabled and status == "ready"
        label = f"{parser_name} (`{parser_id}`)"
        help_text = f"status={status}, enabled={enabled}"

        selected = st.checkbox(
            label,
            value=selectable,
            disabled=not selectable,
            help=help_text,
            key=f"parser_compare_{parser_id}",
        )
        if selected and selectable:
            selected_parser_ids.append(parser_id)

        if not selectable:
            st.caption(f"{parser_name} is not selectable yet: {help_text}")

    return selected_parser_ids


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
            "document_name": str(metadata.get("document_name", "")),
            "page_number": metadata.get("page_number", ""),
            "section_title": metadata.get("section_title", ""),
            "chunk_index": metadata.get("chunk_index", ""),
            "text": str(documents[index]),
        }

        if distances:
            chunk["distance"] = distances[index]

        retrieved_chunks.append(chunk)

    return retrieved_chunks


@st.cache_data(show_spinner=False)
def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"Missing config file: {path}"

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        return {}, f"Could not parse {path}: {exc}"
    except OSError as exc:
        return {}, f"Could not read {path}: {exc}"

    if not isinstance(config, dict):
        return {}, f"Config file must contain a YAML mapping: {path}"

    return config, None


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> Any:
    from embeddings import EmbeddingModel

    return EmbeddingModel(model_name)


@st.cache_resource(show_spinner=False)
def load_chroma_collection(vector_db_dir: str, collection_name: str) -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=vector_db_dir)
    return client.get_collection(name=collection_name)


def manual_query_config(config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    try:
        settings = {
            "vector_db_dir": str(config["paths"]["vector_db_dir"]),
            "pymupdf_collection_name": str(config["embedding"]["collection_name"]),
            "docling_collection_name": str(config["embedding"]["docling_collection_name"]),
            "embedding_model_name": str(config["embedding"]["model_name"]),
            "top_k": int(config["retrieval"]["top_k"]),
            "ollama_model_name": str(config["ollama"]["model_name"]),
            "ollama_temperature": float(config["ollama"]["temperature"]),
            "num_expanded_queries": int(config["query_expansion"]["num_queries"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {}, f"Missing or invalid manual query config value: {exc}"

    if settings["top_k"] <= 0:
        return {}, "Configured retrieval.top_k must be greater than 0."
    if settings["num_expanded_queries"] <= 0:
        return {}, "Configured query_expansion.num_queries must be greater than 0."

    return settings, None


def manual_source_options(settings: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "PyMuPDF baseline": {
            "source": "pymupdf",
            "collection_name": settings["pymupdf_collection_name"],
        },
        "Docling structured parser": {
            "source": "docling",
            "collection_name": settings["docling_collection_name"],
        },
    }


def expand_manual_query(query_text: str, settings: dict[str, Any]) -> list[str]:
    from query_expansion import generate_expanded_queries

    expanded_queries, _raw_response = generate_expanded_queries(
        question=query_text,
        model_name=settings["ollama_model_name"],
        num_queries=settings["num_expanded_queries"],
        temperature=settings["ollama_temperature"],
    )
    return expanded_queries


def chunk_evidence_label(chunk: dict[str, Any]) -> str:
    parts = [
        f"chunk_id={chunk.get('chunk_id', '')}",
        f"rank={chunk.get('rank', '')}",
    ]
    if chunk.get("source"):
        parts.append(f"source={chunk.get('source')}")
    if chunk.get("retrieval_type"):
        parts.append(f"retrieval_type={chunk.get('retrieval_type')}")
    if chunk.get("page_number") not in ("", None):
        parts.append(f"page_number={chunk.get('page_number')}")
    if chunk.get("section_title"):
        parts.append(f"section_title={chunk.get('section_title')}")
    return ", ".join(parts)


def build_answer_prompt(query_text: str, chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"[{chunk_evidence_label(chunk)}]",
                    str(chunk.get("text", "")),
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""You answer technical-document questions using only retrieved context.

Rules:
- Use only the retrieved context below.
- Do not use outside knowledge.
- Do not infer fields, values, sections, or requirements that are not explicitly supported by the context.
- If the context is insufficient, clearly say what is missing.
- Keep the answer concise and technical.
- Cite evidence using chunk_id, rank, source/parser, retrieval type, page_number if available, and section_title if available.

Question:
{query_text}

Retrieved context:
{context}

Return exactly these sections:
## Answer Summary
## Evidence Used
## Missing or Uncertain Information
"""


def generate_grounded_answer(
    query_text: str,
    chunks: list[dict[str, Any]],
    settings: dict[str, Any],
) -> str:
    if not chunks:
        return (
            "## Answer Summary\n"
            "The retrieved context is insufficient because no chunks were provided.\n\n"
            "## Evidence Used\n"
            "- None\n\n"
            "## Missing or Uncertain Information\n"
            "- No retrieved chunks were available for answer generation.\n"
        )

    import ollama

    try:
        response = ollama.generate(
            model=settings["ollama_model_name"],
            prompt=build_answer_prompt(query_text, chunks),
            options={"temperature": settings["ollama_temperature"]},
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to generate answer with Ollama. "
            f"Check that Ollama is running and model '{settings['ollama_model_name']}' is available."
        ) from exc

    return str(response["response"]).strip()


def retrieve_manual_query(
    query_text: str,
    top_k: int,
    settings: dict[str, Any],
    collection_name: str,
    source: str,
) -> list[dict[str, Any]]:
    embedding_model = load_embedding_model(settings["embedding_model_name"])
    try:
        collection = load_chroma_collection(settings["vector_db_dir"], collection_name)
    except Exception as exc:
        if source == "docling":
            raise RuntimeError(
                "Docling collection is not available. Build it before using this mode: "
                "python src/vector_store.py --source docling"
            ) from exc
        raise RuntimeError(f"Selected collection does not exist: {collection_name}") from exc

    metadata = collection.metadata or {}
    existing_model_name = metadata.get("embedding_model")
    if existing_model_name != settings["embedding_model_name"]:
        raise ValueError(
            "Chroma collection embedding model mismatch: "
            f"collection={collection_name}, existing={existing_model_name}, "
            f"configured={settings['embedding_model_name']}"
        )
    if source == "docling" and metadata.get("source") != "docling":
        raise ValueError(
            "Selected collection is not marked as a Docling collection. "
            f"collection={collection_name}, source={metadata.get('source')}"
        )

    query_embedding = embedding_model.embed_texts([query_text])[0]
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return format_retrieved_chunks(query_result)


@st.cache_data(show_spinner=False)
def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"Missing expected file: {path}"

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if not isinstance(record, dict):
                    return [], f"Expected JSON object at {path}:{line_number}"
                records.append(record)
    except json.JSONDecodeError as exc:
        return [], f"Could not parse {path}:{exc.lineno}: {exc.msg}"
    except OSError as exc:
        return [], f"Could not read {path}: {exc}"

    return records, None


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"Missing expected file: {path}"

    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pandas can raise several parser and IO errors.
        return pd.DataFrame(), f"Could not read {path}: {exc}"


@st.cache_data(show_spinner=False)
def read_markdown(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "", f"Missing expected file: {path}"

    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return "", f"Could not read {path}: {exc}"


def group_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        question_id = str(record.get("question_id", ""))
        if question_id:
            grouped[question_id].append(record)
    return dict(grouped)


def query_expansion_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("question_id")): record
        for record in records
        if record.get("question_id") is not None
    }


def benchmark_question_map(records: list[dict[str, Any]]) -> dict[str, str]:
    questions: dict[str, str] = {}
    for record in records:
        question_id = record.get("id") or record.get("question_id")
        question = record.get("question")
        if question_id is not None and question:
            questions[str(question_id)] = str(question)
    return questions


def first_record_for_question(
    records_by_question: dict[str, list[dict[str, Any]]],
    question_id: str,
) -> dict[str, Any] | None:
    records = records_by_question.get(question_id, [])
    return records[0] if records else None


def annotated_chunk(
    chunk: dict[str, Any],
    source: str,
    retrieval_type: str,
) -> dict[str, Any]:
    annotated = dict(chunk)
    annotated["source"] = source
    annotated["retrieval_type"] = retrieval_type
    return annotated


def original_answer_chunks(
    records_by_question: dict[str, list[dict[str, Any]]],
    question_id: str,
    source: str,
    top_k: int,
) -> list[dict[str, Any]]:
    record = first_record_for_question(records_by_question, question_id)
    if not record:
        return []

    chunks = record.get("retrieved_chunks", [])[:top_k]
    return [annotated_chunk(chunk, source=source, retrieval_type="original") for chunk in chunks]


def rank_value(chunk: dict[str, Any]) -> int:
    return safe_int(chunk.get("rank"), default=10_000)


def retrieval_quality_value(chunk: dict[str, Any]) -> float:
    if chunk.get("distance") is not None:
        try:
            return float(chunk["distance"])
        except (TypeError, ValueError):
            return float("inf")

    if chunk.get("score") is not None:
        try:
            return -float(chunk["score"])
        except (TypeError, ValueError):
            return float("inf")

    return float("inf")


def expanded_answer_chunks(
    records_by_question: dict[str, list[dict[str, Any]]],
    question_id: str,
    source: str,
    top_k: int,
) -> list[dict[str, Any]]:
    best_chunks: dict[str, dict[str, Any]] = {}
    for record in records_by_question.get(question_id, []):
        for chunk in record.get("retrieved_chunks", []):
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue

            annotated = annotated_chunk(chunk, source=source, retrieval_type="expanded")
            existing = best_chunks.get(chunk_id)
            if existing is None:
                best_chunks[chunk_id] = annotated
                continue

            existing_key = (rank_value(existing), retrieval_quality_value(existing))
            candidate_key = (rank_value(annotated), retrieval_quality_value(annotated))
            if candidate_key < existing_key:
                best_chunks[chunk_id] = annotated

    ranked_chunks = sorted(
        best_chunks.values(),
        key=lambda chunk: (
            rank_value(chunk),
            retrieval_quality_value(chunk),
            str(chunk.get("chunk_id", "")),
        ),
    )
    return ranked_chunks[:top_k]


def first_question_text(
    question_id: str,
    original_by_question: dict[str, list[dict[str, Any]]],
    expansions_by_question: dict[str, dict[str, Any]],
    expanded_by_question: dict[str, list[dict[str, Any]]],
) -> str:
    original_records = original_by_question.get(question_id, [])
    if original_records:
        return str(original_records[0].get("question", ""))

    expansion_record = expansions_by_question.get(question_id, {})
    if expansion_record:
        return str(expansion_record.get("original_question", ""))

    expanded_records = expanded_by_question.get(question_id, [])
    if expanded_records:
        return str(expanded_records[0].get("original_question", ""))

    return ""


def all_question_ids(
    original_records: list[dict[str, Any]],
    expansion_records: list[dict[str, Any]],
    expanded_records: list[dict[str, Any]],
) -> list[str]:
    question_ids = {
        str(record.get("question_id"))
        for record in original_records + expansion_records + expanded_records
        if record.get("question_id") is not None
    }
    return sorted(question_ids)


def chunk_rows(chunks: list[dict[str, Any]], highlight_ids: set[str] | None = None) -> pd.DataFrame:
    highlight_ids = highlight_ids or set()
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        score_or_distance = chunk.get("score", chunk.get("distance"))
        rows.append(
            {
                "rank": chunk.get("rank", ""),
                "chunk_id": chunk_id,
                "document_name": chunk.get("document_name", ""),
                "page_number": chunk.get("page_number", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "score_or_distance": score_or_distance,
                "expanded_only": chunk_id in highlight_ids,
                "text_preview": preview_text(chunk.get("text", "")),
            }
        )
    return pd.DataFrame(rows)


def unique_chunks(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for record in records:
        for chunk in record.get("retrieved_chunks", []):
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id:
                chunks.setdefault(chunk_id, chunk)
    return chunks


def render_chunk_table(
    chunks: list[dict[str, Any]],
    empty_message: str,
    highlight_ids: set[str] | None = None,
) -> None:
    if not chunks:
        st.info(empty_message)
        return

    st.dataframe(
        chunk_rows(chunks, highlight_ids=highlight_ids),
        hide_index=True,
        use_container_width=True,
        column_config={
            "text_preview": st.column_config.TextColumn("text_preview", width="large"),
            "score_or_distance": st.column_config.NumberColumn(
                "score_or_distance",
                format="%.4f",
            ),
        },
    )


def render_file_warnings(errors: list[str]) -> None:
    if not errors:
        return

    st.warning("Some expected MVP output files are missing or unreadable.")
    for error in errors:
        st.caption(error)


def first_original_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[0] if records else None


def comparison_question_ids(
    comparison_df: pd.DataFrame,
    *record_groups: dict[str, list[dict[str, Any]]],
) -> list[str]:
    question_ids: set[str] = set()
    if not comparison_df.empty and "question_id" in comparison_df.columns:
        question_ids.update(str(question_id) for question_id in comparison_df["question_id"].dropna())

    for group in record_groups:
        question_ids.update(group)

    return sorted(question_ids)


def question_label(
    question_id: str,
    parser_comparison_df: pd.DataFrame,
    *record_groups: dict[str, list[dict[str, Any]]],
) -> str:
    if not parser_comparison_df.empty and "question_id" in parser_comparison_df.columns:
        rows = parser_comparison_df[parser_comparison_df["question_id"].astype(str) == question_id]
        if not rows.empty and "question" in rows.columns:
            question = str(rows.iloc[0]["question"])
            if question:
                return f"{question_id} - {question}"

    for group in record_groups:
        records = group.get(question_id, [])
        if records:
            question = records[0].get("question") or records[0].get("original_question")
            if question:
                return f"{question_id} - {question}"

    return question_id


def parser_metric_value(row: pd.Series | None, column: str) -> int:
    if row is None or column not in row:
        return 0
    return safe_int(row[column])


def render_expanded_records(
    records: list[dict[str, Any]],
    empty_message: str,
) -> None:
    if not records:
        st.info(empty_message)
        return

    for record in sorted(records, key=lambda item: safe_int(item.get("expanded_query_index"))):
        query_index = safe_int(record.get("expanded_query_index")) + 1
        query_text = str(record.get("query_text", ""))
        with st.expander(f"Expanded query {query_index}: {query_text}", expanded=False):
            render_chunk_table(
                record.get("retrieved_chunks", []),
                "No retrieved chunks found for this expanded query.",
            )


def render_answer_generation(
    label: str,
    query_text: str,
    chunks: list[dict[str, Any]],
    settings: dict[str, Any],
) -> None:
    answer_key = f"manual_answer_{label}"
    button_key = f"generate_answer_{label}"

    if st.button(f"Generate Answer - {label}", key=button_key):
        try:
            with st.spinner(f"Generating grounded answer from {label} chunks..."):
                st.session_state[answer_key] = generate_grounded_answer(
                    query_text=query_text,
                    chunks=chunks,
                    settings=settings,
                )
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")

    if answer_key in st.session_state:
        st.markdown(st.session_state[answer_key])


def render_manual_query_results(
    source_choice: str,
    use_query_expansion: bool,
    expanded_queries: list[str],
    manual_results: dict[str, dict[str, Any]],
    query_text: str,
    settings: dict[str, Any],
) -> None:
    if use_query_expansion:
        st.markdown("#### Expanded Queries")
        if expanded_queries:
            for index, expanded_query in enumerate(expanded_queries, start=1):
                st.write(f"{index}. {expanded_query}")
        else:
            st.info("Ollama did not return any expanded queries.")

    if source_choice == "Compare both":
        result_cols = st.columns(2)
        for column, (label, result) in zip(result_cols, manual_results.items()):
            with column:
                st.markdown(f"#### {label}")
                st.caption(f"Collection: `{result['collection_name']}`")
                st.markdown("##### Original Retrieval")
                render_chunk_table(
                    result["original_chunks"],
                    "No chunks were retrieved for the original query.",
                )
                render_answer_generation(
                    label=label,
                    query_text=query_text,
                    chunks=result["original_chunks"],
                    settings=settings,
                )

                if use_query_expansion:
                    st.markdown("##### Expanded Retrieval")
                    render_expanded_records(
                        result["expanded_records"],
                        "No expanded retrieval records were generated.",
                    )
        return

    result = next(iter(manual_results.values()))
    st.caption(f"Collection: `{result['collection_name']}`")
    st.markdown("#### Original Retrieval")
    render_chunk_table(
        result["original_chunks"],
        "No chunks were retrieved for the original query.",
    )
    render_answer_generation(
        label=source_choice,
        query_text=query_text,
        chunks=result["original_chunks"],
        settings=settings,
    )

    if use_query_expansion:
        original_chunk_ids = {
            str(chunk.get("chunk_id", "")) for chunk in result["original_chunks"]
        }
        expanded_chunks_by_id = unique_chunks(result["expanded_records"])
        expanded_only_ids = set(expanded_chunks_by_id) - original_chunk_ids
        expanded_only_chunks = [
            chunk
            for chunk_id, chunk in expanded_chunks_by_id.items()
            if chunk_id in expanded_only_ids
        ]

        st.markdown("#### Expanded-Only Chunks")
        render_chunk_table(
            expanded_only_chunks,
            "No chunks were found only by expanded queries.",
            highlight_ids=expanded_only_ids,
        )

        st.markdown("#### Expanded Retrieval")
        render_expanded_records(
            result["expanded_records"],
            "No expanded retrieval records were generated.",
        )


def render_answer_panel(title: str, answer: str | None, chunks: list[dict[str, Any]]) -> None:
    st.markdown(f"#### {title}")
    st.caption(f"Context chunks: {len(chunks)}")
    with st.expander("Context Chunks", expanded=False):
        render_chunk_table(chunks, "No chunks available for this answer.")

    if answer:
        st.markdown(answer)
    else:
        st.info("No answer generated yet.")


def render_manual_review_checklist() -> None:
    st.markdown("#### Manual Review Checklist")
    st.checkbox("Does the answer use the retrieved evidence?", key="review_uses_evidence")
    st.checkbox("Does the answer include unsupported claims?", key="review_unsupported_claims")
    st.checkbox("Does the answer miss important field/table details?", key="review_misses_details")
    st.text_input("Which answer is most useful?", key="review_most_useful")


def render_benchmark_tool(config: dict[str, Any]) -> None:
    experiment_type = st.selectbox(
        "Experiment Type",
        options=[
            "Parser Comparison",
            "Embedding Model Comparison",
            "Answer Model Comparison",
            "Full Auto Recommendation (coming soon)",
        ],
    )

    st.info(
        "Query expansion is intentionally excluded from this phase and will be added later."
    )

    if experiment_type == "Full Auto Recommendation (coming soon)":
        st.warning("Full Auto Recommendation is disabled for now.")
        st.button("Run Full Auto Recommendation", disabled=True)
        return

    uploaded_pdf = st.file_uploader(
        "PDF upload",
        type=["pdf"],
        key=f"{experiment_type}_pdf_upload",
    )
    question_mode, uploaded_questions = benchmark_question_file_control(experiment_type)

    embedding_model_options, answer_model_options, ollama_error = benchmark_model_options()
    if ollama_error:
        st.caption(f"{ollama_error}. Using default model options.")
    else:
        st.caption(
            f"Loaded {len(embedding_model_options)} embedding options and "
            f"{len(answer_model_options)} answer options from defaults plus local Ollama models."
        )

    default_embedding = str(
        config.get("embedding", {}).get("model_name", embedding_model_options[0])
    )
    default_answer = str(config.get("ollama", {}).get("model_name", answer_model_options[0]))
    default_top_k = safe_int(config.get("retrieval", {}).get("top_k"), default=5)

    if experiment_type == "Parser Comparison":
        st.caption(
            "Parser Comparison changes only the document parser while keeping the embedding model "
            "and answer model fixed. This helps isolate the effect of document parsing on retrieval "
            "and answer quality."
        )
        selected_parser_ids = render_parser_checkboxes(config)
        embedding_model = st.selectbox(
            "Embedding model",
            options=embedding_model_options,
            index=embedding_model_options.index(default_embedding)
            if default_embedding in embedding_model_options
            else 0,
        )
        answer_model = st.selectbox(
            "Answer model",
            options=answer_model_options,
            index=answer_model_options.index(default_answer)
            if default_answer in answer_model_options
            else 0,
        )
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=50,
            value=default_top_k,
            step=1,
            key="parser_benchmark_top_k",
        )

        run_disabled = len(selected_parser_ids) < 2
        if run_disabled:
            st.warning("Select at least two parsers to run Parser Comparison.")

        if st.button("Run Parser Benchmark", type="primary", disabled=run_disabled):
            validation_errors = validate_benchmark_inputs(uploaded_pdf, uploaded_questions, question_mode)
            if len(selected_parser_ids) < 2:
                validation_errors.append("Select at least two parsers to run Parser Comparison.")
            if validation_errors:
                for error in validation_errors:
                    st.warning(error)
            else:
                run_config = base_run_config(experiment_type, int(top_k), question_mode)
                run_config.update(
                    {
                        "experiment_type": "parser_comparison",
                        "purpose": "Compare selected parsers while keeping embedding model and answer model fixed.",
                        "selected_parsers": selected_parser_ids,
                        "embedding_model": embedding_model,
                        "answer_model": answer_model,
                        "chunk_size": safe_int(
                            config.get("chunking", {}).get("chunk_size"),
                            default=800,
                        ),
                        "chunk_overlap": safe_int(
                            config.get("chunking", {}).get("chunk_overlap"),
                            default=150,
                        ),
                        "controlled_variables": {
                            "embedding_model": embedding_model,
                            "answer_model": answer_model,
                        },
                        "variable_under_test": "parser",
                    }
                )
                run_id, run_dir, config_path = save_benchmark_run_config(
                    run_config,
                    uploaded_pdf,
                    uploaded_questions,
                )
                run_config["uploaded_pdf_path"] = run_config["pdf"]["saved_path"]
                run_config["benchmark_questions_path"] = run_config["benchmark_questions"][
                    "saved_path" if uploaded_questions is not None else "path"
                ]
                config_path.write_text(
                    yaml.safe_dump(run_config, sort_keys=False),
                    encoding="utf-8",
                )
                render_placeholder_status()
                render_saved_run(run_id, run_dir, config_path, run_config)
        return

    if experiment_type == "Embedding Model Comparison":
        st.caption("Compare embedding models while keeping parser and answer model fixed.")
        parser = st.selectbox("Parser", options=PARSER_OPTIONS)
        embedding_models = st.multiselect(
            "Embedding models",
            options=embedding_model_options,
            default=[default_embedding] if default_embedding in embedding_model_options else [],
        )
        answer_model = st.selectbox(
            "Answer model",
            options=answer_model_options,
            index=answer_model_options.index(default_answer)
            if default_answer in answer_model_options
            else 0,
        )
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=50,
            value=default_top_k,
            step=1,
            key="embedding_benchmark_top_k",
        )

        if st.button("Run Embedding Benchmark", type="primary"):
            validation_errors = validate_benchmark_inputs(uploaded_pdf, uploaded_questions, question_mode)
            if len(embedding_models) < 2:
                validation_errors.append("Select at least two embedding models for comparison.")
            if validation_errors:
                for error in validation_errors:
                    st.warning(error)
            else:
                run_config = base_run_config(experiment_type, int(top_k), question_mode)
                run_config.update(
                    {
                        "purpose": "Compare multiple embedding models while keeping parser and answer model fixed.",
                        "parser": parser.lower(),
                        "embedding_models": embedding_models,
                        "answer_models": [answer_model],
                        "controlled_variables": {
                            "parser": parser.lower(),
                            "answer_model": answer_model,
                        },
                        "variable_under_test": "embedding_model",
                    }
                )
                run_id, run_dir, config_path = save_benchmark_run_config(
                    run_config,
                    uploaded_pdf,
                    uploaded_questions,
                )
                render_placeholder_status()
                render_saved_run(run_id, run_dir, config_path, run_config)
        return

    st.caption("Compare answer models while keeping parser, embedding model, and retrieved chunks fixed.")
    parser = st.selectbox("Parser", options=PARSER_OPTIONS, key="answer_benchmark_parser")
    embedding_model = st.selectbox(
        "Embedding model",
        options=embedding_model_options,
        index=embedding_model_options.index(default_embedding)
        if default_embedding in embedding_model_options
        else 0,
        key="answer_benchmark_embedding",
    )
    answer_models = st.multiselect(
        "Answer models",
        options=answer_model_options,
        default=[default_answer] if default_answer in answer_model_options else [],
    )
    top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="answer_benchmark_top_k",
    )

    if st.button("Run Answer Benchmark", type="primary"):
        validation_errors = validate_benchmark_inputs(uploaded_pdf, uploaded_questions, question_mode)
        if len(answer_models) < 2:
            validation_errors.append("Select at least two answer models for comparison.")
        if validation_errors:
            for error in validation_errors:
                st.warning(error)
        else:
            run_config = base_run_config(experiment_type, int(top_k), question_mode)
            run_config.update(
                {
                    "purpose": "Compare multiple answer generation models while keeping parser, embedding model, and retrieved chunks fixed.",
                    "parser": parser.lower(),
                    "embedding_models": [embedding_model],
                    "answer_models": answer_models,
                    "controlled_variables": {
                        "parser": parser.lower(),
                        "embedding_model": embedding_model,
                        "retrieved_chunks": "fixed",
                    },
                    "variable_under_test": "answer_model",
                }
            )
            run_id, run_dir, config_path = save_benchmark_run_config(
                run_config,
                uploaded_pdf,
                uploaded_questions,
            )
            render_placeholder_status()
            render_saved_run(run_id, run_dir, config_path, run_config)


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def benchmark_defaults(config: dict[str, Any]) -> tuple[list[str], list[str], str, str, int]:
    embedding_model_options, answer_model_options, ollama_error = benchmark_model_options()
    if ollama_error:
        st.caption(f"{ollama_error}. Using default model options.")
    else:
        st.caption(
            f"Loaded {len(embedding_model_options)} embedding options and "
            f"{len(answer_model_options)} answer options from defaults plus local Ollama models."
        )

    default_embedding = str(
        config.get("embedding", {}).get("model_name", embedding_model_options[0])
    )
    default_answer = str(config.get("ollama", {}).get("model_name", answer_model_options[0]))
    default_top_k = safe_int(config.get("retrieval", {}).get("top_k"), default=5)
    return embedding_model_options, answer_model_options, default_embedding, default_answer, default_top_k


def setup_run_config(experiment_type: str, top_k: int) -> dict[str, Any]:
    return {
        "run_id": generate_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
        "top_k": int(top_k),
        "uploaded_pdf_path": None,
        "benchmark_questions_path": str(BENCHMARK_QUESTIONS_PATH),
        "execution": {
            "status": "configured_only",
            "note": "Benchmark execution is intentionally not run from this UI yet.",
        },
    }


def validate_setup_inputs(uploaded_pdf: Any, required_errors: list[str]) -> list[str]:
    errors = []
    if uploaded_pdf is None:
        errors.append("Upload a PDF before saving a benchmark run.")
    _questions, questions_error = load_benchmark_questions_for_ui()
    if questions_error and questions_error not in required_errors:
        errors.append(questions_error)
    errors.extend(required_errors)
    return errors


def load_benchmark_questions_for_ui(path: Path = BENCHMARK_QUESTIONS_PATH) -> tuple[list[dict[str, str]], str | None]:
    records, error = read_jsonl(path)
    if error:
        return [], error
    if not records:
        return [], f"Benchmark question file is empty: {path}"

    questions: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        question_id = record.get("id")
        question = record.get("question")
        if question_id is None or question is None:
            return [], f"Missing id/question in benchmark question record {index}: {path}"
        questions.append({"id": str(question_id), "question": str(question)})

    return questions, None


def render_benchmark_questions_preview(questions: list[dict[str, str]], error: str | None) -> None:
    st.markdown("#### Benchmark Questions")
    st.caption(f"Default source: `{BENCHMARK_QUESTIONS_PATH}`")
    if error:
        st.error(error)
        return

    st.dataframe(
        pd.DataFrame(questions),
        hide_index=True,
        use_container_width=True,
    )


def selected_parser_ids_from_checkboxes(key_prefix: str) -> list[str]:
    selected: list[str] = []
    for parser_label in PARSER_OPTIONS:
        checked = st.checkbox(
            parser_label,
            value=True,
            key=f"{key_prefix}_{PARSER_ID_BY_LABEL[parser_label]}",
        )
        if checked:
            selected.append(PARSER_ID_BY_LABEL[parser_label])
    return selected


def selected_options_from_checkboxes(
    label: str,
    options: list[str],
    default_selected: list[str],
    key_prefix: str,
) -> list[str]:
    st.markdown(label)
    selected: list[str] = []
    default_selected_set = set(default_selected)
    for option in options:
        checked = st.checkbox(
            option,
            value=option in default_selected_set,
            key=f"{key_prefix}_{option}",
        )
        if checked:
            selected.append(option)
    return selected


def render_fixed_retrieval_strategy(key: str) -> str:
    return st.selectbox(
        "Retrieval strategy",
        options=[DEFAULT_RETRIEVAL_STRATEGY],
        key=key,
        help="Fixed for now. Retrieval strategy comparison is coming later.",
    )


def parser_result_records(runner_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records_by_parser: dict[str, list[dict[str, Any]]] = {}
    for parser_id, result in runner_result.get("parser_results", {}).items():
        answer_path = Path(str(result["answer_results"]))
        records, error = read_jsonl(answer_path)
        if error:
            st.warning(error)
            records_by_parser[str(parser_id)] = []
        else:
            records_by_parser[str(parser_id)] = records
    return records_by_parser


def render_parser_compare_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Parser Compare Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "chroma_dir": runner_result.get("chroma_dir"),
            "reports": runner_result.get("reports"),
        }
    )

    records_by_parser = parser_result_records(runner_result)
    question_ids = sorted(
        {
            str(record.get("question_id"))
            for records in records_by_parser.values()
            for record in records
            if record.get("question_id") is not None
        }
    )
    if not question_ids:
        st.info("No answer results were found for this run.")
        return

    question_text_by_id = {
        str(record.get("question_id")): str(record.get("question", ""))
        for records in records_by_parser.values()
        for record in records
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"parser_compare_result_question_{runner_result.get('run_id')}",
    )

    parser_ids = list(records_by_parser)
    columns = st.columns(len(parser_ids))
    for column, parser_id in zip(columns, parser_ids):
        with column:
            st.markdown(f"#### {parser_id}")
            record = next(
                (
                    item
                    for item in records_by_parser[parser_id]
                    if str(item.get("question_id")) == selected_question_id
                ),
                None,
            )
            if record is None:
                st.info("No result for this question.")
                continue

            st.markdown("##### Generated Answer")
            st.markdown(str(record.get("generated_answer", "")))
            st.markdown("##### Retrieved Chunks")
            render_chunk_table(
                record.get("retrieved_chunks", []),
                "No retrieved chunks found.",
            )


def chunking_result_records(runner_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for strategy, result in runner_result.get("strategy_results", {}).items():
        answer_path = Path(str(result["answer_results"]))
        records, error = read_jsonl(answer_path)
        if error:
            st.warning(error)
            records_by_strategy[str(strategy)] = []
        else:
            records_by_strategy[str(strategy)] = records
    return records_by_strategy


def render_chunking_compare_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Chunking Compare Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "chroma_dir": runner_result.get("chroma_dir"),
            "parser": runner_result.get("parser"),
            "reports": runner_result.get("reports"),
        }
    )

    strategy_results = dict(runner_result.get("strategy_results", {}))
    warnings = {
        strategy: result.get("warnings", [])
        for strategy, result in strategy_results.items()
        if result.get("warnings")
    }
    if warnings:
        st.warning("Some chunking strategies emitted warnings.")
        st.write(warnings)

    records_by_strategy = chunking_result_records(runner_result)
    question_ids = sorted(
        {
            str(record.get("question_id"))
            for records in records_by_strategy.values()
            for record in records
            if record.get("question_id") is not None
        }
    )
    if not question_ids:
        st.info("No answer results were found for this run.")
        return

    question_text_by_id = {
        str(record.get("question_id")): str(record.get("question", ""))
        for records in records_by_strategy.values()
        for record in records
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"chunking_compare_result_question_{runner_result.get('run_id')}",
    )

    strategies = list(records_by_strategy)
    columns = st.columns(len(strategies))
    for column, strategy in zip(columns, strategies):
        with column:
            result = strategy_results.get(strategy, {})
            st.markdown(f"#### {strategy}")
            st.caption(
                f"Chunks: {result.get('chunk_count', 0)} | "
                f"Avg length: {float(result.get('average_chunk_length', 0.0)):.1f}"
            )
            record = next(
                (
                    item
                    for item in records_by_strategy[strategy]
                    if str(item.get("question_id")) == selected_question_id
                ),
                None,
            )
            if record is None:
                st.info("No result for this question.")
                continue

            st.markdown("##### Generated Answer")
            st.markdown(str(record.get("generated_answer", "")))
            st.markdown("##### Retrieved Chunks")
            render_chunk_table(
                record.get("retrieved_chunks", []),
                "No retrieved chunks found.",
            )


def embedding_result_records(runner_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name, result in runner_result.get("model_results", {}).items():
        answer_path = Path(str(result["answer_results"]))
        records, error = read_jsonl(answer_path)
        if error:
            st.warning(error)
            records_by_model[str(model_name)] = []
        else:
            records_by_model[str(model_name)] = records
    return records_by_model


def render_embedding_compare_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Embedding Compare Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "chroma_dir": runner_result.get("chroma_dir"),
            "parser": runner_result.get("parser"),
            "chunking_strategy": runner_result.get("chunking_strategy"),
            "reports": runner_result.get("reports"),
        }
    )

    model_results = dict(runner_result.get("model_results", {}))
    warnings = {
        model_name: result.get("warnings", [])
        for model_name, result in model_results.items()
        if result.get("warnings")
    }
    if warnings:
        st.warning("Some embedding models emitted warnings.")
        st.write(warnings)

    records_by_model = embedding_result_records(runner_result)
    question_ids = sorted(
        {
            str(record.get("question_id"))
            for records in records_by_model.values()
            for record in records
            if record.get("question_id") is not None
        }
    )
    if not question_ids:
        st.info("No answer results were found for this run.")
        return

    question_text_by_id = {
        str(record.get("question_id")): str(record.get("question", ""))
        for records in records_by_model.values()
        for record in records
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"embedding_compare_result_question_{runner_result.get('run_id')}",
    )

    model_names = list(records_by_model)
    columns = st.columns(len(model_names))
    for column, model_name in zip(columns, model_names):
        with column:
            result = model_results.get(model_name, {})
            st.markdown(f"#### {model_name}")
            st.caption(
                f"Runtime: {float(result.get('embedding_runtime_seconds', 0.0)):.2f}s | "
                f"Folder: `{result.get('safe_model_name', '')}`"
            )
            record = next(
                (
                    item
                    for item in records_by_model[model_name]
                    if str(item.get("question_id")) == selected_question_id
                ),
                None,
            )
            if record is None:
                st.info("No result for this question.")
                continue

            st.markdown("##### Generated Answer")
            st.markdown(str(record.get("generated_answer", "")))
            st.markdown("##### Retrieved Chunks")
            render_chunk_table(
                record.get("retrieved_chunks", []),
                "No retrieved chunks found.",
            )


def render_retrieval_fusion_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Retrieval Fusion Compare Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "chroma_dir": runner_result.get("chroma_dir"),
            "parser": runner_result.get("parser"),
            "chunking_strategy": runner_result.get("chunking_strategy"),
            "fusion_method": runner_result.get("fusion_method"),
            "reports": runner_result.get("reports"),
        }
    )

    model_warnings = dict(runner_result.get("model_warnings", {}))
    visible_warnings = {
        model_name: warnings
        for model_name, warnings in model_warnings.items()
        if warnings
    }
    if visible_warnings:
        st.warning("Some embedding models emitted warnings.")
        st.write(visible_warnings)

    fusion_results = dict(runner_result.get("fusion_results", {}))
    fused_records, fused_error = read_jsonl(Path(str(fusion_results.get("fused_retrieval_results", ""))))
    answer_records, answer_error = read_jsonl(Path(str(fusion_results.get("answer_results", ""))))
    if fused_error:
        st.warning(fused_error)
    if answer_error:
        st.warning(answer_error)
    if not fused_records:
        st.info("No fused retrieval results were found for this run.")
        return

    answers_by_question = {
        str(record.get("question_id")): record for record in answer_records
    }
    question_ids = [str(record.get("question_id")) for record in fused_records]
    question_text_by_id = {
        str(record.get("question_id")): str(record.get("question", ""))
        for record in fused_records
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"retrieval_fusion_result_question_{runner_result.get('run_id')}",
    )
    fused_record = next(
        record for record in fused_records if str(record.get("question_id")) == selected_question_id
    )
    answer_record = answers_by_question.get(selected_question_id, {})

    st.markdown("#### Generated Answer")
    st.markdown(str(answer_record.get("generated_answer", "")))

    st.markdown("#### Fused Chunks")
    fused_chunks = fused_record.get("fused_chunks", [])
    if not fused_chunks:
        st.info("No fused chunks found for this question.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "final_rank": chunk.get("final_rank"),
                    "chunk_id": chunk.get("chunk_id"),
                    "page_number": chunk.get("page_number", ""),
                    "section_title": chunk.get("section_title", ""),
                    "fusion_score": chunk.get("fusion_score"),
                    "retrieved_by_embedding_models": ", ".join(
                        chunk.get("retrieved_by_embedding_models", [])
                    ),
                    "original_ranks_by_model": json.dumps(
                        chunk.get("original_ranks_by_model", {}),
                        ensure_ascii=False,
                    ),
                    "text_preview": preview_text(chunk.get("text", "")),
                }
                for chunk in fused_chunks
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_parser_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Parser Compare")
    st.info(
        "Change only the document parser. Chunking strategy, embedding model, answer model, "
        "retrieval strategy, benchmark questions, and top_k stay fixed."
    )

    embedding_options, answer_options, default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="parser_compare_pdf")
    selected_parsers = selected_parser_ids_from_checkboxes("parser_compare_parser")
    chunking_strategy = st.selectbox("Chunking strategy", options=["fixed-size"], key="parser_compare_chunking")
    retrieval_strategy = render_fixed_retrieval_strategy("parser_compare_retrieval_strategy")
    embedding_model = st.selectbox(
        "Embedding model",
        options=embedding_options,
        index=default_index(embedding_options, default_embedding),
        key="parser_compare_embedding",
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="parser_compare_answer",
    )
    top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="parser_compare_top_k",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)

    validation_errors = []
    if len(selected_parsers) < 2:
        validation_errors.append("Select at least two parsers.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Parser Benchmark", type="primary", key="run_parser_compare"):
        errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("parser_compare", int(top_k))
        run_config.update(
            {
                "selected_parsers": selected_parsers,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": retrieval_strategy,
                "embedding_model": embedding_model,
                "answer_model": answer_model,
                "chunk_size": safe_int(
                    config.get("chunking", {}).get("chunk_size"),
                    default=800,
                ),
                "chunk_overlap": safe_int(
                    config.get("chunking", {}).get("chunk_overlap"),
                    default=150,
                ),
            }
        )
        run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
        render_saved_run(run_id, run_dir, config_path, run_config)

        progress_bar = st.progress(0)
        with st.status("Running parser comparison benchmark", expanded=True) as status:
            try:
                from benchmark_runner import run_parser_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_parser_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Parser benchmark failed", state="error")
                st.error(f"Parser benchmark failed: {exc}")
                return

            status.update(label="Parser benchmark complete", state="complete")

        st.session_state["last_parser_compare_result"] = runner_result
        render_parser_compare_results(runner_result)
        rendered_current_results = True

    if "last_parser_compare_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Parser Compare Results", expanded=False):
            render_parser_compare_results(dict(st.session_state["last_parser_compare_result"]))


def render_chunking_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Chunking Compare")
    st.info(
        "Change only the chunking strategy. Parser, embedding model, answer model, "
        "retrieval strategy, benchmark questions, and top_k stay fixed."
    )

    embedding_options, answer_options, default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="chunking_compare_pdf")
    parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="chunking_compare_parser")
    selected_chunking_strategies = selected_options_from_checkboxes(
        "Chunking strategies",
        CHUNKING_STRATEGY_OPTIONS,
        ["fixed-size", "page-based"],
        "chunking_compare_strategy",
    )
    retrieval_strategy = render_fixed_retrieval_strategy("chunking_compare_retrieval_strategy")
    embedding_model = st.selectbox(
        "Embedding model",
        options=embedding_options,
        index=default_index(embedding_options, default_embedding),
        key="chunking_compare_embedding",
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="chunking_compare_answer",
    )
    top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="chunking_compare_top_k",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)

    validation_errors = []
    if len(selected_chunking_strategies) < 2:
        validation_errors.append("Select at least two chunking strategies.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Chunking Benchmark", type="primary", key="run_chunking_compare"):
        errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("chunking_compare", int(top_k))
        run_config.update(
            {
                "parser": PARSER_ID_BY_LABEL[parser_label],
                "selected_chunking_strategies": selected_chunking_strategies,
                "retrieval_strategy": retrieval_strategy,
                "embedding_model": embedding_model,
                "answer_model": answer_model,
                "chunk_size": safe_int(
                    config.get("chunking", {}).get("chunk_size"),
                    default=800,
                ),
                "chunk_overlap": safe_int(
                    config.get("chunking", {}).get("chunk_overlap"),
                    default=150,
                ),
            }
        )
        run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
        render_saved_run(run_id, run_dir, config_path, run_config)

        progress_bar = st.progress(0)
        with st.status("Running chunking comparison benchmark", expanded=True) as status:
            try:
                from benchmark_runner import run_chunking_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_chunking_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Chunking benchmark failed", state="error")
                st.error(f"Chunking benchmark failed: {exc}")
                return

            status.update(label="Chunking benchmark complete", state="complete")

        st.session_state["last_chunking_compare_result"] = runner_result
        render_chunking_compare_results(runner_result)
        rendered_current_results = True

    if "last_chunking_compare_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Chunking Compare Results", expanded=False):
            render_chunking_compare_results(dict(st.session_state["last_chunking_compare_result"]))


def render_embedding_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Embedding Compare")
    st.info(
        "Change only the embedding model. Parser, chunking strategy, answer model, "
        "retrieval strategy, benchmark questions, and top_k stay fixed."
    )

    embedding_options, answer_options, _default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    default_embeddings = [
        model for model in EMBEDDING_MODEL_OPTIONS[:2] if model in embedding_options
    ]
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="embedding_compare_pdf")
    parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="embedding_compare_parser")
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=CHUNKING_STRATEGY_OPTIONS,
        key="embedding_compare_chunking",
    )
    retrieval_strategy = render_fixed_retrieval_strategy("embedding_compare_retrieval_strategy")
    selected_embedding_models = selected_options_from_checkboxes(
        "Embedding models",
        embedding_options,
        default_embeddings,
        "embedding_compare_model",
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="embedding_compare_answer",
    )
    top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="embedding_compare_top_k",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)

    validation_errors = []
    if len(selected_embedding_models) < 2:
        validation_errors.append("Select at least two embedding models.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Embedding Benchmark", type="primary", key="run_embedding_compare"):
        errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("embedding_compare", int(top_k))
        run_config.update(
            {
                "parser": PARSER_ID_BY_LABEL[parser_label],
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": retrieval_strategy,
                "selected_embedding_models": selected_embedding_models,
                "answer_model": answer_model,
                "chunk_size": safe_int(
                    config.get("chunking", {}).get("chunk_size"),
                    default=800,
                ),
                "chunk_overlap": safe_int(
                    config.get("chunking", {}).get("chunk_overlap"),
                    default=150,
                ),
            }
        )
        run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
        render_saved_run(run_id, run_dir, config_path, run_config)

        progress_bar = st.progress(0)
        with st.status("Running embedding comparison benchmark", expanded=True) as status:
            try:
                from benchmark_runner import run_embedding_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_embedding_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Embedding benchmark failed", state="error")
                st.error(f"Embedding benchmark failed: {exc}")
                return

            status.update(label="Embedding benchmark complete", state="complete")

        st.session_state["last_embedding_compare_result"] = runner_result
        render_embedding_compare_results(runner_result)
        rendered_current_results = True

    if "last_embedding_compare_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Embedding Compare Results", expanded=False):
            render_embedding_compare_results(dict(st.session_state["last_embedding_compare_result"]))


def render_retrieval_fusion_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Retrieval Fusion Compare")
    st.info(
        "Change only the retrieval fusion strategy over multiple embedding-model-specific vector DBs. "
        "Parser, chunking strategy, answer model, benchmark questions, and final_top_k stay fixed."
    )

    _embedding_options, answer_options, _default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    default_embeddings = [
        model for model in EMBEDDING_MODEL_OPTIONS[:2]
    ]
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="retrieval_fusion_pdf")
    parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="retrieval_fusion_parser")
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=CHUNKING_STRATEGY_OPTIONS,
        key="retrieval_fusion_chunking",
    )
    selected_embedding_models = st.multiselect(
        "Embedding models",
        options=EMBEDDING_MODEL_OPTIONS,
        default=default_embeddings,
        key="retrieval_fusion_embedding_models",
    )
    fusion_method = st.selectbox(
        "Fusion method",
        options=["union_dedup", "rrf"],
        key="retrieval_fusion_method",
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="retrieval_fusion_answer",
    )
    per_model_top_k = st.number_input(
        "per_model_top_k",
        min_value=1,
        max_value=100,
        value=max(default_top_k, 10),
        step=1,
        key="retrieval_fusion_per_model_top_k",
    )
    final_top_k = st.number_input(
        "final_top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="retrieval_fusion_final_top_k",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)

    validation_errors = []
    if len(selected_embedding_models) < 2:
        validation_errors.append("Select at least two embedding models.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    if int(per_model_top_k) < int(final_top_k):
        st.warning("per_model_top_k is smaller than final_top_k; fused results may contain fewer final chunks.")
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Retrieval Fusion Benchmark", type="primary", key="run_retrieval_fusion"):
        errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("retrieval_fusion_compare", int(final_top_k))
        run_config.update(
            {
                "parser": PARSER_ID_BY_LABEL[parser_label],
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
                "selected_embedding_models": selected_embedding_models,
                "fusion_method": fusion_method,
                "answer_model": answer_model,
                "per_model_top_k": int(per_model_top_k),
                "final_top_k": int(final_top_k),
                "rrf_k": 60,
                "chunk_size": safe_int(
                    config.get("chunking", {}).get("chunk_size"),
                    default=800,
                ),
                "chunk_overlap": safe_int(
                    config.get("chunking", {}).get("chunk_overlap"),
                    default=150,
                ),
            }
        )
        run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
        render_saved_run(run_id, run_dir, config_path, run_config)

        progress_bar = st.progress(0)
        with st.status("Running retrieval fusion benchmark", expanded=True) as status:
            try:
                from benchmark_runner import run_retrieval_fusion_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_retrieval_fusion_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Retrieval fusion benchmark failed", state="error")
                st.error(f"Retrieval fusion benchmark failed: {exc}")
                return

            status.update(label="Retrieval fusion benchmark complete", state="complete")

        st.session_state["last_retrieval_fusion_result"] = runner_result
        render_retrieval_fusion_results(runner_result)
        rendered_current_results = True

    if "last_retrieval_fusion_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Retrieval Fusion Results", expanded=False):
            render_retrieval_fusion_results(dict(st.session_state["last_retrieval_fusion_result"]))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Configure and run controlled benchmark comparisons. Query expansion, judging, reranking, "
        "and answer-model comparison are intentionally excluded."
    )

    config, config_error = load_config(CONFIG_PATH)
    if config_error:
        st.warning(config_error)

    if BENCHMARK_QUESTIONS_PATH.exists():
        st.caption(f"Benchmark questions: `{BENCHMARK_QUESTIONS_PATH}`")
    else:
        st.warning(f"Missing benchmark questions file: {BENCHMARK_QUESTIONS_PATH}")

    parser_tab, chunking_tab, embedding_tab, retrieval_fusion_tab = st.tabs(
        [
            "Parser Compare",
            "Chunking Compare",
            "Embedding Compare",
            "Retrieval Fusion Compare",
        ]
    )

    with parser_tab:
        render_parser_compare_tab(config)

    with chunking_tab:
        render_chunking_compare_tab(config)

    with embedding_tab:
        render_embedding_compare_tab(config)

    with retrieval_fusion_tab:
        render_retrieval_fusion_compare_tab(config)


if __name__ == "__main__":
    main()
