from __future__ import annotations

import json
import hashlib
import re
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
DOCUMENT_PROFILES_PATH = OUTPUT_DIR / "document_profiles.json"
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
TABLE_AWARE_CHUNKING_STRATEGY_OPTIONS = [
    "fixed-size",
    "page-based",
    "section-aware",
    "table-aware",
    "parent-child table context",
]
TABLE_METADATA_KEYS = [
    "chunk_type",
    "table_id",
    "parent_table_id",
    "parent_table_title",
    "field_name",
    "field_aliases",
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


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "value"


def benchmark_collection_name(parser_id: str, embedding_model_name: str, run_id: str) -> str:
    digest = hashlib.sha1(f"{parser_id}|{embedding_model_name}|{run_id}".encode("utf-8")).hexdigest()
    return f"{slugify(parser_id)[:16]}_{digest[:16]}"


def benchmark_strategy_collection_name(
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


def get_default_table_handling_config() -> dict[str, bool]:
    return {
        "detect_tables": True,
        "preserve_table_markdown": True,
        "include_full_table_context": True,
        "include_nearby_context": True,
        "include_table_json": False,
        "use_parent_child_table_context": False,
    }


def render_table_handling_options(
    key_prefix: str,
    title: str = "Table Handling Options",
    show_detection_options: bool = True,
    include_answer_context_options: bool = True,
    full_context_label: str = "Include full table markdown when retrieved",
) -> dict[str, bool]:
    defaults = get_default_table_handling_config()
    st.markdown(f"#### {title}")
    detect_tables = defaults["detect_tables"]
    preserve_table_markdown = defaults["preserve_table_markdown"]
    if show_detection_options:
        detect_tables = st.checkbox(
            "Detect table-like chunks",
            value=defaults["detect_tables"],
            key=f"{key_prefix}_detect_tables",
        )
        preserve_table_markdown = st.checkbox(
            "Preserve Markdown tables if available",
            value=defaults["preserve_table_markdown"],
            key=f"{key_prefix}_preserve_table_markdown",
        )

    include_full_table_context = defaults["include_full_table_context"]
    include_nearby_context = defaults["include_nearby_context"]
    include_table_json = defaults["include_table_json"]
    if include_answer_context_options:
        include_full_table_context = st.checkbox(
            full_context_label,
            value=defaults["include_full_table_context"],
            key=f"{key_prefix}_include_full_table_context",
        )
        include_nearby_context = st.checkbox(
            "Include nearby section context",
            value=defaults["include_nearby_context"],
            key=f"{key_prefix}_include_nearby_context",
        )
        include_table_json = st.checkbox(
            "Include raw table JSON if available",
            value=defaults["include_table_json"],
            key=f"{key_prefix}_include_table_json",
        )

    return {
        "detect_tables": bool(detect_tables),
        "preserve_table_markdown": bool(preserve_table_markdown),
        "include_full_table_context": bool(include_full_table_context),
        "include_nearby_context": bool(include_nearby_context),
        "include_table_json": bool(include_table_json),
        "use_parent_child_table_context": defaults["use_parent_child_table_context"],
    }


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


def save_config_only_run_config(run_config: dict[str, Any]) -> tuple[str, Path, Path]:
    run_id = str(run_config["run_id"])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    questions_path = run_dir / "benchmark_questions.jsonl"
    questions_path.write_bytes(BENCHMARK_QUESTIONS_PATH.read_bytes())

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
        for key in TABLE_METADATA_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                chunk[key] = value

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
        score_or_distance = chunk.get("score", chunk.get("distance", chunk.get("fusion_score")))
        table_label = table_chunk_label(chunk)
        rows.append(
            {
                "rank": chunk.get("rank", chunk.get("final_rank", "")),
                "chunk_type": table_label,
                "chunk_id": chunk_id,
                "document_name": chunk.get("document_name", ""),
                "page_number": chunk.get("page_number", ""),
                "section_title": chunk.get("section_title", ""),
                "caption": chunk.get("caption", chunk.get("table_caption", "")),
                "parser_source": chunk.get("parser", chunk.get("source", "")),
                "chunk_index": chunk.get("chunk_index", ""),
                "retrieval_query_source": chunk.get(
                    "retrieval_query_source",
                    display_value(chunk.get("retrieval_query_sources", {})),
                ),
                "retrieval_query": chunk.get(
                    "retrieval_query",
                    display_value(chunk.get("retrieved_by_queries", [])),
                ),
                "score_or_distance": score_or_distance,
                "expanded_only": chunk_id in highlight_ids,
                "text_preview": preview_text(chunk.get("text", "")),
            }
        )
    return pd.DataFrame(rows)


def is_possible_table_chunk(text: Any) -> bool:
    normalized_text = str(text or "")
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) >= 2:
        return True
    if normalized_text.count("|") >= 4:
        return True
    if any(re.search(r"\|\s*:?-{3,}:?\s*\|", line) for line in lines):
        return True
    table_terms = ["Value", "Description", "Bit", "Bits", "Field", "Fields", "Reserved"]
    term_count = sum(1 for term in table_terms if re.search(rf"\b{re.escape(term)}\b", normalized_text))
    return term_count >= 2


def get_chunk_type_label(chunk: dict[str, Any]) -> str:
    chunk_type = str(chunk.get("chunk_type") or chunk.get("content_type") or "").casefold()
    if chunk_type == "table":
        return "[Table Chunk]"
    if chunk_type == "table_fragment":
        return "[Table Fragment]"
    if chunk_type == "table_field":
        return "[Table Field Chunk]"
    if chunk.get("contains_table") is True or is_possible_table_chunk(chunk.get("text", "")):
        return "[Possible Table Chunk]"
    return "[Text Chunk]"


def table_chunk_label(chunk: dict[str, Any]) -> str:
    return get_chunk_type_label(chunk)


def count_table_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    table_chunks = 0
    possible_table_chunks = 0
    text_chunks = 0
    table_pages: set[str] = set()

    for chunk in chunks:
        label = table_chunk_label(chunk)
        if label in {"[Table Chunk]", "[Table Fragment]", "[Table Field Chunk]"}:
            table_chunks += 1
        elif label == "[Possible Table Chunk]":
            possible_table_chunks += 1
        else:
            text_chunks += 1

        if label != "[Text Chunk]" and chunk.get("page_number") not in (None, ""):
            table_pages.add(str(chunk.get("page_number")))

    return {
        "total_retrieved_chunks": len(chunks),
        "table_chunks": table_chunks,
        "possible_table_chunks": possible_table_chunks,
        "text_chunks": text_chunks,
        "table_evidence_pages": ", ".join(sorted(table_pages, key=lambda value: safe_int(value, 0))) or "N/A",
    }


def render_table_summary(chunks: list[dict[str, Any]]) -> None:
    summary = count_table_chunks(chunks)
    st.markdown("##### Table-aware Summary")
    st.markdown(
        "\n".join(
            [
                f"- Total retrieved chunks: {summary['total_retrieved_chunks']}",
                f"- Table chunks: {summary['table_chunks']}",
                f"- Possible table chunks: {summary['possible_table_chunks']}",
                f"- Text chunks: {summary['text_chunks']}",
                f"- Table evidence pages: {summary['table_evidence_pages']}",
            ]
        )
    )


def table_like_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        chunk
        for chunk in chunks
        if table_chunk_label(chunk) in {
            "[Table Chunk]",
            "[Table Fragment]",
            "[Possible Table Chunk]",
            "[Table Field Chunk]",
        }
    ]


def chunk_source_parsers(chunk: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for key in ["retrieved_by_parsers", "parser_sources"]:
        value = chunk.get(key)
        if isinstance(value, list):
            sources.extend(str(item) for item in value if item not in (None, ""))
        elif value not in (None, ""):
            sources.append(str(value))
    for key in ["parser", "source_parser", "source"]:
        value = chunk.get(key)
        if value not in (None, ""):
            sources.append(str(value))
    return unique_preserve_order(sources)


def summarize_table_evidence(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed = [
        chunk
        for chunk in chunks
        if table_chunk_label(chunk) in {"[Table Chunk]", "[Table Fragment]", "[Table Field Chunk]"}
    ]
    possible = [chunk for chunk in chunks if table_chunk_label(chunk) == "[Possible Table Chunk]"]
    evidence_chunks = confirmed + possible
    pages = {
        str(chunk.get("page_number"))
        for chunk in evidence_chunks
        if chunk.get("page_number") not in (None, "")
    }
    chunk_ids = [
        str(chunk.get("chunk_id"))
        for chunk in evidence_chunks
        if chunk.get("chunk_id") not in (None, "")
    ]
    source_parsers: list[str] = []
    for chunk in evidence_chunks:
        source_parsers.extend(chunk_source_parsers(chunk))

    return {
        "confirmed_table_chunks": len(confirmed),
        "possible_table_chunks": len(possible),
        "table_evidence_pages": ", ".join(sorted(pages, key=lambda value: safe_int(value, 0))) or "N/A",
        "table_chunk_ids": ", ".join(chunk_ids) or "N/A",
        "source_parsers": ", ".join(unique_preserve_order(source_parsers)) or "N/A",
        "has_table_evidence": bool(evidence_chunks),
        "has_table_markdown": any(chunk.get("table_markdown") for chunk in evidence_chunks),
    }


def render_table_evidence_summary(chunks: list[dict[str, Any]]) -> None:
    summary = summarize_table_evidence(chunks)
    st.markdown(
        "\n".join(
            [
                f"- Confirmed table chunks: {summary['confirmed_table_chunks']}",
                f"- Possible table chunks: {summary['possible_table_chunks']}",
                f"- Table evidence pages: {summary['table_evidence_pages']}",
                f"- Table chunk IDs: {summary['table_chunk_ids']}",
                f"- Source parsers: {summary['source_parsers']}",
            ]
        )
    )
    if summary["has_table_evidence"] and not summary["has_table_markdown"]:
        st.warning("Table evidence was detected, but table_markdown is not available. Showing raw text fallback.")


def render_parser_fusion_table_diagnostics_section(
    title: str,
    chunks: list[dict[str, Any]],
    fused: bool = False,
) -> None:
    with st.expander(title, expanded=False):
        render_table_evidence_summary(chunks)
        evidence_chunks = table_like_chunks(chunks)
        if not evidence_chunks:
            st.info("No table-like evidence detected in this result set.")
            return
        for chunk in evidence_chunks:
            render_table_chunk_card(chunk, fused=fused)


def table_markdown_for_chunk(chunk: dict[str, Any]) -> str:
    return str(chunk.get("table_markdown") or chunk.get("markdown_table") or chunk.get("text") or "")


def render_table_chunk_details(chunks: list[dict[str, Any]]) -> None:
    evidence_chunks = table_like_chunks(chunks)
    if not evidence_chunks:
        return

    st.markdown("##### Table Chunk Details")
    for index, chunk in enumerate(evidence_chunks, start=1):
        label = table_chunk_label(chunk)
        chunk_id = str(chunk.get("chunk_id", f"chunk-{index}"))
        caption = chunk.get("caption") or chunk.get("table_caption") or "No caption"
        with st.expander(f"{label} {chunk_id} - {caption}", expanded=False):
            st.write(
                {
                    "chunk_id": chunk_id,
                    "page_number": chunk.get("page_number", "N/A"),
                    "section_title": chunk.get("section_title", "N/A"),
                    "caption": caption,
                    "parser_source": chunk.get("parser", chunk.get("source", "N/A")),
                    "rank": chunk.get("rank", chunk.get("final_rank", "N/A")),
                    "score_distance_or_fusion": chunk.get(
                        "score",
                        chunk.get("distance", chunk.get("fusion_score", "N/A")),
                    ),
                }
            )
            with st.expander("Show full table markdown", expanded=False):
                st.markdown(table_markdown_for_chunk(chunk))
            with st.expander("Show nearby context", expanded=False):
                nearby_context = chunk.get("nearby_context") or chunk.get("parent_context")
                if nearby_context:
                    st.markdown(str(nearby_context))
                else:
                    st.write("N/A")
            with st.expander("Show raw chunk JSON", expanded=False):
                st.json(chunk)


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
    show_table_summary: bool = True,
) -> None:
    if not chunks:
        st.info(empty_message)
        return

    if show_table_summary:
        render_table_summary(chunks)
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
    render_table_chunk_details(chunks)


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
        "table_handling": get_default_table_handling_config(),
        "uploaded_pdf_path": None,
        "benchmark_questions_path": str(BENCHMARK_QUESTIONS_PATH),
        "notes": "Configured from Streamlit Benchmark Tool. Execution is not started by this cleanup UI flow.",
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


def render_experiment_context(variable: str, fixed: list[str]) -> None:
    st.info(
        f"Variable under test: {variable}. Fixed variables: {', '.join(fixed)}."
    )
    st.caption(f"Benchmark questions path: `{BENCHMARK_QUESTIONS_PATH}`")


def render_result_area(run_dir: Path, experiment_type: str) -> None:
    st.markdown("#### Result Area")
    result_paths_by_experiment = {
        "parser_compare": [
            run_dir / "reports" / "parser_comparison_report.md",
            run_dir / "reports" / "parser_comparison.csv",
        ],
        "chunking_compare": [
            run_dir / "reports" / "chunking_comparison_report.md",
            run_dir / "reports" / "chunking_comparison.csv",
        ],
        "embedding_compare": [
            run_dir / "reports" / "embedding_comparison_report.md",
            run_dir / "reports" / "embedding_comparison.csv",
        ],
        "retrieval_fusion_compare": [
            run_dir / "fusion" / "fused_retrieval_results.jsonl",
            run_dir / "fusion" / "answer_results.jsonl",
            run_dir / "reports" / "retrieval_fusion_comparison_report.md",
        ],
        "parser_fusion_compare": [
            run_dir / "parser_fusion" / "fused_retrieval_results.jsonl",
            run_dir / "parser_fusion" / "answer_results.jsonl",
            run_dir / "reports" / "parser_fusion_comparison_report.md",
        ],
        "multi_query_retrieval_compare": [
            run_dir / "original" / "retrieval_results.jsonl",
            run_dir / "multi_query" / "fused_retrieval_results.jsonl",
            run_dir / "multi_query" / "answer_results.jsonl",
            run_dir / "reports" / "multi_query_retrieval_comparison_report.md",
        ],
    }
    existing_paths = [
        path for path in result_paths_by_experiment.get(experiment_type, []) if path.exists()
    ]
    if not existing_paths:
        st.info("Benchmark execution is not implemented or not completed for this run yet.")
        return

    for path in existing_paths:
        st.caption(f"Found result file: `{path}`")
        if path.suffix == ".md":
            markdown, error = read_markdown(path)
            if error:
                st.warning(error)
            else:
                st.markdown(markdown)
        elif path.suffix == ".csv":
            dataframe, error = read_csv(path)
            if error:
                st.warning(error)
            else:
                st.dataframe(dataframe, hide_index=True, use_container_width=True)
        elif path.suffix == ".jsonl":
            records, error = read_jsonl(path)
            if error:
                st.warning(error)
            else:
                st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


def save_configured_run(
    run_config: dict[str, Any],
    uploaded_pdf: Any,
    experiment_type: str,
) -> None:
    run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
    render_placeholder_status()
    render_saved_run(run_id, run_dir, config_path, run_config)
    render_result_area(run_dir, experiment_type)


def load_run_config(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return {}, f"Missing run_config.yaml: {config_path}"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"Could not read run_config.yaml: {exc}"
    if not isinstance(config, dict):
        return {}, f"run_config.yaml must contain a mapping: {config_path}"
    return config, None


def list_parser_fusion_runs() -> list[tuple[str, Path, dict[str, Any]]]:
    if not RUNS_DIR.exists():
        return []

    runs: list[tuple[str, Path, dict[str, Any]]] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run_config, error = load_run_config(run_dir)
        if error:
            continue
        if run_config.get("experiment_type") == "parser_fusion_compare":
            runs.append((run_dir.name, run_dir, run_config))
    return runs


def load_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    return read_jsonl(path)


def question_id_value(record: dict[str, Any]) -> str:
    return str(record.get("question_id") or record.get("id") or "")


def find_record_by_question_id(
    records: list[dict[str, Any]],
    question_id: str,
) -> dict[str, Any] | None:
    for record in records:
        if question_id_value(record) == question_id:
            return record
    return None


def record_chunks(record: dict[str, Any] | None, fused: bool = False) -> list[dict[str, Any]]:
    if not record:
        return []
    if fused:
        chunks = record.get("fused_chunks") or record.get("retrieved_chunks") or record.get("chunks") or []
    else:
        chunks = record.get("retrieved_chunks") or record.get("chunks") or []
    return chunks if isinstance(chunks, list) else []


def first_present(mapping: dict[str, Any], keys: list[str], default: Any = "N/A") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def display_value(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parser_fusion_chunk_rows(chunks: list[dict[str, Any]], fused: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if fused:
            rows.append(
                {
                    "final_rank": first_present(chunk, ["final_rank", "rank"], index),
                    "chunk_id": first_present(chunk, ["chunk_id"]),
                    "retrieved_by_parsers": display_value(
                        first_present(chunk, ["retrieved_by_parsers"], [])
                    ),
                    "original_ranks_by_parser": display_value(
                        first_present(chunk, ["original_ranks_by_parser"], {})
                    ),
                    "parser_sources": display_value(first_present(chunk, ["parser_sources"], [])),
                    "page_number": first_present(chunk, ["page_number"]),
                    "section_title": first_present(chunk, ["section_title"]),
                    "field_name": first_present(chunk, ["field_name"]),
                    "parent_table_title": first_present(chunk, ["parent_table_title"]),
                    "fusion_score": first_present(chunk, ["fusion_score", "score", "distance", "similarity"]),
                    "text_preview": preview_text(first_present(chunk, ["text"], "")),
                }
            )
        else:
            rows.append(
                {
                    "rank": first_present(chunk, ["rank"], index),
                    "chunk_id": first_present(chunk, ["chunk_id"]),
                    "page_number": first_present(chunk, ["page_number"]),
                    "section_title": first_present(chunk, ["section_title"]),
                    "field_name": first_present(chunk, ["field_name"]),
                    "parent_table_title": first_present(chunk, ["parent_table_title"]),
                    "chunk_index": first_present(chunk, ["chunk_index"]),
                    "score_or_distance": first_present(
                        chunk,
                        ["score", "distance", "similarity"],
                    ),
                    "text_preview": preview_text(first_present(chunk, ["text"], "")),
                }
            )
    return pd.DataFrame(rows)


def parser_fusion_chunk_metadata(chunk: dict[str, Any], fused: bool) -> dict[str, Any]:
    metadata = {
        "chunk_type": table_chunk_label(chunk),
        "chunk_id": first_present(chunk, ["chunk_id"]),
        "table_id": first_present(chunk, ["table_id"]),
        "parent_table_id": first_present(chunk, ["parent_table_id"]),
        "parent_table_title": first_present(chunk, ["parent_table_title"]),
        "field_name": first_present(chunk, ["field_name"]),
        "field_aliases": display_value(first_present(chunk, ["field_aliases"], [])),
        "table_group_index": first_present(chunk, ["table_group_index"]),
        "table_fragment_index": first_present(chunk, ["table_fragment_index"]),
        "table_value_codes": display_value(first_present(chunk, ["table_value_codes"], [])),
        "rank": first_present(chunk, ["final_rank", "rank"]),
        "parser/source": display_value(chunk_source_parsers(chunk)),
        "page_number": first_present(chunk, ["page_number"]),
        "section_title": first_present(chunk, ["section_title"]),
        "caption": first_present(chunk, ["caption", "table_caption"]),
        "score/distance/fusion_score": first_present(
            chunk,
            ["fusion_score", "score", "distance", "similarity"],
        ),
        "retrieval_query_source": first_present(
            chunk,
            ["retrieval_query_source", "retrieval_query_sources"],
        ),
        "retrieval_query": first_present(
            chunk,
            ["retrieval_query", "retrieved_by_queries"],
        ),
    }
    if fused:
        metadata.update(
            {
                "retrieved_by_parsers": display_value(
                    first_present(chunk, ["retrieved_by_parsers"], [])
                ),
                "parser_sources": display_value(first_present(chunk, ["parser_sources"], [])),
                "original_ranks_by_parser": display_value(
                    first_present(chunk, ["original_ranks_by_parser"], {})
                ),
            }
        )
    return metadata


def render_table_chunk_card(chunk: dict[str, Any], fused: bool = False) -> None:
    label = table_chunk_label(chunk)
    rank = first_present(chunk, ["final_rank", "rank"], "N/A")
    chunk_id = first_present(chunk, ["chunk_id"], "N/A")
    title = f"{label} rank {rank} - {chunk_id}"
    with st.expander(title, expanded=label == "[Table Chunk]"):
        st.write(parser_fusion_chunk_metadata(chunk, fused=fused))

        table_markdown = chunk.get("table_markdown") or chunk.get("markdown_table")
        parent_table_context = chunk.get("full_table_markdown") or chunk.get("parent_table_text")
        table_json = chunk.get("table_json") or chunk.get("raw_table_json")
        text = str(chunk.get("text") or "")

        if label == "[Table Field Chunk]":
            st.markdown("##### Table Field")
            st.write(
                {
                    "field_name": chunk.get("field_name", "N/A"),
                    "field_aliases": chunk.get("field_aliases", "N/A"),
                    "parent_table_title": chunk.get("parent_table_title", "N/A"),
                    "parent_table_id": chunk.get("parent_table_id", "N/A"),
                }
            )
            st.markdown("##### Field Text")
            st.write(preview_text(text, limit=1200))
        if parent_table_context:
            st.markdown("##### Full Parent Table Context")
            st.code(str(parent_table_context), language="markdown")
            with st.expander("Rendered Parent Table Context", expanded=False):
                st.markdown(str(parent_table_context))
        elif table_markdown:
            st.markdown("##### Table Content")
            st.markdown(str(table_markdown))
            with st.expander("Table Content Source", expanded=False):
                st.code(str(table_markdown), language="markdown")
        elif text:
            st.markdown("##### Table-like Text Content")
            st.code(text)
        else:
            st.info("No table_markdown or text content is available for this chunk.")

        if table_json:
            with st.expander("Raw Table JSON", expanded=False):
                st.json(table_json)

        with st.expander("Nearby Context", expanded=False):
            nearby_context = chunk.get("nearby_context") or chunk.get("parent_context")
            if nearby_context:
                st.markdown(str(nearby_context))
            else:
                st.write("N/A")

        with st.expander("Raw Chunk JSON", expanded=False):
            st.json(chunk)


def answer_table_evidence_records(answer_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not answer_record:
        return []
    explicit_evidence = answer_record.get("table_evidence_used")
    if isinstance(explicit_evidence, list):
        return [item for item in explicit_evidence if isinstance(item, dict)]

    chunks = answer_record.get("fused_chunks") or answer_record.get("retrieved_chunks") or []
    if not isinstance(chunks, list):
        return []

    evidence: list[dict[str, Any]] = []
    for chunk in table_like_chunks([item for item in chunks if isinstance(item, dict)]):
        table_markdown = chunk.get("table_markdown")
        preview_source = table_markdown or chunk.get("text") or ""
        evidence_record = {
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_type": table_chunk_label(chunk).strip("[]"),
            "table_id": chunk.get("table_id", ""),
            "parent_table_id": chunk.get("parent_table_id", ""),
            "has_full_parent_table_context": bool(chunk.get("has_full_parent_table_context")),
            "table_context_incomplete": bool(chunk.get("table_context_incomplete")),
            "page_number": chunk.get("page_number", ""),
            "section_title": chunk.get("section_title", ""),
            "source_parser": ", ".join(chunk_source_parsers(chunk)),
            "caption": chunk.get("caption", chunk.get("table_caption", "")),
            "raw_chunk": chunk,
        }
        if chunk.get("final_rank") not in (None, ""):
            evidence_record["final_rank"] = chunk.get("final_rank")
        elif chunk.get("rank") not in (None, ""):
            evidence_record["rank"] = chunk.get("rank")
        if table_markdown:
            evidence_record["table_markdown_preview"] = preview_text(preview_source, limit=360)
        else:
            evidence_record["text_preview"] = preview_text(preview_source, limit=360)
        evidence.append(evidence_record)
    return evidence


def render_table_evidence_used(answer_record: dict[str, Any] | None) -> None:
    evidence_records = answer_table_evidence_records(answer_record)
    with st.expander("Table Evidence Used in Answer", expanded=False):
        if not evidence_records:
            st.info("No table evidence was recorded or detected for this answer.")
            return
        rows = [
            {
                "chunk_id": record.get("chunk_id", "N/A"),
                "chunk_type": record.get("chunk_type", "N/A"),
                "table_id": record.get("table_id", "N/A"),
                "parent_table_id": record.get("parent_table_id", "N/A"),
                "has_full_parent_table_context": record.get("has_full_parent_table_context", "N/A"),
                "table_context_incomplete": record.get("table_context_incomplete", "N/A"),
                "page_number": record.get("page_number", "N/A"),
                "section_title": record.get("section_title", "N/A"),
                "source_parser": record.get("source_parser", "N/A"),
                "rank": record.get("final_rank", record.get("rank", "N/A")),
                "adjacent_context_chunk_ids": display_value(
                    record.get("adjacent_context_chunk_ids", [])
                ),
                "caption": record.get("caption", "N/A"),
                "table_markdown_preview": record.get(
                    "table_markdown_preview",
                    record.get("text_preview", ""),
                ),
            }
            for record in evidence_records
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for record in evidence_records:
            chunk_id = record.get("chunk_id", "N/A")
            with st.expander(f"Raw table evidence JSON - {chunk_id}", expanded=False):
                st.json(record)


def render_answer_intent(answer_record: dict[str, Any] | None) -> None:
    if not answer_record:
        return
    intent = str(answer_record.get("answer_intent") or "").strip().lower()
    if intent == "comparison":
        st.caption("Comparison Answer")
    elif intent:
        st.caption(f"Answer intent: `{intent}`")


def render_expanded_context_chunks(answer_record: dict[str, Any] | None) -> None:
    records = []
    if answer_record and isinstance(answer_record.get("expanded_context_chunks"), list):
        records = [
            item
            for item in answer_record.get("expanded_context_chunks", [])
            if isinstance(item, dict)
        ]

    with st.expander("Expanded Table Context Used", expanded=False):
        if not records:
            st.info("No expanded table context was recorded for this answer.")
            return

        rows = [
            {
                "expanded_chunk_id": record.get("chunk_id", "N/A"),
                "expanded_from_chunk_id": record.get("expanded_from_chunk_id", "N/A"),
                "context_expansion_reason": record.get("context_expansion_reason", "N/A"),
                "table_id": record.get("table_id", "N/A"),
                "parent_table_id": record.get("parent_table_id", "N/A"),
                "parent_table_title": record.get("parent_table_title", "N/A"),
                "field_name": record.get("field_name", "N/A"),
                "field_aliases": display_value(record.get("field_aliases", "")),
                "page_number": record.get("page_number", "N/A"),
                "section_title": record.get("section_title", "N/A"),
                "source_parser": record.get("source_parser", "N/A"),
                "text_preview": record.get("text_preview", ""),
                "table_markdown_preview": record.get("table_markdown_preview", ""),
            }
            for record in records
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        value_codes: list[str] = []
        grouped_chunk_ids: list[str] = []
        following_records = [
            record
            for record in records
            if record.get("context_expansion_reason") == "following_table_reference"
        ]
        for record in records:
            if record.get("chunk_id") not in (None, ""):
                grouped_chunk_ids.append(str(record.get("chunk_id")))
            preview = str(record.get("text_preview") or "")
            if record.get("table_markdown_preview"):
                preview = f"{preview}\n{record.get('table_markdown_preview')}"
            value_codes.extend(re.findall(r"\b(?:[01]{2,8}b|[0-9A-Fa-f]{2,8}h|\d{1,2}:\d{1,2})\b", preview))

        st.markdown("##### Table Group Diagnostics")
        st.markdown(
            "\n".join(
                [
                    f"- Following table expansion applied: {'Yes' if following_records else 'No'}",
                    f"- Expanded table fragments: {len(records)}",
                    f"- Expanded chunk IDs: {', '.join(unique_preserve_order(grouped_chunk_ids)) or 'N/A'}",
                    "- Expanded following table chunk IDs: "
                    f"{', '.join(unique_preserve_order([str(record.get('chunk_id')) for record in following_records if record.get('chunk_id') not in (None, '')])) or 'N/A'}",
                    f"- Detected value codes: {', '.join(unique_preserve_order(value_codes)) or 'N/A'}",
                ]
            )
        )

        for record in records:
            chunk_id = record.get("chunk_id", "N/A")
            with st.expander(f"Raw expanded context JSON - {chunk_id}", expanded=False):
                st.json(record)


def render_text_chunk_card(chunk: dict[str, Any], fused: bool = False) -> None:
    rank = first_present(chunk, ["final_rank", "rank"], "N/A")
    chunk_id = first_present(chunk, ["chunk_id"], "N/A")
    with st.expander(f"[Text Chunk] rank {rank} - {chunk_id}", expanded=False):
        st.write(parser_fusion_chunk_metadata(chunk, fused=fused))
        st.markdown("##### Text Preview")
        st.write(preview_text(chunk.get("text", "")))
        with st.expander("Raw Chunk JSON", expanded=False):
            st.json(chunk)


def render_chunk_card(chunk: dict[str, Any], fused: bool = False) -> None:
    if table_chunk_label(chunk) in {
        "[Table Chunk]",
        "[Table Fragment]",
        "[Possible Table Chunk]",
        "[Table Field Chunk]",
    }:
        render_table_chunk_card(chunk, fused=fused)
    else:
        render_text_chunk_card(chunk, fused=fused)


def render_parser_fusion_chunk_cards(chunks: list[dict[str, Any]], fused: bool = False) -> None:
    if not chunks:
        return
    st.markdown("##### Evidence Cards")
    for chunk in chunks:
        render_chunk_card(chunk, fused=fused)


def render_parser_fusion_chunk_table(
    title: str,
    record: dict[str, Any] | None,
    fused: bool = False,
) -> None:
    st.markdown(f"#### {title}")
    chunks = record_chunks(record, fused=fused)
    render_chunk_table(
        chunks,
        "No chunks found for this question.",
        show_table_summary=False,
    )


def render_raw_json_expander(title: str, record: dict[str, Any] | None) -> None:
    with st.expander(title, expanded=False):
        if record is None:
            st.info("No record found.")
        else:
            st.json(record)


def render_parser_fusion_existing_results_browser() -> None:
    st.markdown("### Existing Parser Fusion Results")
    parser_fusion_runs = list_parser_fusion_runs()
    if not parser_fusion_runs:
        st.info("No parser_fusion_compare runs were found under outputs/runs/.")
        return

    run_lookup = {run_id: (run_dir, run_config) for run_id, run_dir, run_config in parser_fusion_runs}
    selected_run_id = st.selectbox(
        "Parser Fusion run_id",
        options=list(run_lookup),
        key="parser_fusion_existing_run",
    )
    run_dir, run_config = run_lookup[selected_run_id]

    paths = {
        "PyMuPDF retrieval file missing": run_dir / "pymupdf" / "retrieval_results.jsonl",
        "Docling retrieval file missing": run_dir / "docling" / "retrieval_results.jsonl",
        "Fused retrieval file missing": run_dir / "parser_fusion" / "fused_retrieval_results.jsonl",
        "Answer results file missing": run_dir / "parser_fusion" / "answer_results.jsonl",
        "Report file missing": run_dir / "reports" / "parser_fusion_comparison_report.md",
    }
    for warning, path in paths.items():
        if not path.exists():
            st.warning(f"{warning}: `{path}`")

    pymupdf_records, pymupdf_error = load_jsonl_records(paths["PyMuPDF retrieval file missing"])
    docling_records, docling_error = load_jsonl_records(paths["Docling retrieval file missing"])
    fused_records, fused_error = load_jsonl_records(paths["Fused retrieval file missing"])
    answer_records, answer_error = load_jsonl_records(paths["Answer results file missing"])
    for error in [pymupdf_error, docling_error, fused_error, answer_error]:
        if error:
            st.warning(error)

    question_ids = sorted(
        {
            question_id_value(record)
            for record in fused_records + answer_records
            if question_id_value(record)
        }
    )
    if not question_ids:
        st.info("No question records were found in fused retrieval or answer results.")
        return

    question_text_by_id = {
        question_id_value(record): str(record.get("question", ""))
        for record in fused_records + answer_records
        if question_id_value(record)
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"parser_fusion_existing_question_{selected_run_id}",
    )

    st.markdown("#### Run Configuration")
    st.write(
        {
            "experiment_type": run_config.get("experiment_type", "N/A"),
            "selected_parsers": run_config.get("selected_parsers", "N/A"),
            "chunking_strategy": run_config.get("chunking_strategy", "N/A"),
            "embedding_model": run_config.get("embedding_model", "N/A"),
            "fusion_method": run_config.get("fusion_method", "N/A"),
            "answer_model": run_config.get("answer_model", "N/A"),
            "per_parser_top_k": run_config.get("per_parser_top_k", "N/A"),
            "final_top_k": run_config.get("final_top_k", "N/A"),
            "benchmark_questions_path": run_config.get("benchmark_questions_path", "N/A"),
        }
    )

    fused_record = find_record_by_question_id(fused_records, selected_question_id)
    answer_record = find_record_by_question_id(answer_records, selected_question_id)
    pymupdf_record = find_record_by_question_id(pymupdf_records, selected_question_id)
    docling_record = find_record_by_question_id(docling_records, selected_question_id)
    question_text = (
        (fused_record or {}).get("question")
        or (answer_record or {}).get("question")
        or (pymupdf_record or {}).get("question")
        or (docling_record or {}).get("question")
        or "N/A"
    )

    st.markdown("#### Selected Question")
    st.write({"question_id": selected_question_id, "question": question_text})

    columns = st.columns(3)
    with columns[0]:
        render_parser_fusion_chunk_table("PyMuPDF Only", pymupdf_record)
    with columns[1]:
        render_parser_fusion_chunk_table("Docling Only", docling_record)
    with columns[2]:
        render_parser_fusion_chunk_table("Fused Final Top-K", fused_record, fused=True)

    st.markdown("#### Table Evidence Diagnostics")
    st.caption(
        "This section shows table-like evidence found within the selected question's retrieval results. "
        "It is diagnostic information, not a separate table-aware benchmark."
    )
    diagnostics_columns = st.columns(3)
    with diagnostics_columns[0]:
        render_parser_fusion_table_diagnostics_section(
            "PyMuPDF table evidence",
            record_chunks(pymupdf_record),
        )
    with diagnostics_columns[1]:
        render_parser_fusion_table_diagnostics_section(
            "Docling table evidence",
            record_chunks(docling_record),
        )
    with diagnostics_columns[2]:
        render_parser_fusion_table_diagnostics_section(
            "Fused final top-k table evidence",
            record_chunks(fused_record, fused=True),
            fused=True,
        )

    st.markdown("#### Generated Answer")
    if answer_record is None:
        st.info("No answer record found for this question.")
    else:
        st.caption(f"Answer model: `{answer_record.get('answer_model', 'N/A')}`")
        render_answer_intent(answer_record)
        st.markdown(str(answer_record.get("generated_answer") or "N/A"))
        render_table_evidence_used(answer_record)
        render_expanded_context_chunks(answer_record)
        evidence = answer_record.get("evidence_used") or answer_record.get("evidence")
        if evidence:
            st.markdown("##### Evidence Used")
            st.write(evidence)

    with st.expander("Parser Fusion Report", expanded=False):
        report_path = paths["Report file missing"]
        if report_path.exists():
            report_markdown, report_error = read_markdown(report_path)
            if report_error:
                st.warning(report_error)
            else:
                st.markdown(report_markdown)
        else:
            st.info("Report file is not available for this run.")

    render_raw_json_expander("Raw PyMuPDF record", pymupdf_record)
    render_raw_json_expander("Raw Docling record", docling_record)
    render_raw_json_expander("Raw fused retrieval record", fused_record)
    render_raw_json_expander("Raw answer record", answer_record)


def safe_profile_name(value: str) -> str:
    name = value.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_") or "value"


def list_benchmark_runs() -> list[tuple[str, Path, dict[str, Any]]]:
    if not RUNS_DIR.exists():
        return []

    runs: list[tuple[str, Path, dict[str, Any]]] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run_config, error = load_run_config(run_dir)
        if error:
            continue
        runs.append((run_dir.name, run_dir, run_config))
    return runs


def multi_query_index_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_id, run_dir, run_config in list_benchmark_runs():
        chroma_path = run_dir / "chroma"
        if not chroma_path.exists():
            continue
        experiment_type = str(run_config.get("experiment_type", ""))
        chunking_strategy = str(run_config.get("chunking_strategy") or "")

        def add_candidate(
            *,
            parser: str,
            embedding_model: str,
            chunks_path: Path,
            collection_name: str,
            label_suffix: str = "",
        ) -> None:
            if not parser or not embedding_model or not chunks_path.exists():
                return
            candidates.append(
                {
                    "label": (
                        f"{run_id} | {experiment_type} | {parser} | "
                        f"{chunking_strategy} | {embedding_model}{label_suffix}"
                    ),
                    "source_run_id": run_id,
                    "uploaded_pdf_path": str(run_config.get("uploaded_pdf_path") or ""),
                    "source_chroma_path": str(chroma_path),
                    "source_chunks_path": str(chunks_path),
                    "source_collection_name": collection_name,
                    "collection_name": collection_name,
                    "parser": parser,
                    "chunking_strategy": chunking_strategy,
                    "embedding_model": embedding_model,
                    "experiment_type": experiment_type,
                }
            )

        if experiment_type in {"parser_compare", "parser_fusion_compare"}:
            embedding_model = str(run_config.get("embedding_model") or "")
            parsers = run_config.get("selected_parsers") or []
            for parser in parsers:
                add_candidate(
                    parser=str(parser),
                    embedding_model=embedding_model,
                    chunks_path=run_dir / str(parser) / "chunks.jsonl",
                    collection_name=(
                        benchmark_collection_name(str(parser), embedding_model, run_id)
                        if experiment_type == "parser_compare"
                        else benchmark_strategy_collection_name(
                            run_id,
                            str(parser),
                            chunking_strategy,
                            embedding_model,
                        )
                    ),
                )
        elif experiment_type == "embedding_compare":
            parser = str(run_config.get("parser") or "")
            for model in run_config.get("selected_embedding_models") or []:
                add_candidate(
                    parser=parser,
                    embedding_model=str(model),
                    chunks_path=run_dir / "chunks" / "chunks.jsonl",
                    collection_name=benchmark_strategy_collection_name(
                        run_id,
                        parser,
                        chunking_strategy,
                        str(model),
                    ),
                    label_suffix=f" | {safe_profile_name(str(model))}",
                )
        elif experiment_type == "retrieval_fusion_compare":
            parser = str(run_config.get("parser") or "")
            for model in run_config.get("selected_embedding_models") or []:
                add_candidate(
                    parser=parser,
                    embedding_model=str(model),
                    chunks_path=run_dir / "chunks" / "chunks.jsonl",
                    collection_name=benchmark_strategy_collection_name(
                        run_id,
                        parser,
                        chunking_strategy,
                        str(model),
                    ),
                    label_suffix=f" | {safe_profile_name(str(model))}",
                )
        elif experiment_type == "chunking_compare":
            parser = str(run_config.get("parser") or "")
            embedding_model = str(run_config.get("embedding_model") or "")
            for strategy in run_config.get("selected_chunking_strategies") or []:
                old_strategy = chunking_strategy
                chunking_strategy = str(strategy)
                add_candidate(
                    parser=parser,
                    embedding_model=embedding_model,
                    chunks_path=run_dir / str(strategy) / "chunks.jsonl",
                    collection_name=benchmark_strategy_collection_name(
                        run_id,
                        parser,
                        str(strategy),
                        embedding_model,
                    ),
                )
                chunking_strategy = old_strategy
    return candidates


def build_multi_query_preview(
    benchmark_questions: list[dict[str, Any]],
    max_sub_queries: int,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    from benchmark_runner import decompose_query_rule_based, reject_sub_query_reason

    accepted_by_question: dict[str, list[str]] = {}
    rejected_by_question: dict[str, list[dict[str, str]]] = {}
    preview_by_question: dict[str, Any] = {}

    st.markdown("#### Controlled Query Expansion Preview")
    st.caption(
        "Review conservative phrase-level sub-queries before retrieval. "
        "The original query is always included, and unchecked phrases are saved as rejected."
    )

    for question in benchmark_questions:
        question_id = str(question.get("id", ""))
        question_text = str(question.get("question", ""))
        if not question_id or not question_text:
            continue
        decomposition = decompose_query_rule_based(question_text, max_sub_queries=int(max_sub_queries))
        generated = [str(query) for query in decomposition.get("generated_sub_queries", [])]
        rejected = [
            item for item in decomposition.get("rejected_sub_queries", [])
            if isinstance(item, dict)
        ]
        anchors = [str(anchor) for anchor in decomposition.get("anchors", [])]
        required_anchors = [str(anchor) for anchor in decomposition.get("required_anchors", [])]
        accepted: list[str] = []

        with st.expander(f"{question_id}: {question_text}", expanded=False):
            st.write(
                {
                    "required_anchors": required_anchors,
                    "detected_anchors": anchors,
                }
            )
            for index, query in enumerate(generated):
                source = decomposition.get("query_sources", {}).get(query, "phrase_decomposition")
                rejection_reason = reject_sub_query_reason(query, question_text, anchors)
                is_original = source == "original" or query.casefold() == question_text.casefold()
                if rejection_reason and not is_original:
                    st.warning(f"Rejected: {query} ({rejection_reason})")
                    rejected.append({"query": query, "reason": rejection_reason})
                    continue
                checked = st.checkbox(
                    f"{query} [{source}]",
                    value=True,
                    disabled=is_original,
                    key=f"multi_query_accept_{question_id}_{index}",
                )
                if checked or is_original:
                    accepted.append(query)
                else:
                    rejected.append({"query": query, "reason": "user_unchecked"})

            if rejected:
                with st.expander("Rejected or unchecked sub-queries", expanded=False):
                    st.write(rejected)

        accepted_by_question[question_id] = unique_preserve_order(accepted)
        rejected_by_question[question_id] = rejected
        preview_by_question[question_id] = {
            "question": question_text,
            "accepted_sub_queries": accepted_by_question[question_id],
            "rejected_sub_queries": rejected,
            "anchors": anchors,
            "required_anchors": required_anchors,
            "query_sources": decomposition.get("query_sources", {}),
        }

    return accepted_by_question, rejected_by_question, preview_by_question


def discover_answer_result_files(
    run_dir: Path,
    run_config: dict[str, Any],
) -> list[tuple[str, Path]]:
    experiment_type = str(run_config.get("experiment_type", ""))

    if experiment_type == "parser_compare":
        parsers = run_config.get("selected_parsers") or ["pymupdf", "docling"]
        return [(str(parser_id), run_dir / str(parser_id) / "answer_results.jsonl") for parser_id in parsers]

    if experiment_type == "chunking_compare":
        strategies = run_config.get("selected_chunking_strategies") or []
        return [
            (str(strategy), run_dir / str(strategy) / "answer_results.jsonl")
            for strategy in strategies
        ]

    if experiment_type == "embedding_compare":
        models = run_config.get("selected_embedding_models") or []
        return [
            (
                safe_profile_name(str(model_name)),
                run_dir / "embeddings" / safe_profile_name(str(model_name)) / "answer_results.jsonl",
            )
            for model_name in models
        ]

    if experiment_type == "retrieval_fusion_compare":
        return [("fusion result", run_dir / "fusion" / "answer_results.jsonl")]

    if experiment_type == "parser_fusion_compare":
        return [("parser fusion result", run_dir / "parser_fusion" / "answer_results.jsonl")]

    if experiment_type == "multi_query_retrieval_compare":
        return [("multi-query result", run_dir / "multi_query" / "answer_results.jsonl")]

    return []


def load_answer_results_for_run(
    run_dir: Path,
    run_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for case_name, path in discover_answer_result_files(run_dir, run_config):
        if not path.exists():
            warnings.append(f"Missing answer result file for {case_name}: {path}")
            continue
        case_records, error = load_jsonl_records(path)
        if error:
            warnings.append(error)
            continue
        for record in case_records:
            enriched = dict(record)
            enriched["case_name"] = case_name
            enriched["answer_results_path"] = str(path)
            records.append(enriched)
    return records, warnings


def group_answers_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        question_id = question_id_value(record)
        if question_id:
            grouped[question_id].append(record)
    return dict(grouped)


def make_answer_preview(answer: Any, limit: int = 260) -> str:
    return preview_text(answer, limit=limit)


def answer_context_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = record.get("retrieved_chunks") or record.get("fused_chunks") or record.get("chunks") or []
    return chunks if isinstance(chunks, list) else []


def answer_comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "question_id": question_id_value(record),
                "case_name": record.get("case_name", "N/A"),
                "parser": record.get("parser", "N/A"),
                "chunking_strategy": record.get("chunking_strategy", "N/A"),
                "embedding_model": record.get("embedding_model", "N/A"),
                "retrieval_strategy": record.get("retrieval_strategy", "N/A"),
                "fusion_method": record.get("fusion_method", "N/A"),
                "answer_model": record.get("answer_model", "N/A"),
                "answer_preview": make_answer_preview(record.get("generated_answer", "")),
            }
        )
    return rows


def load_document_profiles() -> dict[str, Any]:
    if not DOCUMENT_PROFILES_PATH.exists():
        return {"documents": []}
    try:
        profiles = json.loads(DOCUMENT_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"documents": []}
    if not isinstance(profiles, dict):
        return {"documents": []}
    documents = profiles.get("documents")
    if not isinstance(documents, list):
        profiles["documents"] = []
    return profiles


def save_document_profiles(profiles: dict[str, Any]) -> None:
    DOCUMENT_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_PROFILES_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def upsert_document_profile(profile: dict[str, Any]) -> None:
    profiles = load_document_profiles()
    documents = profiles.setdefault("documents", [])
    document_id = str(profile["document_id"])
    for index, existing in enumerate(documents):
        if str(existing.get("document_id")) == document_id:
            documents[index] = profile
            save_document_profiles(profiles)
            return
    documents.append(profile)
    save_document_profiles(profiles)


def saved_profile_rows(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in profiles.get("documents", []):
        selected_profile = document.get("selected_profile", {})
        rows.append(
            {
                "document_id": document.get("document_id", ""),
                "file_name": document.get("file_name", ""),
                "source_run_id": document.get("source_run_id", ""),
                "parser_mode": selected_profile.get("parser_mode", ""),
                "chunking_strategy": selected_profile.get("chunking_strategy", ""),
                "embedding_mode": selected_profile.get("embedding_mode", ""),
                "retrieval_strategy": selected_profile.get("retrieval_strategy", ""),
                "fusion_method": selected_profile.get("fusion_method", ""),
                "answer_model": selected_profile.get("answer_model", ""),
                "selected_at": document.get("selected_at", ""),
                "notes_preview": preview_text(document.get("notes", ""), limit=120),
            }
        )
    return rows


def profile_defaults(run_id: str, run_config: dict[str, Any]) -> dict[str, Any]:
    experiment_type = str(run_config.get("experiment_type", ""))
    uploaded_pdf_path = str(run_config.get("uploaded_pdf_path", ""))
    file_name = Path(uploaded_pdf_path).name if uploaded_pdf_path else ""
    selected_parsers = run_config.get("selected_parsers") or (
        [run_config["parser"]] if run_config.get("parser") else []
    )
    selected_embedding_models = run_config.get("selected_embedding_models") or (
        [run_config["embedding_model"]] if run_config.get("embedding_model") else []
    )

    parser_mode = "single_parser"
    embedding_mode = "single_embedding"
    retrieval_strategy = str(run_config.get("retrieval_strategy", DEFAULT_RETRIEVAL_STRATEGY))
    if experiment_type == "parser_fusion_compare":
        parser_mode = "parser_fusion"
        retrieval_strategy = "parser_fusion"
    if experiment_type == "retrieval_fusion_compare":
        embedding_mode = "multi_embedding_fusion"
        retrieval_strategy = "multi_embedding_fusion"
    if experiment_type == "multi_query_retrieval_compare":
        retrieval_strategy = "multi_query_rrf"

    return {
        "document_id": Path(file_name).stem or run_id,
        "file_name": file_name,
        "source_run_id": run_id,
        "parser_mode": parser_mode,
        "selected_parsers": selected_parsers,
        "chunking_strategy": run_config.get("chunking_strategy", ""),
        "embedding_mode": embedding_mode,
        "embedding_model": run_config.get("embedding_model", ""),
        "selected_embedding_models": selected_embedding_models,
        "retrieval_strategy": retrieval_strategy,
        "fusion_method": run_config.get("fusion_method", ""),
        "top_k": run_config.get("top_k", run_config.get("final_top_k", "")),
        "per_model_top_k": run_config.get("per_model_top_k", ""),
        "per_parser_top_k": run_config.get("per_parser_top_k", ""),
        "per_query_top_k": run_config.get("per_query_top_k", ""),
        "final_top_k": run_config.get("final_top_k", ""),
        "max_sub_queries": run_config.get("max_sub_queries", ""),
        "answer_model": run_config.get("answer_model", ""),
        "table_handling": run_config.get("table_handling", get_default_table_handling_config()),
        "notes": "",
    }


def render_run_config_summary(run_config: dict[str, Any]) -> None:
    summary_keys = [
        "run_id",
        "experiment_type",
        "uploaded_pdf_path",
        "benchmark_questions_path",
        "parser",
        "selected_parsers",
        "chunking_strategy",
        "selected_chunking_strategies",
        "embedding_model",
        "selected_embedding_models",
        "retrieval_strategy",
        "use_existing_index",
        "source_index",
        "fusion_method",
        "table_handling",
        "answer_model",
        "top_k",
        "per_model_top_k",
        "per_parser_top_k",
        "per_query_top_k",
        "max_sub_queries",
        "final_top_k",
    ]
    st.write({key: run_config.get(key, "N/A") for key in summary_keys if key in run_config})


def render_answer_case(case_record: dict[str, Any]) -> None:
    st.markdown(f"#### {case_record.get('case_name', 'N/A')}")
    st.write(
        {
            "question_id": question_id_value(case_record) or "N/A",
            "question": case_record.get("question", "N/A"),
            "parser": case_record.get("parser", "N/A"),
            "chunking_strategy": case_record.get("chunking_strategy", "N/A"),
            "embedding_model": case_record.get("embedding_model", "N/A"),
            "fusion_method": case_record.get("fusion_method", "N/A"),
            "answer_model": case_record.get("answer_model", "N/A"),
        }
    )
    st.markdown("##### Answer Preview")
    st.write(make_answer_preview(case_record.get("generated_answer", "")))
    with st.expander("Full generated answer", expanded=False):
        render_answer_intent(case_record)
        st.markdown(str(case_record.get("generated_answer") or "N/A"))
    render_table_evidence_used(case_record)
    render_expanded_context_chunks(case_record)
    with st.expander("Retrieved or fused chunks", expanded=False):
        chunks = answer_context_chunks(case_record)
        if chunks:
            render_chunk_table(chunks, "No chunks found.")
        else:
            st.info("No retrieved_chunks or fused_chunks field found for this answer record.")


def run_config_list_value(run_config: dict[str, Any], plural_key: str, singular_key: str) -> list[str]:
    values = run_config.get(plural_key)
    if isinstance(values, list):
        return [str(value) for value in values]
    value = run_config.get(singular_key)
    return [str(value)] if value not in (None, "") else []


def render_best_profile_form(run_id: str, run_config: dict[str, Any]) -> None:
    defaults = profile_defaults(run_id, run_config)
    selected_parsers_from_config = run_config_list_value(
        run_config,
        "selected_parsers",
        "parser",
    )
    selected_embeddings_from_config = run_config_list_value(
        run_config,
        "selected_embedding_models",
        "embedding_model",
    )
    chunking_options = unique_preserve_order(
        CHUNKING_STRATEGY_OPTIONS
        + [str(run_config.get("chunking_strategy", ""))]
        + [str(value) for value in run_config.get("selected_chunking_strategies", [])]
    )
    chunking_options = [value for value in chunking_options if value]
    embedding_options = unique_preserve_order(
        EMBEDDING_MODEL_OPTIONS
        + selected_embeddings_from_config
        + [str(defaults.get("embedding_model", ""))]
    )
    embedding_options = [value for value in embedding_options if value]
    answer_options = unique_preserve_order(
        ANSWER_MODEL_OPTIONS + [str(defaults.get("answer_model", ""))]
    )
    answer_options = [value for value in answer_options if value]
    parser_options = unique_preserve_order(["pymupdf", "docling"] + selected_parsers_from_config)
    table_handling = defaults.get("table_handling", get_default_table_handling_config())
    if not isinstance(table_handling, dict):
        table_handling = get_default_table_handling_config()

    st.markdown("### Manual Best Profile Selection")
    with st.form(f"best_profile_form_{run_id}"):
        document_id = st.text_input("document_id", value=str(defaults["document_id"]))
        file_name = st.text_input("file_name", value=str(defaults["file_name"]))
        source_run_id = st.text_input("source_run_id", value=str(defaults["source_run_id"]))
        parser_mode = st.selectbox(
            "parser_mode",
            options=["single_parser", "parser_fusion"],
            index=default_index(["single_parser", "parser_fusion"], str(defaults["parser_mode"])),
        )
        selected_parsers = st.multiselect(
            "selected_parsers",
            options=parser_options,
            default=[value for value in selected_parsers_from_config if value in parser_options],
        )
        chunking_strategy = st.selectbox(
            "chunking_strategy",
            options=chunking_options or CHUNKING_STRATEGY_OPTIONS,
            index=default_index(chunking_options or CHUNKING_STRATEGY_OPTIONS, str(defaults["chunking_strategy"])),
        )
        embedding_mode = st.selectbox(
            "embedding_mode",
            options=["single_embedding", "multi_embedding_fusion"],
            index=default_index(
                ["single_embedding", "multi_embedding_fusion"],
                str(defaults["embedding_mode"]),
            ),
        )
        embedding_model = st.selectbox(
            "embedding_model",
            options=embedding_options or EMBEDDING_MODEL_OPTIONS,
            index=default_index(embedding_options or EMBEDDING_MODEL_OPTIONS, str(defaults["embedding_model"])),
        )
        selected_embedding_models = st.multiselect(
            "selected_embedding_models",
            options=embedding_options or EMBEDDING_MODEL_OPTIONS,
            default=[
                value
                for value in selected_embeddings_from_config
                if value in (embedding_options or EMBEDDING_MODEL_OPTIONS)
            ],
        )
        retrieval_strategy = st.text_input(
            "retrieval_strategy",
            value=str(defaults["retrieval_strategy"]),
        )
        fusion_method = st.selectbox(
            "fusion_method",
            options=unique_preserve_order(
                ["", "union_dedup", "rrf", str(defaults.get("fusion_method", ""))]
            ),
            index=default_index(
                unique_preserve_order(["", "union_dedup", "rrf", str(defaults.get("fusion_method", ""))]),
                str(defaults.get("fusion_method", "")),
            ),
        )
        top_k = st.number_input("top_k", min_value=0, value=safe_int(defaults["top_k"], 0), step=1)
        per_model_top_k = st.number_input(
            "per_model_top_k",
            min_value=0,
            value=safe_int(defaults["per_model_top_k"], 0),
            step=1,
        )
        per_parser_top_k = st.number_input(
            "per_parser_top_k",
            min_value=0,
            value=safe_int(defaults["per_parser_top_k"], 0),
            step=1,
        )
        final_top_k = st.number_input(
            "final_top_k",
            min_value=0,
            value=safe_int(defaults["final_top_k"], 0),
            step=1,
        )
        answer_model = st.selectbox(
            "answer_model",
            options=answer_options or ANSWER_MODEL_OPTIONS,
            index=default_index(answer_options or ANSWER_MODEL_OPTIONS, str(defaults["answer_model"])),
        )
        st.markdown("#### table_handling")
        st.json(table_handling)
        notes = st.text_area("notes", value=str(defaults["notes"]))
        submitted = st.form_submit_button("Save Best Profile", type="primary")

    if submitted:
        profile = {
            "document_id": document_id,
            "file_name": file_name,
            "source_run_id": source_run_id,
            "selected_profile": {
                "parser_mode": parser_mode,
                "selected_parsers": selected_parsers,
                "chunking_strategy": chunking_strategy,
                "embedding_mode": embedding_mode,
                "embedding_model": embedding_model,
                "selected_embedding_models": selected_embedding_models,
                "retrieval_strategy": retrieval_strategy,
                "fusion_method": fusion_method,
                "top_k": int(top_k),
                "per_model_top_k": int(per_model_top_k),
                "per_parser_top_k": int(per_parser_top_k),
                "final_top_k": int(final_top_k),
                "answer_model": answer_model,
                "table_handling": table_handling,
            },
            "selected_by": "manual",
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        upsert_document_profile(profile)
        st.success(f"Saved best profile for document_id `{document_id}`.")


def render_saved_profiles() -> None:
    st.markdown("### Saved Profiles")
    profiles = load_document_profiles()
    rows = saved_profile_rows(profiles)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No saved document profiles yet.")
    with st.expander("Raw saved document_profiles.json", expanded=False):
        st.json(profiles)


def render_best_profile_manager_tab() -> None:
    st.subheader("Best Profile Manager")
    st.caption(
        "Review generated answers from completed benchmark runs, then manually save the "
        "document profile you want to use later."
    )

    benchmark_runs = list_benchmark_runs()
    if not benchmark_runs:
        st.info("No benchmark runs with run_config.yaml were found under outputs/runs/.")
        render_saved_profiles()
        return

    run_labels = {
        f"{run_id} ({run_config.get('experiment_type', 'unknown')})": (run_id, run_dir, run_config)
        for run_id, run_dir, run_config in benchmark_runs
    }
    selected_label = st.selectbox(
        "Benchmark run",
        options=list(run_labels),
        key="best_profile_run_selector",
    )
    selected_run_id, selected_run_dir, selected_run_config = run_labels[selected_label]

    st.markdown("### Run Config Summary")
    render_run_config_summary(selected_run_config)

    answer_records, answer_warnings = load_answer_results_for_run(
        selected_run_dir,
        selected_run_config,
    )
    for warning in answer_warnings:
        st.warning(warning)

    grouped_answers = group_answers_by_question(answer_records)
    if grouped_answers:
        question_text_by_id = {
            question_id: str(records[0].get("question", ""))
            for question_id, records in grouped_answers.items()
            if records
        }
        selected_question_id = st.selectbox(
            "Question",
            options=sorted(grouped_answers),
            format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
            key=f"best_profile_question_{selected_run_id}",
        )
        selected_records = grouped_answers[selected_question_id]

        st.markdown("### Selected Question")
        st.write(
            {
                "question_id": selected_question_id,
                "question": question_text_by_id.get(selected_question_id, "N/A"),
            }
        )

        st.markdown("### Answer Comparison Table")
        st.dataframe(
            pd.DataFrame(answer_comparison_rows(selected_records)),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Answer Cases")
        for case_record in selected_records:
            render_answer_case(case_record)
    else:
        st.info("No generated answer results were found for this run.")

    render_best_profile_form(selected_run_id, selected_run_config)
    render_saved_profiles()

    with st.expander("Raw run_config.yaml", expanded=False):
        st.code(yaml.safe_dump(selected_run_config, sort_keys=False), language="yaml")
    with st.expander("Raw answer result records for selected run", expanded=False):
        st.json(answer_records)


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
            render_answer_intent(record)
            st.markdown(str(record.get("generated_answer", "")))
            render_table_evidence_used(record)
            render_expanded_context_chunks(record)
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
            render_answer_intent(record)
            st.markdown(str(record.get("generated_answer", "")))
            render_table_evidence_used(record)
            render_expanded_context_chunks(record)
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
            render_answer_intent(record)
            st.markdown(str(record.get("generated_answer", "")))
            render_table_evidence_used(record)
            render_expanded_context_chunks(record)
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
    render_answer_intent(answer_record)
    st.markdown(str(answer_record.get("generated_answer", "")))
    render_table_evidence_used(answer_record)
    render_expanded_context_chunks(answer_record)

    st.markdown("#### Fused Chunks")
    fused_chunks = fused_record.get("fused_chunks", [])
    render_chunk_table(fused_chunks, "No fused chunks found for this question.")


def render_parser_fusion_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Parser Fusion Compare Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "chroma_dir": runner_result.get("chroma_dir"),
            "selected_parsers": runner_result.get("selected_parsers"),
            "chunking_strategy": runner_result.get("chunking_strategy"),
            "embedding_model": runner_result.get("embedding_model"),
            "fusion_method": runner_result.get("fusion_method"),
            "reports": runner_result.get("reports"),
        }
    )

    parser_warnings = {
        parser_id: warnings
        for parser_id, warnings in dict(runner_result.get("parser_warnings", {})).items()
        if warnings
    }
    if parser_warnings:
        st.warning("Some parsers emitted warnings.")
        st.write(parser_warnings)

    fusion_results = dict(runner_result.get("fusion_results", {}))
    fused_records, fused_error = read_jsonl(Path(str(fusion_results.get("fused_retrieval_results", ""))))
    answer_records, answer_error = read_jsonl(Path(str(fusion_results.get("answer_results", ""))))
    if fused_error:
        st.warning(fused_error)
    if answer_error:
        st.warning(answer_error)
    if not fused_records:
        st.info("No fused parser retrieval results were found for this run.")
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
        key=f"parser_fusion_result_question_{runner_result.get('run_id')}",
    )
    fused_record = next(
        record for record in fused_records if str(record.get("question_id")) == selected_question_id
    )
    answer_record = answers_by_question.get(selected_question_id, {})

    st.markdown("#### Generated Answer")
    render_answer_intent(answer_record)
    st.markdown(str(answer_record.get("generated_answer", "")))
    render_table_evidence_used(answer_record)
    render_expanded_context_chunks(answer_record)

    st.markdown("#### Fused Chunks")
    fused_chunks = fused_record.get("fused_chunks", [])
    render_chunk_table(fused_chunks, "No fused chunks found for this question.")


def render_multi_query_retrieval_results(runner_result: dict[str, Any]) -> None:
    st.subheader("Controlled Query Expansion Results")
    st.write(
        {
            "run_id": runner_result.get("run_id"),
            "run_dir": runner_result.get("run_dir"),
            "parser": runner_result.get("parser"),
            "chunking_strategy": runner_result.get("chunking_strategy"),
            "embedding_model": runner_result.get("embedding_model"),
            "use_existing_index": runner_result.get("use_existing_index"),
            "source_index": runner_result.get("source_index"),
            "reports": runner_result.get("reports"),
        }
    )
    if runner_result.get("warnings"):
        st.warning("Some extraction or chunking warnings were recorded.")
        st.write(runner_result.get("warnings"))

    original_results = dict(runner_result.get("original_results", {}))
    multi_query_results = dict(runner_result.get("multi_query_results", {}))
    original_records, original_error = read_jsonl(Path(str(original_results.get("retrieval_results", ""))))
    sub_query_records, sub_query_error = read_jsonl(
        Path(str(multi_query_results.get("sub_query_retrieval_results", "")))
    )
    fused_records, fused_error = read_jsonl(Path(str(multi_query_results.get("fused_retrieval_results", ""))))
    answer_records, answer_error = read_jsonl(Path(str(multi_query_results.get("answer_results", ""))))
    for error in [original_error, sub_query_error, fused_error, answer_error]:
        if error:
            st.warning(error)
    if not fused_records:
        st.info("No multi-query fused retrieval results were found for this run.")
        return

    answers_by_question = {str(record.get("question_id")): record for record in answer_records}
    original_by_question = {str(record.get("question_id")): record for record in original_records}
    question_ids = [str(record.get("question_id")) for record in fused_records]
    question_text_by_id = {
        str(record.get("question_id")): str(record.get("question", ""))
        for record in fused_records
    }
    selected_question_id = st.selectbox(
        "Question",
        options=question_ids,
        format_func=lambda question_id: f"{question_id} - {question_text_by_id.get(question_id, '')}",
        key=f"multi_query_result_question_{runner_result.get('run_id')}",
    )
    fused_record = next(
        record for record in fused_records if str(record.get("question_id")) == selected_question_id
    )
    original_record = original_by_question.get(selected_question_id, {})
    answer_record = answers_by_question.get(selected_question_id, {})

    st.markdown("#### Generated Answer")
    render_answer_intent(answer_record)
    st.markdown(str(answer_record.get("generated_answer", "")))
    render_table_evidence_used(answer_record)
    render_expanded_context_chunks(answer_record)

    st.markdown("#### Sub-queries")
    st.write(
        {
            "original_query": fused_record.get("original_query", fused_record.get("question", "")),
            "generated_sub_queries": fused_record.get(
                "generated_sub_queries",
                fused_record.get("sub_queries", []),
            ),
            "accepted_sub_queries": fused_record.get("accepted_sub_queries", []),
            "rejected_sub_queries": fused_record.get("rejected_sub_queries", []),
            "query_anchors": fused_record.get("query_anchors", []),
            "required_anchors": fused_record.get("required_anchors", []),
            "query_sources": fused_record.get("query_sources", {}),
        }
    )

    columns = st.columns(2)
    with columns[0]:
        st.markdown("#### Original Single-query Top-K")
        render_chunk_table(
            original_record.get("retrieved_chunks", []),
            "No original single-query chunks found.",
        )
    with columns[1]:
        st.markdown("#### Multi-query RRF Final Top-K")
        render_chunk_table(
            fused_record.get("fused_chunks", []),
            "No fused multi-query chunks found.",
        )

    with st.expander("Sub-query Retrieval Debug", expanded=False):
        question_sub_records = [
            record for record in sub_query_records if str(record.get("question_id")) == selected_question_id
        ]
        rows = []
        for record in question_sub_records:
            rows.append(
                {
                    "retrieval_query_index": record.get("retrieval_query_index", "N/A"),
                    "retrieval_query": record.get("retrieval_query", record.get("sub_query", "")),
                    "retrieval_query_source": record.get("retrieval_query_source", "N/A"),
                    "retrieved_chunk_count": len(record.get("retrieved_chunks", [])),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for record in question_sub_records:
            label = (
                f"{record.get('retrieval_query_index', 'N/A')}: "
                f"{record.get('retrieval_query', record.get('sub_query', ''))}"
            )
            with st.expander(label, expanded=False):
                render_chunk_table(
                    record.get("retrieved_chunks", []),
                    "No chunks found for this sub-query.",
                )


def render_parser_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Parser Compare")
    render_experiment_context(
        variable="parser",
        fixed=[
            "chunking_strategy",
            "embedding_model",
            "answer_model",
            "top_k",
            "benchmark_questions_path",
        ],
    )

    embedding_options, answer_options, default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="parser_compare_pdf")
    selected_parsers = selected_parser_ids_from_checkboxes("parser_compare_parser")
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=CHUNKING_STRATEGY_OPTIONS,
        key="parser_compare_chunking",
    )
    table_handling = render_table_handling_options(
        "parser_compare_table",
        title="Table Handling Options",
        include_answer_context_options=True,
        full_context_label="Include full table context in answer generation",
    )
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
                "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
                "table_handling": table_handling,
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
        save_configured_run(run_config, uploaded_pdf, "parser_compare")


def render_chunking_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Chunking Compare")
    render_experiment_context(
        variable="chunking_strategy",
        fixed=[
            "parser",
            "embedding_model",
            "answer_model",
            "top_k",
            "benchmark_questions_path",
        ],
    )

    embedding_options, answer_options, default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="chunking_compare_pdf")
    parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="chunking_compare_parser")
    selected_chunking_strategies = selected_options_from_checkboxes(
        "Chunking strategies",
        TABLE_AWARE_CHUNKING_STRATEGY_OPTIONS,
        ["fixed-size", "page-based"],
        "chunking_compare_strategy",
    )
    table_handling = get_default_table_handling_config()
    if "parent-child table context" in selected_chunking_strategies:
        table_handling["use_parent_child_table_context"] = True
    if "table-aware" in selected_chunking_strategies:
        table_handling["detect_tables"] = True
        table_handling["preserve_table_markdown"] = True
        table_handling["include_full_table_context"] = True
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
                "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
                "table_handling": table_handling,
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
        save_configured_run(run_config, uploaded_pdf, "chunking_compare")


def render_embedding_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Embedding Compare")
    render_experiment_context(
        variable="embedding_model",
        fixed=[
            "parser",
            "chunking_strategy",
            "answer_model",
            "top_k",
            "benchmark_questions_path",
        ],
    )

    _embedding_options, answer_options, _default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    default_embeddings = [
        model for model in EMBEDDING_MODEL_OPTIONS[:2]
    ]
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="embedding_compare_pdf")
    parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="embedding_compare_parser")
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=CHUNKING_STRATEGY_OPTIONS,
        key="embedding_compare_chunking",
    )
    selected_embedding_models = selected_options_from_checkboxes(
        "Embedding models",
        EMBEDDING_MODEL_OPTIONS,
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
                "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
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
        save_configured_run(run_config, uploaded_pdf, "embedding_compare")


def render_retrieval_fusion_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Retrieval Fusion Compare")
    render_experiment_context(
        variable="retrieval fusion strategy",
        fixed=[
            "parser",
            "chunking_strategy",
            "answer_model",
            "benchmark_questions_path",
        ],
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
    selected_embedding_models = selected_options_from_checkboxes(
        "Embedding models",
        EMBEDDING_MODEL_OPTIONS,
        default_embeddings,
        "retrieval_fusion_embedding_model",
    )
    fusion_method = st.selectbox(
        "Fusion method",
        options=["union_dedup", "rrf"],
        key="retrieval_fusion_method",
    )
    table_handling = render_table_handling_options(
        "retrieval_fusion_table",
        title="Answer Context Options",
        show_detection_options=False,
        include_answer_context_options=True,
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
    if int(final_top_k) > int(per_model_top_k) * max(len(selected_embedding_models), 1):
        st.warning(
            "final_top_k is larger than per_model_top_k multiplied by the number of embedding models; "
            "fused results may contain fewer final chunks."
        )
    for error in validation_errors:
        st.warning(error)

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
                "retrieval_strategy": "fusion",
                "selected_embedding_models": selected_embedding_models,
                "fusion_method": fusion_method,
                "table_handling": table_handling,
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
        save_configured_run(run_config, uploaded_pdf, "retrieval_fusion_compare")


def render_parser_fusion_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Parser Fusion Compare")
    render_experiment_context(
        variable="parser retrieval strategy",
        fixed=[
            "chunking_strategy",
            "embedding_model",
            "answer_model",
            "benchmark_questions_path",
            "final_top_k",
            "retrieval_strategy",
        ],
    )
    render_parser_fusion_existing_results_browser()
    st.markdown("### Configure New Parser Fusion Run")

    _embedding_options, answer_options, _default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], key="parser_fusion_pdf")
    selected_parsers = selected_parser_ids_from_checkboxes("parser_fusion_parser")
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=CHUNKING_STRATEGY_OPTIONS,
        key="parser_fusion_chunking",
    )
    embedding_model = st.selectbox(
        "Embedding model",
        options=EMBEDDING_MODEL_OPTIONS,
        key="parser_fusion_embedding",
    )
    fusion_method = st.selectbox(
        "Fusion method",
        options=["union_dedup", "rrf"],
        key="parser_fusion_method",
    )
    table_handling = render_table_handling_options(
        "parser_fusion_table",
        title="Answer Context Options",
        show_detection_options=False,
        include_answer_context_options=True,
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="parser_fusion_answer",
    )
    per_parser_top_k = st.number_input(
        "per_parser_top_k",
        min_value=1,
        max_value=100,
        value=max(default_top_k, 10),
        step=1,
        key="parser_fusion_per_parser_top_k",
    )
    final_top_k = st.number_input(
        "final_top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="parser_fusion_final_top_k",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)

    validation_errors = []
    if len(selected_parsers) < 2:
        validation_errors.append("Select at least two parsers.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    if int(final_top_k) > int(per_parser_top_k) * max(len(selected_parsers), 1):
        st.warning(
            "final_top_k is larger than per_parser_top_k multiplied by the number of selected parsers; "
            "fused results may contain fewer final chunks."
        )
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Parser Fusion Benchmark", type="primary", key="run_parser_fusion"):
        errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("parser_fusion_compare", int(final_top_k))
        run_config.update(
            {
                "selected_parsers": selected_parsers,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": "parser_fusion",
                "embedding_model": embedding_model,
                "fusion_method": fusion_method,
                "table_handling": table_handling,
                "answer_model": answer_model,
                "per_parser_top_k": int(per_parser_top_k),
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
        with st.status("Running parser fusion benchmark", expanded=True) as status:
            try:
                from benchmark_runner import run_parser_fusion_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_parser_fusion_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Parser fusion benchmark failed", state="error")
                st.error(f"Parser fusion benchmark failed: {exc}")
                return

            status.update(label="Parser fusion benchmark complete", state="complete")

        st.session_state["last_parser_fusion_result"] = runner_result
        render_parser_fusion_results(runner_result)
        rendered_current_results = True

    if "last_parser_fusion_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Parser Fusion Results", expanded=False):
            render_parser_fusion_results(dict(st.session_state["last_parser_fusion_result"]))


def render_multi_query_retrieval_compare_tab(config: dict[str, Any]) -> None:
    st.subheader("Controlled Query Expansion Preview")
    render_experiment_context(
        variable="query strategy",
        fixed=[
            "parser",
            "chunking_strategy",
            "embedding_model",
            "answer_model",
            "benchmark_questions_path",
        ],
    )
    st.caption(
        "Previews conservative phrase-level query expansions before retrieval. "
        "The original query is always included, word-level splitting is disabled, "
        "and no LLM query expansion is used."
    )

    embedding_options, answer_options, default_embedding, default_answer, default_top_k = benchmark_defaults(config)
    index_mode = st.selectbox(
        "Index mode",
        options=["Use exist index", "Build new index if missing"],
        index=0,
        key="multi_query_index_mode",
        help="Use exist index is retrieval-only. Build new index if missing allows a fallback build.",
    )
    use_existing_index = True
    build_new_index_if_missing = index_mode == "Build new index if missing"
    source_index: dict[str, Any] = {}
    uploaded_pdf = None

    candidates = multi_query_index_candidates()
    if not candidates:
        st.warning("No existing index candidates were found under outputs/runs/.")
    else:
        selected_label = st.selectbox(
            "Existing index source",
            options=[candidate["label"] for candidate in candidates],
            key="multi_query_source_index",
        )
        source_index = next(candidate for candidate in candidates if candidate["label"] == selected_label)
        st.markdown("#### Source Index")
        st.write(
            {
                "source_run_id": source_index.get("source_run_id"),
                "uploaded_pdf_path": source_index.get("uploaded_pdf_path"),
                "parser": source_index.get("parser"),
                "chunking_strategy": source_index.get("chunking_strategy"),
                "embedding_model": source_index.get("embedding_model"),
                "source_chroma_path": source_index.get("source_chroma_path"),
                "source_chunks_path": source_index.get("source_chunks_path"),
                "source_collection_name": source_index.get("source_collection_name"),
            }
        )
    parser_id = str(source_index.get("parser", ""))
    chunking_strategy = str(source_index.get("chunking_strategy", ""))
    embedding_model = str(source_index.get("embedding_model", default_embedding))

    if build_new_index_if_missing:
        uploaded_pdf = st.file_uploader(
            "Fallback PDF upload",
            type=["pdf"],
            key="multi_query_pdf",
            help="Only used if the selected source index files are missing.",
        )
    if not source_index and build_new_index_if_missing:
        parser_label = st.selectbox("Parser", options=PARSER_OPTIONS, key="multi_query_parser")
        parser_id = PARSER_ID_BY_LABEL[parser_label]
        chunking_strategy = st.selectbox(
            "Chunking strategy",
            options=CHUNKING_STRATEGY_OPTIONS,
            key="multi_query_chunking",
        )
        embedding_model = st.selectbox(
            "Embedding model",
            options=embedding_options,
            index=default_index(embedding_options, default_embedding),
            key="multi_query_embedding",
        )
    table_handling = render_table_handling_options(
        "multi_query_table",
        title="Answer Context Options",
        show_detection_options=False,
        include_answer_context_options=True,
    )
    answer_model = st.selectbox(
        "Answer model",
        options=answer_options,
        index=default_index(answer_options, default_answer),
        key="multi_query_answer",
    )
    per_query_top_k = st.number_input(
        "per_query_top_k",
        min_value=1,
        max_value=100,
        value=max(default_top_k, 10),
        step=1,
        key="multi_query_per_query_top_k",
    )
    final_top_k = st.number_input(
        "final_top_k",
        min_value=1,
        max_value=50,
        value=default_top_k,
        step=1,
        key="multi_query_final_top_k",
    )
    max_sub_queries = st.number_input(
        "max_sub_queries",
        min_value=1,
        max_value=12,
        value=5,
        step=1,
        key="multi_query_max_sub_queries",
    )
    benchmark_questions, benchmark_questions_error = load_benchmark_questions_for_ui()
    render_benchmark_questions_preview(benchmark_questions, benchmark_questions_error)
    accepted_sub_queries_by_question: dict[str, list[str]] = {}
    rejected_sub_queries_by_question: dict[str, list[dict[str, str]]] = {}
    query_preview_by_question: dict[str, Any] = {}
    if benchmark_questions and not benchmark_questions_error:
        (
            accepted_sub_queries_by_question,
            rejected_sub_queries_by_question,
            query_preview_by_question,
        ) = build_multi_query_preview(
            benchmark_questions,
            max_sub_queries=int(max_sub_queries),
        )

    validation_errors = []
    if use_existing_index and not source_index and not build_new_index_if_missing:
        validation_errors.append("Select an existing index source.")
    if benchmark_questions_error:
        validation_errors.append(benchmark_questions_error)
    for question in benchmark_questions:
        question_id = str(question.get("id", ""))
        if question_id and not accepted_sub_queries_by_question.get(question_id):
            validation_errors.append(f"No accepted query phrases for question {question_id}.")
    for error in validation_errors:
        st.warning(error)

    rendered_current_results = False
    if st.button("Run Controlled Query Expansion Retrieval", type="primary", key="run_multi_query"):
        errors = list(validation_errors)
        if build_new_index_if_missing and not source_index:
            errors = validate_setup_inputs(uploaded_pdf, validation_errors)
        if errors:
            for error in errors:
                st.warning(error)
            return

        run_config = setup_run_config("multi_query_retrieval_compare", int(final_top_k))
        run_config.update(
            {
                "parser": parser_id,
                "chunking_strategy": chunking_strategy,
                "retrieval_strategy": "multi_query_rrf",
                "embedding_model": embedding_model,
                "answer_model": answer_model,
                "table_handling": table_handling,
                "use_existing_index": bool(source_index),
                "build_new_index_if_missing": bool(build_new_index_if_missing),
                "source_index": source_index if use_existing_index else {},
                "per_query_top_k": int(per_query_top_k),
                "final_top_k": int(final_top_k),
                "max_sub_queries": int(max_sub_queries),
                "rrf_k": 60,
                "query_decomposition": {
                    "method": "rule_based",
                    "mode": "controlled_preview",
                    "always_include_original": True,
                    "word_level_splitting": False,
                    "llm_query_expansion": False,
                    "accepted_sub_queries_by_question": accepted_sub_queries_by_question,
                    "rejected_sub_queries_by_question": rejected_sub_queries_by_question,
                    "preview_by_question": query_preview_by_question,
                },
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
        if uploaded_pdf is None:
            run_id, run_dir, config_path = save_config_only_run_config(run_config)
        else:
            run_id, run_dir, config_path = save_setup_only_run_config(run_config, uploaded_pdf)
        render_saved_run(run_id, run_dir, config_path, run_config)

        progress_bar = st.progress(0)
        with st.status("Running controlled query expansion retrieval", expanded=True) as status:
            try:
                from benchmark_runner import run_multi_query_retrieval_compare

                def update_progress(message: str, step: int, total: int) -> None:
                    progress_bar.progress(step / total)
                    st.write(message)

                runner_result = run_multi_query_retrieval_compare(
                    config_path,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                status.update(label="Controlled query expansion retrieval failed", state="error")
                st.error(f"Controlled query expansion retrieval failed: {exc}")
                return

            status.update(label="Controlled query expansion retrieval complete", state="complete")

        st.session_state["last_multi_query_result"] = runner_result
        render_multi_query_retrieval_results(runner_result)
        rendered_current_results = True

    if "last_multi_query_result" in st.session_state and not rendered_current_results:
        with st.expander("Latest Controlled Query Expansion Results", expanded=False):
            render_multi_query_retrieval_results(dict(st.session_state["last_multi_query_result"]))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Configure controlled benchmark comparisons and save run_config.yaml files. "
        "Query expansion, judging, reranking, BM25, and answer-model comparison are intentionally excluded."
    )

    config, config_error = load_config(CONFIG_PATH)
    if config_error:
        st.warning(config_error)

    if BENCHMARK_QUESTIONS_PATH.exists():
        st.caption(f"Benchmark questions: `{BENCHMARK_QUESTIONS_PATH}`")
    else:
        st.warning(f"Missing benchmark questions file: {BENCHMARK_QUESTIONS_PATH}")

    (
        parser_tab,
        chunking_tab,
        embedding_tab,
        retrieval_fusion_tab,
        parser_fusion_tab,
        multi_query_tab,
        best_profile_tab,
    ) = st.tabs(
        [
            "Parser Compare",
            "Chunking Compare",
            "Embedding Compare",
            "Retrieval Fusion Compare",
            "Parser Fusion Compare",
            "Controlled Query Expansion Preview",
            "Best Profile Manager",
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

    with parser_fusion_tab:
        render_parser_fusion_compare_tab(config)

    with multi_query_tab:
        render_multi_query_retrieval_compare_tab(config)

    with best_profile_tab:
        render_best_profile_manager_tab()


if __name__ == "__main__":
    main()
