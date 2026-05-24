from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


APP_TITLE = "SOTA RAG - MVP Result Viewer"
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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    benchmark_records, benchmark_error = read_jsonl(BENCHMARK_QUESTIONS_PATH)
    original_records, original_error = read_jsonl(ORIGINAL_RESULTS_PATH)
    expansion_records, expansion_error = read_jsonl(QUERY_EXPANSIONS_PATH)
    expanded_records, expanded_error = read_jsonl(EXPANDED_RESULTS_PATH)
    docling_original_records, docling_original_error = read_jsonl(DOCLING_ORIGINAL_RESULTS_PATH)
    docling_expanded_records, docling_expanded_error = read_jsonl(DOCLING_EXPANDED_RESULTS_PATH)
    comparison_df, comparison_error = read_csv(COMPARISON_CSV_PATH)
    report_markdown, report_error = read_markdown(COMPARISON_REPORT_PATH)
    parser_comparison_df, parser_comparison_error = read_csv(PARSER_COMPARISON_CSV_PATH)
    parser_report_markdown, parser_report_error = read_markdown(PARSER_COMPARISON_REPORT_PATH)
    config, config_error = load_config(CONFIG_PATH)
    manual_settings, manual_config_error = manual_query_config(config) if config else ({}, None)

    errors = [
        error
        for error in [
            original_error,
            expansion_error,
            expanded_error,
            comparison_error,
            report_error,
            docling_original_error,
            docling_expanded_error,
            parser_comparison_error,
            parser_report_error,
            benchmark_error,
        ]
        if error
    ]
    render_file_warnings(errors)

    benchmark_questions = benchmark_question_map(benchmark_records)
    original_by_question = group_by_question(original_records)
    expansions_by_question = query_expansion_map(expansion_records)
    expanded_by_question = group_by_question(expanded_records)
    docling_original_by_question = group_by_question(docling_original_records)
    docling_expanded_by_question = group_by_question(docling_expanded_records)
    question_ids = all_question_ids(original_records, expansion_records, expanded_records)

    st.subheader("Overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Benchmark questions", len(question_ids))
    metric_cols[1].metric("Original retrieval records", len(original_records))
    metric_cols[2].metric("Expanded retrieval records", len(expanded_records))
    metric_cols[3].metric("Query expansion records", len(expansion_records))

    details_tab, manual_tab, benchmark_answer_tab, parser_tab, comparison_tab, report_tab = st.tabs(
        [
            "Question Details",
            "Manual Query",
            "Benchmark Answer Comparison",
            "Parser Comparison",
            "Comparison CSV",
            "Markdown Report",
        ]
    )

    with details_tab:
        if not question_ids:
            st.info("No question records were found in the available output files.")
        else:
            labels = {
                question_id: f"{question_id} - {first_question_text(question_id, original_by_question, expansions_by_question, expanded_by_question)}"
                for question_id in question_ids
            }
            selected_question_id = st.selectbox(
                "Question",
                options=question_ids,
                format_func=lambda question_id: labels[question_id],
            )

            original_question = first_question_text(
                selected_question_id,
                original_by_question,
                expansions_by_question,
                expanded_by_question,
            )
            selected_original_records = original_by_question.get(selected_question_id, [])
            selected_expansion_record = expansions_by_question.get(selected_question_id, {})
            selected_expanded_records = expanded_by_question.get(selected_question_id, [])

            original_chunks = (
                selected_original_records[0].get("retrieved_chunks", [])
                if selected_original_records
                else []
            )
            original_chunk_ids = {str(chunk.get("chunk_id", "")) for chunk in original_chunks}
            expanded_chunks_by_id = unique_chunks(selected_expanded_records)
            expanded_only_ids = set(expanded_chunks_by_id) - original_chunk_ids
            expanded_only_chunks = [
                chunk
                for chunk_id, chunk in expanded_chunks_by_id.items()
                if chunk_id in expanded_only_ids
            ]

            st.markdown("#### Original Question")
            st.write(original_question or "No original question text found.")

            st.markdown("#### Expanded Queries")
            expanded_queries = selected_expansion_record.get("expanded_queries", [])
            if expanded_queries:
                for index, query in enumerate(expanded_queries, start=1):
                    st.write(f"{index}. {query}")
            else:
                st.info("No expanded queries found for this question.")

            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("#### Original Retrieved Chunks")
                render_chunk_table(
                    original_chunks,
                    "No original retrieved chunks found for this question.",
                )

            with right_col:
                st.markdown("#### Expanded-Only Chunks")
                render_chunk_table(
                    expanded_only_chunks,
                    "No chunks found only by expanded queries for this question.",
                    highlight_ids=expanded_only_ids,
                )

            st.markdown("#### Expanded Retrieved Chunks By Query")
            if not selected_expanded_records:
                st.info("No expanded retrieval records found for this question.")
            else:
                for record in sorted(
                    selected_expanded_records,
                    key=lambda item: safe_int(item.get("expanded_query_index")),
                ):
                    query_index = safe_int(record.get("expanded_query_index")) + 1
                    query_text = str(record.get("query_text", ""))
                    with st.expander(f"Expanded query {query_index}: {query_text}", expanded=False):
                        render_chunk_table(
                            record.get("retrieved_chunks", []),
                            "No retrieved chunks found for this expanded query.",
                            highlight_ids=expanded_only_ids,
                        )

    with manual_tab:
        st.subheader("Manual Query")

        if config_error:
            st.warning(config_error)
        elif manual_config_error:
            st.warning(manual_config_error)
        else:
            config_cols = st.columns(3)
            config_cols[0].caption(f"Embedding model: `{manual_settings['embedding_model_name']}`")
            config_cols[1].caption(f"Chroma path: `{manual_settings['vector_db_dir']}`")
            config_cols[2].caption(
                "Collections: "
                f"`{manual_settings['pymupdf_collection_name']}`, "
                f"`{manual_settings['docling_collection_name']}`"
            )
            st.caption(
                "Ollama expansion model: "
                f"`{manual_settings['ollama_model_name']}` "
                f"({manual_settings['num_expanded_queries']} queries)"
            )

            source_choice = st.selectbox(
                "Parser/source",
                options=["PyMuPDF baseline", "Docling structured parser", "Compare both"],
            )
            manual_query = st.text_area(
                "Custom technical document question",
                placeholder="Type a question to retrieve matching chunks from the current Chroma DB.",
                height=120,
            )
            top_k = st.number_input(
                "top_k",
                min_value=1,
                max_value=50,
                value=manual_settings["top_k"],
                step=1,
            )
            use_query_expansion = st.checkbox("Use Ollama query expansion")

            if st.button("Search", type="primary"):
                query_text = manual_query.strip()
                if not query_text:
                    st.warning("Enter a query before searching.")
                elif not Path(manual_settings["vector_db_dir"]).exists():
                    st.warning(f"Chroma DB path does not exist: {manual_settings['vector_db_dir']}")
                else:
                    try:
                        source_options = manual_source_options(manual_settings)
                        selected_sources = (
                            list(source_options.items())
                            if source_choice == "Compare both"
                            else [(source_choice, source_options[source_choice])]
                        )

                        expanded_queries: list[str] = []
                        if use_query_expansion:
                            with st.spinner("Generating expanded queries with Ollama..."):
                                expanded_queries = expand_manual_query(query_text, manual_settings)

                        manual_results: dict[str, dict[str, Any]] = {}
                        for label, source_config in selected_sources:
                            with st.spinner(f"Retrieving chunks from {label}..."):
                                original_chunks = retrieve_manual_query(
                                    query_text=query_text,
                                    top_k=int(top_k),
                                    settings=manual_settings,
                                    collection_name=source_config["collection_name"],
                                    source=source_config["source"],
                                )

                                expanded_records: list[dict[str, Any]] = []
                                if use_query_expansion:
                                    for index, expanded_query in enumerate(expanded_queries):
                                        expanded_records.append(
                                            {
                                                "query_text": expanded_query,
                                                "expanded_query_index": index,
                                                "retrieved_chunks": retrieve_manual_query(
                                                    query_text=expanded_query,
                                                    top_k=int(top_k),
                                                    settings=manual_settings,
                                                    collection_name=source_config["collection_name"],
                                                    source=source_config["source"],
                                                ),
                                            }
                                        )

                            manual_results[label] = {
                                "original_chunks": original_chunks,
                                "expanded_records": expanded_records,
                                "collection_name": source_config["collection_name"],
                            }
                        st.session_state["manual_query_text"] = query_text
                        st.session_state["manual_results"] = manual_results
                        st.session_state["manual_source_choice"] = source_choice
                        st.session_state["manual_use_query_expansion"] = use_query_expansion
                        st.session_state["manual_expanded_queries"] = expanded_queries
                        for key in list(st.session_state):
                            if str(key).startswith("manual_answer_"):
                                del st.session_state[key]
                    except Exception as exc:
                        st.error(f"Manual retrieval failed: {exc}")
                    else:
                        st.success("Retrieval complete.")

            if "manual_results" in st.session_state:
                render_manual_query_results(
                    source_choice=str(st.session_state["manual_source_choice"]),
                    use_query_expansion=bool(st.session_state["manual_use_query_expansion"]),
                    expanded_queries=list(st.session_state["manual_expanded_queries"]),
                    manual_results=dict(st.session_state["manual_results"]),
                    query_text=str(st.session_state["manual_query_text"]),
                    settings=manual_settings,
                )

    with benchmark_answer_tab:
        st.subheader("Benchmark Answer Comparison")
        st.caption(
            "Neutral comparison of answers generated from existing retrieval outputs."
        )

        if config_error:
            st.warning(config_error)
        elif manual_config_error:
            st.warning(manual_config_error)
        elif not benchmark_questions:
            st.info("No benchmark questions were found.")
        else:
            benchmark_question_ids = sorted(benchmark_questions)
            selected_benchmark_question_id = st.selectbox(
                "Benchmark question",
                options=benchmark_question_ids,
                format_func=lambda question_id: f"{question_id} - {benchmark_questions[question_id]}",
                key="benchmark_answer_question",
            )
            selected_benchmark_question = benchmark_questions[selected_benchmark_question_id]

            st.write(
                {
                    "question_id": selected_benchmark_question_id,
                    "question": selected_benchmark_question,
                }
            )

            top_k = manual_settings["top_k"]
            answer_contexts = {
                "PyMuPDF Original": original_answer_chunks(
                    original_by_question,
                    selected_benchmark_question_id,
                    source="pymupdf",
                    top_k=top_k,
                ),
                "PyMuPDF Expanded": expanded_answer_chunks(
                    expanded_by_question,
                    selected_benchmark_question_id,
                    source="pymupdf",
                    top_k=top_k,
                ),
                "Docling Original": original_answer_chunks(
                    docling_original_by_question,
                    selected_benchmark_question_id,
                    source="docling",
                    top_k=top_k,
                ),
                "Docling Expanded": expanded_answer_chunks(
                    docling_expanded_by_question,
                    selected_benchmark_question_id,
                    source="docling",
                    top_k=top_k,
                ),
            }

            context_cols = st.columns(4)
            for column, (label, chunks) in zip(context_cols, answer_contexts.items()):
                column.metric(label, len(chunks))

            if st.button("Generate Answers for Selected Question", type="primary"):
                answer_key = f"benchmark_answers_{selected_benchmark_question_id}"
                try:
                    generated_answers: dict[str, str] = {}
                    for label, chunks in answer_contexts.items():
                        with st.spinner(f"Generating answer: {label}..."):
                            generated_answers[label] = generate_grounded_answer(
                                query_text=selected_benchmark_question,
                                chunks=chunks,
                                settings=manual_settings,
                            )
                    st.session_state[answer_key] = generated_answers
                except Exception as exc:
                    st.error(f"Benchmark answer generation failed: {exc}")

            answers = st.session_state.get(
                f"benchmark_answers_{selected_benchmark_question_id}",
                {},
            )

            first_row = st.columns(2)
            second_row = st.columns(2)
            panels = list(answer_contexts.items())
            for column, (label, chunks) in zip(first_row + second_row, panels):
                with column:
                    render_answer_panel(
                        title=label,
                        answer=answers.get(label),
                        chunks=chunks,
                    )

            render_manual_review_checklist()

    with parser_tab:
        st.subheader("Parser Comparison")
        st.caption(
            "Neutral side-by-side view of existing PyMuPDF and Docling retrieval outputs."
        )

        parser_question_ids = comparison_question_ids(
            parser_comparison_df,
            original_by_question,
            docling_original_by_question,
            expanded_by_question,
            docling_expanded_by_question,
        )

        total_pymupdf_only = (
            int(parser_comparison_df["pymupdf_only_count"].sum())
            if not parser_comparison_df.empty and "pymupdf_only_count" in parser_comparison_df
            else 0
        )
        total_docling_only = (
            int(parser_comparison_df["docling_only_count"].sum())
            if not parser_comparison_df.empty and "docling_only_count" in parser_comparison_df
            else 0
        )
        total_overlap = (
            int(parser_comparison_df["overlap_count"].sum())
            if not parser_comparison_df.empty and "overlap_count" in parser_comparison_df
            else 0
        )

        parser_metric_cols = st.columns(4)
        parser_metric_cols[0].metric("Questions compared", len(parser_question_ids))
        parser_metric_cols[1].metric("PyMuPDF-only chunks", total_pymupdf_only)
        parser_metric_cols[2].metric("Docling-only chunks", total_docling_only)
        parser_metric_cols[3].metric("Overlap", total_overlap)

        if not parser_question_ids:
            st.info("No parser comparison records were found.")
        else:
            selected_parser_question_id = st.selectbox(
                "Parser comparison question",
                options=parser_question_ids,
                format_func=lambda question_id: question_label(
                    question_id,
                    parser_comparison_df,
                    original_by_question,
                    docling_original_by_question,
                    expanded_by_question,
                    docling_expanded_by_question,
                ),
            )

            selected_rows = (
                parser_comparison_df[
                    parser_comparison_df["question_id"].astype(str) == selected_parser_question_id
                ]
                if not parser_comparison_df.empty and "question_id" in parser_comparison_df.columns
                else pd.DataFrame()
            )
            selected_row = selected_rows.iloc[0] if not selected_rows.empty else None

            question_cols = st.columns(4)
            question_cols[0].metric(
                "PyMuPDF unique",
                parser_metric_value(selected_row, "pymupdf_unique_chunk_count"),
            )
            question_cols[1].metric(
                "Docling unique",
                parser_metric_value(selected_row, "docling_unique_chunk_count"),
            )
            question_cols[2].metric(
                "PyMuPDF only",
                parser_metric_value(selected_row, "pymupdf_only_count"),
            )
            question_cols[3].metric(
                "Docling only",
                parser_metric_value(selected_row, "docling_only_count"),
            )

            if selected_row is not None:
                st.write(
                    {
                        "overlap_count": parser_metric_value(selected_row, "overlap_count"),
                        "top_pymupdf_pages": selected_row.get("top_pymupdf_pages", ""),
                        "top_docling_pages": selected_row.get("top_docling_pages", ""),
                        "top_docling_section_titles": selected_row.get(
                            "top_docling_section_titles",
                            "",
                        ),
                    }
                )

            pymupdf_original_record = first_original_record(
                original_by_question.get(selected_parser_question_id, [])
            )
            docling_original_record = first_original_record(
                docling_original_by_question.get(selected_parser_question_id, [])
            )
            pymupdf_expanded_records = expanded_by_question.get(selected_parser_question_id, [])
            selected_docling_expanded_records = docling_expanded_by_question.get(
                selected_parser_question_id,
                [],
            )

            original_cols = st.columns(2)
            with original_cols[0]:
                st.markdown("#### PyMuPDF Original Retrieval")
                render_chunk_table(
                    pymupdf_original_record.get("retrieved_chunks", [])
                    if pymupdf_original_record
                    else [],
                    "No PyMuPDF original retrieval chunks found for this question.",
                )
            with original_cols[1]:
                st.markdown("#### Docling Original Retrieval")
                render_chunk_table(
                    docling_original_record.get("retrieved_chunks", [])
                    if docling_original_record
                    else [],
                    "No Docling original retrieval chunks found for this question.",
                )

            expanded_cols = st.columns(2)
            with expanded_cols[0]:
                st.markdown("#### PyMuPDF Expanded Retrieval")
                render_expanded_records(
                    pymupdf_expanded_records,
                    "No PyMuPDF expanded retrieval records found for this question.",
                )
            with expanded_cols[1]:
                st.markdown("#### Docling Expanded Retrieval")
                render_expanded_records(
                    selected_docling_expanded_records,
                    "No Docling expanded retrieval records found for this question.",
                )

        with st.expander("Parser Comparison Report", expanded=False):
            if parser_report_markdown:
                st.markdown(parser_report_markdown)
            else:
                st.info("No parser comparison report available.")

    with comparison_tab:
        st.subheader("Retrieval Comparison CSV")
        if comparison_df.empty:
            st.info("No comparison CSV data available.")
        else:
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    with report_tab:
        st.subheader("Retrieval Comparison Report")
        if report_markdown:
            st.markdown(report_markdown)
        else:
            st.info("No Markdown report available.")

    with st.sidebar:
        st.header("Expected Files")
        for path in EXPECTED_FILES:
            status = "Found" if path.exists() else "Missing"
            st.write(f"{status}: `{path}`")


if __name__ == "__main__":
    main()
