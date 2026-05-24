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
OUTPUT_DIR = Path("outputs")
ORIGINAL_RESULTS_PATH = OUTPUT_DIR / "original_retrieval_results.jsonl"
QUERY_EXPANSIONS_PATH = OUTPUT_DIR / "query_expansions.jsonl"
EXPANDED_RESULTS_PATH = OUTPUT_DIR / "expanded_retrieval_results.jsonl"
COMPARISON_CSV_PATH = OUTPUT_DIR / "retrieval_comparison.csv"
COMPARISON_REPORT_PATH = OUTPUT_DIR / "retrieval_comparison_report.md"
EXPECTED_FILES = [
    ORIGINAL_RESULTS_PATH,
    QUERY_EXPANSIONS_PATH,
    EXPANDED_RESULTS_PATH,
    COMPARISON_CSV_PATH,
    COMPARISON_REPORT_PATH,
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
            "collection_name": str(config["embedding"]["collection_name"]),
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


def expand_manual_query(query_text: str, settings: dict[str, Any]) -> list[str]:
    from query_expansion import generate_expanded_queries

    expanded_queries, _raw_response = generate_expanded_queries(
        question=query_text,
        model_name=settings["ollama_model_name"],
        num_queries=settings["num_expanded_queries"],
        temperature=settings["ollama_temperature"],
    )
    return expanded_queries


def retrieve_manual_query(query_text: str, top_k: int, settings: dict[str, Any]) -> list[dict[str, Any]]:
    embedding_model = load_embedding_model(settings["embedding_model_name"])
    collection = load_chroma_collection(settings["vector_db_dir"], settings["collection_name"])

    metadata = collection.metadata or {}
    existing_model_name = metadata.get("embedding_model")
    if existing_model_name != settings["embedding_model_name"]:
        raise ValueError(
            "Chroma collection embedding model mismatch: "
            f"collection={settings['collection_name']}, existing={existing_model_name}, "
            f"configured={settings['embedding_model_name']}"
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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    original_records, original_error = read_jsonl(ORIGINAL_RESULTS_PATH)
    expansion_records, expansion_error = read_jsonl(QUERY_EXPANSIONS_PATH)
    expanded_records, expanded_error = read_jsonl(EXPANDED_RESULTS_PATH)
    comparison_df, comparison_error = read_csv(COMPARISON_CSV_PATH)
    report_markdown, report_error = read_markdown(COMPARISON_REPORT_PATH)
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
        ]
        if error
    ]
    render_file_warnings(errors)

    original_by_question = group_by_question(original_records)
    expansions_by_question = query_expansion_map(expansion_records)
    expanded_by_question = group_by_question(expanded_records)
    question_ids = all_question_ids(original_records, expansion_records, expanded_records)

    st.subheader("Overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Benchmark questions", len(question_ids))
    metric_cols[1].metric("Original retrieval records", len(original_records))
    metric_cols[2].metric("Expanded retrieval records", len(expanded_records))
    metric_cols[3].metric("Query expansion records", len(expansion_records))

    details_tab, manual_tab, comparison_tab, report_tab = st.tabs(
        ["Question Details", "Manual Query", "Comparison CSV", "Markdown Report"]
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
            config_cols[2].caption(f"Collection: `{manual_settings['collection_name']}`")
            st.caption(
                "Ollama expansion model: "
                f"`{manual_settings['ollama_model_name']}` "
                f"({manual_settings['num_expanded_queries']} queries)"
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
                        with st.spinner("Retrieving chunks for the original query..."):
                            original_manual_chunks = retrieve_manual_query(
                                query_text=query_text,
                                top_k=int(top_k),
                                settings=manual_settings,
                            )

                        expanded_queries: list[str] = []
                        expanded_records: list[dict[str, Any]] = []
                        if use_query_expansion:
                            with st.spinner("Generating expanded queries with Ollama..."):
                                expanded_queries = expand_manual_query(query_text, manual_settings)

                            with st.spinner("Retrieving chunks for expanded queries..."):
                                for index, expanded_query in enumerate(expanded_queries):
                                    expanded_records.append(
                                        {
                                            "query_text": expanded_query,
                                            "expanded_query_index": index,
                                            "retrieved_chunks": retrieve_manual_query(
                                                query_text=expanded_query,
                                                top_k=int(top_k),
                                                settings=manual_settings,
                                            ),
                                        }
                                    )
                    except Exception as exc:
                        st.error(f"Manual retrieval failed: {exc}")
                    else:
                        st.markdown("#### Original Retrieval")
                        render_chunk_table(
                            original_manual_chunks,
                            "No chunks were retrieved for the original query.",
                        )

                        if use_query_expansion:
                            st.markdown("#### Expanded Queries")
                            if expanded_queries:
                                for index, expanded_query in enumerate(expanded_queries, start=1):
                                    st.write(f"{index}. {expanded_query}")
                            else:
                                st.info("Ollama did not return any expanded queries.")

                            original_chunk_ids = {
                                str(chunk.get("chunk_id", "")) for chunk in original_manual_chunks
                            }
                            expanded_chunks_by_id = unique_chunks(expanded_records)
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
                            if not expanded_records:
                                st.info("No expanded retrieval records were generated.")
                            else:
                                for record in expanded_records:
                                    query_index = safe_int(record.get("expanded_query_index")) + 1
                                    query_text = str(record.get("query_text", ""))
                                    with st.expander(
                                        f"Expanded query {query_index}: {query_text}",
                                        expanded=False,
                                    ):
                                        render_chunk_table(
                                            record.get("retrieved_chunks", []),
                                            "No chunks were retrieved for this expanded query.",
                                            highlight_ids=expanded_only_ids,
                                        )

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
