from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
ORIGINAL_RESULTS_FILENAME = "original_retrieval_results.jsonl"
EXPANDED_RESULTS_FILENAME = "expanded_retrieval_results.jsonl"
DOCLING_ORIGINAL_RESULTS_FILENAME = "original_retrieval_results_docling.jsonl"
DOCLING_EXPANDED_RESULTS_FILENAME = "expanded_retrieval_results_docling.jsonl"
COMPARISON_CSV_FILENAME = "retrieval_comparison.csv"
COMPARISON_REPORT_FILENAME = "retrieval_comparison_report.md"
PARSER_COMPARISON_CSV_FILENAME = "parser_comparison.csv"
PARSER_COMPARISON_REPORT_FILENAME = "parser_comparison_report.md"
PREVIEW_CHUNK_LIMIT = 5
PREVIEW_CHAR_LIMIT = 280
SUPPORTED_MODES = {"retrieval-comparison", "parser-comparison"}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line must be an object at {input_path}:{line_number}")
            records.append(record)

    return records


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "original_retrieved_count",
        "expanded_unique_retrieved_count",
        "overlap_count",
        "new_expanded_chunk_count",
        "overlap_chunk_ids",
        "new_expanded_chunk_ids",
        "top_original_pages",
        "top_expanded_pages",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_parser_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "pymupdf_unique_chunk_count",
        "docling_unique_chunk_count",
        "overlap_count",
        "pymupdf_only_count",
        "docling_only_count",
        "overlap_keys",
        "pymupdf_only_chunk_ids",
        "docling_only_chunk_ids",
        "top_pymupdf_pages",
        "top_docling_pages",
        "top_docling_section_titles",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def preview_text(text: str, char_limit: int = PREVIEW_CHAR_LIMIT) -> str:
    text = normalize_text(text)
    if len(text) <= char_limit:
        return text
    return text[: char_limit - 3].rstrip() + "..."


def chunk_identity(chunk: dict[str, Any]) -> str:
    normalized = normalize_text(str(chunk.get("text", ""))).casefold()
    if not normalized:
        return str(chunk.get("chunk_id", ""))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def chunk_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for record in records:
        for chunk in record["retrieved_chunks"]:
            chunks.setdefault(str(chunk["chunk_id"]), chunk)
    return chunks


def chunk_map_by_identity(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for record in records:
        for chunk in record["retrieved_chunks"]:
            chunks.setdefault(chunk_identity(chunk), chunk)
    return chunks


def chunk_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        for chunk in record["retrieved_chunks"]:
            ids.append(str(chunk["chunk_id"]))
    return ids


def chunk_identity_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for record in records:
        for chunk in record["retrieved_chunks"]:
            keys.append(chunk_identity(chunk))
    return keys


def page_summary(records: list[dict[str, Any]], limit: int = 5) -> list[str]:
    page_counts: Counter[int] = Counter()
    for record in records:
        for chunk in record["retrieved_chunks"]:
            if "page_number" in chunk:
                page_counts[int(chunk["page_number"])] += 1

    ranked_pages = sorted(page_counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{page} ({count})" for page, count in ranked_pages[:limit]]


def section_title_summary(records: list[dict[str, Any]], limit: int = 5) -> list[str]:
    section_counts: Counter[str] = Counter()
    for record in records:
        for chunk in record["retrieved_chunks"]:
            section_title = str(chunk.get("section_title", "")).strip()
            if section_title:
                section_counts[section_title] += 1

    ranked_sections = sorted(section_counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{section} ({count})" for section, count in ranked_sections[:limit]]


def group_expanded_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["question_id"])].append(record)
    return dict(grouped)


def group_original_by_question(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["question_id"]): record for record in records}


def question_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str(record.get("question") or record.get("original_question") or "")


def chunk_preview(chunk: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "chunk_index": chunk.get("chunk_index", ""),
        "preview": preview_text(str(chunk.get("text", ""))),
    }

    if "page_number" in chunk:
        preview["page_number"] = chunk["page_number"]
    if "section_title" in chunk:
        preview["section_title"] = chunk["section_title"]

    return preview


def compare_retrieval_results(
    original_records: list[dict[str, Any]],
    expanded_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded_by_question = group_expanded_by_question(expanded_records)
    comparisons: list[dict[str, Any]] = []

    for original_record in original_records:
        question_id = str(original_record["question_id"])
        expanded_question_records = expanded_by_question.get(question_id, [])

        original_ids = set(chunk_ids([original_record]))
        expanded_ids = set(chunk_ids(expanded_question_records))
        overlap_ids = sorted(original_ids & expanded_ids)
        new_expanded_ids = sorted(expanded_ids - original_ids)
        all_chunks = chunk_map([original_record] + expanded_question_records)

        comparisons.append(
            {
                "question_id": question_id,
                "question": str(original_record["question"]),
                "original_retrieved_count": len(original_ids),
                "expanded_unique_retrieved_count": len(expanded_ids),
                "overlap_count": len(overlap_ids),
                "new_expanded_chunk_count": len(new_expanded_ids),
                "overlap_chunk_ids": "|".join(overlap_ids),
                "new_expanded_chunk_ids": "|".join(new_expanded_ids),
                "top_original_pages": "|".join(page_summary([original_record])),
                "top_expanded_pages": "|".join(page_summary(expanded_question_records)),
                "new_chunk_previews": [
                    {
                        "chunk_id": chunk_id,
                        "page_number": int(all_chunks[chunk_id]["page_number"]),
                        "chunk_index": int(all_chunks[chunk_id]["chunk_index"]),
                        "preview": preview_text(str(all_chunks[chunk_id]["text"])),
                    }
                    for chunk_id in new_expanded_ids[:PREVIEW_CHUNK_LIMIT]
                    if chunk_id in all_chunks
                ],
            }
        )

    return comparisons


def compare_parser_results(
    pymupdf_original_records: list[dict[str, Any]],
    pymupdf_expanded_records: list[dict[str, Any]],
    docling_original_records: list[dict[str, Any]],
    docling_expanded_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pymupdf_original_by_question = group_original_by_question(pymupdf_original_records)
    docling_original_by_question = group_original_by_question(docling_original_records)
    pymupdf_expanded_by_question = group_expanded_by_question(pymupdf_expanded_records)
    docling_expanded_by_question = group_expanded_by_question(docling_expanded_records)

    question_ids = sorted(
        set(pymupdf_original_by_question)
        | set(docling_original_by_question)
        | set(pymupdf_expanded_by_question)
        | set(docling_expanded_by_question)
    )

    comparisons: list[dict[str, Any]] = []
    for question_id in question_ids:
        pymupdf_records = []
        docling_records = []

        pymupdf_original = pymupdf_original_by_question.get(question_id)
        docling_original = docling_original_by_question.get(question_id)

        if pymupdf_original:
            pymupdf_records.append(pymupdf_original)
        if docling_original:
            docling_records.append(docling_original)

        pymupdf_records.extend(pymupdf_expanded_by_question.get(question_id, []))
        docling_records.extend(docling_expanded_by_question.get(question_id, []))

        pymupdf_chunks = chunk_map_by_identity(pymupdf_records)
        docling_chunks = chunk_map_by_identity(docling_records)
        pymupdf_keys = set(pymupdf_chunks)
        docling_keys = set(docling_chunks)
        overlap_keys = sorted(pymupdf_keys & docling_keys)
        pymupdf_only_keys = sorted(pymupdf_keys - docling_keys)
        docling_only_keys = sorted(docling_keys - pymupdf_keys)

        question = (
            question_text(pymupdf_original)
            or question_text(docling_original)
            or question_text(next(iter(pymupdf_records), None))
            or question_text(next(iter(docling_records), None))
        )

        comparisons.append(
            {
                "question_id": question_id,
                "question": question,
                "pymupdf_unique_chunk_count": len(pymupdf_keys),
                "docling_unique_chunk_count": len(docling_keys),
                "overlap_count": len(overlap_keys),
                "pymupdf_only_count": len(pymupdf_only_keys),
                "docling_only_count": len(docling_only_keys),
                "overlap_keys": "|".join(overlap_keys),
                "pymupdf_only_chunk_ids": "|".join(
                    str(pymupdf_chunks[key].get("chunk_id", "")) for key in pymupdf_only_keys
                ),
                "docling_only_chunk_ids": "|".join(
                    str(docling_chunks[key].get("chunk_id", "")) for key in docling_only_keys
                ),
                "top_pymupdf_pages": "|".join(page_summary(pymupdf_records)),
                "top_docling_pages": "|".join(page_summary(docling_records)),
                "top_docling_section_titles": "|".join(section_title_summary(docling_records)),
                "pymupdf_only_previews": [
                    chunk_preview(pymupdf_chunks[key])
                    for key in pymupdf_only_keys[:PREVIEW_CHUNK_LIMIT]
                    if key in pymupdf_chunks
                ],
                "docling_only_previews": [
                    chunk_preview(docling_chunks[key])
                    for key in docling_only_keys[:PREVIEW_CHUNK_LIMIT]
                    if key in docling_chunks
                ],
            }
        )

    return comparisons


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def table_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(table_cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(table_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def generate_markdown_report(comparisons: list[dict[str, Any]]) -> str:
    total_questions = len(comparisons)
    total_original = sum(row["original_retrieved_count"] for row in comparisons)
    total_expanded_unique = sum(row["expanded_unique_retrieved_count"] for row in comparisons)
    total_overlap = sum(row["overlap_count"] for row in comparisons)
    total_new = sum(row["new_expanded_chunk_count"] for row in comparisons)

    summary_rows = [
        [
            row["question_id"],
            row["original_retrieved_count"],
            row["expanded_unique_retrieved_count"],
            row["overlap_count"],
            row["new_expanded_chunk_count"],
            row["top_original_pages"],
            row["top_expanded_pages"],
        ]
        for row in comparisons
    ]

    lines = [
        "# Retrieval Comparison Report",
        "",
        "## Summary",
        "",
        f"- Questions compared: {total_questions}",
        f"- Original retrieved chunks: {total_original}",
        f"- Expanded unique retrieved chunks: {total_expanded_unique}",
        f"- Overlapping chunks: {total_overlap}",
        f"- New chunks found only by expanded queries: {total_new}",
        "",
        markdown_table(
            [
                "Question",
                "Original Chunks",
                "Expanded Unique Chunks",
                "Overlap",
                "New Expanded Chunks",
                "Top Original Pages",
                "Top Expanded Pages",
            ],
            summary_rows,
        ),
        "",
        "## Per-Question Comparison",
    ]

    for row in comparisons:
        lines.extend(
            [
                "",
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"- Original retrieved chunks: {row['original_retrieved_count']}",
                f"- Expanded unique retrieved chunks: {row['expanded_unique_retrieved_count']}",
                f"- Overlap count: {row['overlap_count']}",
                f"- New expanded-only chunks: {row['new_expanded_chunk_count']}",
                f"- Top pages from original query: {row['top_original_pages'] or 'None'}",
                f"- Top pages from expanded queries: {row['top_expanded_pages'] or 'None'}",
                "",
                "New expanded-only chunk previews:",
            ]
        )

        if row["new_chunk_previews"]:
            for preview in row["new_chunk_previews"]:
                lines.extend(
                    [
                        "",
                        (
                            f"- `{preview['chunk_id']}` "
                            f"(page {preview['page_number']}, chunk {preview['chunk_index']}): "
                            f"{preview['preview']}"
                        ),
                    ]
                )
        else:
            lines.append("")
            lines.append("- None")

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- Expanded query retrieval increases the candidate evidence pool when new expanded-only chunks are found.",
            "- Overlap shows where original and expanded queries converge on the same evidence.",
            "- Page concentration can help identify whether expansions are broadening retrieval or repeatedly targeting the same sections.",
            "",
            "## Notes For Manual Review",
            "",
            "- This report does not judge correctness because no gold evidence labels are available yet.",
            "- Review whether new expanded-only chunks are actually relevant to the original question.",
            "- Watch for query expansions that drift away from the technical terms in the original question.",
            "- Use these results to decide whether benchmark questions need expected pages, sections, or evidence chunk IDs.",
        ]
    )

    return "\n".join(lines) + "\n"


def format_preview_item(preview: dict[str, Any]) -> str:
    location_parts = [f"chunk {preview['chunk_index']}"]
    if "page_number" in preview:
        location_parts.insert(0, f"page {preview['page_number']}")
    if "section_title" in preview:
        location_parts.insert(0, f"section {preview['section_title']}")

    return f"- `{preview['chunk_id']}` ({', '.join(location_parts)}): {preview['preview']}"


def generate_parser_comparison_report(comparisons: list[dict[str, Any]]) -> str:
    total_questions = len(comparisons)
    total_pymupdf = sum(row["pymupdf_unique_chunk_count"] for row in comparisons)
    total_docling = sum(row["docling_unique_chunk_count"] for row in comparisons)
    total_overlap = sum(row["overlap_count"] for row in comparisons)
    total_pymupdf_only = sum(row["pymupdf_only_count"] for row in comparisons)
    total_docling_only = sum(row["docling_only_count"] for row in comparisons)

    summary_rows = [
        [
            row["question_id"],
            row["pymupdf_unique_chunk_count"],
            row["docling_unique_chunk_count"],
            row["overlap_count"],
            row["pymupdf_only_count"],
            row["docling_only_count"],
            row["top_pymupdf_pages"] or "None",
            row["top_docling_section_titles"] or "None",
        ]
        for row in comparisons
    ]

    lines = [
        "# Parser Retrieval Comparison Report",
        "",
        "## Overall Summary",
        "",
        f"- Questions compared: {total_questions}",
        f"- Unique PyMuPDF chunks retrieved: {total_pymupdf}",
        f"- Unique Docling chunks retrieved: {total_docling}",
        f"- Overlapping retrieved chunks by normalized text: {total_overlap}",
        f"- PyMuPDF-only retrieved chunks: {total_pymupdf_only}",
        f"- Docling-only retrieved chunks: {total_docling_only}",
        "",
        "Docling retrieved additional chunks that may be useful for manual review.",
        "",
        markdown_table(
            [
                "Question",
                "PyMuPDF Chunks",
                "Docling Chunks",
                "Overlap",
                "PyMuPDF Only",
                "Docling Only",
                "Top PyMuPDF Pages",
                "Top Docling Sections",
            ],
            summary_rows,
        ),
        "",
        "## Per-Question Comparison",
    ]

    for row in comparisons:
        lines.extend(
            [
                "",
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"- Unique PyMuPDF chunks: {row['pymupdf_unique_chunk_count']}",
                f"- Unique Docling chunks: {row['docling_unique_chunk_count']}",
                f"- Overlap by normalized text: {row['overlap_count']}",
                f"- PyMuPDF-only chunks: {row['pymupdf_only_count']}",
                f"- Docling-only chunks: {row['docling_only_count']}",
                f"- Top PyMuPDF pages: {row['top_pymupdf_pages'] or 'None'}",
                f"- Top Docling pages: {row['top_docling_pages'] or 'None'}",
                f"- Top Docling section titles: {row['top_docling_section_titles'] or 'None'}",
                "",
                "#### PyMuPDF-Only Findings",
            ]
        )

        if row["pymupdf_only_previews"]:
            lines.extend(format_preview_item(preview) for preview in row["pymupdf_only_previews"])
        else:
            lines.append("- None")

        lines.append("")
        lines.append("#### Docling-Only Findings")

        if row["docling_only_previews"]:
            lines.extend(format_preview_item(preview) for preview in row["docling_only_previews"])
        else:
            lines.append("- None")

    lines.extend(
        [
            "",
            "## PyMuPDF-Only Findings",
            "",
            "- PyMuPDF-only chunks show evidence retrieved by the baseline parser that did not appear as exact normalized text matches in Docling retrieval.",
            "- These may reflect parser-specific chunk boundaries, page-local extraction, or content that Docling represented differently.",
            "",
            "## Docling-Only Findings",
            "",
            "- Docling-only chunks show additional structured Markdown-derived chunks that may be useful for manual review.",
            "- These may reflect section-aware extraction, table conversion, or chunk boundaries that differ from the PyMuPDF baseline.",
            "",
            "## Manual Review Notes",
            "",
            "- This report does not judge parser quality because no gold evidence labels are available yet.",
            "- Review whether parser-only chunks are relevant to the original benchmark question.",
            "- Overlap is computed by normalized retrieved text, not by chunk ID, because parser-specific chunk IDs are intentionally different.",
            "- Do not treat higher unique chunk counts as automatically better.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_markdown_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def run_retrieval_comparison(config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(config["paths"]["output_dir"])
    original_path = output_dir / ORIGINAL_RESULTS_FILENAME
    expanded_path = output_dir / EXPANDED_RESULTS_FILENAME
    csv_path = output_dir / COMPARISON_CSV_FILENAME
    report_path = output_dir / COMPARISON_REPORT_FILENAME

    original_records = read_jsonl(original_path)
    expanded_records = read_jsonl(expanded_path)
    comparisons = compare_retrieval_results(original_records, expanded_records)

    write_csv(comparisons, csv_path)
    write_markdown_report(generate_markdown_report(comparisons), report_path)

    print(f"Loaded {len(original_records)} original retrieval records from {original_path}")
    print(f"Loaded {len(expanded_records)} expanded retrieval records from {expanded_path}")
    print(f"Compared retrieval results for {len(comparisons)} questions")
    print(f"Saved CSV comparison to {csv_path}")
    print(f"Saved Markdown report to {report_path}")

    return csv_path, report_path


def run_parser_comparison(config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(config["paths"]["output_dir"])
    pymupdf_original_path = output_dir / ORIGINAL_RESULTS_FILENAME
    pymupdf_expanded_path = output_dir / EXPANDED_RESULTS_FILENAME
    docling_original_path = output_dir / DOCLING_ORIGINAL_RESULTS_FILENAME
    docling_expanded_path = output_dir / DOCLING_EXPANDED_RESULTS_FILENAME
    csv_path = output_dir / PARSER_COMPARISON_CSV_FILENAME
    report_path = output_dir / PARSER_COMPARISON_REPORT_FILENAME

    pymupdf_original_records = read_jsonl(pymupdf_original_path)
    pymupdf_expanded_records = read_jsonl(pymupdf_expanded_path)
    docling_original_records = read_jsonl(docling_original_path)
    docling_expanded_records = read_jsonl(docling_expanded_path)
    comparisons = compare_parser_results(
        pymupdf_original_records=pymupdf_original_records,
        pymupdf_expanded_records=pymupdf_expanded_records,
        docling_original_records=docling_original_records,
        docling_expanded_records=docling_expanded_records,
    )

    write_parser_comparison_csv(comparisons, csv_path)
    write_markdown_report(generate_parser_comparison_report(comparisons), report_path)

    print(f"Loaded {len(pymupdf_original_records)} PyMuPDF original retrieval records from {pymupdf_original_path}")
    print(f"Loaded {len(pymupdf_expanded_records)} PyMuPDF expanded retrieval records from {pymupdf_expanded_path}")
    print(f"Loaded {len(docling_original_records)} Docling original retrieval records from {docling_original_path}")
    print(f"Loaded {len(docling_expanded_records)} Docling expanded retrieval records from {docling_expanded_path}")
    print(f"Compared parser retrieval results for {len(comparisons)} questions")
    print(f"Saved parser comparison CSV to {csv_path}")
    print(f"Saved parser comparison Markdown report to {report_path}")

    return csv_path, report_path


def run(config_path: Path = DEFAULT_CONFIG_PATH, mode: str = "retrieval-comparison") -> tuple[Path, Path]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    config = load_config(config_path)

    if mode == "retrieval-comparison":
        return run_retrieval_comparison(config)
    if mode == "parser-comparison":
        return run_parser_comparison(config)

    raise ValueError(f"Unsupported evaluation mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOTA RAG evaluation.")
    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="retrieval-comparison",
        help="Evaluation mode to run.",
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
