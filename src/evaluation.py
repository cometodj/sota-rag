from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
ORIGINAL_RESULTS_FILENAME = "original_retrieval_results.jsonl"
EXPANDED_RESULTS_FILENAME = "expanded_retrieval_results.jsonl"
COMPARISON_CSV_FILENAME = "retrieval_comparison.csv"
COMPARISON_REPORT_FILENAME = "retrieval_comparison_report.md"
PREVIEW_CHUNK_LIMIT = 5
PREVIEW_CHAR_LIMIT = 280


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


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def preview_text(text: str, char_limit: int = PREVIEW_CHAR_LIMIT) -> str:
    text = normalize_text(text)
    if len(text) <= char_limit:
        return text
    return text[: char_limit - 3].rstrip() + "..."


def chunk_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for record in records:
        for chunk in record["retrieved_chunks"]:
            chunks.setdefault(str(chunk["chunk_id"]), chunk)
    return chunks


def chunk_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        for chunk in record["retrieved_chunks"]:
            ids.append(str(chunk["chunk_id"]))
    return ids


def page_summary(records: list[dict[str, Any]], limit: int = 5) -> list[str]:
    page_counts: Counter[int] = Counter()
    for record in records:
        for chunk in record["retrieved_chunks"]:
            page_counts[int(chunk["page_number"])] += 1

    ranked_pages = sorted(page_counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{page} ({count})" for page, count in ranked_pages[:limit]]


def group_expanded_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["question_id"])].append(record)
    return dict(grouped)


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


def write_markdown_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[Path, Path]:
    config = load_config(config_path)

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


if __name__ == "__main__":
    run()
