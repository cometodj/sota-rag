from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from table_aware import add_table_metadata, extract_markdown_table_blocks


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DOCLING_EXTRACTED_FILENAME = "docling_extracted.jsonl"
DOCLING_CHUNKS_FILENAME = "chunks_docling.jsonl"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

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


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks


def create_chunks(
    records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    per_document_chunk_counts: dict[str, int] = {}

    for record_index, record in enumerate(records):
        document_name = str(record["document_name"])
        section_title = record.get("section_title")
        text = str(record.get("text", ""))
        chunk_start_index = per_document_chunk_counts.get(document_name, 0)
        table_blocks = extract_markdown_table_blocks(text)
        table_ranges = {
            (int(block["start_line"]), int(block["end_line"]))
            for block in table_blocks
        }
        lines = text.splitlines()

        text_segments: list[str] = []
        cursor = 0
        for start, end in sorted(table_ranges):
            segment = "\n".join(lines[cursor:start]).strip()
            if segment:
                text_segments.append(segment)
            cursor = end + 1
        trailing_segment = "\n".join(lines[cursor:]).strip()
        if trailing_segment:
            text_segments.append(trailing_segment)

        if not table_blocks:
            text_segments = [text]

        offset = 0
        for segment in text_segments:
            for chunk_text in split_text(segment, chunk_size, chunk_overlap):
                chunk_index = chunk_start_index + offset
                chunk = {
                    "chunk_id": f"{document_name}:docling:r{record_index:04d}:c{offset:04d}",
                    "document_name": document_name,
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "source": "docling",
                }
                add_table_metadata(chunk, source_parser="docling")
                chunks.append(chunk)
                offset += 1

        for table_offset, table_block in enumerate(table_blocks):
            table_markdown = str(table_block["table_markdown"])
            nearby_context = str(table_block.get("nearby_context") or "")
            caption = table_block.get("caption")
            chunk_text = "\n\n".join(
                part
                for part in [
                    f"Section: {section_title}" if section_title else "",
                    f"Caption: {caption}" if caption else "",
                    nearby_context,
                    table_markdown,
                ]
                if part
            )
            chunk_index = chunk_start_index + offset
            chunk = {
                "chunk_id": f"{document_name}:docling:r{record_index:04d}:c{offset:04d}",
                "document_name": document_name,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "source": "docling",
            }
            add_table_metadata(
                chunk,
                source_parser="docling",
                table_id=f"{document_name}:docling:r{record_index:04d}:t{table_offset:04d}",
                table_markdown=table_markdown,
                nearby_context=nearby_context,
                caption=str(caption) if caption else None,
            )
            chunks.append(chunk)
            offset += 1

        per_document_chunk_counts[document_name] = chunk_start_index + offset

    return chunks


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)

    output_dir = Path(config["paths"]["output_dir"])
    input_path = output_dir / DOCLING_EXTRACTED_FILENAME
    output_path = output_dir / DOCLING_CHUNKS_FILENAME
    chunk_size = int(config["chunking"]["chunk_size"])
    chunk_overlap = int(config["chunking"]["chunk_overlap"])

    records = read_jsonl(input_path)
    chunks = create_chunks(records, chunk_size, chunk_overlap)
    write_jsonl(chunks, output_path)

    print(f"Loaded {len(records)} Docling extraction records from {input_path}")
    print(f"Created {len(chunks)} Docling chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"Saved Docling chunks to {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chunks from Docling extraction output.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(config_path=args.config)
