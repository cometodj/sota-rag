from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
EXTRACTED_PAGES_FILENAME = "extracted_pages.jsonl"
CHUNKS_FILENAME = "chunks.jsonl"


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
    pages: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for page in pages:
        document_name = str(page["document_name"])
        page_number = int(page["page_number"])
        text = str(page.get("text", ""))

        for chunk_index, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document_name}:p{page_number:04d}:c{chunk_index:04d}",
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )

    return chunks


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)

    output_dir = Path(config["paths"]["output_dir"])
    input_path = output_dir / EXTRACTED_PAGES_FILENAME
    output_path = output_dir / CHUNKS_FILENAME
    chunk_size = int(config["chunking"]["chunk_size"])
    chunk_overlap = int(config["chunking"]["chunk_overlap"])

    pages = read_jsonl(input_path)
    chunks = create_chunks(pages, chunk_size, chunk_overlap)
    write_jsonl(chunks, output_path)

    print(f"Loaded {len(pages)} pages from {input_path}")
    print(f"Created {len(chunks)} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"Saved chunks to {output_path}")

    return output_path


if __name__ == "__main__":
    run()
