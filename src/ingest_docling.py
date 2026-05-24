from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from docling.document_converter import DocumentConverter


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
MARKDOWN_OUTPUT_FILENAME = "docling_document.md"
JSONL_OUTPUT_FILENAME = "docling_extracted.jsonl"
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()

    if not isinstance(markdown, str):
        raise TypeError("Docling markdown export did not return a string.")

    return markdown


def markdown_sections(markdown: str, document_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def append_current_section() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return

        records.append(
            {
                "document_name": document_name,
                "section_title": current_title,
                "text": text,
                "char_count": len(text),
                "source": "docling",
            }
        )

    for line in markdown.splitlines():
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            append_current_section()
            current_title = heading_match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    append_current_section()

    if records:
        return records

    fallback_text = markdown.strip()
    if not fallback_text:
        return []

    return [
        {
            "document_name": document_name,
            "section_title": None,
            "text": fallback_text,
            "char_count": len(fallback_text),
            "source": "docling",
        }
    ]


def write_markdown(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[Path, Path]:
    config = load_config(config_path)

    pdf_path = Path(config["document"]["pdf_path"])
    output_dir = Path(config["paths"]["output_dir"])
    markdown_path = output_dir / MARKDOWN_OUTPUT_FILENAME
    jsonl_path = output_dir / JSONL_OUTPUT_FILENAME

    markdown = convert_pdf_to_markdown(pdf_path)
    records = markdown_sections(markdown, document_name=pdf_path.name)

    write_markdown(markdown, markdown_path)
    write_jsonl(records, jsonl_path)

    total_chars = sum(record["char_count"] for record in records)
    print(f"Converted {pdf_path} with Docling")
    print(f"Saved Docling Markdown to {markdown_path}")
    print(f"Saved {len(records)} Docling extraction records to {jsonl_path}")
    print(f"Total extracted characters: {total_chars}")

    return markdown_path, jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Docling document extraction.")
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
