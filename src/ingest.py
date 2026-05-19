from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_FILENAME = "extracted_pages.jsonl"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict[str, Any]] = []
    document_name = pdf_path.name

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text()
            pages.append(
                {
                    "document_name": document_name,
                    "page_number": page_index,
                    "text": text,
                    "char_count": len(text),
                }
            )

    return pages


def write_pages_jsonl(pages: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for page in pages:
            file.write(json.dumps(page, ensure_ascii=False) + "\n")


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)

    pdf_path = Path(config["document"]["pdf_path"])
    output_dir = Path(config["paths"]["output_dir"])
    output_path = output_dir / OUTPUT_FILENAME

    pages = extract_pdf_pages(pdf_path)
    write_pages_jsonl(pages, output_path)

    total_chars = sum(page["char_count"] for page in pages)
    print(f"Extracted {len(pages)} pages from {pdf_path}")
    print(f"Total characters: {total_chars}")
    print(f"Saved page-level text to {output_path}")

    return output_path


if __name__ == "__main__":
    run()
