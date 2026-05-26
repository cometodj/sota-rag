from __future__ import annotations

import re
from typing import Any


TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
PIPE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
TABLE_TERMS = ["Value", "Description", "Bit", "Bits", "Field", "Fields", "Reserved"]


def is_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    table_rows = [line for line in lines if PIPE_ROW_PATTERN.match(line)]
    has_separator = any(TABLE_SEPARATOR_PATTERN.match(line) for line in lines)
    return len(table_rows) >= 2 and has_separator


def is_possible_table_text(text: str) -> bool:
    value = str(text or "")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if is_markdown_table(value):
        return True
    if value.count("|") >= 4:
        return True
    table_like_lines = [
        line
        for line in lines
        if len(re.split(r"\s{2,}|\t", line)) >= 3
    ]
    if len(table_like_lines) >= 3:
        return True
    term_count = sum(1 for term in TABLE_TERMS if re.search(rf"\b{re.escape(term)}\b", value))
    return term_count >= 2


def get_nearby_context(
    lines: list[str],
    table_start: int,
    table_end: int,
    context_lines: int = 3,
) -> str:
    before_start = max(0, table_start - context_lines)
    after_end = min(len(lines), table_end + 1 + context_lines)
    context = lines[before_start:table_start] + lines[table_end + 1:after_end]
    return "\n".join(line for line in context if line.strip()).strip()


def table_caption(lines: list[str], table_start: int) -> str | None:
    for index in range(table_start - 1, max(-1, table_start - 5), -1):
        candidate = lines[index].strip()
        if not candidate:
            continue
        if re.search(r"\b(Table|Figure)\b", candidate, flags=re.IGNORECASE):
            return candidate.lstrip("#").strip()
        if len(candidate) <= 160:
            return candidate.lstrip("#").strip()
    return None


def extract_markdown_table_blocks(markdown_text: str) -> list[dict[str, Any]]:
    lines = str(markdown_text or "").splitlines()
    blocks: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        if not PIPE_ROW_PATTERN.match(lines[index]):
            index += 1
            continue

        start = index
        current = index
        has_separator = False
        while current < len(lines) and PIPE_ROW_PATTERN.match(lines[current]):
            if TABLE_SEPARATOR_PATTERN.match(lines[current]):
                has_separator = True
            current += 1

        end = current - 1
        if has_separator and end > start:
            markdown = "\n".join(lines[start : end + 1]).strip()
            blocks.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "table_markdown": markdown,
                    "nearby_context": get_nearby_context(lines, start, end),
                    "caption": table_caption(lines, start),
                }
            )
        index = max(current, index + 1)

    return blocks


def classify_chunk_text(text: str) -> str:
    if is_markdown_table(text):
        return "table"
    if is_possible_table_text(text):
        return "possible_table"
    return "text"


def add_table_metadata(
    chunk: dict[str, Any],
    *,
    source_parser: str,
    text: str | None = None,
    table_id: str | None = None,
    table_markdown: str | None = None,
    nearby_context: str | None = None,
    caption: str | None = None,
) -> dict[str, Any]:
    chunk_text = str(text if text is not None else chunk.get("text", ""))
    chunk_type = "table" if table_markdown else classify_chunk_text(chunk_text)
    chunk["chunk_type"] = chunk_type
    chunk["source_parser"] = source_parser

    if table_id:
        chunk["table_id"] = table_id
    if table_markdown:
        chunk["table_markdown"] = table_markdown
    if nearby_context:
        chunk["nearby_context"] = nearby_context
    if caption:
        chunk["caption"] = caption

    return chunk
