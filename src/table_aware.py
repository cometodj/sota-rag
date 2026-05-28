from __future__ import annotations

import re
from typing import Any


TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
PIPE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
TABLE_TERMS = ["Value", "Description", "Bit", "Bits", "Field", "Fields", "Reserved"]
VALUE_CODE_PATTERN = re.compile(r"\b(?:[01]{2,8}b|[0-9A-Fa-f]{2,8}h|\d{1,2}:\d{1,2})\b")


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


def slugify_table_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or "table"


def extract_value_codes(text: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for match in VALUE_CODE_PATTERN.findall(str(text or "")):
        normalized = match.strip()
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        codes.append(normalized)
    return codes


def is_table_like_chunk(chunk: dict[str, Any]) -> bool:
    if not isinstance(chunk, dict):
        return False
    chunk_type = str(chunk.get("chunk_type") or "").casefold()
    if chunk_type in {"table", "table_fragment", "possible_table"}:
        return True
    return is_possible_table_text(str(chunk.get("text", "")))


def table_group_base(chunk: dict[str, Any], group_index: int) -> str:
    document_name = slugify_table_id(str(chunk.get("document_name") or "document"))
    source_parser = slugify_table_id(
        str(chunk.get("source_parser") or chunk.get("parser") or chunk.get("source") or "unknown")
    )
    section_or_caption = chunk.get("caption") or chunk.get("section_title") or ""
    if section_or_caption:
        context = slugify_table_id(str(section_or_caption))[:48]
    elif chunk.get("page_number") not in (None, ""):
        context = f"p{int(chunk['page_number']):04d}"
    else:
        context = "unsectioned"
    return f"table_{document_name}_{source_parser}_{context}_g{group_index:04d}"


def same_table_group_candidate(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if str(previous.get("document_name") or "") != str(current.get("document_name") or ""):
        return False

    previous_source = str(previous.get("source_parser") or previous.get("parser") or previous.get("source") or "")
    current_source = str(current.get("source_parser") or current.get("parser") or current.get("source") or "")
    if previous_source and current_source and previous_source != current_source:
        return False

    previous_index = previous.get("chunk_index")
    current_index = current.get("chunk_index")
    try:
        if int(current_index) - int(previous_index) != 1:
            return False
    except (TypeError, ValueError):
        return False

    previous_section = previous.get("section_title")
    current_section = current.get("section_title")
    if previous_section and current_section and previous_section != current_section:
        return False

    previous_page = previous.get("page_number")
    current_page = current.get("page_number")
    if previous_page not in (None, "") and current_page not in (None, ""):
        if str(previous_page) == str(current_page):
            return True
        if previous_section and previous_section == current_section:
            return True
        previous_codes = extract_value_codes(str(previous.get("text", "")))
        current_codes = extract_value_codes(str(current.get("text", "")))
        return bool(previous_codes and current_codes)

    return True


def parent_table_text(group: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    dict_group = [chunk for chunk in group if isinstance(chunk, dict)]
    for chunk in sorted(dict_group, key=lambda item: int(item.get("chunk_index") or 0)):
        text = str(chunk.get("table_markdown") or chunk.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def assign_table_group_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue
        if not is_table_like_chunk(chunk):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        if current_group and same_table_group_candidate(current_group[-1], chunk):
            current_group.append(chunk)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [chunk]

    if current_group:
        groups.append(current_group)

    for group_index, group in enumerate(groups):
        existing_group_id = next(
            (
                str(chunk.get("parent_table_id") or chunk.get("table_id"))
                for chunk in group
                if chunk.get("parent_table_id") or chunk.get("table_id")
            ),
            "",
        )
        group_id = existing_group_id or table_group_base(group[0], group_index)
        combined_text = parent_table_text(group)
        value_codes = extract_value_codes(combined_text)

        for fragment_index, chunk in enumerate(group):
            if not chunk.get("table_id"):
                chunk["table_id"] = group_id
            if not chunk.get("parent_table_id"):
                chunk["parent_table_id"] = group_id
            chunk["table_group_index"] = group_index
            chunk["table_fragment_index"] = fragment_index
            if len(group) > 1 and str(chunk.get("chunk_type") or "").casefold() != "table":
                chunk["chunk_type"] = "table_fragment"
            elif not chunk.get("chunk_type"):
                chunk["chunk_type"] = classify_chunk_text(str(chunk.get("text", "")))
            if combined_text:
                chunk["parent_table_text"] = combined_text
            if value_codes:
                chunk["table_value_codes"] = value_codes

    return chunks


def add_table_metadata(
    chunk: dict[str, Any],
    *,
    source_parser: str,
    text: str | None = None,
    table_id: str | None = None,
    parent_table_id: str | None = None,
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
        chunk["parent_table_id"] = parent_table_id or table_id
    elif parent_table_id:
        chunk["parent_table_id"] = parent_table_id
    if table_markdown:
        searchable_parts = [
            str(chunk.get("section_title") or ""),
            str(caption or ""),
            str(nearby_context or ""),
            table_markdown,
        ]
        searchable_text = "\n\n".join(part for part in searchable_parts if part.strip())
        if searchable_text and table_markdown not in chunk_text:
            chunk["text"] = "\n\n".join(part for part in [chunk_text, searchable_text] if part.strip())
            chunk["char_count"] = len(str(chunk["text"]))
        chunk["table_markdown"] = table_markdown
        chunk.setdefault("full_table_markdown", table_markdown)
    if nearby_context:
        chunk["nearby_context"] = nearby_context
    if caption:
        chunk["caption"] = caption

    return chunk
