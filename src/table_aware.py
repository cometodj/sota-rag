from __future__ import annotations

import re
from typing import Any


TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
PIPE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
TABLE_TERMS = ["Value", "Description", "Bit", "Bits", "Field", "Fields", "Reserved"]
VALUE_CODE_PATTERN = re.compile(r"\b(?:[01]{2,8}b|[0-9A-Fa-f]{2,8}h|\d{1,2}:\d{1,2})\b")
FIELD_ALIAS_MAP = {
    "sanitize capabilities": [
        "SANCAP",
        "Sanitize Capabilities",
        "sanitize operation capabilities",
        "Identify Controller Sanitize Capabilities",
    ],
    "secure erase settings": [
        "SES",
        "Secure Erase Settings",
        "Format NVM Secure Erase Settings",
    ],
}


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
    if chunk_type in {"table", "table_fragment", "possible_table", "table_field"}:
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
            existing_type = str(chunk.get("chunk_type") or "").casefold()
            if len(group) > 1 and existing_type not in {"table", "table_field"}:
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


def split_markdown_row(row: str) -> list[str]:
    value = str(row or "").strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip().replace("<br>", " ") for cell in value.split("|")]


def markdown_table_rows(table_markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(table_markdown or "").splitlines():
        if not PIPE_ROW_PATTERN.match(line):
            continue
        if TABLE_SEPARATOR_PATTERN.match(line):
            continue
        cells = split_markdown_row(line)
        if any(cell for cell in cells):
            rows.append(cells)
    return rows


def field_aliases_for_name(field_name: str) -> list[str]:
    normalized = str(field_name or "").casefold()
    aliases: list[str] = []
    for key, values in FIELD_ALIAS_MAP.items():
        if key in normalized or any(str(alias).casefold() in normalized for alias in values):
            aliases.extend(values)
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        key = str(alias).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(str(alias))
    return unique


def canonical_field_name(row_text: str, cells: list[str]) -> str:
    normalized = str(row_text or "").casefold()
    if "sanitize capabilities" in normalized or "sancap" in normalized:
        return "Sanitize Capabilities"
    if "secure erase settings" in normalized or re.search(r"\bSES\b", row_text):
        return "Secure Erase Settings"
    for cell in cells:
        candidate = re.sub(r"\s+", " ", str(cell or "")).strip()
        if not candidate:
            continue
        if re.fullmatch(r"[0-9A-Fa-f:]+h?|[01]{2,8}b|\d+:\d+", candidate):
            continue
        if len(candidate) > 120:
            continue
        if re.search(r"[A-Za-z][A-Za-z ]{2,}", candidate):
            return candidate
    return re.sub(r"\s+", " ", str(row_text or "")).strip()[:120]


def meaningful_table_field_row(row_text: str, cells: list[str]) -> bool:
    normalized = str(row_text or "").casefold()
    if any(key in normalized for key in FIELD_ALIAS_MAP):
        return True
    if re.search(r"\b(?:SANCAP|SES)\b", row_text):
        return True
    alpha_cells = [cell for cell in cells if re.search(r"[A-Za-z]{3,}", cell)]
    if len(alpha_cells) >= 2 and len(row_text.strip()) >= 24:
        return True
    return False


def create_table_field_chunks(
    *,
    parent_chunk: dict[str, Any],
    table_markdown: str,
    parent_table_id: str,
    parent_table_title: str,
    source_parser: str,
    start_chunk_index: int,
) -> list[dict[str, Any]]:
    rows = markdown_table_rows(table_markdown)
    if len(rows) < 2:
        return []
    field_chunks: list[dict[str, Any]] = []
    header = rows[0]
    for row_index, cells in enumerate(rows[1:]):
        row_text = " | ".join(cell for cell in cells if cell).strip()
        if not meaningful_table_field_row(row_text, cells):
            continue
        field_name = canonical_field_name(row_text, cells)
        aliases = field_aliases_for_name(field_name + " " + row_text)
        text_for_embedding = "\n".join(
            part
            for part in [
                parent_table_title,
                f"Section: {parent_chunk.get('section_title')}" if parent_chunk.get("section_title") else "",
                f"Field: {field_name}",
                f"Aliases: {', '.join(aliases)}" if aliases else "",
                f"Columns: {' | '.join(header)}" if header else "",
                f"Row: {row_text}",
                str(parent_chunk.get("nearby_context") or ""),
            ]
            if part
        )
        chunk = {
            "chunk_id": f"{parent_chunk['chunk_id']}:f{row_index:04d}",
            "document_name": parent_chunk.get("document_name", ""),
            "section_title": parent_chunk.get("section_title", ""),
            "chunk_index": start_chunk_index + len(field_chunks),
            "chunk_type": "table_field",
            "table_id": parent_table_id,
            "parent_table_id": parent_table_id,
            "parent_table_title": parent_table_title,
            "field_name": field_name,
            "field_aliases": aliases,
            "text": text_for_embedding,
            "text_for_embedding": text_for_embedding,
            "char_count": len(text_for_embedding),
            "source": parent_chunk.get("source", source_parser),
            "source_parser": source_parser,
            "nearby_context": parent_chunk.get("nearby_context", ""),
            "table_markdown": table_markdown,
        }
        if parent_chunk.get("page_number") not in (None, ""):
            chunk["page_number"] = parent_chunk.get("page_number")
        field_chunks.append(chunk)
    return field_chunks
