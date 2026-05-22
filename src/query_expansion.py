from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_FILENAME = "query_expansions.jsonl"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def read_benchmark_questions(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark questions JSONL not found: {input_path}")

    questions: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line must be an object at {input_path}:{line_number}")
            if "id" not in record or "question" not in record:
                raise ValueError(f"Missing id or question at {input_path}:{line_number}")

            questions.append(
                {
                    "id": str(record["id"]),
                    "question": str(record["question"]),
                }
            )

    return questions


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(question: str, num_queries: int) -> str:
    return f"""You generate search queries for technical document retrieval.

Task:
- Rewrite the original question into exactly {num_queries} expanded search queries.
- Do not answer the question.
- Only generate search queries.
- Preserve technical terms from the original question.
- Make each query useful for retrieving relevant sections from a technical specification.
- Return only a JSON array of strings. Do not include Markdown, explanations, keys, or numbering.

Original question:
{question}
"""


def parse_expanded_queries(raw_response: str, num_queries: int) -> list[str]:
    parsed = parse_json_array(raw_response)
    if parsed:
        return normalize_queries(parsed, num_queries)

    fallback_queries = []
    for line in raw_response.splitlines():
        line = line.strip()
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line and not line.startswith("[") and not line.endswith("]"):
            fallback_queries.append(line)

    return normalize_queries(fallback_queries, num_queries)


def parse_json_array(raw_response: str) -> list[str]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw_response)
        if not match:
            return []

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, str)]


def normalize_queries(queries: list[str], num_queries: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for query in queries:
        query = " ".join(query.split())
        if not query:
            continue

        key = query.casefold()
        if key in seen:
            continue

        normalized.append(query)
        seen.add(key)

        if len(normalized) == num_queries:
            break

    return normalized


def generate_expanded_queries(
    question: str,
    model_name: str,
    num_queries: int,
    temperature: float,
) -> tuple[list[str], str]:
    prompt = build_prompt(question, num_queries)
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to generate query expansions with Ollama. "
            f"Check that Ollama is running and model '{model_name}' is available."
        ) from exc

    raw_response = str(response["response"]).strip()

    return parse_expanded_queries(raw_response, num_queries), raw_response


def expand_questions(
    questions: list[dict[str, str]],
    model_name: str,
    num_queries: int,
    temperature: float,
) -> list[dict[str, Any]]:
    if num_queries <= 0:
        raise ValueError("query_expansion.num_queries must be greater than 0")

    records: list[dict[str, Any]] = []
    for question in questions:
        expanded_queries, raw_response = generate_expanded_queries(
            question=question["question"],
            model_name=model_name,
            num_queries=num_queries,
            temperature=temperature,
        )
        records.append(
            {
                "question_id": question["id"],
                "original_question": question["question"],
                "model_name": model_name,
                "expanded_queries": expanded_queries,
                "raw_response": raw_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return records


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)

    benchmark_path = Path(config["paths"]["benchmark_questions"])
    output_dir = Path(config["paths"]["output_dir"])
    output_path = output_dir / OUTPUT_FILENAME
    model_name = str(config["ollama"]["model_name"])
    temperature = float(config["ollama"]["temperature"])
    num_queries = int(config["query_expansion"]["num_queries"])

    questions = read_benchmark_questions(benchmark_path)
    records = expand_questions(
        questions=questions,
        model_name=model_name,
        num_queries=num_queries,
        temperature=temperature,
    )
    write_jsonl(records, output_path)

    print(f"Loaded {len(questions)} benchmark questions from {benchmark_path}")
    print(f"Generated up to {num_queries} expanded queries per question with {model_name}")
    print(f"Saved query expansions to {output_path}")

    return output_path


if __name__ == "__main__":
    run()
