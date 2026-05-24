# SOTA RAG

SOTA RAG is an Agentic RAG Evaluator for technical documents and technical specifications.

The goal is not just to build a RAG chatbot. The goal is to evaluate which retrieval strategies work best
for technical document search.

This MVP tests whether Ollama-based query expansion improves retrieval coverage for technical document
RAG. It compares chunks retrieved by the original benchmark questions against chunks retrieved by
expanded search queries.

## Current MVP Scope

The current pipeline:

1. Loads one technical PDF from `data/sample.pdf`.
2. Extracts page-level text with PyMuPDF.
3. Chunks extracted text using configured chunk settings.
4. Generates embeddings with one configured Sentence Transformers model.
5. Stores chunks and embeddings in Chroma.
6. Loads five benchmark questions from JSONL.
7. Retrieves top-k chunks for each original question.
8. Uses one local Ollama model to generate expanded search queries.
9. Retrieves top-k chunks for each expanded query.
10. Compares original retrieval coverage with expanded-query retrieval coverage.

This MVP does not generate final answers. It only evaluates retrieval behavior.

## Environment Setup

Use Python 3.11+ in the existing conda environment:

```bash
conda activate agentic_rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The MVP uses:

- PyMuPDF for PDF extraction
- Sentence Transformers for embeddings
- Chroma for the local vector database
- Ollama for local query expansion
- PyYAML for configuration

For query expansion, Ollama must be running locally and the model configured in `configs/config.yaml`
must be available.

## Required Input Files

Prepare the input PDF:

```text
data/sample.pdf
```

Prepare benchmark questions:

```text
benchmark/benchmark_questions.jsonl
```

Each benchmark record should include:

```json
{"id": "q001", "question": "Example technical question?"}
```

Do not use proprietary or confidential documents.

## Configuration

The configuration file is:

```text
configs/config.yaml
```

It controls:

- PDF path
- Output directory
- Benchmark question path
- Chroma database directory
- Chunk size and overlap
- Embedding model name
- Chroma collection name
- Ollama model name and temperature
- Retrieval top-k
- Number of expanded queries

Model names, paths, chunk size, and top-k should be changed in the config file rather than hard-coded.

## MVP Execution Guide

Run the pipeline from the repository root.

1. Activate conda environment:

```bash
conda activate agentic_rag
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare input PDF:

```text
data/sample.pdf
```

4. Run PDF extraction:

```bash
python src/ingest.py
```

5. Run chunking:

```bash
python src/chunking.py
```

6. Build Chroma vector DB:

```bash
python src/vector_store.py
```

7. Run original query retrieval:

```bash
python src/retrieval.py --mode original
```

8. Run Ollama query expansion:

```bash
python src/query_expansion.py
```

9. Run expanded query retrieval:

```bash
python src/retrieval.py --mode expanded
```

10. Generate retrieval comparison report:

```bash
python src/evaluation.py
```

## Expected Output Files

The pipeline writes:

```text
outputs/extracted_pages.jsonl
outputs/chunks.jsonl
outputs/chroma_db/
outputs/original_retrieval_results.jsonl
outputs/query_expansions.jsonl
outputs/expanded_retrieval_results.jsonl
outputs/retrieval_comparison.csv
outputs/retrieval_comparison_report.md
```

The final Markdown report is:

```text
outputs/retrieval_comparison_report.md
```

It summarizes:

- Number of chunks retrieved by original questions
- Number of unique chunks retrieved by expanded queries
- Overlap between original and expanded retrieval
- New chunks found only by expanded queries
- Top pages retrieved by each method
- Text previews for manual review

## Current Limitations

- No gold evidence labels are used yet.
- The comparison measures retrieval coverage, not answer correctness.
- There is no ChatGPT Judge or LLM-based grading.
- Expanded query results are not fused or reranked.
- Chroma search uses the configured embedding model only.
- The MVP uses one PDF and five benchmark questions.
- Streamlit and LangGraph are intentionally not included.

## Phase 2: Docling Comparison Plan

Phase 2 will add Docling as an alternative document processing pipeline. Docling should not replace the
current PyMuPDF baseline. The project should compare:

- PyMuPDF baseline extraction and chunking
- Docling structured extraction and structure-aware chunking

Docling is not an embedding model. It should be used for structured document parsing, preserving document
layout, sections, tables, and field context before chunking. Embedding models remain separate and
configurable.

For a fair comparison, both pipelines should use:

- The same input PDF
- The same benchmark questions
- The same embedding model
- The same retrieval `top_k`
- The same evaluation/reporting format

Parser-specific outputs should be stored separately. PyMuPDF chunks and Docling chunks should not be
mixed in the same Chroma collection. Each parser and embedding model combination should use its own
collection, for example:

- `sample_doc_pymupdf_all_minilm_l6_v2`
- `sample_doc_docling_all_minilm_l6_v2`

The goal of the Docling comparison is to measure whether structured parsing and structure-aware chunking
improve:

- Retrieval coverage
- Section preservation
- Table and field context
- Manual answerability of retrieved evidence

Docling should be added only after the PyMuPDF baseline remains reproducible.

## Future Extension Plan

Possible next steps:

1. Add benchmark gold evidence labels such as expected pages, sections, or chunk IDs.
2. Add retrieval metrics such as recall@k, hit rate, and MRR.
3. Add a report section that compares original and expanded retrieval against gold evidence.
4. Add Docling structured parsing as a separate pipeline for comparison with PyMuPDF.
5. Add answer generation after retrieval quality is measurable.
6. Add optional judge-based evaluation after deterministic retrieval metrics exist.
7. Test additional embedding models using separate Chroma collections.
8. Add reranking or fusion methods after the baseline query expansion comparison is stable.
9. Add a UI only after the command-line MVP is reliable.
