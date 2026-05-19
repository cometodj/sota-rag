# AGENTS.md

This project is called SOTA RAG.

It is an Agentic RAG Evaluator for technical documents and technical specifications.

The first MVP should evaluate whether query expansion improves retrieval quality.

Rules:
- Use Python 3.11+.
- Keep the code modular and easy to test.
- Do not hard-code model names.
- Use config files for model names, paths, chunk size, and top-k.
- Do not mix vectors from different embedding models in the same vector DB collection.
- Store experiment outputs as JSONL and Markdown.
- Do not use proprietary or confidential documents.
- Do not build Streamlit yet.
- Do not add LangGraph yet.
- Start with a minimal MVP.

MVP scope:
1. Load one technical PDF from the data/ folder.
2. Extract text using PyMuPDF.
3. Chunk the extracted text.
4. Generate embeddings using one embedding model.
5. Store chunks in Chroma.
6. Load five benchmark questions.
7. Use one Ollama local LLM to generate query expansions.
8. Retrieve top-k chunks using both the original question and expanded queries.
9. Save retrieval results.
10. Generate a simple Markdown report.