# SOTA RAG

SOTA RAG is an Agentic RAG Evaluator for technical documents and technical specifications.

The goal is not just to build a RAG chatbot.  
The goal is to evaluate which models and retrieval strategies work best for technical document search.

Initial MVP:
- One PDF
- Five benchmark questions
- One embedding model
- One local LLM through Ollama
- Chroma vector database
- Original query vs expanded query retrieval comparison

## MVP Workflow

1. Load PDF
2. Extract text
3. Chunk text
4. Generate embeddings
5. Store chunks in Chroma
6. Load benchmark questions
7. Generate query expansions with Ollama
8. Retrieve chunks using original and expanded queries
9. Save results
10. Generate report
