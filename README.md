# Research Assistant LLM Chatbot

*April 2025*

An LLM-powered research assistant that discovers academic papers, ingests PDF content, and answers questions grounded in that material—not generic web knowledge.

## Problem and motivation

Researchers spend significant time finding and reading papers before they can discuss or compare ideas. This project reduces that overhead by automating discovery from arXiv, structuring full-text content for retrieval, and supporting follow-up dialogue that stays tied to the papers in scope.

## Capabilities

- **Automated paper retrieval** — Fetch papers from arXiv using filters such as query terms, authors, categories, abstract keywords, submission dates, and exclusion terms. Duplicates are skipped across sessions by tracking previously retrieved titles.
- **PDF upload** — Ingest private, unpublished, or non-indexed PDFs through the same chunking and indexing pipeline as arXiv papers.
- **Context-aware Q&A** — Combine retrieved document chunks with recent conversation history so users can ask follow-up questions and explore paper content in depth.
- **Per-chat isolation** — Each chat session uses its own vector index so retrieval only covers papers requested or uploaded in that session.

## How it works

At a high level, the system follows a retrieval-augmented generation (RAG) pipeline:

```
arXiv search / PDF upload → text extraction → chunking + metadata → embeddings → Pinecone → retrieval + prompt → GPT-4o-mini → Gradio UI
```

**Data preparation.** Papers are downloaded from arXiv (or supplied as uploads), parsed with LangChain’s `PyPDFLoader`, and split into overlapping chunks (300 characters, 100 overlap) with metadata: title, author(s), publication date, chunk index, and source. Uploaded PDFs use the filename as title and placeholder metadata where formal fields are missing.

**Embedding and indexing.** Chunks are embedded in batches (up to 200 per batch) with OpenAI `text-embedding-3-small` and stored in Pinecone with metadata. Indexes are scoped per chat so unrelated sessions do not leak into retrieval.

**Retrieval and generation.** User queries are classified and optionally filtered by metadata (e.g. author, date range). Relevant chunks are retrieved from Pinecone (with a short retry on cold-start), formatted into a prompt with up to ten prior exchanges, and truncated if needed to respect token limits. Answers are produced with `gpt-4o-mini` and appended to the conversation history for coherent multi-turn dialogue.

## Technology stack


| Layer          | Technologies                                                                        |
| -------------- | ----------------------------------------------------------------------------------- |
| Language & API | Python, FastAPI                                                                     |
| UI             | Gradio                                                                              |
| Storage        | MongoDB (sessions/users), Pinecone (vectors)                                        |
| ML / NLP       | OpenAI embeddings (`text-embedding-3-small`), ChatOpenAI (`gpt-4o-mini`), LangChain |
| Quality        | Pytest, ClearML, Evidently (evaluation notebooks)                                   |


## Repository layout


| Path                | Role                                                      |
| ------------------- | --------------------------------------------------------- |
| `app/`              | FastAPI application and routes                            |
| `backend/`          | Ingestion, embedding, retrieval, database, PDF processing |
| `gradio_app.py`     | Chat UI                                                   |
| `app.py`            | Entry point (FastAPI + Gradio)                            |
| `pytest_scripts/`   | Automated tests                                           |
| `model_evaluation/` | Embedding and LLM evaluation experiments                  |


## Evaluation

The system was evaluated on a manually curated benchmark of 30 academic Q&A pairs with reference paragraphs.

- **Embeddings** — Compared `text-embedding-3-small`, `text-embedding-3-large`, and `all-MiniLM-L6-v2` using cosine similarity to ground-truth passages, including paraphrased queries for robustness.
- **LLM** — Assessed with Evidently (relevance to retrieved docs, alignment with the question, similarity to reference answers) under conservative, balanced, and open generation settings, plus manual edge-case testing (long queries, typos, ambiguous prompts, bundled questions).

Tracked experiments and plots: [ClearML evaluation dashboard](https://app.5ccsagap.er.kcl.ac.uk/projects/4581c1b73366408b92ac361df3110a7c/experiments/a58d7c69df8444c291f584f73153a336/info-output/metrics/plots?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-started&filter=).

## Known limitations

- Does not reliably correct users when paper titles are wrong or misspelled.
- When asked for direct excerpts, responses may summarize metadata rather than quote or paraphrase specific passages.
- Very long inputs and strict output formats (e.g. JSON) are handled inconsistently.
- Figures, tables, and other visual content in PDFs are not analyzed.

## Team

Omed Amiri, Khaled Albuainain, Jingwen Guo, Abdullah AL Mu'adh, Takumi Matsukura, Mateo Marthoz Blanco

