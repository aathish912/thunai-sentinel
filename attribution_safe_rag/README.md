# Thunai Sentinel

Every Answer. Every Source. Fully Verified.

Thunai Sentinel is an enterprise knowledge verification platform for grounded AI answers across support content, internal documentation, policies, knowledge bases, and customer intelligence. It combines retrieval, verification, attribution tracking, and evidence grounding so organizations can trust what the system returns.

## Problem

Enterprise AI systems often retrieve text that is semantically similar but belongs to the wrong person, document, or business context. That creates attribution drift, source confusion, and hallucinated ownership.

Thunai Sentinel is designed around a stricter principle:

**Correct attribution matters more than semantic similarity.**

The platform preserves document lineage, stores Knowledge Evidence in MongoDB Atlas, verifies ownership and metadata consistency, and only lets the LLM present facts that are already supported.

## Architecture

```text
Documents
  ↓
Offline Processing
  ↓
Knowledge Evidence
  ↓
MongoDB Atlas
  ↓
Vector Search
  ↓
Hybrid Retrieval
  ↓
Source Context Expansion
  ↓
Query Router
  ↓
Sentinel Verification Engine
  ↓
Sentinel Response Engine
  ↓
Grounded Answer
```

## Retrieval Pipeline

Thunai Sentinel uses the following retrieval path:

- Offline ingestion converts `.txt` files and PDFs into Knowledge Evidence.
- Each evidence record preserves `doc_id`, `workspace_id`, `source_filename`, `document_type`, ownership fields, section metadata, and embeddings.
- MongoDB Atlas Vector Search provides top candidates.
- Hybrid scoring combines vector similarity and keyword matching.
- Intent-aware reranking improves outcomes for ownership, procedure, summary, comparison, and general queries.
- Child evidence is expanded into Source Context for answer construction and review.

Terminology used in the platform:

- `Retriever` → `Sentinel Discovery Engine`
- `Verifier` → `Sentinel Verification Engine`
- `Generator` → `Sentinel Response Engine`
- `Evidence Units` → `Knowledge Evidence`
- `Parent Context` → `Source Context`

## Verification Layer

The Sentinel Verification Engine decides truth. It does not let the LLM determine ownership or provenance.

It performs:

- metadata consistency checks across retrieved evidence
- entity detection for ownership-sensitive queries
- ownership verification for candidate, project, or entity attribution
- insufficient-evidence handling when support is weak or ambiguous

The Sentinel Response Engine only rewrites verified facts into clearer language. If the evidence is insufficient, the answer must say `insufficient evidence`.

## Evaluation Results

Current benchmark performance:

- Wrong Attribution Rate: `0.00%`
- Retrieval Hit Rate: `100%`
- Ownership Accuracy: `100%`
- Query Type Accuracy: `100%`

The evaluation framework checks:

- retrieval accuracy
- attribution correctness
- wrong-owner prevention
- grounded answer behavior
- comparison handling

Run evaluation with:

```bash
python eval.py --top-k 5
```

## Setup

```bash
cd attribution_safe_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Example `.env` values:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=attribution_safe_rag
WORKSPACE_ID=thunai-demo
```

## How to Run

Ingest sample documents:

```bash
python ingest.py --docs-dir sample_docs
```

Run retrieval:

```bash
python retrieve.py "Which resume has the Fraud Detection System project?" --top-k 5
python retrieve.py "Who built the Credit Risk Dashboard?" --top-k 5
python retrieve.py "Where is SSO setup explained?" --top-k 5
```

Build a grounded answer:

```bash
python answer.py "Which resume has the Fraud Detection System project?" --top-k 5
python answer.py "Who built the Credit Risk Dashboard?" --top-k 5
python answer.py "Where is SSO setup explained?" --top-k 5
```

Run the Streamlit dashboard:

```bash
python -m streamlit run demo_app.py --server.fileWatcherType none
```

The ingestion pipeline supports text files and PDFs. PDFs are processed offline into Knowledge Evidence, and the same metadata, retrieval, and verification pipeline applies to both formats.

If a PDF has no extractable text, the ingester prints:

`No extractable text found. OCR is not supported yet.`

## Screenshots

Suggested screenshots for the product overview:

- dashboard home view
- uploaded Knowledge Sources list
- grounded answer with Knowledge Evidence citations
- Sentinel Verification Engine panel
- evaluation snapshot with benchmark metrics

## Future Work

- OCR support for scanned PDFs
- deeper enterprise policy evaluation sets
- access control and workspace-level audit views
- richer dashboard analytics and usage insights
- production deployment workflows and monitoring

## Product Summary

Thunai Sentinel is built to feel like an enterprise AI platform rather than a generic RAG demo. It preserves existing ingestion, MongoDB Atlas integration, upload workflows, evaluation, retrieval, and verification behavior while emphasizing trust, source lineage, and fully verified answers.
