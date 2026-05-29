# LLM-KG (knowledge graph)

LLM-KG is a persistent reasoning layer for AI agents. It turns raw documents into an inspectable LLM Wiki, then compiles that wiki into a lightweight knowledge graph made of claims, evidence, entities, and typed relations.

The goal is not just to summarize documents. The goal is to make knowledge reusable, traceable, and computable across future agent workflows.

```text
TXT/MD/DOCX/PDF source
  -> Markdown wiki pages
  -> evidence-backed claims
  -> entities and typed relations
  -> PostgreSQL + pgvector retrieval store
  -> basic/local query, reasoning traces, lint, and evolution workflows
```

## System Architecture

![LLM-KG system architecture](docs/system-architecture.svg)

## Problem

Traditional RAG is query-time synthesis: every question searches raw chunks and asks an LLM to assemble an answer from scratch. That works for quick lookup, but it has weak memory. Important facts, contradictions, entity relationships, and reasoning paths are repeatedly rediscovered instead of becoming durable product knowledge.

LLM Wiki improves this by compiling new sources into maintained Markdown pages at ingest time. The limitation is that wiki links are usually weak: `[[SB 330]] affects [[Housing Project]]` is readable, but it does not tell a system what kind of effect exists, which evidence supports it, whether the claim is active, or how confident the system should be.

LLM-KG adds the next layer: every useful source can become structured claims, evidence quotes, entities, and typed relations. This makes the knowledge base both human-readable and machine-reasonable.

## How LLM-KG Differs From GraphRAG

LLM-KG is not another GraphRAG clone. GraphRAG builds graph indexes to improve retrieval. LLM-KG builds an evidence-governed knowledge system for persistent reasoning, with a built-in Wiki compilation layer and a handoff contract to LLM-KEE for safe knowledge evolution.

```text
GraphRAG improves retrieval.
LLM-KG governs reasoning.
LLM-KEE evolves knowledge.
```

In this repository, GraphRAG-like retrieval is one layer: chunking, entity/relation/claim extraction, embeddings, and basic/local search. The differentiating layer is evidence governance:

- No evidence, no claim.
- No source trace, no relation.
- No unreviewed correction directly mutates governed knowledge.
- Corrections, conflicts, and outdated claims become proposal drafts for LLM-KEE.
- Approved update plans can be applied through a narrow, auditable adapter boundary.
- Query answers can be persisted as reasoning traces and exported to LLM-KEE as reusable learning signals.

## Why Build This

AI agents need durable context. They should not re-read the same PDFs, staff reports, policies, meeting notes, or project files every time a user asks a related question. LLM-KG gives agents a local memory artifact that can compound over time:

- Raw sources remain immutable as the source of truth.
- Wiki pages make knowledge readable and maintainable.
- Claims make facts explicit and reviewable.
- Evidence keeps answers grounded in source material.
- Entities and relations make dependency paths computable.
- Lint checks keep the knowledge base healthy as it grows.
- Audit events and review states make knowledge changes governable.

The system keeps Markdown and JSONL exports for inspection, while PostgreSQL+pgvector provides the primary retrieval and structured storage backend when configured.

## Product Examples

### Housing Intelligence

A city planning team, real estate analyst, or permitting agent can ingest zoning codes, staff reports, public comments, application packets, and policy memos. LLM-KG can turn them into a graph such as:

```text
Project A
  -> located_on -> Parcel X
  -> submitted_to -> Palo Alto Planning Department
  -> governed_by -> Zoning Code Section Y
  -> affected_by -> SB 330
  -> missing -> Traffic Study
  -> has_risk -> Incomplete Filing Risk
  -> supported_by -> Staff Report 2025-03-12
```

Example questions:

- "Which policies affect this housing project?"
- "What evidence supports the incomplete filing risk?"
- "Which city documents mention traffic study requirements?"
- "What changed between the March and May staff reports?"

### Legal and Policy Research

Teams can ingest statutes, guidance, memos, contracts, or policy updates. Instead of only retrieving paragraphs, the system can track which rule constrains which activity, which claim is outdated, and which source supports each answer.

Example questions:

- "Which active claims support this compliance conclusion?"
- "Where do two sources contradict each other?"
- "Which obligations apply to this project type?"

### Agent Memory for Complex Workflows

An internal agent can use LLM-KG as long-term memory for product, research, operations, or due diligence workflows. Good answers can later be written back as synthesis pages or graph records, turning one-off analysis into reusable knowledge.

## Current MVP

This repository implements the first local version:

- Source readers for `.txt`, `.md`, `.docx`, and `.pdf`.
- Pydantic models for documents, text units, wiki pages, claims, evidence, entities, relations, embeddings, query hits, and lint issues.
- Deterministic `mock` LLM provider for offline development and tests.
- Optional OpenAI provider for wiki generation, strict structured extraction, and answers.
- Optional OpenAI Vision OCR fallback for PDF pages where native text extraction fails or times out.
- Deterministic `mock` embedding provider and optional OpenAI embeddings.
- PostgreSQL+pgvector storage for text units, graph records, and embeddings.
- Evidence governance fields for claims, evidence, entities, relations, and wiki pages.
- Strict ingest governance that turns claims without evidence and invalid relations into proposals instead of active graph records.
- Generic ontology validation for common entity types and relation predicates.
- Source, entity, concept, synthesis, and comparison wiki compilation where extracted knowledge supports it.
- Reasoning traces for query answers, including used claims, relations, evidence, hits, confidence, and answer output.
- Verification, trace, proposal export, reasoning trace export, and approved apply-plan commands.
- Optional direct LLM-KEE adapter for approved proposal application.
- Markdown wiki output in `wiki/`.
- JSONL graph output in `graph_store/`.
- CLI commands for ingest, query, lint, stats, and database migration/status.

This is a GraphRAG-inspired Local+Vector MVP. It implements chunking, extraction, embeddings, basic vector RAG, and local entity-aware retrieval. It does not yet implement community detection, global search, or DRIFT search.

The default LLM provider is deterministic `mock`, so the project can run offline. Set `LLM_KG_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI.

OpenAI extraction uses Structured Outputs for claims, evidence, entities, and relations. Malformed OpenAI extraction responses fail clearly by default; set `LLM_KG_OPENAI_FALLBACK_TO_MOCK=true` only when you explicitly want deterministic mock fallback for local experiments.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## PostgreSQL + pgvector

Start the local pgvector database:

```bash
docker compose up -d postgres
```

Set the database URL:

```bash
export LLM_KG_DATABASE_URL="postgresql://llm_kg:llm_kg@localhost:54329/llm_kg"
```

Apply migrations:

```bash
python -m llm_kg db init
python -m llm_kg db status
```

When `LLM_KG_DATABASE_URL` or `[database].url` is configured, ingest writes to PostgreSQL+pgvector and also exports Markdown/JSONL for inspection. Without a database URL, the project falls back to the original local Markdown/JSONL behavior.

## CLI

```bash
python -m llm_kg ingest raw_sources/markdown/example.md
python -m llm_kg query "What policies affect Housing Project Alpha?" --mode local
python -m llm_kg query "What evidence mentions SB 330?" --mode basic
python -m llm_kg verify claim claim_123
python -m llm_kg verify relation rel_123
python -m llm_kg trace claim claim_123
python -m llm_kg trace query trace_123
python -m llm_kg traces list
python -m llm_kg traces export trace_123 --format llm-kee
python -m llm_kg propose relation rel_123 --change change.json
python -m llm_kg export-proposal prop_123 --format llm-kee
python -m llm_kg apply-plan approved-plan.json
python -m llm_kg lint
python -m llm_kg stats
```

Use `--json` for structured output:

```bash
python -m llm_kg --json query "What evidence mentions SB 330?"
```

## Environment

- `LLM_KG_WORKSPACE`: workspace path; defaults to current directory.
- `LLM_KG_CONFIG`: optional path to a TOML config file; defaults to `llm_kg.toml` in the workspace.
- `LLM_KG_PROVIDER`: `mock` or `openai`; overrides config and defaults to `openai` only when `OPENAI_API_KEY` is present.
- `LLM_KG_OPENAI_MODEL`: OpenAI model; defaults to `gpt-4.1-mini`.
- `LLM_KG_OPENAI_FALLBACK_TO_MOCK`: set `true` to allow mock extraction fallback after OpenAI structured extraction errors.
- `LLM_KG_DATABASE_URL`: PostgreSQL connection string.
- `LLM_KG_EMBEDDING_PROVIDER`: `mock` or `openai`.
- `LLM_KG_EMBEDDING_MODEL`: embedding model; defaults to `text-embedding-3-small`.
- `LLM_KG_EMBEDDING_DIMENSIONS`: embedding size; defaults to `1536`.
- `LLM_KG_TOP_K`: default query hit count.
- `LLM_KG_QUERY_MODE`: `basic` or `local`; defaults to `local`.
- `LLM_KG_ONTOLOGY_PROFILE`: ontology profile; currently `generic`.
- `LLM_KG_ENFORCE_EVIDENCE`: require claims to have traceable evidence.
- `LLM_KG_ENFORCE_RELATION_TRACE`: require relations to have entity refs plus claim/evidence trace.
- `LLM_KG_KEE_WORKSPACE`: optional sibling LLM-KEE workspace.
- `LLM_KG_KEE_ENABLE_DIRECT_ADAPTER`: enables direct adapter assumptions in config.
- `LLM_KG_OCR_PROVIDER`: `none` or `openai`; defaults to `none`.
- `LLM_KG_OCR_MODEL`: OpenAI Vision OCR model; defaults to the configured OpenAI model.
- `LLM_KG_OCR_MAX_PAGES`: maximum timeout/failed PDF pages to send to OCR; defaults to `25`.
- `LLM_KG_OCR_TIMEOUT_SECONDS`: per-page OCR call timeout; defaults to `30`.

## PDF OCR And Extraction Quality

PDF ingest first tries native text extraction with page-level timeout protection. When OCR is enabled, pages that fail or time out are rendered with PyMuPDF and sent to OpenAI Vision OCR. The resulting source text keeps explicit page markers such as:

```text
[Page 12 | ocr_text | provider=openai]
...
```

Evidence records preserve `page_number` and `source_mode`, and source wiki pages include an extraction coverage summary. `python -m llm_kg stats` reports extraction quality metrics such as PDF page coverage, OCR evidence count, claims with evidence, claims with page numbers, noisy entity ratio, and relation validity. `python -m llm_kg lint` warns when PDF evidence lacks page numbers or page coverage is low.

## Config File

The project config file is TOML. Copy `llm_kg.toml.example` to `llm_kg.toml` when you want checked-in defaults for local runs:

```toml
[llm]
# "mock" runs fully offline. Use "openai" with OPENAI_API_KEY for model-backed runs.
provider = "mock"
openai_model = "gpt-4.1-mini"
fallback_to_mock = false

[database]
# Docker Compose default:
# postgresql://llm_kg:llm_kg@localhost:54329/llm_kg
url = ""

[embedding]
provider = "mock"
model = "text-embedding-3-small"
dimensions = 1536

[query]
top_k = 5
default_mode = "local"

[ontology]
profile = "generic"

[governance]
enforce_evidence = true
enforce_relation_trace = true

[ocr]
provider = "none"
model = ""
max_pages = 25
timeout_seconds = 30

[kee]
workspace = "../LLM-KEE"
enable_direct_adapter = true
```

Resolution order:

1. Explicit CLI or API workspace argument.
2. `LLM_KG_WORKSPACE`.
3. Current directory.
4. `<workspace>/llm_kg.toml`, unless `LLM_KG_CONFIG` points elsewhere.
5. Environment variables override matching TOML values.

Do not put secrets in `llm_kg.toml`. Keep `OPENAI_API_KEY` in the environment.

## Storage

- Raw sources are not modified.
- Wiki pages are written to `wiki/`.
- Graph records are written to `graph_store/nodes.jsonl`, `edges.jsonl`, `claims.jsonl`, `evidence.jsonl`, `wiki_pages.jsonl`, `reasoning_traces.jsonl`, `proposals.jsonl`, and `audit_events.jsonl`.
- When configured, PostgreSQL stores `documents`, `text_units`, `wiki_pages`, `claims`, `evidence`, `entities`, `relationships`, `embeddings`, `reasoning_traces`, `ontology_schemas`, `update_proposals`, and `audit_events`.

## LLM-KEE Evolution Loop

LLM-KG owns governed knowledge storage and approved mutation. LLM-KEE owns feedback interpretation, evaluator aggregation, learning gate decisions, and reusable pattern learning.

```bash
python -m llm_kg export-proposal prop_123 --format llm-kee
python -m llm_kee evaluate prop_123
python -m llm_kee apply prop_123
```

When LLM-KEE is configured with:

```toml
[kg]
enable_direct_adapter = true
workspace = "../LLM-KG"
project_path = "../LLM-KG"
```

`llm-kee apply` uses the direct LLM-KG adapter. Unapproved, rejected, or need-more-evidence proposals do not mutate LLM-KG. Approved plans are applied through `apply_update_plan`, versioned, and recorded in audit events.

## Python API

```python
from pathlib import Path
from llm_kg import ingest_source, query_knowledge, lint_workspace, verify_object, trace_object, trace_query

ingest_source(Path("raw_sources/markdown/example.md"), workspace=Path("."))
result = query_knowledge("What affects the project?", workspace=Path("."))
verification = verify_object("claim", "claim_123", workspace=Path("."))
trace = trace_object("claim", "claim_123", workspace=Path("."))
query_trace = trace_query(result.trace_id, workspace=Path(".")) if result.trace_id else None
issues = lint_workspace(Path("."))
```
