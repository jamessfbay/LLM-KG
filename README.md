# LLM-KG (knowledge graph)

LLM-KG is a persistent reasoning layer for AI agents. It turns raw documents into an inspectable LLM Wiki, then compiles that wiki into a lightweight knowledge graph made of claims, evidence, entities, and typed relations.

The goal is not just to summarize documents. The goal is to make knowledge reusable, traceable, and computable across future agent workflows.

```text
TXT/MD/DOCX/PDF source
  -> Markdown wiki pages
  -> evidence-backed claims
  -> entities and typed relations
  -> JSONL knowledge graph
  -> query, lint, and reasoning workflows
```

## Problem

Traditional RAG is query-time synthesis: every question searches raw chunks and asks an LLM to assemble an answer from scratch. That works for quick lookup, but it has weak memory. Important facts, contradictions, entity relationships, and reasoning paths are repeatedly rediscovered instead of becoming durable product knowledge.

LLM Wiki improves this by compiling new sources into maintained Markdown pages at ingest time. The limitation is that wiki links are usually weak: `[[SB 330]] affects [[Housing Project]]` is readable, but it does not tell a system what kind of effect exists, which evidence supports it, whether the claim is active, or how confident the system should be.

LLM-KG adds the next layer: every useful source can become structured claims, evidence quotes, entities, and typed relations. This makes the knowledge base both human-readable and machine-reasonable.

## Why Build This

AI agents need durable context. They should not re-read the same PDFs, staff reports, policies, meeting notes, or project files every time a user asks a related question. LLM-KG gives agents a local memory artifact that can compound over time:

- Raw sources remain immutable as the source of truth.
- Wiki pages make knowledge readable and maintainable.
- Claims make facts explicit and reviewable.
- Evidence keeps answers grounded in source material.
- Entities and relations make dependency paths computable.
- Lint checks keep the knowledge base healthy as it grows.

The MVP intentionally starts with Markdown and JSONL instead of a graph database. This keeps the system easy to inspect, Git-friendly, and useful before adding SQLite, Neo4j, vector search, or a UI.

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
- Pydantic models for documents, wiki pages, claims, evidence, entities, relations, query hits, and lint issues.
- Deterministic `mock` LLM provider for offline development and tests.
- Optional OpenAI provider for wiki generation, extraction, and answers.
- Markdown wiki output in `wiki/`.
- JSONL graph output in `graph_store/`.
- CLI commands for ingest, query, lint, and stats.

The default LLM provider is deterministic `mock`, so the project can run offline. Set `LLM_KG_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## CLI

```bash
python -m llm_kg ingest raw_sources/markdown/example.md
python -m llm_kg query "What policies affect Housing Project Alpha?"
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
- `LLM_KG_TOP_K`: default query hit count.

## Config File

The project config file is TOML. Copy `llm_kg.toml.example` to `llm_kg.toml` when you want checked-in defaults for local runs:

```toml
[llm]
# "mock" runs fully offline. Use "openai" with OPENAI_API_KEY for model-backed runs.
provider = "mock"
openai_model = "gpt-4.1-mini"

[query]
top_k = 5
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
- Graph records are written to `graph_store/nodes.jsonl`, `edges.jsonl`, `claims.jsonl`, and `evidence.jsonl`.

## Python API

```python
from pathlib import Path
from llm_kg import ingest_source, query_knowledge, lint_workspace

ingest_source(Path("raw_sources/markdown/example.md"), workspace=Path("."))
result = query_knowledge("What affects the project?", workspace=Path("."))
issues = lint_workspace(Path("."))
```
