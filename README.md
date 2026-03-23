# Eidetic Memory

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" />
  <img src="https://github.com/CodeNinjaSarthak/eidetic-memory/actions/workflows/ci.yml/badge.svg" />
  <img src="https://img.shields.io/badge/LLM-Claude%20%7C%20Gemini%20%7C%20Azure%20%7C%20Groq-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/vector%20store-Qdrant-red?style=flat-square" />
</p>

<p align="center">
  <strong>Long-term memory for AI agents.</strong><br/>
  Extracts facts from conversations, evolves them over time,
  and retrieves the right context when it matters.
</p>

---

## Demo

> 🎥 Demo GIF coming soon — chat UI with live memory extraction

<!-- Replace with actual demo GIF -->

---

## How It Works

Every conversation turn passes through a four-stage pipeline:

```
User message
│
▼
┌─────────────────┐
│ ExtractionPipeline │  LLM extracts candidate facts from the conversation pair
└────────┬────────┘
         │  candidates[]
         ▼
┌─────────────────┐
│  EvolutionEngine  │  Compares each candidate against existing memories
└────────┬────────┘  → ADD / UPDATE / DELETE / NOOP
         │
         ▼
┌─────────────────┐
│  QdrantMemoryStore│  Executes operations, stores embeddings + payloads
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MemoryRetriever  │  Embeds query, searches Qdrant, reranks by importance
└─────────────────┘
         │
         ▼
Retrieved context → injected into LLM system prompt
```

---

## Features

- **Fact extraction** — identifies atomic facts from conversation pairs using LLM tool calling
- **Memory evolution** — decides ADD, UPDATE, DELETE, or NOOP by comparing candidates against semantically similar existing memories
- **Importance scoring** — recency × frequency scoring surfaces the most relevant memories at retrieval time
- **Multi-LLM** — plug in Claude, Gemini, Azure OpenAI, or Groq with a single env var change
- **Async throughout** — every I/O operation is async; no blocking calls anywhere in the stack
- **Chat UI** — Next.js frontend with live memory extraction and a memory browser

---

## Evaluation

Evaluated on the [LoCoMo benchmark](https://github.com/snap-research/locomo) —
long-form multi-session conversations with QA pairs across 4 categories.

### Component accuracy

| Metric | Score | Details |
|--------|-------|---------|
| Fact extraction recall | **95.0%** | n=100, QA pairs with evidence |
| Fact extraction precision | **58.6%** | Relevant facts / total extracted |
| Conflict resolution accuracy | **100%** | 26 test cases (ADD/UPDATE/DELETE/NOOP) |

### Retrieval accuracy (LoCoMo, n=100)

| K | Hit@K |
|---|-------|
| 1 | 11% |
| 5 | 25% |
| 10 | 38% |
| 20 | **56%** |

### End-to-end QA accuracy (LoCoMo conv-30, n=81)

| Category | Accuracy |
|----------|----------|
| Temporal | 15.4% |
| Open-domain | 20.5% |
| Single-hop | 9.1% |
| **Overall** | **14.8%** |

> LoCoMo is a challenging multi-party benchmark designed for human-to-human
> conversations. The system was designed for user/assistant pairs — the
> end-to-end numbers reflect the domain gap, not production performance.

---

## Quick Start

**Prerequisites:** Python 3.13+, Node 18+, [uv](https://docs.astral.sh/uv/),
[Qdrant Cloud](https://qdrant.tech/) account, Google API key.

```bash
# 1. Clone and install
git clone https://github.com/CodeNinjaSarthak/eidetic-memory.git
cd eidetic-memory
uv sync --all-packages

# 2. Configure
cp .env.development.example .env.development
# Fill in: GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Start backend
make run

# 4. Start frontend (separate terminal)
make frontend-install && make frontend-dev
```

Backend: `http://localhost:8000` · Frontend: `http://localhost:3000`

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memories/` | Extract and store facts from a conversation turn |
| `POST` | `/memories/search` | Semantic similarity search |
| `GET` | `/memories/{user_id}` | List all memories for a user |
| `DELETE` | `/memories/{memory_id}` | Delete a memory |
| `POST` | `/chat/` | Memory-augmented chat turn |
| `GET` | `/health` | Health check |

### Extract memories

```bash
curl -X POST http://localhost:8000/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "current_message": {
      "role": "user",
      "content": "I just moved to Berlin for a new job at a startup.",
      "user_id": "user-1",
      "session_id": "session-1"
    },
    "previous_message": {
      "role": "assistant",
      "content": "That sounds exciting! Tell me more.",
      "user_id": "user-1",
      "session_id": "session-1"
    }
  }'
```

### Search memories

```bash
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Where does the user live?", "user_id": "user-1", "top_k": 5}'
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `claude` | `claude` · `gemini` · `azure` · `groq` |
| `ANTHROPIC_API_KEY` | If claude | — | Anthropic API key |
| `GOOGLE_API_KEY` | If gemini | — | Google AI key — also used for embeddings |
| `GROQ_API_KEY` | If groq | — | Groq API key |
| `AZURE_OPENAI_API_KEY` | If azure | — | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | If azure | — | Azure endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | If azure | — | Deployment name |
| `QDRANT_URL` | Yes | — | Qdrant instance URL |
| `QDRANT_API_KEY` | No | — | Required for Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | No | `eidetic_memories` | Collection name |
| `EMBEDDING_MODEL` | No | `gemini-embedding-exp-03-07` | Embedding model |
| `EMBEDDING_DIMENSION` | No | `768` | Vector dimension |
| `MEMORY_EXTRACTION_MODEL` | No | `gemini-2.0-flash` | Extraction + evolution model |
| `RECENCY_WINDOW` | No | `10` | Recent messages window size |
| `SIMILARITY_TOP_K` | No | `10` | Top-k for similarity search |
| `API_PORT` | No | `8000` | API server port |

**Switching to Groq:**

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
MEMORY_EXTRACTION_MODEL=llama-3.3-70b-versatile
```

---

## Project Structure

```
eidetic-memory/
├── backend/
│   ├── apps/api/              # FastAPI routes, schemas, DI
│   ├── services/
│   │   ├── llm/               # Provider adapters (Claude, Gemini, Azure, Groq)
│   │   ├── memory/            # Extraction, evolution, lifecycle pipeline
│   │   ├── retrieval/         # Semantic search + context building
│   │   └── storage/           # Qdrant abstraction
│   └── packages/config/       # Shared settings — single source of truth
├── frontend/                  # Next.js chat UI + memory browser
├── eval/                      # LoCoMo evaluation harness + scripts
├── Makefile
└── pyproject.toml             # uv workspace root
```

**Dependency graph** (no circular deps allowed):

```
config → storage → llm → retrieval → memory → api
```

---

## Development

```bash
make test        # run 142 tests
make lint        # ruff check
make format      # ruff format
make check       # lint + test
make run         # start API with hot reload
```

Tests follow Google-style: behavior-focused, one reason to fail,
no mocking unless external I/O. See [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

[Apache License 2.0](LICENSE)
