# 🧠 Eidetic Memory

**Long-term memory for AI agents — extracts, evolves, and retrieves facts across conversations.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" />
  <img src="https://github.com/CodeNinjaSarthak/eidetic-memory/actions/workflows/ci.yml/badge.svg" />
  <img src="https://img.shields.io/badge/LLM-Claude%20%7C%20Gemini%20%7C%20Azure%20%7C%20Groq-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/vector%20store-Qdrant-red?style=flat-square" />
  <img src="https://img.shields.io/badge/LoCoMo_QA-53.2%25-brightgreen?style=flat-square" />
</p>

---

## ✨ Why Eidetic Memory

- 🧠 **Remembers what matters** — extracts atomic facts from every conversation turn, not raw chat logs
- ⚡ **Evolves over time** — ADD, UPDATE, DELETE, NOOP decisions keep memory fresh and conflict-free
- 🎯 **Retrieves the right context** — semantic search + importance reranking surfaces relevant facts at query time
- 🗣️ **Multi-party ready** — per-speaker memory isolation prevents cross-speaker contamination in group conversations
- 🔌 **Any LLM, any time** — swap Claude, Gemini, Azure OpenAI, or Groq with a single env var
- 📊 **Benchmark-validated** — 53.2% QA accuracy on LoCoMo, 3.54x over baseline

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

## 📊 Evaluation

Evaluated on the [LoCoMo benchmark](https://github.com/snap-research/locomo) (conv-26 + conv-30, n=233 QA pairs) — long-form multi-session conversations with per-speaker memory isolation.

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

### End-to-end QA accuracy

| Category | Accuracy |
|----------|----------|
| Temporal | **68.3%** |
| Open-domain | 52.6% |
| Single-hop | 37.2% |
| Multi-hop | 38.5% |
| **Overall** | **53.2%** |

### Progress

| Run | Score | Details |
|-----|-------|---------|
| Baseline | 14.8% | Gemini, conv-30 only, top-k=10 |
| + Per-speaker isolation | 31.8% | Multi-namespace retrieval |
| + Fixed merge | 52.4% | Round-robin interleaving |
| **+ Named entities + two-pass** | **53.2%** | Current best |
| mem0 paper | ~70% | Target ceiling |

---

## 🚀 Quick Start

**Prerequisites:** Python 3.13+, Node 18+, [uv](https://docs.astral.sh/uv/), [Qdrant Cloud](https://qdrant.tech/) account, Google API key.

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

Backend: `http://localhost:8000` · Frontend: `http://localhost:3000` · API docs: `http://localhost:8000/docs`

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

142 tests · behavior-focused · no mocks except external I/O

Tests follow Google-style testing principles. See [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Contributing

- 🐛 Found a bug? Open an issue with reproduction steps
- 💡 Have an idea? Check open issues first, then open a discussion
- 🔧 Want to contribute? PRs welcome — run `make check` before submitting

---

## License

[Apache License 2.0](LICENSE)
