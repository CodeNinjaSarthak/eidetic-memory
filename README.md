# Eidetic Memory

![CI](https://github.com/CodeNinjaSarthak/eidetic-memory/actions/workflows/ci.yml/badge.svg)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Production-ready implementation of the Mem0 paper — scalable long-term memory for AI agents.

**Paper:** [Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv:2504.19413v1)](https://arxiv.org/abs/2504.19413v1)

## Demo

<!-- Add demo GIF here -->

**Chat interface** — conversations stream via SSE while the memory pipeline extracts and evolves facts in the background.

**Memory browser** — browse, search, and inspect all stored memories with importance scores and timestamps.

## What This Implements

- **Fact extraction** — new conversation pairs are processed by `ExtractionPipeline` to produce candidate facts
- **Memory evolution** — each candidate is compared against existing memories by `EvolutionEngine`, which decides ADD, UPDATE, DELETE, or NOOP via tool calling
- **Semantic retrieval** — `MemoryRetriever` embeds the query, searches Qdrant for top-k matches, and optionally reranks by importance score
- **Lifecycle scoring** — `LifecycleManager` computes recency-weighted scores to surface the most relevant memories

## Quick Start

**Prerequisites:** Python 3.13+, Node 18+, [uv](https://docs.astral.sh/uv/), a [Qdrant Cloud](https://qdrant.tech/) account, and a Google API key (for Gemini).

```bash
# 1. Clone and install
git clone https://github.com/CodeNinjaSarthak/eidetic-memory.git
cd eidetic-memory
uv sync --all-packages

# 2. Copy the env template
cp .env.development.example .env.development

# 3. Fill in required keys
#    GOOGLE_API_KEY   — from Google AI Studio
#    QDRANT_URL       — from Qdrant Cloud dashboard
#    QDRANT_API_KEY   — from Qdrant Cloud dashboard

# 4. Start the backend
make run

# 5. Start the frontend (separate terminal)
make frontend-install && make frontend-dev
```

The API runs at `http://localhost:8000` and the UI at `http://localhost:3000`.

## API Reference

| Method   | Path                  | Description                |
|----------|-----------------------|----------------------------|
| `POST`   | `/memories/`          | Extract and store facts    |
| `POST`   | `/memories/search`    | Semantic similarity search |
| `GET`    | `/memories/{user_id}` | List all memories          |
| `DELETE` | `/memories/{memory_id}` | Delete a memory          |
| `GET`    | `/health`             | Health check               |

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
      "content": "That sounds exciting! Tell me more about the move.",
      "user_id": "user-1",
      "session_id": "session-1"
    },
    "conversation_summary": null
  }'
```

### Search memories

```bash
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where does the user live?",
    "user_id": "user-1",
    "top_k": 5
  }'
```

## Configuration

All configuration is loaded from `.env.development` via a single `Settings` class.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `claude` | LLM backend: `claude`, `gemini`, `azure`, or `groq` |
| `ANTHROPIC_API_KEY` | When provider=claude | — | Anthropic API key |
| `GOOGLE_API_KEY` | When provider=gemini | — | Google AI API key (also used for embeddings) |
| `GROQ_API_KEY` | When provider=groq | — | Groq API key |
| `AZURE_OPENAI_API_KEY` | When provider=azure | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | When provider=azure | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | When provider=azure | — | Azure OpenAI deployment name |
| `QDRANT_URL` | Yes | — | Qdrant instance URL |
| `QDRANT_API_KEY` | No | — | Qdrant API key (required for Qdrant Cloud) |
| `QDRANT_COLLECTION_NAME` | No | `eidetic_memories` | Qdrant collection name |
| `EMBEDDING_MODEL` | No | `gemini-embedding-exp-03-07` | Embedding model name |
| `EMBEDDING_DIMENSION` | No | `768` | Embedding vector dimension |
| `MEMORY_EXTRACTION_MODEL` | No | `gemini-2.0-flash` | Model used for fact extraction and evolution |
| `RECENCY_WINDOW` | No | `10` | Number of recent messages considered (paper param m) |
| `SIMILARITY_TOP_K` | No | `10` | Top-k results for similarity search (paper param s) |
| `API_HOST` | No | `0.0.0.0` | API server bind address |
| `API_PORT` | No | `8000` | API server port |
| `API_ENV` | No | `development` | Environment: `development`, `production`, or `test` |
| `EVAL_LLM_JUDGE_MODEL` | No | `gemini-2.0-flash` | Model for evaluation runs (not needed for normal use) |

## LLM Providers

| Provider | Required Env Vars |
|----------|------------------|
| `claude` | `ANTHROPIC_API_KEY` |
| `gemini` | `GOOGLE_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

**Example: switching to Groq**

```bash
# In .env.development, change two variables:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
# Optionally set the extraction model:
MEMORY_EXTRACTION_MODEL=llama-3.3-70b-versatile
```

## Development

### Make commands

| Command | Description |
|---------|-------------|
| `make install` | Install all packages (`uv sync --all-packages`) |
| `make sync` | Same as install |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make lint-fix` | Auto-fix lint issues |
| `make test` | Run pytest |
| `make run` | Start API server with hot reload |
| `make dev` | Copy env template and print reminder |
| `make check` | Lint + test in one command |
| `make frontend-install` | Install frontend dependencies |
| `make frontend-dev` | Start Next.js dev server |
| `make frontend-build` | Build frontend for production |

### Running tests

```bash
make test
```

Tests follow Google-style testing: behavior-focused, one assertion per failure reason, no mocking unless external I/O.

### Adding a new LLM provider

1. Implement `AbstractLLMService` in a new file under `backend/services/llm/src/llm/generation/`
2. Register the provider in `settings.py` and `dependencies.py`
3. Export from `llm/__init__.py` and add tests

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full step-by-step guide.

## Project Structure

```
eidetic-memory/
├── backend/
│   ├── apps/
│   │   └── api/              # FastAPI HTTP layer — routes, schemas, dependencies
│   ├── services/
│   │   ├── llm/              # LLM provider adapters (Claude, Gemini, Azure, Groq)
│   │   ├── memory/           # Core pipeline: extraction, evolution, lifecycle
│   │   ├── retrieval/        # Semantic search and context building
│   │   └── storage/          # Qdrant vector store abstraction
│   └── packages/
│       └── config/           # Shared pydantic-settings, single source of truth
├── frontend/                 # Next.js chat UI and memory browser
├── eval/                     # LOCOMO + synthetic evaluation harness
├── Makefile                  # Dev commands
├── pyproject.toml            # uv workspace root
└── .env.development.example  # Env var template
```

## License

[Apache License 2.0](LICENSE)
