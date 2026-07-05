# Architecture

Eidetic Memory is a long-term memory system for AI agents. Conversation pairs are
processed to extract facts, evolve them against existing memories, and retrieve them
via semantic search. The codebase is a monorepo with clear service boundaries,
structured like microservices but deployed as a monolith.

## Dependency Graph

```
config
  └── storage
        └── llm
              └── retrieval
                    └── memory
                          └── api
```

Each layer depends only on the layers below it:

- **config** — no dependencies. Provides `Settings` to every other package.
- **storage** — depends on config for Qdrant connection details. Defines `AbstractMemoryStore` and `QdrantMemoryStore`.
- **llm** — depends on config for API keys and model names. Defines `AbstractLLMService` and provider implementations.
- **retrieval** — depends on storage (to query Qdrant) and llm (for embeddings). Contains `MemoryRetriever`.
- **memory** — depends on all lower layers. Contains `ExtractionPipeline`, `EvolutionEngine`, `LifecycleManager`, and `MemoryManager`.
- **api** — depends on memory and retrieval. HTTP-only layer with FastAPI routes, no business logic.

Circular dependencies between these layers are not allowed.

## Data Flow

### POST /memories/ — extract and store facts

1. Request arrives with `user_id`, `session_id`, `current_message`, `previous_message`,
   `current_speaker`, `conversation_summary`, and `recent_messages`
2. The router builds a `ConversationPair` from the two `MessageSchema` objects
3. `MemoryManager.add_memory()` is called with `current_speaker`, `conversation_summary`,
   and `recent_messages`
4. Context is assembled via `_build_user_content()`:
   - **Current Speaker header** — identifies whose turn is being processed so the LLM
     attributes facts by name at generation time, not via post-hoc rewriting
   - **Conversation Summary** — a 3–5 sentence summary of key entities, relationships,
     and dated events, refreshed every 15 turns by a single LLM call over the last 30 lines
   - **Recent Messages window** — the last 10 messages providing local conversational context
   - **Current Turn** — the (previous, current) conversation pair being processed
5. `ExtractionPipeline.extract()` sends the assembled context to the LLM, which returns
   atomic, self-contained candidate facts with speaker names attributed at generation time
6. Each candidate fact is embedded via `AbstractEmbeddingService`
7. For each candidate, existing memories are retrieved from Qdrant by similarity
8. `EvolutionEngine.decide()` compares each candidate against existing memories and returns
   a `MemoryUpdate` with action: ADD, UPDATE, DELETE, or NOOP
9. `QdrantMemoryStore` executes the decided action (insert, overwrite, or delete)
10. The response returns all newly added or updated `MemoryFact` objects

### POST /memories/search — semantic retrieval (legacy path — not the demo or paper surface)

> This endpoint is **not** part of the demo or the paper's reported numbers. It predates
> the local cross-encoder work and still reranks via the external Jina Reranker API
> (`retriever.py` → `jina-reranker-v2-base-multilingual`, falling back to raw vector order
> if no Jina key is set). The demo (`/demo/query`) and all eval scripts use the local
> cross-encoder (`reranker.py`, `ms-marco-MiniLM-L-6-v2`) instead — see below.

1. Request arrives with `query`, `user_id`, and `top_k`
2. `MemoryRetriever.retrieve()` embeds the query string
3. Qdrant returns the top-k most similar memories for that user
4. Results are reranked by the Jina Reranker API if `jina_api_key` is configured,
   otherwise returned in raw vector-similarity order (no local cross-encoder here)
5. The response returns the matched `MemoryFact` objects in reranked order

### Demo and eval retrieval path — local cross-encoder (this is what the paper reports)

The `/demo/query` endpoint and all `eval/eval_qa_accuracy.py` runs (via `--local-rerank`)
use `reranker.py`'s local cross-encoder (`ms-marco-MiniLM-L-6-v2`, 66M params, CPU
inference, ~5ms per pair once warm) instead of `MemoryRetriever`/Jina. This is the
reranker behind every accuracy number in this document and in the paper.

## Key Design Decisions

1. **Monorepo with service boundaries.** Code is organized as independent packages with
   explicit dependencies, but everything runs in a single process. This gives clear ownership
   without deployment overhead.

2. **Single settings source of truth.** All configuration flows through
   `backend/packages/config/src/config/settings.py`. No service reads environment variables
   directly.

3. **Abstract base classes for LLM and storage.** `AbstractLLMService` and
   `AbstractMemoryStore` let the pipeline code stay provider-agnostic. Swapping providers
   requires no changes to business logic.

4. **SecretStr for all API keys.** Pydantic's `SecretStr` prevents keys from appearing in
   logs, repr output, or serialized settings.

5. **Pydantic v2 for all data models.** Request schemas, internal models, and storage models
   all use Pydantic v2 with `field_serializer` instead of deprecated `json_encoders`.

6. **Async by default.** All I/O operations (LLM calls, Qdrant queries, embedding requests)
   are async. Sync wrappers are never used.

7. **Prompts as text files.** All LLM prompts live in
   `backend/services/memory/src/memory/prompts/` as `.txt` files. No prompt strings are
   hardcoded in Python.

8. **Google-style testing.** Tests describe behavior, not implementation. Each test has one
   reason to fail. Mocking is used only for external I/O.

9. **uv workspaces.** Python dependencies are managed by uv with workspace-level resolution.
   Each service has its own `pyproject.toml` declaring only its direct dependencies.

10. **Ruff for linting and formatting.** A single tool replaces flake8, black, and isort.
    Configuration lives in the root `pyproject.toml`.

11. **FastAPI as HTTP-only layer.** Routers convert request schemas to domain objects and
    call service methods. No business logic lives in the API layer.

12. **Qdrant as vector store.** Chosen for managed cloud hosting, filtering by `user_id`,
    and payload storage alongside vectors. The collection name is configurable. A full-text
    index on the `content` payload field is created at collection initialization. Client
    timeout is set to 60s with retry on `ResponseHandlingException` to handle transient
    cloud latency.

13. **Pipeline parameters as config, partially.** `recency_window` and `similarity_top_k`
    are environment variables (`backend/packages/config/src/config/settings.py`) with
    sensible defaults. `summary_interval` (15 turns) and `reranker_fetch_multiplier` (3x)
    are **not** env-configurable: `summary_interval` is a hardcoded `% 15` check in
    `eval/ingest_locomo_production.py`, and `reranker_fetch_multiplier` is a Python
    default argument (`=3`) in `retriever.py` and `dependencies.py`. Changing either
    requires a code edit, not an env var.

14. **Embedding model decoupled from LLM provider.** `text-embedding-3-small` (1536-dim,
    via Azure OpenAI) is used for all vector operations regardless of which LLM provider
    handles generation. This avoids tying embedding quality to provider choice and keeps
    the embedding pipeline stable across provider swaps.

## Memory Isolation

Multi-party conversations require per-speaker memory to prevent cross-speaker
contamination. Eidetic Memory achieves this through user ID namespacing, speaker-aware
extraction context, and dual-namespace retrieval.

### Per-speaker user IDs

Each speaker in a conversation gets their own `user_id`, constructed as:

```
{base_user_id}_{speaker_name_lowercase}
```

For example, a conversation between Jon and Gina produces two namespaces:
`user_conv_30_jon` and `user_conv_30_gina`. Every conversation turn is routed to the
correct speaker's namespace during ingestion.

### Speaker-aware extraction context

Each extraction call receives a `Current Speaker` header identifying whose turn is being
processed. The extraction prompt instructs the LLM to use that exact speaker name in every
fact produced, ensuring facts like "Jon lives in Berlin" are attributed at generation time
rather than via brittle post-hoc string replacement. This reduced speaker misattribution
from 60% of extraction errors (v1) to 15% (v2), lifting overall attribution accuracy
to 97.0% (Wilson 95% CI [91.5, 99.0]).

A rolling conversation summary (generated every 15 turns by a single LLM call over the
last 30 lines) and a recent-message window (last 10 messages) are prepended to every
extraction call, giving the extractor session-wide context needed to produce complete,
dated, speaker-attributed facts. For example, "I started the new job today" without
context yields the vague fact "Jon started a new job"; with context it yields "Jon left
Acme Corp and started at [company] on [date]."

### Dual-namespace retrieval merge

At query time, memories are retrieved from both speaker namespaces independently, then
merged using round-robin interleaving:

1. Retrieve top-k×3 from speaker A's namespace (90 candidates at k=30)
2. Retrieve top-k×3 from speaker B's namespace (90 candidates at k=30)
3. Interleave results alternately (via `itertools.zip_longest`), producing 180 candidates
4. Deduplicate by content
5. Rerank all 180 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
   (66M parameters, local CPU, no API key, ~5ms per pair once warm)
6. Take the top-30 reranked facts as the final context

This strategy ensures both speakers are represented in the final context while the
cross-encoder correctly prioritises recent evidence over semantically similar but stale
facts — distinguishing "Jon works at Acme Corp now" from "Jon worked at Acme Corp last
year" in a way embedding similarity cannot.

## Eval Pipeline

Two scripts in `eval/` form the evaluation pipeline. Both require a running Qdrant
instance and configured `.env.development`.

### ingest_locomo_production.py

Ingests LoCoMo conversations through the production `MemoryManager` pipeline, storing
memories under per-speaker user IDs. Each turn is processed with the full extraction
context: current speaker header, rolling conversation summary (refreshed every 15 turns),
and a recent-message window of the last 10 messages. Ingesting all 10 LoCoMo conversations
(4,431 turns, ~7,574 facts) takes roughly 60 minutes at standard API rate limits; rolling
summary generation adds approximately 295 extra LLM calls across all conversations.

```bash
# Ingest default conversations (conv-26, conv-30)
uv run python eval/ingest_locomo_production.py

# Specific conversations
uv run python eval/ingest_locomo_production.py --conv-ids conv-26 conv-30

# Preview without calling LLMs
uv run python eval/ingest_locomo_production.py --dry-run

# Delete all eval memories and exit
uv run python eval/ingest_locomo_production.py --cleanup
```

### eval_qa_accuracy.py

End-to-end QA accuracy evaluation. Requires memories to be ingested first. For each QA
pair, it retrieves from both speaker namespaces, reranks with the local cross-encoder,
generates an answer via LLM (max 200 tokens for factual questions, 350 for open-domain),
and judges correctness against the gold answer using LLM-as-judge at temperature=0.

Implements two-pass retrieval: if the first-pass answer contains "don't know" on a
non-open-domain question, the query is rephrased (5-word keyword query, max 50 tokens)
and retrieval is attempted again. In v2, this fires on 1.7% of non-open-domain queries,
giving an average of 1.02 LLM calls per query across the full benchmark.

> **Note:** Eval scripts use a separate venv (`~/eval-venv`) with sentence-transformers
> and ONNX runtime. Do not use `uv run` for eval commands.

```bash
# Run full evaluation
~/eval-venv/bin/python eval/eval_qa_accuracy.py --local-rerank

# Limit to N QA pairs (useful for quick checks)
~/eval-venv/bin/python eval/eval_qa_accuracy.py --limit 10 --local-rerank

# Custom output path
~/eval-venv/bin/python eval/eval_qa_accuracy.py \
  --output eval/results/my_results.json --local-rerank
```

Results are saved as JSON with overall accuracy, per-category breakdown, and per-pair
details. Canonical result files:

| File | Description | Score |
|------|-------------|-------|
| `eval/results/prompt_fix_full_v1.json` | v2 canonical, full benchmark | 66.6% (n=1540) |
| `eval/results/heldout_prompt_fix.json` | v2 held-out validation | 65.2% (n=718) |
| `eval/results/improvement_week2_full_v1.json` | v2 extraction only, pre-prompt-fix | 64.6% (n=1540) |
| `eval/results/final_eidetic_memory_local_reranker_m3.json` | v1 canonical | 56.3% (n=1540) |

## Adding a New LLM Provider

1. **Create the service class.** Add
   `backend/services/llm/src/llm/generation/{provider}.py` with a class that extends
   `AbstractLLMService`. Implement `complete()` and `complete_with_tool()`.

2. **Register in settings.** Add the provider name to the `Literal` type in
   `backend/packages/config/src/config/settings.py` (line 24) and add a validation
   branch in the `_validate_provider_credentials` model validator.

3. **Wire up the factory.** Add a `case` branch to `_build_llm_service()` in
   `backend/apps/api/src/api/dependencies.py`.

4. **Export the class.** Add the import and `__all__` entry in
   `backend/services/llm/src/llm/__init__.py`.

5. **Add tests.** Create
   `backend/services/llm/tests/test_{provider}_service.py`. Test that `complete()` and
   `complete_with_tool()` return expected formats. Mock only the HTTP client.

## Testing Philosophy

Tests are living documentation. They describe what the system does, not how it does it.

**Rules:**

- Test names read as full sentences: `test_memory_fact_excludes_embedding_from_qdrant_payload`
- Each test has exactly one reason to fail
- Never test private methods or internal state
- Assert on outcomes, not implementation details
- Arrange / Act / Assert separated by blank lines
- No mocking unless absolutely necessary (external I/O only)
- Every new module has a corresponding test file
- Tests live in `tests/` inside each service package

**Good:**

```python
def test_conversation_pair_rejects_messages_from_different_sessions():
    user_message = Message(user_id="u1", session_id="session-A", role="user", content="hello")
    assistant_message = Message(user_id="u1", session_id="session-B", role="assistant", content="hi")

    with pytest.raises(ValidationError):
        ConversationPair(current=user_message, previous=assistant_message)
```

**Bad:**

```python
def test_validator():  # What validator? What behavior?
    assert ConversationPair._validate_sessions(...) == ...
```