# Architecture

Eidetic Memory is a long-term memory system for AI agents. Conversation pairs are processed to extract facts, evolve them against existing memories, and retrieve them via semantic search. The codebase is a monorepo with clear service boundaries, structured like microservices but deployed as a monolith.

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

1. Request arrives with `user_id`, `session_id`, `current_message`, `previous_message`, and optional `conversation_summary`
2. The router builds a `ConversationPair` from the two `MessageSchema` objects
3. `MemoryManager.add_memory()` is called
4. `ExtractionPipeline.extract()` sends the conversation pair to the LLM, which returns candidate fact strings
5. Each candidate fact is embedded via `AbstractEmbeddingService`
6. For each candidate, existing memories are retrieved from Qdrant by similarity
7. `EvolutionEngine.decide()` compares each candidate against existing memories and returns a `MemoryUpdate` with action: ADD, UPDATE, DELETE, or NOOP
8. `QdrantMemoryStore` executes the decided action (insert, overwrite, or delete)
9. The response returns all newly added or updated `MemoryFact` objects

### POST /memories/search — semantic retrieval

1. Request arrives with `query`, `user_id`, and `top_k`
2. `MemoryRetriever.retrieve()` embeds the query string
3. Qdrant returns the top-k most similar memories for that user
4. Results are optionally reranked by importance score
5. The response returns the matched `MemoryFact` objects

## Key Design Decisions

1. **Monorepo with service boundaries.** Code is organized as independent packages with explicit dependencies, but everything runs in a single process. This gives clear ownership without deployment overhead.

2. **Single settings source of truth.** All configuration flows through `backend/packages/config/src/config/settings.py`. No service reads environment variables directly.

3. **Abstract base classes for LLM and storage.** `AbstractLLMService` and `AbstractMemoryStore` let the pipeline code stay provider-agnostic. Swapping providers requires no changes to business logic.

4. **SecretStr for all API keys.** Pydantic's `SecretStr` prevents keys from appearing in logs, repr output, or serialized settings.

5. **Pydantic v2 for all data models.** Request schemas, internal models, and storage models all use Pydantic v2 with `field_serializer` instead of deprecated `json_encoders`.

6. **Async by default.** All I/O operations (LLM calls, Qdrant queries, embedding requests) are async. Sync wrappers are never used.

7. **Prompts as text files.** All LLM prompts live in `backend/services/memory/src/memory/prompts/` as `.txt` files. No prompt strings are hardcoded in Python.

8. **Google-style testing.** Tests describe behavior, not implementation. Each test has one reason to fail. Mocking is used only for external I/O.

9. **uv workspaces.** Python dependencies are managed by uv with workspace-level resolution. Each service has its own `pyproject.toml` declaring only its direct dependencies.

10. **Ruff for linting and formatting.** A single tool replaces flake8, black, and isort. Configuration lives in the root `pyproject.toml`.

11. **FastAPI as HTTP-only layer.** Routers convert request schemas to domain objects and call service methods. No business logic lives in the API layer.

12. **Qdrant as vector store.** Chosen for managed cloud hosting, filtering by `user_id`, and payload storage alongside vectors. The collection name is configurable.

13. **Pipeline parameters as config.** `recency_window` and `similarity_top_k` are environment variables with sensible defaults, not hardcoded constants. This makes the pipeline tunable without code changes.

14. **Embedding model decoupled from LLM provider.** Gemini embeddings are used regardless of which LLM provider handles generation. This avoids tying embedding quality to provider choice.

## Memory Isolation

Multi-party conversations require per-speaker memory to prevent cross-speaker contamination. Eidetic Memory achieves this through user ID namespacing and dual-namespace retrieval.

### Per-speaker user IDs

Each speaker in a conversation gets their own `user_id`, constructed as:

```
{base_user_id}_{speaker_name_lowercase}
```

For example, a conversation between Jon and Gina produces two namespaces: `user_conv_30_jon` and `user_conv_30_gina`. Every conversation turn is routed to the correct speaker's namespace during ingestion.

### Named entity attribution

After facts are extracted, generic "User" references are replaced with the actual speaker name. This ensures facts like "User lives in Berlin" become "Jon lives in Berlin", making retrieval and deduplication speaker-aware.

### Dual-namespace retrieval merge

At query time, memories are retrieved from both speaker namespaces independently, then merged using round-robin interleaving:

1. Retrieve top-k from speaker A's namespace
2. Retrieve top-k from speaker B's namespace
3. Interleave results alternately (via `itertools.zip_longest`), preserving per-speaker relevance order
4. Deduplicate by content
5. Take the top-k merged results

This strategy ensures both speakers are represented in the final context while maintaining semantic relevance ordering within each namespace.

## Eval Pipeline

Two scripts in `eval/` form the evaluation pipeline. Both require a running Qdrant instance and configured `.env.development`.

### ingest_locomo_production.py

Ingests LoCoMo conversations through the production `MemoryManager` pipeline, storing memories under per-speaker user IDs.

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

End-to-end QA accuracy evaluation. Requires memories to be ingested first. For each QA pair, it retrieves from both speaker namespaces, generates an answer via LLM, and judges correctness against the gold answer. Implements two-pass retrieval: if the first-pass answer contains "don't know", the query is rephrased and retrieval is attempted again.

```bash
# Run full evaluation
uv run python eval/eval_qa_accuracy.py

# Limit to N QA pairs (useful for quick checks)
uv run python eval/eval_qa_accuracy.py --limit 10

# Custom output path
uv run python eval/eval_qa_accuracy.py --output eval/results/my_results.json
```

Results are saved as JSON with overall accuracy, per-category breakdown, and per-pair details.

## Adding a New LLM Provider

1. **Create the service class.** Add `backend/services/llm/src/llm/generation/{provider}.py` with a class that extends `AbstractLLMService`. Implement `complete()` and `complete_with_tool()`.

2. **Register in settings.** Add the provider name to the `Literal` type in `backend/packages/config/src/config/settings.py` (line 24) and add a validation branch in the `_validate_provider_credentials` model validator.

3. **Wire up the factory.** Add a `case` branch to `_build_llm_service()` in `backend/apps/api/src/api/dependencies.py`.

4. **Export the class.** Add the import and `__all__` entry in `backend/services/llm/src/llm/__init__.py`.

5. **Add tests.** Create `backend/services/llm/tests/test_{provider}_service.py`. Test that `complete()` and `complete_with_tool()` return expected formats. Mock only the HTTP client.

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
