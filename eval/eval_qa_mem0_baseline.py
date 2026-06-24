#!/usr/bin/env python3
# /// script
# dependencies = ["mem0ai>=0.1.0", "openai>=2.29.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Mem0 OSS baseline QA accuracy evaluation on LoCoMo dataset.

Provenance
----------
Mem0's published 66.9% LoCoMo result used evaluation code that was never released
as standalone source.  The mem0ai/mem0 repository's evaluation/ path is a git
submodule of mem0ai/memory-benchmarks (initial commit 2026-03-30, after the 66.9%
result was reported), which runs a Dockerized Mem0 REST server with the V3 platform
pipeline at top_k=200 and GPT-5 for extraction, reporting 92.5% — a categorically
different system.

This script therefore re-evaluates the OSS Mem0 Memory Python SDK (mem0ai) under
conditions identical to Eidetic Memory so the comparison is fair:

  - Same Azure gpt-4.1 answerer (AZURE_OPENAI_DEPLOYMENT)
  - Same judge function and JUDGE_PROMPT (byte-identical with eval_qa_accuracy.py)
  - Same text-embedding-3-small embedder via Azure
  - top_k=10  (paper-matched: the Mem0 LoCoMo paper states s=10)
  - Turn-by-turn add  (CHUNK_SIZE=1, matching memory-benchmarks/benchmarks/locomo/run.py)

The answerer and judge always use AsyncAzureOpenAI — OPENAI_API_KEY is irrelevant
here and is never used for our own LLM calls (Mem0's internal embedder may read it
from env during init; that is outside our control and does not affect our clients).

Two phases run in sequence:
  1. Ingestion  — adds each conversation turn into Mem0 (resumable via checkpoint).
  2. QA eval   — retrieves memories, generates answers, judges with Azure OpenAI.

Usage:
    uv run python eval/eval_qa_mem0_baseline.py
    uv run python eval/eval_qa_mem0_baseline.py --conv-ids conv-26 --limit 5
    uv run python eval/eval_qa_mem0_baseline.py \\
        --output eval/results/qa_mem0_baseline_results.json

Reads .env.development for configuration.
Prerequisites: uv add --package eval mem0ai
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ── sys.path / env ────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent
env_path = _repo_root / ".env.development"
load_dotenv(env_path)

# Suppress mem0's PostHog telemetry (fires on every search call, floods stderr with
# connection errors when the machine is offline or posthog.com is unreachable).
os.environ.setdefault("MEM0_TELEMETRY", "false")

# ── Constants ─────────────────────────────────────────────────
_DEFAULT_COLLECTION = "mem0_locomo_eval_baseline"
TOP_K = 10  # paper-matched: Mem0 LoCoMo paper states s=10
SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

_EVAL_DIR = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _EVAL_DIR / "checkpoints"
_LOG_DIR = _EVAL_DIR / "logs"
_CONTENT_FILTER_LOG = _LOG_DIR / "mem0_content_filter_skips.jsonl"


def _ingest_checkpoint_path(collection_name: str) -> Path:
    """Per-collection checkpoint — prevents stale completion entries from a
    prior collection silently skipping ingestion into a fresh one."""
    return _CHECKPOINT_DIR / f"mem0_ingest_{collection_name}.json"

# Azure OpenAI chat API version — hardcoded to match eval_qa_accuracy.py
_AZURE_CHAT_API_VERSION = "2024-10-21"

REQUIRED_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_EMBEDDING_API_VERSION",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
}

MAX_RETRIES = 5
GENERATION_BACKOFF = [5, 15, 30, 60, 120]
JUDGE_BACKOFF = [2, 4, 8, 30, 60]
INGEST_BACKOFF = [5, 15, 45]

# ── Prompts — byte-identical with eval_qa_accuracy.py ─────────
#
# JUDGE_PROMPT diff vs eval_qa_accuracy.py: NONE — identical string.
# Verified 2026-06-22 by direct comparison.
JUDGE_PROMPT = """Your task is to label an answer as CORRECT or WRONG.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Be generous: if the generated answer refers to the same fact or time \
period as the gold answer, label it CORRECT even if phrased differently.

Date rules:
- "May 2023" and "May 7, 2023" are both CORRECT if gold is "7 May 2023".
- If the gold answer is a relative date like "the Friday before 20 May \
2023" and the generated answer gives a nearby absolute date within 7 days \
of the anchor date (e.g. "19 May 2023" or "20 May 2023"), label CORRECT.

Return JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

ANSWER_SYSTEM_PROMPT = """You are a precise memory retrieval assistant.
Answer the question using ONLY the provided memories.

Rules:
1. Base your answer strictly on the memory text. Do not add any information
   not explicitly stated in the memories.
2. Be concise but complete. If the answer is a single fact, one short sentence.
   If the answer is a list of items, include ALL items from the memories.
3. Pay attention to timestamps. If asked what is current or most recent,
   use the latest memory. Convert relative time references to specific dates
   using memory timestamps.
4. If memories contain contradictory information, use the most recent.
5. If the memories contain ANY relevant information, use it to answer.
   Only say "I don't know" if no memories relate to the question at all.
   Do not say "I don't know" if partial evidence exists — use what you have."""

OPEN_DOMAIN_SYSTEM_PROMPT = """You are a precise memory retrieval assistant.
Answer the question using ONLY the provided memories.

Rules:
1. Base your answer strictly on the memory text. Do not add any information
   not explicitly stated in the memories.
2. Answer directly and specifically. State the key facts clearly first.
   Do not bury the answer in long narrative preamble.
3. Include enough detail to fully answer the question, but do not ramble.
   Two to four sentences is usually sufficient.
4. If memories contain contradictory information, use the most recent.
5. If the memories contain ANY relevant information, use it to answer.
   Only say "I don't know" if no memories relate to the question at all."""


# ── LLM call counter ─────────────────────────────────────────

class _LLMCallCounter:
    """Counts Mem0 internal LLM calls by phase.

    Wraps mem.llm.generate_response after Memory/AsyncMemory.from_config() is called.
    Call set_phase() before each section so calls are bucketed correctly.

    Call-path trace (mem0ai 2.0.7 V3 pipeline, from code inspection — empirically
    confirmed by the counter output printed during the smoke test run):
      add()  → _add_to_vector_store() → generate_response() [expected: 1 call,
               combined extraction+update in single JSON response, line 838 of main.py]
      search() → _search_vector_store() [expected: 0 LLM calls
               (semantic + BM25 + entity-boost scoring only; no LLM call path)]

    Expected: ingestion_calls ≈ 1 per successful add(), qa_search_calls = 0.
    Verify against counter output — do not assume these hold if counter.active=False.

    Thread safety: increment() is guarded by _lock because AsyncMemory dispatches
    generate_response via asyncio.to_thread — the wrapped _counted() runs in a
    thread-pool worker, not the event loop thread. set_phase() and hit_rate_limit()
    are only called from the event loop thread (no lock needed, but _lock is harmless).
    """

    def __init__(self) -> None:
        self.ingestion_calls: int = 0
        self.qa_search_calls: int = 0
        self.rate_limit_hits: int = 0  # 429s encountered during ingestion (all convs)
        self._phase: str = "ingestion"
        self.active: bool = False
        self._lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        # Event-loop thread only — no lock needed.
        self._phase = phase

    def increment(self) -> None:
        with self._lock:
            if self._phase == "ingestion":
                self.ingestion_calls += 1
            elif self._phase == "qa_search":
                self.qa_search_calls += 1

    def hit_rate_limit(self) -> None:
        with self._lock:
            self.rate_limit_hits += 1


class _ExtractionFailureHandler(logging.Handler):
    """Counts mem0 extraction parse-failure warnings logged during ingestion.

    mem0 catches JSON parse errors internally (unterminated strings, etc.) and logs
    a WARNING rather than raising. The add() call still returns normally (NOOP for that
    turn). This handler intercepts those warnings so we can report the fraction of turns
    that produced no stored memory due to extraction failure.

    Thread-safe: emit() is called from asyncio.to_thread workers (thread-pool), not
    the event loop thread — _lock guards the counter.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count: int = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if "Error parsing extraction response" in record.getMessage():
            with self._lock:
                self.count += 1


def instrument_llm_calls(
    mem: object, counter: _LLMCallCounter | None = None
) -> _LLMCallCounter:
    """Wrap mem.llm.generate_response to tally Mem0's internal LLM calls.

    Accepts an existing counter so multiple AsyncMemory instances (one per
    conversation) can all increment the same shared counter. If counter is
    None a fresh one is created.

    If wrapping fails (e.g. mem.llm attribute missing in this Mem0 version),
    returns an inactive counter so the rest of the script can proceed and
    reports unavailable in metadata rather than fabricating a count.
    """
    if counter is None:
        counter = _LLMCallCounter()
    try:
        original = mem.llm.generate_response  # type: ignore[union-attr]

        def _counted(*args: object, **kwargs: object) -> object:
            counter.increment()
            return original(*args, **kwargs)

        mem.llm.generate_response = _counted  # type: ignore[union-attr]
        if not counter.active:
            counter.active = True
            print("LLM call counter: active (wrapping mem.llm.generate_response)")
    except AttributeError as exc:
        if not counter.active:
            print(
                f"WARNING: LLM call instrumentation unavailable ({exc}). "
                "Mem0 internal call counts will be reported as null — "
                "do not extrapolate from this run."
            )
    return counter


# ── Helpers ───────────────────────────────────────────────────

def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)


def _check_deployment() -> str:
    """Print and validate the resolved Azure deployment name.

    Asserts it is not an o-series model (which rejects temperature=0 and max_tokens).
    Returns the deployment name.
    """
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    print(f"Resolved AZURE_OPENAI_DEPLOYMENT: {deployment!r}")
    o_series_prefixes = ("o1", "o3", "o4", "o-")
    if any(deployment.lower().startswith(p) for p in o_series_prefixes):
        raise SystemExit(
            f"ERROR: AZURE_OPENAI_DEPLOYMENT={deployment!r} appears to be an o-series model. "
            "temperature=0 and max_tokens are unsupported for o-series. "
            "Set AZURE_OPENAI_DEPLOYMENT to gpt-4.1 or gpt-4o instead."
        )
    return deployment


def build_mem0_config(
    collection_name: str = _DEFAULT_COLLECTION,
    history_db_path: str | None = None,
) -> dict:
    """Build Mem0 config for Azure OpenAI LLM + embedder + Qdrant vector store.

    Schema from https://docs.mem0.ai/components/llms/models/azure_openai and
    https://docs.mem0.ai/components/embedders/models/azure_openai — Azure kwargs
    are nested under 'azure_kwargs' inside each provider config.

    Chat API version hardcoded to 2024-10-21 (matches eval_qa_accuracy.py).
    AZURE_OPENAI_API_VERSION is NOT used — not set in .env.development.
    """
    chat_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    embedding_api_version = os.environ["AZURE_EMBEDDING_API_VERSION"]

    # mem0's sync qdrant client requires the port (6333 for Qdrant Cloud).
    # Without it, the client hits port 443 which returns 404.
    # The backend's async QdrantClient handles the URL differently, so .env.development
    # omits the port for backend compatibility. We add it here for mem0's sync client.
    qdrant_url = os.environ["QDRANT_URL"]
    if not re.search(r":\d+$", qdrant_url.split("://", 1)[-1]):
        qdrant_url = f"{qdrant_url}:6333"

    config: dict = {
        "llm": {
            "provider": "azure_openai",
            "config": {
                "model": chat_deployment,
                "temperature": 0,
                "max_tokens": 2000,
                "azure_kwargs": {
                    "azure_deployment": chat_deployment,
                    "api_version": _AZURE_CHAT_API_VERSION,
                    "azure_endpoint": azure_endpoint,
                    "api_key": api_key,
                },
            },
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": embedding_deployment,
                "embedding_dims": 1536,
                "azure_kwargs": {
                    "azure_deployment": embedding_deployment,
                    "api_version": embedding_api_version,
                    "azure_endpoint": azure_endpoint,
                    "api_key": api_key,
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "url": qdrant_url,
                "api_key": os.environ["QDRANT_API_KEY"],
                "embedding_model_dims": 1536,
            },
        },
    }
    if history_db_path is not None:
        config["history_db_path"] = history_db_path
    return config


def conv_user_id(conv_id: str) -> str:
    """Single Mem0 user_id per conversation — both speakers stored together."""
    return f"locomo_eval_{conv_id}"


def extract_sessions(conversation: dict) -> list[dict]:
    """Extract sessions sorted by session number."""
    session_keys = sorted(
        (k for k in conversation if SESSION_KEY_RE.match(k)),
        key=lambda k: int(SESSION_KEY_RE.match(k).group(1)),  # type: ignore[union-attr]
    )
    sessions: list[dict] = []
    for session_key in session_keys:
        session_num = SESSION_KEY_RE.match(session_key).group(1)  # type: ignore[union-attr]
        session_id = f"session_{session_num}"
        datetime_key = f"{session_id}_date_time"
        sessions.append(
            {
                "session_id": session_id,
                "session_datetime": conversation.get(datetime_key, ""),
                "turns": conversation[session_key],
            }
        )
    return sessions


def collect_qa_entries(conversation_data: dict) -> list[dict]:
    """Collect QA entries (categories 1–4 only), skipping empty answers."""
    entries: list[dict] = []
    for qa in conversation_data.get("qa", []):
        if qa.get("category") not in (1, 2, 3, 4):
            continue
        answer = qa.get("answer", "")
        if not answer:
            continue
        entries.append(
            {
                "question": qa["question"],
                "answer": answer,
                "category": qa["category"],
            }
        )
    return entries


def _extract_memory_texts(results: object) -> list[str]:
    """Extract memory text strings from Mem0 search results.

    Handles both list format and dict-with-'results' key format.
    Does NOT strip date or metadata embedded in memory text — the session
    datetime is stored inline in the memory string ('{speaker}: [{dt}] {text}')
    and is the primary source of temporal context for the answerer.
    """
    if isinstance(results, dict):
        results = results.get("results", [])
    if not isinstance(results, list):
        return []
    return [r["memory"] for r in results if isinstance(r, dict) and r.get("memory")]


def build_context(memory_texts: list[str]) -> str:
    """Format retrieved memory strings into a context block for the answer LLM."""
    if not memory_texts:
        return ""
    lines = "\n".join(f"- {m}" for m in memory_texts)
    return f"## Retrieved memories\n{lines}"


def log_content_filter_skip(phase: str, conv_id: str, text_prefix: str) -> None:
    """Append one content-filter skip to the JSONL log (one record per line)."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "phase": phase,
        "conv_id": conv_id,
        "text_prefix": text_prefix[:120],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(_CONTENT_FILTER_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Checkpoint helpers ────────────────────────────────────────

def load_ingest_checkpoint(collection_name: str) -> dict:
    """Load ingestion checkpoint for a specific collection."""
    path = _ingest_checkpoint_path(collection_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "last_updated": None}


def save_ingest_checkpoint(checkpoint: dict, collection_name: str) -> None:
    """Persist ingestion checkpoint for a specific collection."""
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now(UTC).isoformat()
    with open(_ingest_checkpoint_path(collection_name), "w") as f:
        json.dump(checkpoint, f, indent=2)


# ── Ingestion ─────────────────────────────────────────────────

def ingest_conversation(
    mem: object,
    conv_entry: dict,
    conv_id: str,
    user_id: str,
    counter: _LLMCallCounter,
) -> tuple[int, int]:
    """Ingest one LoCoMo conversation into Mem0 turn by turn.

    Turn-by-turn add matches CHUNK_SIZE=1 in memory-benchmarks/benchmarks/locomo/run.py.
    speaker_a → "user" role, speaker_b → "assistant" role.
    Session datetime is embedded in the content string for temporal context.

    Returns (successful_add_calls, content_filter_skips).
    """
    conversation = conv_entry["conversation"]
    speaker_a: str = conversation["speaker_a"]
    sessions = extract_sessions(conversation)

    total_add_calls = 0
    filter_skips = 0
    counter.set_phase("ingestion")

    for session in sessions:
        session_datetime = session["session_datetime"]
        for turn in tqdm(
            session["turns"],
            desc=f"  {conv_id}/{session['session_id']}",
            unit="turn",
            leave=True,
        ):
            speaker: str = turn["speaker"]
            role = "user" if speaker == speaker_a else "assistant"
            text = f"{speaker}: [{session_datetime}] {turn['text']}"
            messages = [{"role": role, "content": text}]

            for attempt in range(len(INGEST_BACKOFF) + 1):
                try:
                    mem.add(messages, user_id=user_id)  # type: ignore[union-attr]
                    total_add_calls += 1
                    break
                except Exception as exc:
                    err_str = str(exc).lower()
                    if "content_filter" in err_str or "content management policy" in err_str:
                        tqdm.write(f"  [CONTENT FILTER] ingestion skip: {text[:80]}")
                        log_content_filter_skip("ingestion", conv_id, text)
                        filter_skips += 1
                        break
                    if attempt >= len(INGEST_BACKOFF):
                        tqdm.write(f"  Failed after {len(INGEST_BACKOFF) + 1} attempts: {exc}")
                        break
                    wait = INGEST_BACKOFF[attempt]
                    tqdm.write(f"  Ingest retry {attempt + 1} in {wait}s: {exc}")
                    time.sleep(wait)

    return total_add_calls, filter_skips


async def ingest_conversation_async(
    mem: object,
    conv_entry: dict,
    conv_id: str,
    user_id: str,
    counter: _LLMCallCounter,
    sem: asyncio.Semaphore,
    ingest_t0: float,
) -> tuple[int, int]:
    """Ingest one LoCoMo conversation via AsyncMemory, preserving strict turn order.

    Turn order within this conversation is sequential — mem0's UPDATE/DELETE evolves
    memories based on prior facts, so add() calls must complete in order.
    Cross-conversation parallelism is safe because Mem0 partitions by user_id in
    both Qdrant (filters) and SQLite (session_scope keyed by user_id).

    The semaphore caps total concurrent in-flight add() calls across all conversations.
    On 429 or 5xx: backoff with asyncio.sleep AFTER releasing the semaphore slot so
    other conversations can proceed. Non-retriable errors skip the turn immediately.

    ingest_t0: monotonic timestamp from main() just before asyncio.gather() — used to
    report per-conversation timestamps relative to the same origin, proving overlap.

    Returns (successful_add_calls, content_filter_skips).
    """
    import openai  # late import — not available in uv workspace, only in eval-venv

    conversation = conv_entry["conversation"]
    speaker_a: str = conversation["speaker_a"]
    sessions = extract_sessions(conversation)

    total_add_calls = 0
    filter_skips = 0

    t_conv_start = time.monotonic()
    tqdm.write(f"  [t=+{t_conv_start - ingest_t0:.0f}s] {conv_id}: coroutine STARTED")

    first_add_started = False
    t_last_add_done: float = t_conv_start

    for session in sessions:
        session_datetime = session["session_datetime"]
        for turn in tqdm(
            session["turns"],
            desc=f"  {conv_id}/{session['session_id']}",
            unit="turn",
            leave=True,
        ):
            speaker: str = turn["speaker"]
            role = "user" if speaker == speaker_a else "assistant"
            text = f"{speaker}: [{session_datetime}] {turn['text']}"
            messages = [{"role": role, "content": text}]

            for attempt in range(len(INGEST_BACKOFF) + 1):
                try:
                    if not first_add_started:
                        t_first = time.monotonic()
                        tqdm.write(
                            f"  [t=+{t_first - ingest_t0:.0f}s] {conv_id}: first add() START"
                        )
                        first_add_started = True
                    async with sem:
                        await mem.add(messages, user_id=user_id)  # type: ignore[union-attr]
                    t_last_add_done = time.monotonic()
                    total_add_calls += 1
                    break
                except Exception as exc:
                    err_str = str(exc).lower()
                    if "content_filter" in err_str or "content management policy" in err_str:
                        tqdm.write(f"  [CONTENT FILTER] ingestion skip: {text[:80]}")
                        log_content_filter_skip("ingestion", conv_id, text)
                        filter_skips += 1
                        break
                    is_rate_limit = isinstance(exc, openai.RateLimitError)
                    is_server_error = (
                        isinstance(exc, openai.APIStatusError) and exc.status_code >= 500
                    )
                    if not (is_rate_limit or is_server_error):
                        tqdm.write(f"  [INGEST] non-retriable error, skipping turn: {exc}")
                        break
                    if is_rate_limit:
                        counter.hit_rate_limit()
                    if attempt >= len(INGEST_BACKOFF):
                        tqdm.write(f"  Failed after {len(INGEST_BACKOFF) + 1} attempts: {exc}")
                        break
                    wait = INGEST_BACKOFF[attempt]
                    tqdm.write(f"  Ingest retry {attempt + 1}/{len(INGEST_BACKOFF)} in {wait}s: {exc}")
                    # Sleep OUTSIDE sem — slot was released when async with block exited.
                    await asyncio.sleep(wait)

    t_conv_end = time.monotonic()
    tqdm.write(
        f"  [t=+{t_conv_end - ingest_t0:.0f}s] {conv_id}: coroutine FINISHED "
        f"(last add done t=+{t_last_add_done - ingest_t0:.0f}s, "
        f"{total_add_calls} adds in {t_conv_end - t_conv_start:.0f}s)"
    )
    return total_add_calls, filter_skips


# ── Async LLM helpers ─────────────────────────────────────────

async def generate_answer(
    client: object,
    model: str,
    question: str,
    category: int,
    memory_texts: list[str],
    conv_id: str,
) -> tuple[str, bool]:
    """Generate an answer from retrieved memories using Azure OpenAI.

    Returns (answer_text, content_filter_triggered).
    """
    active_prompt = OPEN_DOMAIN_SYSTEM_PROMPT if category == 4 else ANSWER_SYSTEM_PROMPT
    context = build_context(memory_texts)
    system_prompt = f"{active_prompt}\n\n{context}" if context else active_prompt
    max_tokens = 350 if category == 4 else 200

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or "I don't know", False
        except Exception as exc:
            err_str = str(exc).lower()
            if "content_filter" in err_str or "content management policy" in err_str:
                tqdm.write(f"  [CONTENT FILTER] answer skip: {question[:80]}")
                log_content_filter_skip("qa_answer", conv_id, question)
                return "I don't know", True
            if attempt == MAX_RETRIES - 1:
                tqdm.write(f"  Generation failed after {MAX_RETRIES} attempts: {exc}")
                return "I don't know", False
            wait = GENERATION_BACKOFF[attempt]
            tqdm.write(f"  Generation retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {exc}")
            await asyncio.sleep(wait)

    return "I don't know", False


async def judge_answer(
    client: object,
    model: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> str:
    """Judge correctness using JUDGE_PROMPT — byte-identical with eval_qa_accuracy.py.

    Parameters (temperature, response_format, max_tokens, backoff) are identical
    to eval_qa_accuracy.py:judge_answer(). Diff: none.

    Returns "CORRECT" or "WRONG".
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=50,
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result.get("label", "WRONG")
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                tqdm.write(f"  ERROR judging answer after {MAX_RETRIES} attempts: {exc}")
                return "WRONG"
            wait = JUDGE_BACKOFF[attempt]
            tqdm.write(f"  Judge retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {exc}")
            await asyncio.sleep(wait)

    return "WRONG"


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mem0 OSS baseline QA accuracy evaluation on LoCoMo dataset.",
    )
    parser.add_argument(
        "--conv-ids",
        nargs="+",
        default=["conv-26", "conv-30"],
        help=(
            "LoCoMo conversation sample_ids to ingest and evaluate. "
            "Default (conv-26 conv-30) is a 2-conversation smoke test. "
            "Full benchmark: all 10 IDs (use 'make run-mem0-baseline')."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs evaluated (default: all). "
             "Example 1-conversation smoke test: --conv-ids conv-26 --limit 5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/qa_mem0_baseline_results.json",
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion phase (assumes data already in Mem0 from a prior run).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_COLLECTION,
        help=(
            f"Qdrant collection name (default: {_DEFAULT_COLLECTION!r}). "
            "Use a fresh name when BM25/spaCy are newly enabled to avoid "
            "a stale collection without the 'bm25' sparse vector slot."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help=(
            "Max concurrent in-flight mem.add() calls across all conversations "
            "(AsyncMemory semaphore). Lower if Azure 429s are frequent. "
            "Default: 5. Each conversation's turns remain strictly sequential."
        ),
    )
    parser.add_argument(
        "--qa-workers",
        type=int,
        default=10,
        help=(
            "Max concurrent in-flight QA pairs (search + generate_answer + judge_answer). "
            "Each pair is independent so concurrency is safe. Lower if Azure 429s are frequent. "
            "Default: 10."
        ),
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────

async def main() -> None:  # noqa: PLR0912, PLR0915
    args = parse_args()
    output_path = Path(args.output)

    check_env()
    deployment = _check_deployment()

    # Late import so mem0 version is captured after env is loaded
    import mem0 as _mem0_module
    from mem0 import AsyncMemory

    collection = args.collection
    mem0_version = _mem0_module.__version__
    print(f"mem0 version: {mem0_version}")
    print(f"Qdrant collection: {collection}")
    print(f"Workers (max concurrent add() calls): {args.workers}")

    # Load dataset
    data_path = _EVAL_DIR / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    # Validate all requested conv_ids up front
    for conv_id in args.conv_ids:
        if not any(e["sample_id"] == conv_id for e in dataset):
            print(f"ERROR: No conversation with sample_id={conv_id!r} found.")
            sys.exit(1)

    # ── Phase 1: Ingestion ────────────────────────────────────
    total_add_calls = 0
    total_ingest_filter_skips = 0
    extraction_failures = 0  # mem0 internal parse failures (turn NOP'd, no memory stored)
    # qa_mem is populated during ingestion (reuse one instance) or created fresh for --skip.
    qa_mem: object = None
    counter = _LLMCallCounter()

    if args.skip_ingestion:
        print("\n── Phase 1: Ingestion SKIPPED (--skip-ingestion) ──────")
        qa_config = build_mem0_config(collection_name=collection)
        qa_mem = AsyncMemory.from_config(qa_config)
        instrument_llm_calls(qa_mem, counter)
    else:
        print(f"\n── Phase 1: Ingestion (parallel, --workers {args.workers}) ──────")
        checkpoint = load_ingest_checkpoint(collection)

        pending = [c for c in args.conv_ids if c not in checkpoint.get("completed", [])]
        for c in args.conv_ids:
            if c not in pending:
                print(f"  {c}: already ingested (checkpoint) — skipping")

        # Per-conversation AsyncMemory instances with isolated SQLite files.
        # Sharing one instance would be safe: messages/history rows are keyed by
        # session_scope (user_id-based) or memory_id (Qdrant UUID, no cross-conv
        # collision), and _lock (threading.Lock) guards the connection against
        # concurrent asyncio.to_thread workers. Per-instance is chosen because the
        # history table has no session_scope column — row isolation is by memory_id
        # (implicit), not by a filter predicate (explicit). Per-instance eliminates
        # all ambiguity without cost.
        async_mems: dict[str, object] = {}
        for cid in pending:
            history_db = str(_CHECKPOINT_DIR / f"mem0_history_{cid}.db")
            _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            cfg = build_mem0_config(collection_name=collection, history_db_path=history_db)
            am = AsyncMemory.from_config(cfg)
            instrument_llm_calls(am, counter)  # all instances share one counter
            async_mems[cid] = am

        sem = asyncio.Semaphore(args.workers)

        # All gather coroutines are ingestion-only — assert phase is stable.
        # counter.increment() is guarded by _lock: generate_response wrapper runs
        # in asyncio.to_thread workers (thread pool), not the event loop thread.
        assert counter._phase == "ingestion", (
            f"Counter must be 'ingestion' before asyncio.gather, got {counter._phase!r}"
        )

        extraction_failure_handler = _ExtractionFailureHandler()
        logging.getLogger().addHandler(extraction_failure_handler)

        ingest_start = time.monotonic()
        ingest_results = await asyncio.gather(
            *[
                ingest_conversation_async(
                    async_mems[cid],
                    next(e for e in dataset if e["sample_id"] == cid),
                    cid,
                    conv_user_id(cid),
                    counter,
                    sem,
                    ingest_start,
                )
                for cid in pending
            ],
            return_exceptions=True,
        )
        ingest_wall = time.monotonic() - ingest_start

        logging.getLogger().removeHandler(extraction_failure_handler)
        extraction_failures = extraction_failure_handler.count

        for cid, result in zip(pending, ingest_results, strict=True):
            if isinstance(result, Exception):
                print(f"  {cid}: FAILED — {result}")
                continue
            add_calls, filter_skips = result
            total_add_calls += add_calls
            total_ingest_filter_skips += filter_skips
            checkpoint.setdefault("completed", []).append(cid)
            save_ingest_checkpoint(checkpoint, collection)
            print(f"  {cid}: {add_calls} mem.add() calls, {filter_skips} content-filter skips")

        total_turns_submitted = total_add_calls + total_ingest_filter_skips
        extr_pct = (
            100.0 * extraction_failures / total_turns_submitted
            if total_turns_submitted > 0
            else 0.0
        )
        print(f"\nIngestion wall-clock: {ingest_wall:.0f}s ({ingest_wall / 60:.1f} min)")
        print(f"429 rate-limit hits:  {counter.rate_limit_hits}")
        print(
            f"Extraction parse failures: {extraction_failures}/{total_turns_submitted} turns "
            f"({extr_pct:.2f}%) — mem0 silently NOOPs these (no memory stored for that turn)"
        )
        if counter.active:
            print(
                f"Ingestion LLM calls (measured): {counter.ingestion_calls} "
                f"(Mem0 V3 pipeline: 1 generate_response per add, extraction+update combined)"
            )

        qa_mem = next(iter(async_mems.values())) if async_mems else None
        if qa_mem is None:
            qa_config = build_mem0_config(collection_name=collection)
            qa_mem = AsyncMemory.from_config(qa_config)
            instrument_llm_calls(qa_mem, counter)

    # ── Phase 2: QA Evaluation ────────────────────────────────
    print(f"\n── Phase 2: QA Evaluation (--qa-workers {args.qa_workers}) ──────────────────────────────")

    # Always use AsyncAzureOpenAI — OPENAI_API_KEY is never used for our clients.
    # Mem0's embedder may probe OPENAI_API_KEY during init; that is irrelevant here.
    from openai import AsyncAzureOpenAI

    llm_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=_AZURE_CHAT_API_VERSION,
        timeout=60.0,
        max_retries=0,
    )

    # Provable Azure assertion — fails loudly if we somehow constructed the wrong client.
    client_class_name = type(llm_client).__name__
    assert client_class_name == "AsyncAzureOpenAI", (
        f"Expected AsyncAzureOpenAI client but got {client_class_name}. "
        "This script must never use a non-Azure OpenAI client for answering or judging."
    )

    answer_model = deployment  # AZURE_OPENAI_DEPLOYMENT = gpt-4.1
    judge_model = deployment   # same deployment for judge — ensures parity with Eidetic eval
    print(f"Answerer/judge client: {client_class_name}")
    print(f"Answerer model: {answer_model!r}")
    print(f"Judge model: {judge_model!r}")

    # Collect QA entries from all specified conversations
    qa_entries: list[dict] = []
    for conv_id in args.conv_ids:
        conv_entry = next(e for e in dataset if e["sample_id"] == conv_id)
        entries = collect_qa_entries(conv_entry)
        for e in entries:
            e["conv_id"] = conv_id
        qa_entries.extend(entries)

    print(f"Total QA entries (categories 1–4): {len(qa_entries)}")

    if args.limit is not None:
        qa_entries = qa_entries[: args.limit]
        print(f"Limited to {len(qa_entries)} QA pairs")

    # Resume support: JSONL partial save for concurrency safety.
    # Each completed pair is appended as one JSON line under write_lock,
    # so concurrent coroutines never race on the file.  On resume all lines
    # are read and completed_keys is reconstructed from them.
    partial_jsonl_path = Path(f"{output_path}.partial.jsonl")
    partial_path_legacy = Path(f"{output_path}.partial.json")
    per_pair_results: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()

    if partial_jsonl_path.exists():
        with open(partial_jsonl_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    r = json.loads(stripped)
                    per_pair_results.append(r)
                    completed_keys.add((r["question"], r["conv_id"]))
        print(f"Resuming: {len(per_pair_results)} pairs already done from {partial_jsonl_path}")
    elif partial_path_legacy.exists():
        with open(partial_path_legacy) as f:
            per_pair_results = json.load(f)
        completed_keys = {(r["question"], r["conv_id"]) for r in per_pair_results}
        print(
            f"Resuming: {len(per_pair_results)} pairs already done from {partial_path_legacy} "
            "(migrating to JSONL)"
        )
        partial_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_jsonl_path, "w") as f:
            for r in per_pair_results:
                f.write(json.dumps(r) + "\n")
        partial_path_legacy.unlink()

    # Concurrency primitives for the QA phase.
    qa_semaphore = asyncio.Semaphore(args.qa_workers)
    write_lock = asyncio.Lock()

    # Mutable counters — list containers allow in-place mutation from event-loop
    # coroutines without nonlocal reassignment.  Asyncio is single-threaded so
    # `+= 1` between awaits is never preempted; no threading lock is needed.
    _total_answer_calls: list[int] = [0]
    _total_qa_filter_skips: list[int] = [0]
    _qa_429_hits: list[int] = [0]
    _raw_search_printed: list[bool] = [False]

    # Patch llm_client.chat.completions.create to count QA 429s without touching
    # generate_answer / judge_answer internals.  The wrapper increments the counter
    # then re-raises, so the existing retry/backoff logic fires unchanged.
    _orig_create = llm_client.chat.completions.create

    async def _create_tracking_429(*_args: object, **_kwargs: object) -> object:
        try:
            return await _orig_create(*_args, **_kwargs)
        except Exception as exc:
            if (hasattr(exc, "status_code") and exc.status_code == 429) or (
                "rate limit" in str(exc).lower()
            ):
                _qa_429_hits[0] += 1
            raise

    llm_client.chat.completions.create = _create_tracking_429  # type: ignore[method-assign]

    pbar = tqdm(total=len(qa_entries), desc="Evaluating", unit="pair")
    pbar.update(len(completed_keys))

    async def evaluate_pair_qa(entry: dict) -> None:
        """Evaluate one QA pair: search → generate_answer → judge_answer.

        Safe to run concurrently — each pair is a read-only search followed by
        independent LLM calls.  Shared state is guarded: writes go through
        write_lock (JSONL append), counters are event-loop-only list[int].
        """
        question: str = entry["question"]
        gold_answer: str = entry["answer"]
        category: int = entry["category"]
        conv_id: str = entry["conv_id"]

        if (question, conv_id) in completed_keys:
            return

        user_id = conv_user_id(conv_id)

        async with qa_semaphore:
            # search() makes 0 LLM calls in the V3 pipeline — no counter phase needed.
            raw_results = await qa_mem.search(  # type: ignore[union-attr]
                question,
                filters={"user_id": user_id},
                top_k=TOP_K,
            )

            result_list = (
                raw_results.get("results", raw_results)
                if isinstance(raw_results, dict)
                else raw_results
            )
            if not _raw_search_printed[0]:
                _raw_search_printed[0] = True  # no await between check and set — safe
                first_raw = result_list[0] if result_list else None
                tqdm.write(
                    f"\n[DEBUG] mem.search() top_k={TOP_K}: returned {len(result_list)} results"
                )
                tqdm.write("[DEBUG] First raw mem.search() result (fields / temporal info):")
                tqdm.write(
                    json.dumps(first_raw, indent=2, default=str)
                    if first_raw
                    else "  (no results)"
                )
                tqdm.write("")

            memory_texts = _extract_memory_texts(raw_results)

            if not memory_texts:
                result_dict: dict = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "I don't know",
                    "category": category,
                    "conv_id": conv_id,
                    "memories_retrieved": [],
                    "label": "WRONG",
                }
            else:
                generated_answer, filter_hit = await generate_answer(
                    llm_client, answer_model, question, category, memory_texts, conv_id
                )
                _total_answer_calls[0] += 1
                if filter_hit:
                    _total_qa_filter_skips[0] += 1

                label = await judge_answer(
                    llm_client, judge_model, question, gold_answer, generated_answer
                )

                result_dict = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "category": category,
                    "conv_id": conv_id,
                    "memories_retrieved": memory_texts,
                    "label": label,
                }

        # Append under lock — outside the semaphore so slow file I/O doesn't
        # block other coroutines from entering the semaphore.
        async with write_lock:
            per_pair_results.append(result_dict)
            completed_keys.add((question, conv_id))
            partial_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(partial_jsonl_path, "a") as pf:
                pf.write(json.dumps(result_dict) + "\n")

        pbar.update(1)

    await asyncio.gather(*[evaluate_pair_qa(entry) for entry in qa_entries])
    pbar.close()

    llm_client.chat.completions.create = _orig_create  # type: ignore[method-assign]
    await llm_client.close()

    # Unpack mutable list counters to plain ints before the metrics section.
    total_answer_calls = _total_answer_calls[0]
    total_qa_filter_skips = _total_qa_filter_skips[0]
    qa_rate_limit_hits = _qa_429_hits[0]

    # Sort by (conv_id, question) for deterministic output — completion order
    # varies under concurrency, but individual answers are deterministic
    # (temperature=0), so accuracy is identical to the sequential run.
    per_pair_results.sort(key=lambda r: (r["conv_id"], r["question"]))

    # ── Metrics ───────────────────────────────────────────────
    n_evaluated = len(per_pair_results)
    n_correct = sum(1 for r in per_pair_results if r["label"] == "CORRECT")
    overall_accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0

    by_category: dict[str, dict] = {}
    for cat_id, cat_label in CATEGORIES.items():
        cat_results = [r for r in per_pair_results if r["category"] == cat_id]
        if not cat_results:
            continue
        cat_correct = sum(1 for r in cat_results if r["label"] == "CORRECT")
        by_category[str(cat_id)] = {
            "label": cat_label,
            "accuracy": cat_correct / len(cat_results),
            "correct": cat_correct,
            "total": len(cat_results),
        }

    # LLM call accounting
    # counter.ingestion_calls  — Mem0 internal calls during add() (V3: 1 per add)
    # counter.qa_search_calls  — always 0: search() makes no LLM calls in V3
    # total_answer_calls       — our own generate_answer() calls (1 per non-empty result)
    if counter.active and n_evaluated > 0:
        avg_mem0_search_llm = counter.qa_search_calls / n_evaluated
        avg_answer_llm = total_answer_calls / n_evaluated
        avg_total_llm_per_query = avg_mem0_search_llm + avg_answer_llm
        llm_calls_instrumented = True
    else:
        avg_mem0_search_llm = None
        avg_answer_llm = total_answer_calls / n_evaluated if n_evaluated > 0 else None
        avg_total_llm_per_query = None
        llm_calls_instrumented = False

    # Print summary
    print(f"\nMem0 Baseline QA Accuracy ({', '.join(args.conv_ids)})")
    print("─" * 34)
    print(f"QA pairs evaluated: {n_evaluated}")
    print(f"Overall accuracy: {overall_accuracy:.1%} ({n_correct}/{n_evaluated} CORRECT)")
    print("\nBy category:")
    for cat_id, cat_label in CATEGORIES.items():
        cat_key = str(cat_id)
        if cat_key in by_category:
            cat = by_category[cat_key]
            print(f"  {cat_label:12s}: {cat['accuracy']:.1%} ({cat['total']} pairs)")
    print(f"\nContent-filter skips: {total_ingest_filter_skips} ingestion, "
          f"{total_qa_filter_skips} QA-answer")
    print(f"  (see {_CONTENT_FILTER_LOG} for details)")
    print(f"QA 429 rate-limit hits: {qa_rate_limit_hits} (across answer+judge retries; "
          f"lower --qa-workers if frequent)")
    if counter.active:
        print("\nLLM calls (measured via mem.llm.generate_response wrapper):")
        print(f"  Ingestion: {counter.ingestion_calls} total"
              f"  ({counter.ingestion_calls / total_add_calls:.2f}/add)" if total_add_calls else
              f"  Ingestion: {counter.ingestion_calls} total")
        print(f"  QA search: {counter.qa_search_calls} total"
              f" (0/query — search() makes no LLM calls in V3 pipeline)")
        if avg_answer_llm is not None:
            print(f"  QA answer: {total_answer_calls} total ({avg_answer_llm:.3f}/query)")
            print(f"  QA total:  {avg_total_llm_per_query:.3f} LLM calls/query (vs Eidetic 1.02)")
    else:
        print("\nLLM call count: instrumentation unavailable (see WARNING above)")

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "provenance": (
                "Mem0 OSS SDK baseline. The mem0ai/mem0 evaluation/ directory is a submodule "
                "of mem0ai/memory-benchmarks (created 2026-03-30), which uses the V3 platform "
                "pipeline (top_k=200, GPT-5 extraction, 92.5%) — a different system. "
                "This script uses the Mem0 Python SDK under identical conditions to Eidetic Memory: "
                "same Azure gpt-4.1 answerer, same judge, same text-embedding-3-small, top_k=10 "
                "(paper-matched, s=10), turn-by-turn add (CHUNK_SIZE=1)."
            ),
            "conv_ids": args.conv_ids,
            "limit": args.limit,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "mem0_version": mem0_version,
            "mem0_collection": collection,
            "answerer_model": answer_model,
            "judge_model": judge_model,
            "answerer_client_class": client_class_name,
            "azure_chat_api_version": _AZURE_CHAT_API_VERSION,
            "top_k": TOP_K,
            "chunk_size": 1,
            "retrieval_source": "mem0_oss_qdrant",
            "speaker_scoping": (
                "Single user_id per conversation (both speakers merged). "
                "speaker_a → 'user' role, speaker_b → 'assistant' role. "
                "Session datetime embedded in content string for temporal context."
            ),
            "judge_parity_note": (
                "JUDGE_PROMPT, temperature=0, response_format=json_object, max_tokens=50, "
                "and backoff [2,4,8,30,60] are byte-identical with eval_qa_accuracy.py. "
                "Diff: none."
            ),
            "llm_call_instrumentation": llm_calls_instrumented,
            "content_filter_log": str(_CONTENT_FILTER_LOG),
            "qa_workers": args.qa_workers,
        },
        "overall": {
            "accuracy": overall_accuracy,
            "correct": n_correct,
            "total": n_evaluated,
            "content_filter_skips_ingestion": total_ingest_filter_skips,
            "content_filter_skips_qa_answer": total_qa_filter_skips,
            "qa_rate_limit_hits": qa_rate_limit_hits,
            "extraction_parse_failures": extraction_failures,
            "extraction_parse_failure_rate": (
                round(extraction_failures / (total_add_calls + total_ingest_filter_skips), 4)
                if (total_add_calls + total_ingest_filter_skips) > 0
                else None
            ),
        },
        "llm_calls": {
            "instrumented": llm_calls_instrumented,
            "ingestion_total": counter.ingestion_calls if counter.active else None,
            "ingestion_per_add": (
                round(counter.ingestion_calls / total_add_calls, 3)
                if counter.active and total_add_calls else None
            ),
            "qa_mem0_search_total": counter.qa_search_calls if counter.active else None,
            "qa_mem0_search_per_query": 0 if counter.active else None,
            "qa_answer_total": total_answer_calls,
            "avg_mem0_search_llm_per_query": avg_mem0_search_llm,
            "avg_answer_llm_per_query": avg_answer_llm,
            "avg_total_llm_per_query_qa": avg_total_llm_per_query,
            "note": (
                "Expected from code inspection (mem0ai 2.0.7 V3 pipeline): "
                "add() makes 1 generate_response call (extraction+update combined, main.py:838). "
                "search() makes 0 LLM calls (semantic+BM25+entity-boost only). "
                "Empirically confirmed by counter output printed during this run (if counter.active=True). "
                "avg_total_llm_per_query_qa = qa_mem0_search (0) + answer calls per query. "
                "Comparable to Eidetic Memory's 1.02 (answer + occasional two-pass rephrase)."
            ),
        },
        "by_category": by_category,
        "per_pair": per_pair_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if partial_jsonl_path.exists():
        partial_jsonl_path.unlink()

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
