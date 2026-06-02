#!/usr/bin/env python3
# /// script
# dependencies = ["google-genai>=1.0.0", "qdrant-client>=1.7.0",
#                 "python-dotenv>=1.0.0", "tqdm>=4.66.0",
#                 "pydantic-settings>=2.0.0", "pydantic>=2.0.0"]
# ///
"""
Ingest a LoCoMo conversation through the real MemoryManager pipeline.

Usage:
    uv run python eval/ingest_locomo_production.py
    uv run python eval/ingest_locomo_production.py --conv-ids conv-26 conv-30
    uv run python eval/ingest_locomo_production.py --cleanup
    uv run python eval/ingest_locomo_production.py --dry-run

Reads .env.development for configuration.
Stores extracted memories in the production eidetic_memories Qdrant collection
under a dedicated eval user ID derived from --conv-id.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# ── sys.path setup for backend imports ───────────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "backend" / "packages" / "config" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "storage" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "llm" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "retrieval" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "memory" / "src"))

from config.settings import Settings  # noqa: E402
from llm.embeddings.azure import AzureEmbeddingService  # noqa: E402
from llm.generation.azure import AzureService  # noqa: E402
from llm.generation.base import LLMError  # noqa: E402
from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402
from llm.generation.claude import ClaudeService  # noqa: E402
from llm.generation.gemini import GeminiService  # noqa: E402
from llm.generation.groq import GroqService  # noqa: E402
from memory.manager import MemoryManager  # noqa: E402
from memory.models.conversation import ConversationPair, Message  # noqa: E402
from storage.qdrant import QdrantMemoryStore  # noqa: E402

# ── Configuration ────────────────────────────────────────────
env_path = _repo_root / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = ["GOOGLE_API_KEY", "QDRANT_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]

SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

_RETRYABLE_ERRORS = (
    asyncio.CancelledError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    openai.APITimeoutError,
    openai.APIConnectionError,
    LLMError,  # Azure occasionally returns malformed JSON in tool calls
    ResponseHandlingException,
)
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [5, 15, 45]

_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
_CHECKPOINT_FILE = _CHECKPOINT_DIR / "ingestion_progress.json"


def load_checkpoint() -> dict:
    """Load checkpoint file, returning empty state if it doesn't exist."""
    if _CHECKPOINT_FILE.exists():
        with open(_CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "last_updated": None}


def save_checkpoint(checkpoint: dict) -> None:
    """Write checkpoint file with current timestamp."""
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now(UTC).isoformat()
    with open(_CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def is_completed(checkpoint: dict, conv_id: str) -> bool:
    """Check if a conversation has already been fully ingested."""
    return conv_id in checkpoint.get("completed", [])


def mark_completed(checkpoint: dict, conv_id: str) -> None:
    """Mark a conversation as fully ingested and persist."""
    if conv_id not in checkpoint["completed"]:
        checkpoint["completed"].append(conv_id)
    save_checkpoint(checkpoint)


def replace_user_with_speaker(content: str, speaker_name: str) -> str:
    """Replace generic 'User'/'user' references with the actual speaker name.

    Replaces:
    - Capitalized 'User' anywhere (always a proper noun in extracted facts)
    - Lowercase 'user' at string start or after sentence boundaries (. ! ?)
    Avoids corrupting 'user' as a common noun mid-sentence (e.g. 'end-user').
    """
    # Capitalized User is always a proper noun reference in extracted facts
    content = re.sub(r"\bUser\b", speaker_name, content)
    # Lowercase user at sentence boundaries
    content = re.sub(r"(?<=[.!?]\s)user\b", speaker_name, content)
    # Lowercase user at string start
    content = re.sub(r"^user\b", speaker_name, content)
    return content


def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)

    if not os.environ.get("AZURE_OPENAI_DEPLOYMENT") and os.environ.get("LLM_PROVIDER") == "azure":
        print("ERROR: AZURE_OPENAI_DEPLOYMENT required when LLM_PROVIDER=azure")
        sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────
def load_conversation(conv_id: str) -> dict:
    """Load locomo10.json and return the full entry matching conv_id."""
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    for entry in dataset:
        if entry["sample_id"] == conv_id:
            return entry

    print(f"ERROR: No conversation with sample_id={conv_id!r} found.")
    sys.exit(1)


def extract_sessions(conversation: dict) -> list[dict]:
    """Extract sessions from conversation, sorted by session number.

    Returns a list of dicts with keys: session_id, session_datetime, turns.
    """
    session_keys = sorted(
        (k for k in conversation if SESSION_KEY_RE.match(k)),
        key=lambda k: int(SESSION_KEY_RE.match(k).group(1)),  # type: ignore[union-attr]
    )

    sessions: list[dict] = []
    for session_key in session_keys:
        session_num = SESSION_KEY_RE.match(session_key).group(1)  # type: ignore[union-attr]
        session_id = f"session_{session_num}"
        datetime_key = f"{session_id}_date_time"
        session_datetime = conversation.get(datetime_key, "")

        sessions.append(
            {
                "session_id": session_id,
                "session_datetime": session_datetime,
                "turns": conversation[session_key],
            }
        )

    return sessions


def _build_llm_service(settings: Settings):
    """Build the LLM service based on the configured provider."""
    match settings.llm_provider:
        case "claude":
            return ClaudeService(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )
        case "gemini":
            return GeminiService(
                api_key=settings.google_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )
        case "azure":
            return AzureService(
                api_key=settings.azure_openai_api_key.get_secret_value(),
                endpoint=settings.azure_openai_endpoint,
                deployment=settings.azure_openai_deployment,
            )
        case "groq":
            return GroqService(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )


async def cleanup(store: QdrantMemoryStore, user_ids: list[str]) -> None:
    """Delete all memories for the given user IDs."""
    for user_id in user_ids:
        facts = await store.list_all(user_id)
        for fact in facts:
            await store.delete(fact.id, user_id)
        print(f"Deleted {len(facts)} memories for {user_id}")


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest a LoCoMo conversation through the production MemoryManager pipeline.",
    )
    parser.add_argument(
        "--conv-ids",
        nargs="+",
        default=["conv-26", "conv-30"],
        help="LoCoMo conversation sample_ids to ingest (default: conv-26 conv-30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be ingested without calling any LLMs",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all memories for the eval user ID and exit",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Ignore checkpoint file and re-ingest all conversations from scratch",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    check_env()

    # Wire services
    settings = Settings()
    store = QdrantMemoryStore.from_settings(settings)

    # Warm up collection before loop
    await store._ensure_collection()

    if args.cleanup:
        for conv_id in args.conv_ids:
            eval_user_id = f"locomo_eval_{conv_id.replace('-', '_')}"
            conv_entry = load_conversation(conv_id)
            conversation = conv_entry["conversation"]
            speaker_a: str = conversation["speaker_a"]
            speaker_b: str = conversation["speaker_b"]
            speaker_a_user_id = f"{eval_user_id}_{speaker_a.lower().replace(' ', '_')}"
            speaker_b_user_id = f"{eval_user_id}_{speaker_b.lower().replace(' ', '_')}"
            await cleanup(store, [eval_user_id, speaker_a_user_id, speaker_b_user_id])
        return

    embedding_service = AzureEmbeddingService(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        endpoint=settings.azure_openai_endpoint,
        deployment="text-embedding-3-small",
    )
    llm_service = _build_llm_service(settings)
    manager = MemoryManager(
        store=store,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_top_k=30,
    )

    checkpoint = load_checkpoint() if not args.no_checkpoint else {"completed": [], "last_updated": None}
    semaphore = asyncio.Semaphore(2)

    async def ingest_conversation(conv_id: str, index: int) -> tuple[int, int]:
        """Ingest a single conversation, returning (turns, facts)."""
        async with semaphore:
            if is_completed(checkpoint, conv_id):
                tqdm.write(f"\n── Skipping {conv_id} (already completed) ──────────────")
                return 0, 0

            eval_user_id = f"locomo_eval_{conv_id.replace('-', '_')}"
            conv_entry = load_conversation(conv_id)
            conversation = conv_entry["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{eval_user_id}_{speaker_a.lower().replace(' ', '_')}"
            speaker_b_user_id = f"{eval_user_id}_{speaker_b.lower().replace(' ', '_')}"

            # Silent cleanup before ingesting to remove stale data
            await cleanup(store, [eval_user_id, speaker_a_user_id, speaker_b_user_id])

            tqdm.write(f"\n── Ingesting {conv_id} ────────────────────────────────")

            sessions = extract_sessions(conversation)
            tqdm.write(f"Found {len(sessions)} sessions for {conv_id}")
            tqdm.write(f"Eval user ID: {eval_user_id}")
            tqdm.write(f"Speaker A: {speaker_a} → {speaker_a_user_id}")
            tqdm.write(f"Speaker B: {speaker_b} → {speaker_b_user_id}")

            if args.dry_run:
                tqdm.write("\n── Dry run ────────────────────────────────────────")

            conv_turns = 0
            conv_facts = 0

            for session in sessions:
                session_id = session["session_id"]
                session_datetime = session["session_datetime"]
                session_turns = session["turns"]
                session_facts = 0
                session_skipped = 0
                recent_messages_window: list[Message] = []
                conversation_summary: str | None = None
                summary_turn_counter: int = 0
                summary_context_lines: list[str] = []

                for i in tqdm(
                    range(len(session_turns)),
                    desc=f"{conv_id}/{session_id}",
                    unit="turn",
                    leave=True,
                    position=index,
                ):
                    # Route turn to the correct speaker's user ID
                    if session_turns[i]["speaker"] == speaker_a:
                        turn_user_id = speaker_a_user_id
                        speaker_name = speaker_a
                    else:
                        turn_user_id = speaker_b_user_id
                        speaker_name = speaker_b

                    if i == 0:
                        previous_user_id = turn_user_id
                    else:
                        prev_speaker = session_turns[i - 1]["speaker"]
                        previous_user_id = speaker_a_user_id if prev_speaker == speaker_a else speaker_b_user_id

                    if i == 0:
                        previous = Message(
                            user_id=previous_user_id,
                            session_id=session_id,
                            role="user",
                            content="[start of conversation]",
                        )
                    else:
                        previous = Message(
                            user_id=previous_user_id,
                            session_id=session_id,
                            role="user",
                            content=session_turns[i - 1]["text"],
                        )

                    current = Message(
                        user_id=turn_user_id,
                        session_id=session_id,
                        role="user",
                        content=f"[Session date: {session_datetime}] {session_turns[i]['text']}",
                    )

                    pair = ConversationPair(current=current, previous=previous)

                    recent_messages_window.append(current)
                    if len(recent_messages_window) > 10:
                        recent_messages_window = recent_messages_window[-10:]
                    summary_context_lines.append(
                        f"{speaker_name}: {session_turns[i]['text']}"
                    )
                    if len(summary_context_lines) > 30:
                        summary_context_lines = summary_context_lines[-30:]
                    summary_turn_counter += 1

                    if args.dry_run:
                        tqdm.write(f"  [{session_id}] turn {i}: {session_turns[i]['text'][:80]}...")
                        continue

                    skipped = False
                    if summary_turn_counter % 15 == 0:
                        _summary_system = (
                            "Summarize the key facts, topics, and events discussed "
                            "in this conversation so far in 3-5 sentences. Focus on "
                            "named entities, relationships, and concrete events. Be "
                            "specific — preserve names, places, dates, and numbers "
                            "exactly."
                        )
                        _context_text = "\n".join(summary_context_lines)
                        try:
                            conversation_summary = await llm_service.complete(
                                messages=[{"role": "user", "content": _context_text}],
                                system=_summary_system,
                            )
                        except Exception as e:
                            tqdm.write(
                                f"  [{session_id}] turn {i}: summary generation failed: {e}"
                            )
                    for attempt in range(1, _MAX_RETRIES + 1):
                        try:
                            facts = await manager.add_memory(
                                pair,
                                user_id=turn_user_id,
                                session_id=session_id,
                                conversation_summary=conversation_summary,
                                recent_messages=recent_messages_window,
                                current_speaker=speaker_name,
                            )
                            break
                        except _RETRYABLE_ERRORS as e:
                            # If it's a content filter error wrapped in LLMError, skip — don't retry
                            if "content_filter" in str(e) or "content management policy" in str(e):
                                tqdm.write(
                                    f"  [{session_id}] turn {i}: skipped (Azure content filter)"
                                )
                                session_skipped += 1
                                skipped = True
                                break
                            if attempt == _MAX_RETRIES:
                                raise
                            wait = _BACKOFF_SECONDS[attempt - 1]
                            tqdm.write(
                                f"  [{session_id}] turn {i}: {type(e).__name__}, "
                                f"retrying in {wait}s (attempt {attempt}/{_MAX_RETRIES})"
                            )
                            await asyncio.sleep(wait)
                        except Exception as e:
                            if "content_filter" in str(e) or "content management policy" in str(e):
                                tqdm.write(
                                    f"  [{session_id}] turn {i}: skipped (Azure content filter)"
                                )
                                session_skipped += 1
                                skipped = True
                                break
                            raise
                    if skipped:
                        continue

                    # Replace generic "User"/"user" references with actual speaker name
                    for fact in facts:
                        replaced = replace_user_with_speaker(fact.content, speaker_name)
                        if replaced != fact.content:
                            fact.content = replaced
                            await store.upsert(fact)

                    session_facts += len(facts)

                conv_turns += len(session_turns)
                conv_facts += session_facts
                tqdm.write(
                    f"  {session_id} ({len(session_turns)} turns): {session_facts} facts extracted"
                    + (f", {session_skipped} turns skipped (content filter)" if session_skipped else "")
                )

            # All sessions for this conv_id completed — mark checkpoint
            if not args.dry_run:
                mark_completed(checkpoint, conv_id)
                tqdm.write(f"  ✓ Checkpoint saved for {conv_id}")

            return conv_turns, conv_facts

    results = await asyncio.gather(
        *(ingest_conversation(conv_id, idx) for idx, conv_id in enumerate(args.conv_ids))
    )
    total_turns = sum(r[0] for r in results)
    total_facts = sum(r[1] for r in results)

    # Final summary
    print("\n── Summary ────────────────────────────────────────")
    print(f"Total: {total_turns} turns processed, {total_facts} facts extracted")


if __name__ == "__main__":
    asyncio.run(main())
