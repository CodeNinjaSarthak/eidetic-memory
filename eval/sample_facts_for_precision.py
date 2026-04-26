"""Sample 100 facts from Qdrant and link each to its source conversation session.

Outputs a CSV with columns:
  fact_text, conversation_id, speaker, session_id, session_datetime,
  source_turns, is_relevant

The is_relevant column is empty — fill in 1 (correct) or 0 (wrong/hallucinated)
during manual review.

Sampling strategy: stratified over user_ids so every speaker-namespace
contributes roughly equally (total target = 100 facts).

Source turns: full dialogue of the session the fact was extracted from,
formatted as "Speaker: text" lines, so reviewers can judge extraction quality
in context.
"""

from __future__ import annotations

import asyncio
import csv
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend" / "packages" / "config" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "backend" / "services" / "storage" / "src"))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from config.settings import Settings  # noqa: E402

load_dotenv(_REPO_ROOT / ".env.development")

FACTS_COLLECTION   = "eidetic_memories"
TURNS_COLLECTION   = "locomo_eval"
OUTPUT_PATH        = Path(__file__).parent / "results" / "precision_sample.csv"
TARGET_TOTAL       = 100
SEED               = 42

# user_id format: locomo_eval_conv_{conv_num}_{speaker_lower}
_USER_ID_RE = re.compile(
    r"^locomo_eval_(conv[-_]\d+)_(.+)$",
    re.IGNORECASE,
)


def parse_user_id(user_id: str) -> tuple[str, str]:
    """Return (conv_id, speaker_slug) from a user_id string.

    e.g. 'locomo_eval_conv_26_caroline' → ('conv-26', 'caroline')
    """
    m = _USER_ID_RE.match(user_id)
    if not m:
        return user_id, "unknown"
    conv_raw, speaker_slug = m.group(1), m.group(2)
    # Normalise underscore separator to hyphen for conv ids
    conv_id = conv_raw.replace("_", "-")
    return conv_id, speaker_slug


async def scroll_all_user_ids(client: AsyncQdrantClient) -> list[str]:
    """Return every unique user_id present in the facts collection."""
    user_ids: set[str] = set()
    offset = None
    while True:
        results, offset = await client.scroll(
            collection_name=FACTS_COLLECTION,
            with_payload=["user_id"],
            with_vectors=False,
            limit=250,
            offset=offset,
        )
        for p in results:
            uid = p.payload.get("user_id", "")
            if uid:
                user_ids.add(uid)
        if offset is None:
            break
    return sorted(user_ids)


async def scroll_facts_for_user(
    client: AsyncQdrantClient,
    user_id: str,
) -> list[dict]:
    """Return all facts (id, user_id, session_id, content) for one user_id."""
    facts: list[dict] = []
    offset = None
    while True:
        results, offset = await client.scroll(
            collection_name=FACTS_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            with_payload=["id", "user_id", "session_id", "content"],
            with_vectors=False,
            limit=250,
            offset=offset,
        )
        for p in results:
            facts.append(p.payload)
        if offset is None:
            break
    return facts


async def fetch_session_turns(
    client: AsyncQdrantClient,
    sample_id: str,
    session_id: str,
) -> list[dict]:
    """Return all conversation turns for a given (sample_id, session_id), ordered by dia_id."""
    turns: list[dict] = []
    offset = None
    while True:
        results, offset = await client.scroll(
            collection_name=TURNS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="sample_id",  match=MatchValue(value=sample_id)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=250,
            offset=offset,
        )
        for p in results:
            turns.append(p.payload)
        if offset is None:
            break
    # Sort by dia_id which encodes turn order (e.g. "D4:11" → session 4, turn 11)
    def _dia_sort_key(t: dict) -> tuple[int, int]:
        raw = t.get("dia_id", "D0:0")
        parts = re.findall(r"\d+", raw)
        return (int(parts[0]) if parts else 0, int(parts[1]) if len(parts) > 1 else 0)

    turns.sort(key=_dia_sort_key)
    return turns


def format_turns(turns: list[dict]) -> tuple[str, str]:
    """Return (session_datetime, formatted_dialogue_string) for a list of turns."""
    if not turns:
        return "", ""
    session_dt = turns[0].get("session_datetime", "")
    lines = [f"{t['speaker']}: {t['text']}" for t in turns if t.get("text")]
    return session_dt, " | ".join(lines)


async def main() -> None:
    settings = Settings()
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None,
    )

    rng = random.Random(SEED)

    print("Fetching all user_ids from facts collection …")
    user_ids = await scroll_all_user_ids(client)
    print(f"  Found {len(user_ids)} namespaces")

    # Per-namespace quota: divide evenly, give remainder to first namespaces
    quota_base  = TARGET_TOTAL // len(user_ids)
    quota_extra = TARGET_TOTAL  % len(user_ids)

    sampled_facts: list[dict] = []

    for i, uid in enumerate(user_ids):
        quota = quota_base + (1 if i < quota_extra else 0)
        facts = await scroll_facts_for_user(client, uid)
        chosen = rng.sample(facts, min(quota, len(facts)))
        sampled_facts.extend(chosen)
        print(f"  {uid}: {len(facts)} facts → sampled {len(chosen)}")

    print(f"\nTotal sampled: {len(sampled_facts)} facts")
    print("Fetching source session turns …")

    # Cache turns per (sample_id, session_id) to avoid redundant requests
    turns_cache: dict[tuple[str, str], list[dict]] = {}
    rows: list[dict] = []

    for fact in sampled_facts:
        uid        = fact.get("user_id", "")
        session_id = fact.get("session_id", "")
        content    = fact.get("content", "")

        conv_id, speaker_slug = parse_user_id(uid)

        cache_key = (conv_id, session_id)
        if cache_key not in turns_cache:
            turns_cache[cache_key] = await fetch_session_turns(client, conv_id, session_id)

        turns = turns_cache[cache_key]
        session_dt, dialogue = format_turns(turns)

        # Resolve the display speaker name from the turns (handles casing)
        speaker_display = speaker_slug
        for t in turns:
            if t.get("speaker", "").lower() == speaker_slug.lower():
                speaker_display = t["speaker"]
                break

        rows.append({
            "fact_text":       content,
            "conversation_id": conv_id,
            "speaker":         speaker_display,
            "session_id":      session_id,
            "session_datetime": session_dt,
            "source_turns":    dialogue,
            "is_relevant":     "",
        })

    await client.close()

    # Write CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fact_text", "conversation_id", "speaker",
        "session_id", "session_datetime", "source_turns", "is_relevant",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {OUTPUT_PATH}")
    print()

    # Summary
    from collections import Counter
    by_conv = Counter(r["conversation_id"] for r in rows)
    print("Rows per conversation:")
    for conv_id, count in sorted(by_conv.items()):
        print(f"  {conv_id}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
