"""Sample 100 facts from the v2 Qdrant collection with stratified sampling.

Stratification:
  - 10 conversations × 10 facts each  = 100 total
  - ~50/50 speakers per conversation  = 5 facts per speaker

For each fact, we resolve the single most-relevant source turn by Jaccard
word overlap with the fact text so human labelers have a concrete utterance to
check against.  Source turns are loaded from the local locomo10.json file
(same data used during ingestion) so no Qdrant turns collection is required.

Output: eval/results/v2_precision_sample.csv

Columns:
  fact_id           — Qdrant point UUID of the extracted fact
  conv_id           — conversation ID (e.g. conv-26)
  speaker           — display name of the speaker the fact belongs to
  fact_text         — the extracted fact
  source_turn_id    — dia_id of the best-matching source turn
  source_utterance  — verbatim text of that turn
  label_correct     — empty, for human labeling (1 = correct, 0 = wrong)
  label_attribution — empty, for human labeling (1 = right speaker, 0 = wrong)
  notes             — empty, for annotator notes
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend" / "packages" / "config" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "backend" / "services" / "storage" / "src"))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from config.settings import Settings  # noqa: E402

load_dotenv(_REPO_ROOT / ".env.development")

FACTS_COLLECTION = "eidetic_memories"
LOCOMO_DATA_PATH = Path(__file__).parent / "data" / "locomo10.json"
OUTPUT_PATH = Path(__file__).parent / "results" / "v2_precision_sample.csv"
TARGET_PER_CONV = 10       # facts per conversation
SEED = 42

_USER_ID_RE = re.compile(r"^locomo_eval_(conv[-_]\d+)_(.+)$", re.IGNORECASE)

FIELDNAMES = [
    "fact_id",
    "conv_id",
    "speaker",
    "fact_text",
    "source_turn_id",
    "source_utterance",
    "label_correct",
    "label_attribution",
    "notes",
]


def parse_user_id(user_id: str) -> tuple[str, str]:
    """Return (conv_id, speaker_slug) from a user_id string.

    e.g. 'locomo_eval_conv_26_caroline' → ('conv-26', 'caroline')
    """
    m = _USER_ID_RE.match(user_id)
    if not m:
        return user_id, "unknown"
    conv_raw, speaker_slug = m.group(1), m.group(2)
    return conv_raw.replace("_", "-"), speaker_slug


def _dia_sort_key(turn: dict) -> tuple[int, int]:
    parts = re.findall(r"\d+", turn.get("dia_id", "D0:0"))
    return (int(parts[0]) if parts else 0, int(parts[1]) if len(parts) > 1 else 0)


def _jaccard(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def find_best_turn(fact_content: str, turns: list[dict]) -> dict | None:
    """Return the turn with the highest Jaccard word overlap against fact_content."""
    if not turns:
        return None
    best = max(turns, key=lambda t: _jaccard(fact_content, t.get("text", "")))
    return best


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


async def scroll_facts_for_user(client: AsyncQdrantClient, user_id: str) -> list[dict]:
    """Return all facts for one user_id (payload + Qdrant point id)."""
    facts: list[dict] = []
    offset = None
    while True:
        results, offset = await client.scroll(
            collection_name=FACTS_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=250,
            offset=offset,
        )
        for p in results:
            fact = dict(p.payload)
            # payload already contains 'id' via to_qdrant_payload(), but we
            # capture the Qdrant point id directly as a safety net
            if "id" not in fact:
                fact["id"] = str(p.id)
            facts.append(fact)
        if offset is None:
            break
    return facts


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def load_turns_index(data_path: Path) -> dict[str, dict[str, list[dict]]]:
    """Build {conv_id: {session_id: [turns]}} from locomo10.json.

    Each turn dict has keys: dia_id, speaker, text.
    """
    with open(data_path, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    index: dict[str, dict[str, list[dict]]] = {}
    for entry in dataset:
        conv_id: str = entry["sample_id"]
        conv: dict = entry["conversation"]
        sessions: dict[str, list[dict]] = {}
        for key, value in conv.items():
            m = _SESSION_KEY_RE.match(key)
            if not m:
                continue
            session_id = f"session_{m.group(1)}"
            turns = [
                {"dia_id": t["dia_id"], "speaker": t["speaker"], "text": t["text"]}
                for t in value
            ]
            turns.sort(key=_dia_sort_key)
            sessions[session_id] = turns
        index[conv_id] = sessions
    return index


def get_session_turns(
    turns_index: dict[str, dict[str, list[dict]]],
    conv_id: str,
    session_id: str,
) -> list[dict]:
    """Return turns for a given (conv_id, session_id) from the local index."""
    return turns_index.get(conv_id, {}).get(session_id, [])


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
    all_user_ids = await scroll_all_user_ids(client)
    print(f"  Found {len(all_user_ids)} namespaces")

    # Group user_ids by conv_id so we can balance speakers per conversation
    conv_to_uids: dict[str, list[str]] = {}
    for uid in all_user_ids:
        conv_id, _ = parse_user_id(uid)
        conv_to_uids.setdefault(conv_id, []).append(uid)

    print(f"  Conversations: {sorted(conv_to_uids)}")

    sampled_facts: list[dict] = []

    for conv_id in sorted(conv_to_uids):
        speaker_uids = sorted(conv_to_uids[conv_id])
        n_speakers = len(speaker_uids)
        per_speaker = TARGET_PER_CONV // n_speakers
        remainder = TARGET_PER_CONV % n_speakers

        print(f"\n  {conv_id} ({n_speakers} speakers):")
        for i, uid in enumerate(speaker_uids):
            quota = per_speaker + (1 if i < remainder else 0)
            facts = await scroll_facts_for_user(client, uid)
            chosen = rng.sample(facts, min(quota, len(facts)))
            sampled_facts.extend(chosen)
            _, slug = parse_user_id(uid)
            print(f"    {slug}: {len(facts)} facts → sampled {len(chosen)}")

    print(f"\nTotal sampled: {len(sampled_facts)} facts")
    print("Loading locomo turns from local JSON …")
    turns_index = load_turns_index(LOCOMO_DATA_PATH)
    print(f"  Loaded turns for {len(turns_index)} conversations")
    print("Resolving source turns …")

    rows: list[dict] = []

    for fact in sampled_facts:
        uid = fact.get("user_id", "")
        session_id = fact.get("session_id", "") or ""
        content = fact.get("content", "")
        fact_id = fact.get("id", "")

        conv_id, speaker_slug = parse_user_id(uid)

        turns = get_session_turns(turns_index, conv_id, session_id)

        # Resolve display speaker name from actual turn data (handles casing)
        speaker_display = speaker_slug
        for t in turns:
            if t.get("speaker", "").lower() == speaker_slug.lower():
                speaker_display = t["speaker"]
                break

        best = find_best_turn(content, turns)
        source_turn_id = best["dia_id"] if best else session_id
        source_utterance = best.get("text", "") if best else ""

        rows.append({
            "fact_id": fact_id,
            "conv_id": conv_id,
            "speaker": speaker_display,
            "fact_text": content,
            "source_turn_id": source_turn_id,
            "source_utterance": source_utterance,
            "label_correct": "",
            "label_attribution": "",
            "notes": "",
        })

    await client.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {OUTPUT_PATH}")

    by_conv = Counter(r["conv_id"] for r in rows)
    print("\nRows per conversation:")
    for cid, cnt in sorted(by_conv.items()):
        # also show speaker breakdown
        speakers = Counter(r["speaker"] for r in rows if r["conv_id"] == cid)
        breakdown = ", ".join(f"{sp}={c}" for sp, c in sorted(speakers.items()))
        print(f"  {cid}: {cnt}  ({breakdown})")

    print("\nFirst 5 rows:")
    for r in rows[:5]:
        fid = r["fact_id"][:8] if r["fact_id"] else "?"
        ft = r["fact_text"][:55]
        su = r["source_utterance"][:45]
        print(f"  {fid}… | {r['conv_id']} | {r['speaker']} | {ft}… | {r['source_turn_id']} | {su}…")


if __name__ == "__main__":
    asyncio.run(main())
