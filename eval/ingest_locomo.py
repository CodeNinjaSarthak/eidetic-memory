#!/usr/bin/env python3
# /// script
# dependencies = ["openai>=2.29.0", "qdrant-client>=1.7.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Ingest LoCoMo dataset into Qdrant for evaluation.

Usage:
    uv run python eval/ingest_locomo.py

Reads .env.development for configuration.
Stores embeddings in Qdrant collection: locomo_eval
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────
COLLECTION_NAME = "locomo_eval"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50

# ── Configuration ────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]


def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────
SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def extract_turns(conversation: dict) -> list[dict]:
    """Extract all turns from all sessions in a conversation."""
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    turns: list[dict] = []

    session_keys = sorted(
        (k for k in conversation if SESSION_KEY_RE.match(k)),
        key=lambda k: int(SESSION_KEY_RE.match(k).group(1)),  # type: ignore[union-attr]
    )

    for session_key in session_keys:
        session_num = SESSION_KEY_RE.match(session_key).group(1)  # type: ignore[union-attr]
        session_id = f"session_{session_num}"
        datetime_key = f"{session_id}_date_time"
        session_datetime = conversation.get(datetime_key, "")

        for turn in conversation[session_key]:
            turns.append(
                {
                    "session_id": session_id,
                    "session_datetime": session_datetime,
                    "dia_id": turn["dia_id"],
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                }
            )

    return turns


def make_point_id(sample_id: str, dia_id: str) -> str:
    """Generate a deterministic UUID for a turn."""
    return str(uuid5(NAMESPACE_DNS, f"{sample_id}:{dia_id}"))


def embed_texts(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Azure OpenAI."""
    response = client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        input=texts,
    )
    return [item.embedding for item in response.data]


# ── Main ─────────────────────────────────────────────────────
def main() -> None:
    check_env()

    # Load dataset
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)
    print(f"Loaded {len(dataset)} conversations from {data_path.name}")

    # Connect to Qdrant
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Create collection if not exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        # Create payload indexes for efficient filtering
        for field in ("sample_id", "dia_id", "session_id"):
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print(f"Created collection '{COLLECTION_NAME}' with payload indexes")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    # Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )

    total_ingested = 0
    total_skipped = 0

    for conv_idx, conversation_data in enumerate(dataset):
        sample_id = conversation_data["sample_id"]
        turns = extract_turns(conversation_data["conversation"])

        # Compute point IDs for all turns
        point_ids = [make_point_id(sample_id, t["dia_id"]) for t in turns]

        # Check which already exist in Qdrant
        existing = set()
        try:
            retrieved = qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing = {str(p.id) for p in retrieved}
        except Exception as e:
            tqdm.write(f"  Warning: could not check existing points: {e}")

        # Filter to new turns only
        new_turns = [(pid, turn) for pid, turn in zip(point_ids, turns) if pid not in existing]

        n_skip = len(turns) - len(new_turns)
        n_new = len(new_turns)
        total_skipped += n_skip

        if not new_turns:
            print(f"Conversation {conv_idx + 1}/10 ({sample_id}): " f"0 new turns, {n_skip} skipped")
            continue

        # Embed and upsert in batches
        ingested_this_conv = 0
        for batch_start in tqdm(
            range(0, len(new_turns), BATCH_SIZE),
            desc=f"Conv {conv_idx + 1}/10 ({sample_id})",
            unit="batch",
            leave=True,
        ):
            batch = new_turns[batch_start : batch_start + BATCH_SIZE]
            batch_ids = [pid for pid, _ in batch]
            batch_turns = [turn for _, turn in batch]
            batch_texts = [turn["text"] for turn in batch_turns]

            vectors: list[list[float]] | None = None
            for attempt in range(3):
                try:
                    vectors = embed_texts(openai_client, batch_texts)
                    break
                except Exception as e:
                    if attempt == 2:
                        tqdm.write(f"  ERROR embedding batch after 3 attempts " f"for {sample_id}: {e}")
                        vectors = None
                        break
                    wait = 2**attempt
                    tqdm.write(f"  Retrying batch in {wait}s " f"(attempt {attempt + 1}/3): {e}")
                    time.sleep(wait)

            if vectors is None:
                continue

            points = [
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "sample_id": sample_id,
                        "conversation_idx": conv_idx,
                        **turn,
                    },
                )
                for pid, vec, turn in zip(batch_ids, vectors, batch_turns)
            ]

            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                ingested_this_conv += len(points)
                time.sleep(1)  # avoid overwhelming the connection
            except Exception as e:
                tqdm.write(f"  ERROR upserting batch {batch_start // BATCH_SIZE + 1} " f"for {sample_id}: {e}")
                continue

        total_ingested += ingested_this_conv
        print(f"Conversation {conv_idx + 1}/10 ({sample_id}): " f"{ingested_this_conv} new turns, {n_skip} skipped")

    # Final summary
    print("\n── Summary ────────────────────────────────────────")
    print(f"Total ingested: {total_ingested}")
    print(f"Total skipped:  {total_skipped}")
    print(f"Collection:     {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
