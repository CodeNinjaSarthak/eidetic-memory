#!/usr/bin/env python3
# /// script
# dependencies = ["openai>=2.29.0", "qdrant-client>=1.7.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Retrieval accuracy evaluation on LoCoMo dataset.

Usage:
    uv run python eval/eval_retrieval.py

Reads .env.development for configuration.
Searches locomo_eval Qdrant collection.
Reports Hit@1, Hit@3, Hit@5, Hit@10 by category.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────
COLLECTION_NAME = "locomo_eval"
RANDOM_SEED = 42
K_VALUES = [1, 3, 5, 10]
CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
    5: "Adversarial",
}

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
def embed_text_with_retry(client: AzureOpenAI, text: str) -> list[float] | None:
    """Embed a single text with 3-attempt exponential backoff."""
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                input=[text],
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  ERROR embedding after 3 attempts: {e}")
                return None
            wait = 2**attempt
            tqdm.write(f"  Retrying embed in {wait}s (attempt {attempt + 1}/3): {e}")
            time.sleep(wait)
    return None


def collect_qa_entries(dataset: list[dict]) -> list[dict]:
    """Collect all QA entries with non-empty evidence across all conversations."""
    entries: list[dict] = []
    for conversation_data in dataset:
        sample_id = conversation_data["sample_id"]
        for qa in conversation_data.get("qa", []):
            if qa.get("category") == 5:
                continue
            evidence = qa.get("evidence", [])
            if not evidence:
                continue
            entries.append(
                {
                    "sample_id": sample_id,
                    "question": qa["question"],
                    "answer": qa.get("answer", ""),
                    "evidence": evidence,
                    "category": qa["category"],
                }
            )
    return entries


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieval accuracy evaluation on LoCoMo dataset.")
    parser.add_argument("--limit", type=int, default=10, help="Number of QA pairs to evaluate (default: 10)")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth (default: 10)")
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/retrieval_results.json",
        help="Path to save results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = args.limit
    top_k = args.top_k
    output_path = Path(args.output)

    check_env()

    # Load dataset
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    # Collect and sample QA entries
    all_entries = collect_qa_entries(dataset)
    print(f"Total QA entries with evidence: {len(all_entries)}")

    random.seed(RANDOM_SEED)
    sampled = random.sample(all_entries, min(sample_size, len(all_entries)))
    print(f"Sampled {len(sampled)} questions (seed={RANDOM_SEED})")

    # Unique conversations in sample
    conv_ids = {e["sample_id"] for e in sampled}
    print(f"Conversations sampled from: {len(conv_ids)}")

    # Connect to Qdrant
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )

    # Evaluate
    results: list[dict] = []
    skipped = 0

    for entry in tqdm(sampled, desc="Evaluating", unit="question"):
        # Embed question
        vector = embed_text_with_retry(openai_client, entry["question"])
        if vector is None:
            skipped += 1
            continue

        # Search Qdrant filtered by conversation
        try:
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="sample_id",
                            match=MatchValue(value=entry["sample_id"]),
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            tqdm.write(f"  ERROR searching Qdrant: {e}")
            skipped += 1
            continue

        # Extract retrieved dia_ids in ranked order
        retrieved_dia_ids = [p.payload["dia_id"] for p in response.points]
        evidence_set = set(entry["evidence"])

        # Compute hit@k
        hits = {}
        for k in K_VALUES:
            top_k_ids = set(retrieved_dia_ids[:k])
            hits[f"hit_at_{k}"] = bool(top_k_ids & evidence_set)

        results.append(
            {
                "question": entry["question"],
                "sample_id": entry["sample_id"],
                "category": entry["category"],
                "evidence": entry["evidence"],
                "retrieved_dia_ids": retrieved_dia_ids,
                **hits,
            }
        )

    # ── Aggregate metrics ────────────────────────────────────
    n_evaluated = len(results)
    if n_evaluated == 0:
        print("No questions were successfully evaluated.")
        return

    overall = {}
    for k in K_VALUES:
        key = f"hit_at_{k}"
        overall[key] = sum(1 for r in results if r[key]) / n_evaluated

    # Hit@5 by category
    by_category: dict[int, dict] = {}
    for cat_id, cat_label in CATEGORIES.items():
        cat_results = [r for r in results if r["category"] == cat_id]
        if not cat_results:
            continue
        hit5 = sum(1 for r in cat_results if r["hit_at_5"]) / len(cat_results)
        by_category[cat_id] = {
            "label": cat_label,
            "hit_at_5": hit5,
            "count": len(cat_results),
        }

    # ── Print results ────────────────────────────────────────
    print("\nRetrieval Accuracy Evaluation")
    print("──────────────────────────────")
    print(f"Questions evaluated: {n_evaluated}")
    if skipped:
        print(f"Questions skipped:   {skipped}")
    print(f"Conversations sampled from: {len(conv_ids)}")

    print("\nOverall Results:")
    for k in K_VALUES:
        key = f"hit_at_{k}"
        print(f"  Hit@{k}:{' ' * (2 - len(str(k)))} {overall[key]:.1%}")

    print("\nResults by Category (Hit@5):")
    for cat_id in sorted(by_category):
        cat = by_category[cat_id]
        label = f"{cat['label']}:"
        print(f"  {label:<14} {cat['hit_at_5']:.1%} ({cat['count']} questions)")

    # ── Save results ─────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "sample_size": n_evaluated,
            "top_k": top_k,
            "seed": RANDOM_SEED,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        "overall": overall,
        "by_category": {str(k): v for k, v in by_category.items()},
        "per_question": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
