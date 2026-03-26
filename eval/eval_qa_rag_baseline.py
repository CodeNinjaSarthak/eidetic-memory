#!/usr/bin/env python3
# /// script
# dependencies = ["google-genai>=1.0.0", "openai>=2.29.0",
#                 "qdrant-client>=1.7.0", "python-dotenv>=1.0.0",
#                 "tqdm>=4.66.0", "pydantic-settings>=2.0.0",
#                 "pydantic>=2.0.0"]
# ///
"""
RAG baseline QA accuracy evaluation on LoCoMo dataset.

Instead of retrieving from per-speaker memory namespaces (the memory pipeline),
this script searches raw conversation turns directly from the locomo_eval
Qdrant collection. This answers: "what accuracy do you get with zero memory
pipeline — just raw vector search on conversation turns?"

Usage:
    uv run python eval/eval_qa_rag_baseline.py
    uv run python eval/eval_qa_rag_baseline.py --conv-ids conv-26 conv-30 --limit 10
    uv run python eval/eval_qa_rag_baseline.py --output eval/results/qa_rag_baseline_results.json

Reads .env.development for configuration.
Requires conversation turns to be ingested into locomo_eval via ingest_locomo.py.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent
env_path = _repo_root / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT",
    "EVAL_LLM_JUDGE_MODEL",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]

COLLECTION_NAME = "locomo_eval"
TOP_K = 20

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
}

ANSWER_SYSTEM_PROMPT = """You are an intelligent memory assistant
tasked with retrieving accurate information from conversation memories.

Instructions:
1. Carefully analyze all provided memories
2. Pay special attention to any timestamps or dates in the memories
3. If the question asks about a specific event or fact, look for
   direct evidence in the memories
4. If memories contain contradictory information, prioritize the
   most recent memory
5. If there is a question about time references (like "last year",
   "two months ago", etc.), calculate the actual date based on
   the memory timestamp
6. Always convert relative time references to specific dates,
   months, or years based on the memory content
7. The answer should be less than 5-6 words

Answer the question using ONLY the provided memories.
If the memories do not contain enough information, say "I don't know"."""

JUDGE_PROMPT = """Your task is to label an answer as CORRECT or WRONG.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Be generous: if the generated answer refers to the same fact or time \
period as the gold answer, label it CORRECT even if phrased differently.
For dates: "May 2023" and "May 7, 2023" are both CORRECT if gold is \
"7 May 2023".

Return JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────
def collect_qa_entries(conversation_data: dict) -> list[dict]:
    """Collect QA entries from a single conversation, categories 1-4 only.

    Skips category 5 (adversarial) and entries with missing/empty answers.
    """
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


def embed_question(client: AzureOpenAI, text: str) -> list[float] | None:
    """Embed a single question with 3-attempt exponential backoff."""
    import time

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


def retrieve_raw_turns(
    qdrant: QdrantClient,
    openai_client: AzureOpenAI,
    question: str,
    conv_id: str,
) -> list[str]:
    """Embed question and search locomo_eval collection filtered by conv_id.

    Returns up to TOP_K conversation turn texts.
    """
    vector = embed_question(openai_client, question)
    if vector is None:
        return []

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="sample_id",
                    match=MatchValue(value=conv_id),
                )
            ]
        ),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    return [p.payload["text"] for p in response.points]


async def judge_answer(
    client: AsyncAzureOpenAI,
    model: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> str:
    """Judge whether a generated answer is correct using Azure OpenAI.

    Returns "CORRECT" or "WRONG".
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result.get("label", "WRONG")
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  ERROR judging answer after 3 attempts: {e}")
                return "WRONG"
            wait = 2**attempt
            await asyncio.sleep(wait)

    return "WRONG"


def build_context_from_turns(turns: list[str]) -> str:
    """Format retrieved conversation turns into a context block for the LLM."""
    if not turns:
        return ""
    lines = [f"- {turn}" for turn in turns]
    return "## Retrieved conversation turns\n" + "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG baseline QA accuracy evaluation on LoCoMo dataset.",
    )
    parser.add_argument(
        "--conv-ids",
        nargs="+",
        default=["conv-26", "conv-30"],
        help="LoCoMo conversation sample_ids to evaluate (default: conv-26 conv-30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/qa_rag_baseline_results.json",
        help="Path to save results JSON",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    check_env()

    # Wire clients
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    openai_sync = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )
    azure_async = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )
    judge_model = os.environ["EVAL_LLM_JUDGE_MODEL"]
    answer_model = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    # Load dataset
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    # Collect QA entries from all specified conversations
    qa_entries: list[dict] = []
    for conv_id in args.conv_ids:
        conv_entry: dict | None = None
        for entry in dataset:
            if entry["sample_id"] == conv_id:
                conv_entry = entry
                break

        if conv_entry is None:
            print(f"ERROR: No conversation with sample_id={conv_id!r} found.")
            sys.exit(1)

        entries = collect_qa_entries(conv_entry)
        for e in entries:
            e["conv_id"] = conv_id
        qa_entries.extend(entries)

    print(f"Total QA entries (categories 1-4): {len(qa_entries)}")

    if args.limit is not None:
        qa_entries = qa_entries[: args.limit]
        print(f"Limited to {len(qa_entries)} QA pairs")

    # Resume support: load partial results if they exist
    partial_path = Path(f"{output_path}.partial.json")
    per_pair_results: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()

    if partial_path.exists():
        with open(partial_path) as f:
            per_pair_results = json.load(f)
        completed_keys = {(r["question"], r["conv_id"]) for r in per_pair_results}
        print(f"Resuming: loaded {len(per_pair_results)} completed pairs from {partial_path}")

    for entry in tqdm(qa_entries, desc="Evaluating", unit="pair"):
        question = entry["question"]
        gold_answer = entry["answer"]
        category = entry["category"]
        conv_id = entry["conv_id"]

        # Skip already-completed pairs (resume support)
        if (question, conv_id) in completed_keys:
            continue

        # Retrieve raw conversation turns from locomo_eval
        turns = retrieve_raw_turns(qdrant, openai_sync, question, conv_id)

        if not turns:
            per_pair_results.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "I don't know",
                    "category": category,
                    "conv_id": conv_id,
                    "turns_retrieved": [],
                    "label": "WRONG",
                }
            )
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with open(partial_path, "w") as pf:
                json.dump(per_pair_results, pf, indent=2)
            continue

        # Generate answer using retrieved turns as context
        context = build_context_from_turns(turns)
        system_prompt = f"{ANSWER_SYSTEM_PROMPT}\n\n{context}"

        try:
            response = await azure_async.chat.completions.create(
                model=answer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0,
            )
            generated_answer = response.choices[0].message.content or "I don't know"
        except Exception as e:
            if "content_filter" in str(e) or "content management policy" in str(e):
                tqdm.write(f"  Skipped (content filter): {question[:60]}")
                generated_answer = "I don't know"
            else:
                raise

        # Judge
        label = await judge_answer(
            azure_async,
            judge_model,
            question,
            gold_answer,
            generated_answer,
        )

        per_pair_results.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "category": category,
                "conv_id": conv_id,
                "turns_retrieved": turns,
                "label": label,
            }
        )

        # Save partial results incrementally
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w") as pf:
            json.dump(per_pair_results, pf, indent=2)

    await azure_async.close()

    # Compute metrics
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

    # Print summary
    print(f"\nRAG Baseline QA Accuracy ({', '.join(args.conv_ids)})")
    print("\u2500" * 34)
    print(f"QA pairs evaluated: {n_evaluated}")
    print(f"Overall accuracy: {overall_accuracy:.1%} ({n_correct}/{n_evaluated} CORRECT)")
    print("\nBy category:")
    for cat_id, cat_label in CATEGORIES.items():
        cat_key = str(cat_id)
        if cat_key in by_category:
            cat = by_category[cat_key]
            print(f"  {cat_label:12s}: {cat['accuracy']:.1%} ({cat['total']} pairs)")

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "conv_ids": args.conv_ids,
            "limit": args.limit,
            "retrieval_source": "locomo_eval (raw conversation turns)",
            "top_k": TOP_K,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        "overall": {
            "accuracy": overall_accuracy,
            "correct": n_correct,
            "total": n_evaluated,
        },
        "by_category": by_category,
        "per_pair": per_pair_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up partial file after successful completion
    if partial_path.exists():
        partial_path.unlink()

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
