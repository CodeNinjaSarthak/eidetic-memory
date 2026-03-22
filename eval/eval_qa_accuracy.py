#!/usr/bin/env python3
# /// script
# dependencies = ["google-genai>=1.0.0", "openai>=2.29.0",
#                 "qdrant-client>=1.7.0", "python-dotenv>=1.0.0",
#                 "tqdm>=4.66.0", "pydantic-settings>=2.0.0",
#                 "pydantic>=2.0.0"]
# ///
"""
End-to-end QA accuracy evaluation on LoCoMo dataset.

Usage:
    uv run python eval/eval_qa_accuracy.py
    uv run python eval/eval_qa_accuracy.py --conv-id conv-26 --limit 10
    uv run python eval/eval_qa_accuracy.py --output eval/results/qa_accuracy_results.json

Reads .env.development for configuration.
Requires memories to be ingested first via ingest_locomo_production.py.
Retrieves memories, generates answers via Gemini, judges correctness via Azure OpenAI.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tqdm import tqdm

# ── sys.path setup for backend imports ───────────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "backend" / "packages" / "config" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "storage" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "llm" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "retrieval" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "memory" / "src"))

from config.settings import Settings  # noqa: E402
from llm.embeddings.gemini import GeminiEmbeddingService  # noqa: E402
from llm.generation.gemini import GeminiService  # noqa: E402
from retrieval.context import ContextBuilder  # noqa: E402
from retrieval.retriever import MemoryRetriever  # noqa: E402
from storage.qdrant import QdrantMemoryStore  # noqa: E402

# ── Configuration ────────────────────────────────────────────
env_path = _repo_root / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "QDRANT_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "EVAL_LLM_JUDGE_MODEL",
]

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
}

ANSWER_SYSTEM_PROMPT = """You are an intelligent memory assistant.
Answer the question using ONLY the provided memories.
Be concise — answer in 5 words or fewer if possible.
If the memories do not contain enough information to answer,
say "I don't know"."""

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


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end QA accuracy evaluation on LoCoMo dataset.",
    )
    parser.add_argument(
        "--conv-id",
        type=str,
        default="conv-30",
        help="LoCoMo conversation sample_id (default: conv-30)",
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
        default="eval/results/qa_accuracy_results.json",
        help="Path to save results JSON",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    eval_user_id = f"locomo_eval_{args.conv_id.replace('-', '_')}"
    output_path = Path(args.output)

    check_env()

    # Wire services
    settings = Settings()
    store = QdrantMemoryStore.from_settings(settings)

    embedding_service = GeminiEmbeddingService(
        api_key=settings.google_api_key.get_secret_value(),
        model=settings.embedding_model,
    )
    retriever = MemoryRetriever(
        store=store,
        embedding_service=embedding_service,
        top_k=10,
    )
    llm_service = GeminiService(
        api_key=settings.google_api_key.get_secret_value(),
        model=settings.memory_extraction_model,
    )
    azure_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )
    judge_model = os.environ["EVAL_LLM_JUDGE_MODEL"]

    # Load conversation and collect QA entries
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    conv_entry: dict | None = None
    for entry in dataset:
        if entry["sample_id"] == args.conv_id:
            conv_entry = entry
            break

    if conv_entry is None:
        print(f"ERROR: No conversation with sample_id={args.conv_id!r} found.")
        sys.exit(1)

    qa_entries = collect_qa_entries(conv_entry)
    print(f"Total QA entries (categories 1-4): {len(qa_entries)}")

    if args.limit is not None:
        qa_entries = qa_entries[: args.limit]
        print(f"Limited to {len(qa_entries)} QA pairs")

    # Warm up collection
    await store._ensure_collection()

    context_builder = ContextBuilder()

    # Evaluate
    per_pair_results: list[dict] = []

    for entry in tqdm(qa_entries, desc="Evaluating", unit="pair"):
        question = entry["question"]
        gold_answer = entry["answer"]
        category = entry["category"]

        # Retrieve memories
        memories = await retriever.retrieve(
            query=question,
            user_id=eval_user_id,
        )

        if not memories:
            per_pair_results.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "I don't know",
                    "category": category,
                    "memories_retrieved": [],
                    "label": "WRONG",
                }
            )
            continue

        # Generate answer
        system_prompt = context_builder.build_system_prompt(
            ANSWER_SYSTEM_PROMPT,
            memories,
        )
        generated_answer = await llm_service.complete(
            messages=[{"role": "user", "content": question}],
            system=system_prompt,
        )

        # Judge
        label = await judge_answer(
            azure_client,
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
                "memories_retrieved": [fact.content for fact in memories],
                "label": label,
            }
        )

    await azure_client.close()

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
    print(f"\nQA Accuracy Evaluation ({args.conv_id})")
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
            "conv_id": args.conv_id,
            "eval_user_id": eval_user_id,
            "limit": args.limit,
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
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
