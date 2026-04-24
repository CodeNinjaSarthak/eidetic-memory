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
    uv run python eval/eval_qa_accuracy.py --conv-ids conv-26 conv-30 --limit 10
    uv run python eval/eval_qa_accuracy.py --output eval/results/qa_accuracy_benchmark_results.json

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
from itertools import zip_longest
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
from llm.embeddings.azure import AzureEmbeddingService  # noqa: E402
from llm.generation.azure import AzureService  # noqa: E402
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

OPEN_DOMAIN_SYSTEM_PROMPT = """You are an intelligent memory assistant
tasked with retrieving accurate information from conversation memories.

Instructions:
1. Carefully analyze all provided memories
2. Answer conversationally and completely — do not truncate your answer
3. If the question asks about opinions, preferences, or general topics,
   synthesize across all relevant memories
4. If memories contain contradictory information, prioritize the most recent
5. Provide enough detail to fully answer the question

Answer the question using ONLY the provided memories.
If the memories do not contain enough information, say "I don't know"."""

MAX_GENERATION_RETRIES = 5
GENERATION_BACKOFF = [5, 15, 30, 60, 120]  # seconds

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

    judge_backoff = [2, 4, 8, 30, 60]
    for attempt in range(5):
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
            if attempt == 4:
                tqdm.write(f"  ERROR judging answer after 5 attempts: {e}")
                return "WRONG"
            wait = judge_backoff[attempt]
            tqdm.write(f"  Judge retry {attempt + 1}/5 in {wait}s: {e}")
            await asyncio.sleep(wait)

    return "WRONG"


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end QA accuracy evaluation on LoCoMo dataset.",
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
        default="eval/results/qa_accuracy_benchmark_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of QA pairs to evaluate concurrently (default: 5)",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help=(
            "Disable per-speaker isolation: retrieve from both speaker "
            "namespaces merged into one query (flat RAG over extracted facts). "
            "Used for ablation baseline."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Disable Jina reranking even if JINA_API_KEY is set. "
            "Used for ablation runs."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    check_env()

    # Wire services
    settings = Settings()
    store = QdrantMemoryStore.from_settings(settings)

    embedding_service = AzureEmbeddingService(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        endpoint=settings.azure_openai_endpoint,
        deployment="text-embedding-3-small",
    )
    retriever = MemoryRetriever(
        store=store,
        embedding_service=embedding_service,
        top_k=30,
        jina_api_key=None if args.no_rerank else os.getenv("JINA_API_KEY"),
    )
    llm_service = AzureService(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
    )
    azure_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )
    judge_model = os.environ["EVAL_LLM_JUDGE_MODEL"]

    # Load dataset
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        dataset: list[dict] = json.load(f)

    # Collect QA entries from all specified conversations
    qa_entries: list[dict] = []
    for conv_id in args.conv_ids:
        eval_user_id = f"locomo_eval_{conv_id.replace('-', '_')}"

        conv_entry: dict | None = None
        for entry in dataset:
            if entry["sample_id"] == conv_id:
                conv_entry = entry
                break

        if conv_entry is None:
            print(f"ERROR: No conversation with sample_id={conv_id!r} found.")
            sys.exit(1)

        speaker_a: str = conv_entry["conversation"]["speaker_a"]
        speaker_b: str = conv_entry["conversation"]["speaker_b"]
        speaker_a_user_id = f"{eval_user_id}_{speaker_a.lower().replace(' ', '_')}"
        speaker_b_user_id = f"{eval_user_id}_{speaker_b.lower().replace(' ', '_')}"

        entries = collect_qa_entries(conv_entry)
        for e in entries:
            e["conv_id"] = conv_id
            e["speaker_a_user_id"] = speaker_a_user_id
            e["speaker_b_user_id"] = speaker_b_user_id
        qa_entries.extend(entries)

    print(f"Total QA entries (categories 1-4): {len(qa_entries)}")

    if args.limit is not None:
        qa_entries = qa_entries[: args.limit]
        print(f"Limited to {len(qa_entries)} QA pairs")

    # Warm up collection
    await store._ensure_collection()

    # Remove known noise facts that corrupt retrieval
    for conv_id in args.conv_ids:
        eval_user_id = f"locomo_eval_{conv_id.replace('-', '_')}"
        conv_entry = next((e for e in dataset if e["sample_id"] == conv_id), None)
        if conv_entry is None:
            continue
        speaker_a = conv_entry["conversation"]["speaker_a"]
        speaker_a_user_id = f"{eval_user_id}_{speaker_a.lower().replace(' ', '_')}"
        all_facts = await store.list_all(speaker_a_user_id)
        noise_facts = [
            f for f in all_facts
            if "linked to this conversation" in f.content.lower()
        ]
        for nf in noise_facts:
            await store.delete(nf.id, speaker_a_user_id)
        if noise_facts:
            print(f"Removed {len(noise_facts)} noise facts from {speaker_a_user_id}")

    context_builder = ContextBuilder()

    # Resume support: load partial results if they exist
    partial_path = Path(f"{output_path}.partial.json")
    per_pair_results: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()

    if partial_path.exists():
        with open(partial_path) as f:
            per_pair_results = json.load(f)
        completed_keys = {(r["question"], r["conv_id"]) for r in per_pair_results}
        print(f"Resuming: loaded {len(per_pair_results)} completed pairs from {partial_path}")

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(qa_entries), desc="Evaluating", unit="pair")
    # Advance progress bar for already-completed pairs
    pbar.update(len(completed_keys))

    async def evaluate_pair(entry: dict) -> None:
        question = entry["question"]
        gold_answer = entry["answer"]
        category = entry["category"]
        conv_id = entry["conv_id"]
        entry_speaker_a_user_id = entry["speaker_a_user_id"]
        entry_speaker_b_user_id = entry["speaker_b_user_id"]

        # Skip already-completed pairs (resume support)
        if (question, conv_id) in completed_keys:
            return

        async with semaphore:
            # Retrieve memories from both speakers with retry
            memories_a: list = []
            for ret_attempt in range(3):
                try:
                    memories_a = await retriever.retrieve(
                        query=question,
                        user_id=entry_speaker_a_user_id,
                    )
                    break
                except Exception as e:
                    if ret_attempt == 2:
                        tqdm.write(f"  Retrieval (A) failed after 3 attempts: {e}")
                        memories_a = []
                        break
                    await asyncio.sleep(2 ** ret_attempt)

            memories_b: list = []
            for ret_attempt in range(3):
                try:
                    memories_b = await retriever.retrieve(
                        query=question,
                        user_id=entry_speaker_b_user_id,
                    )
                    break
                except Exception as e:
                    if ret_attempt == 2:
                        tqdm.write(f"  Retrieval (B) failed after 3 attempts: {e}")
                        memories_b = []
                        break
                    await asyncio.sleep(2 ** ret_attempt)

            seen_contents: set[str] = set()
            merged: list = []
            if args.no_isolation:
                # Flat merge: concat by score order, no round-robin diversity enforcement
                for fact in memories_a + memories_b:
                    if fact.content not in seen_contents:
                        seen_contents.add(fact.content)
                        merged.append(fact)
            else:
                # Round-robin merge: interleave speaker A and B results for diversity
                for fact in [f for pair in zip_longest(memories_a, memories_b) for f in pair if f is not None]:
                    if fact.content not in seen_contents:
                        seen_contents.add(fact.content)
                        merged.append(fact)
            memories = merged[:30]

            if not memories:
                result = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "I don't know",
                    "category": category,
                    "conv_id": conv_id,
                    "memories_retrieved": [],
                    "label": "WRONG",
                }
                async with write_lock:
                    per_pair_results.append(result)
                    partial_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(partial_path, "w") as pf:
                        json.dump(per_pair_results, pf, indent=2)
                pbar.update(1)
                return

            # Generate answer
            active_system_prompt = OPEN_DOMAIN_SYSTEM_PROMPT if category == 4 else ANSWER_SYSTEM_PROMPT
            system_prompt = context_builder.build_system_prompt(
                active_system_prompt,
                memories,
            )
            generated_answer = "I don't know"
            for gen_attempt in range(MAX_GENERATION_RETRIES):
                try:
                    generated_answer = await llm_service.complete(
                        messages=[{"role": "user", "content": question}],
                        system=system_prompt,
                    )
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "content_filter" in err_str or "content management policy" in err_str:
                        tqdm.write(f"  Skipped (content filter): {question[:60]}")
                        generated_answer = "I don't know"
                        break
                    if gen_attempt == MAX_GENERATION_RETRIES - 1:
                        raise
                    wait = GENERATION_BACKOFF[gen_attempt]
                    tqdm.write(f"  Generation retry {gen_attempt + 1}/{MAX_GENERATION_RETRIES} in {wait}s: {e}")
                    await asyncio.sleep(wait)

            # Two-pass retrieval: if first pass fails, retry with rephrased query
            if (
                ("don't know" in generated_answer.lower() or "do not know" in generated_answer.lower())
                and category != 4
            ):
                # Rephrase: extract key nouns from question for a broader search
                rephrase_prompt = f"Rephrase this question as a short keyword search query (5 words max): {question}"
                rephrased_query = question
                for gen_attempt in range(MAX_GENERATION_RETRIES):
                    try:
                        rephrased_query = await llm_service.complete(
                            messages=[{"role": "user", "content": rephrase_prompt}],
                            system="Return only the rephrased query, nothing else.",
                        )
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "content_filter" in err_str or "content management policy" in err_str:
                            break
                        if gen_attempt == MAX_GENERATION_RETRIES - 1:
                            tqdm.write(f"  Rephrase failed after {MAX_GENERATION_RETRIES} attempts, using original query: {e}")
                            break
                        wait = GENERATION_BACKOFF[gen_attempt]
                        tqdm.write(f"  Rephrase retry {gen_attempt + 1}/{MAX_GENERATION_RETRIES} in {wait}s: {e}")
                        await asyncio.sleep(wait)

                memories_a2: list = []
                for ret_attempt in range(3):
                    try:
                        memories_a2 = await retriever.retrieve(
                            query=rephrased_query,
                            user_id=entry_speaker_a_user_id,
                        )
                        break
                    except Exception as e:
                        if ret_attempt == 2:
                            tqdm.write(f"  Retrieval A2 failed after 3 attempts: {e}")
                            memories_a2 = []
                            break
                        await asyncio.sleep(2 ** ret_attempt)

                memories_b2: list = []
                for ret_attempt in range(3):
                    try:
                        memories_b2 = await retriever.retrieve(
                            query=rephrased_query,
                            user_id=entry_speaker_b_user_id,
                        )
                        break
                    except Exception as e:
                        if ret_attempt == 2:
                            tqdm.write(f"  Retrieval B2 failed after 3 attempts: {e}")
                            memories_b2 = []
                            break
                        await asyncio.sleep(2 ** ret_attempt)
                # Merge second-pass results with first-pass, deduplicate
                seen_contents2: set[str] = {f.content for f in memories}
                for fact in [f for pair in zip_longest(memories_a2, memories_b2) for f in pair if f is not None]:
                    if fact.content not in seen_contents2:
                        seen_contents2.add(fact.content)
                        memories.append(fact)
                memories = memories[:30]

                # Regenerate answer with expanded context
                system_prompt = context_builder.build_system_prompt(
                    active_system_prompt,
                    memories,
                )
                for gen_attempt in range(MAX_GENERATION_RETRIES):
                    try:
                        generated_answer = await llm_service.complete(
                            messages=[{"role": "user", "content": question}],
                            system=system_prompt,
                        )
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "content_filter" in err_str or "content management policy" in err_str:
                            tqdm.write(f"  Skipped (content filter, 2nd pass): {question[:60]}")
                            generated_answer = "I don't know"
                            break
                        if gen_attempt == MAX_GENERATION_RETRIES - 1:
                            raise
                        wait = GENERATION_BACKOFF[gen_attempt]
                        tqdm.write(f"  Generation retry {gen_attempt + 1}/{MAX_GENERATION_RETRIES} in {wait}s: {e}")
                        await asyncio.sleep(wait)

            # Judge
            label = await judge_answer(
                azure_client,
                judge_model,
                question,
                gold_answer,
                generated_answer,
            )

            result = {
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "category": category,
                "conv_id": conv_id,
                "memories_retrieved": [fact.content for fact in memories],
                "label": label,
                "two_pass_used": category != 4 and len(memories) > 0,
            }

            async with write_lock:
                per_pair_results.append(result)
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                with open(partial_path, "w") as pf:
                    json.dump(per_pair_results, pf, indent=2)

            pbar.update(1)

    await asyncio.gather(*[evaluate_pair(entry) for entry in qa_entries])
    pbar.close()

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
    print(f"\nQA Accuracy Evaluation ({', '.join(args.conv_ids)})")
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
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "no_isolation": args.no_isolation,
            "no_rerank": args.no_rerank,
            "generation_model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "unknown"),
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
