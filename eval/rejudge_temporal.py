#!/usr/bin/env python3
# /// script
# dependencies = ["openai>=2.29.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Re-judge temporal (category 2) WRONG pairs with the updated JUDGE_PROMPT.

Usage:
    uv run python eval/rejudge_temporal.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tqdm import tqdm

_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env.development")

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

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
}

INPUT_PATH = Path("eval/results/qa_accuracy_all10_jina.json")
OUTPUT_PATH = Path("eval/results/qa_accuracy_all10_jina_v2.json")


async def judge_answer(
    client: AsyncAzureOpenAI,
    model: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> str:
    """Judge whether a generated answer is correct using Azure OpenAI."""
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
                tqdm.write(f"  ERROR judging after 3 attempts: {e}")
                return "WRONG"
            await asyncio.sleep(2**attempt)
    return "WRONG"


async def main() -> None:
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "EVAL_LLM_JUDGE_MODEL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    pairs = data["per_pair"]
    temporal_wrong = [r for r in pairs if r["category"] == 2 and r["label"] == "WRONG"]
    print(f"Temporal WRONG pairs to re-judge: {len(temporal_wrong)}")

    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )
    judge_model = os.environ["EVAL_LLM_JUDGE_MODEL"]

    # Build index from (question, conv_id) -> pair dict for in-place update
    pair_index: dict[tuple[str, str], dict] = {
        (r["question"], r["conv_id"]): r for r in pairs
    }

    semaphore = asyncio.Semaphore(5)
    flipped = 0
    flip_lock = asyncio.Lock()
    pbar = tqdm(total=len(temporal_wrong), desc="Re-judging", unit="pair")

    async def rejudge(entry: dict) -> None:
        nonlocal flipped
        async with semaphore:
            new_label = await judge_answer(
                client,
                judge_model,
                entry["question"],
                entry["gold_answer"],
                entry["generated_answer"],
            )
            if new_label == "CORRECT":
                key = (entry["question"], entry["conv_id"])
                async with flip_lock:
                    pair_index[key]["label"] = "CORRECT"
                    flipped += 1
                    tqdm.write(
                        f"  FLIPPED: Gold={entry['gold_answer']} | "
                        f"Gen={entry['generated_answer'][:60]}"
                    )
            pbar.update(1)

    await asyncio.gather(*[rejudge(entry) for entry in temporal_wrong])
    pbar.close()
    await client.close()

    print(f"\nFlipped {flipped}/{len(temporal_wrong)} temporal pairs from WRONG → CORRECT")

    # Recompute metrics
    n_evaluated = len(pairs)
    n_correct = sum(1 for r in pairs if r["label"] == "CORRECT")
    overall_accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0

    print("\nUpdated Results")
    print("─" * 40)
    print(f"Overall accuracy: {overall_accuracy:.1%} ({n_correct}/{n_evaluated})")
    print("\nBy category:")
    for cat_id, cat_label in CATEGORIES.items():
        cat_results = [r for r in pairs if r["category"] == cat_id]
        if not cat_results:
            continue
        cat_correct = sum(1 for r in cat_results if r["label"] == "CORRECT")
        cat_acc = cat_correct / len(cat_results)
        print(f"  {cat_label:12s}: {cat_acc:.1%} ({cat_correct}/{len(cat_results)})")

    # Save updated results
    data["overall"]["correct"] = n_correct
    data["overall"]["accuracy"] = overall_accuracy
    for cat_id, _cat_label in CATEGORIES.items():
        cat_key = str(cat_id)
        if cat_key in data["by_category"]:
            cat_results = [r for r in pairs if r["category"] == cat_id]
            cat_correct = sum(1 for r in cat_results if r["label"] == "CORRECT")
            data["by_category"][cat_key]["correct"] = cat_correct
            data["by_category"][cat_key]["accuracy"] = cat_correct / len(cat_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
