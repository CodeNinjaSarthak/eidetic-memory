#!/usr/bin/env python3
# /// script
# dependencies = ["google-genai>=1.0.0", "openai>=2.29.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Fact extraction accuracy evaluation on LoCoMo dataset.

Usage:
    uv run python eval/eval_extraction.py --limit 10

Reads .env.development for configuration.
Evaluates whether extracted facts from evidence turns are relevant to QA pairs.
Reports Recall, Precision, and per-category breakdowns.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncAzureOpenAI
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────
RANDOM_SEED = 42
CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
}

EXTRACT_FACTS_TOOL: dict[str, Any] = {
    "name": "extract_facts",
    "description": "Extract a list of factual statements from the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of factual statements extracted from the conversation.",
            }
        },
        "required": ["facts"],
    },
}

# ── Configuration ────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "EVAL_LLM_JUDGE_MODEL",
]


def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)


# ── Dataset helpers ──────────────────────────────────────────
def load_dataset() -> list[dict]:
    """Load the LoCoMo dataset."""
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    with open(data_path) as f:
        return json.load(f)


def collect_qa_entries(dataset: list[dict]) -> list[dict]:
    """Collect all QA entries with non-empty evidence, categories 1-4 only."""
    entries: list[dict] = []
    for conversation_data in dataset:
        sample_id = conversation_data["sample_id"]
        conversation = conversation_data["conversation"]
        for qa in conversation_data.get("qa", []):
            if qa.get("category") not in (1, 2, 3, 4):
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
                    "conversation": conversation,
                }
            )
    return entries


_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")


def parse_evidence_ids(raw: str) -> list[tuple[int, int]]:
    """Parse a raw evidence string into a list of (session_num, turn_index) tuples.

    Handles malformed evidence strings from the dataset:
    - "D" — empty, skipped
    - "D:11:26" — extra colon, skipped (doesn't match D<int>:<int>)
    - "D8:6; D9:17" — semicolon-separated, split and parsed individually
    - "D9:1 D4:4 D4:6" — space-separated, split and parsed individually

    Returns:
        List of (session_num, turn_index) tuples. Empty if no valid IDs found.
    """
    tokens = re.split(r"[;\s]+", raw.strip())
    results: list[tuple[int, int]] = []
    for token in tokens:
        token = token.strip()
        m = _DIA_ID_RE.match(token)
        if m:
            results.append((int(m.group(1)), int(m.group(2))))
    return results


def find_turn_by_dia_id(conversation: dict, dia_id: str) -> tuple[dict | None, list[dict], str]:
    """Find a turn in the conversation by dia_id.

    Returns:
        Tuple of (turn, session_turns, session_datetime).
        turn is None if not found.
    """
    m = _DIA_ID_RE.match(dia_id)
    if not m:
        return None, [], ""
    session_num = int(m.group(1))
    session_key = f"session_{session_num}"
    datetime_key = f"session_{session_num}_date_time"

    session_turns = conversation.get(session_key, [])
    session_datetime = conversation.get(datetime_key, "")

    for turn in session_turns:
        if turn.get("dia_id") == dia_id:
            return turn, session_turns, session_datetime

    return None, session_turns, session_datetime


def build_extraction_context(
    conversation: dict, dia_id: str
) -> tuple[str, str, str, str] | None:
    """Build the extraction context for a given evidence turn.

    Returns:
        Tuple of (previous_role, previous_text, current_role, current_text)
        or None if the turn is not found.
    """
    turn, session_turns, session_datetime = find_turn_by_dia_id(conversation, dia_id)
    if turn is None:
        return None

    # Find index of this turn in session
    turn_idx = None
    for i, t in enumerate(session_turns):
        if t.get("dia_id") == dia_id:
            turn_idx = i
            break

    if turn_idx is None:
        return None

    # Previous turn (or start of conversation placeholder)
    if turn_idx > 0:
        prev_turn = session_turns[turn_idx - 1]
        previous_role = "user"
        previous_text = prev_turn["text"]
    else:
        previous_role = "user"
        previous_text = "[start of conversation]"

    # Current turn with session datetime prepended
    current_role = "user"
    current_text = f"[Session date: {session_datetime}] {turn['text']}"

    return previous_role, previous_text, current_role, current_text


# ── Extraction via Gemini ────────────────────────────────────
def build_gemini_tool() -> types.Tool:
    """Build the Gemini tool definition matching the extract_facts schema."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="extract_facts",
                description="Extract a list of factual statements from the conversation.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "facts": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="STRING"),
                            description="List of factual statements extracted from the conversation.",
                        )
                    },
                    required=["facts"],
                ),
            )
        ]
    )


def extract_facts_from_turn(
    client: genai.Client,
    model: str,
    system_prompt: str,
    previous_role: str,
    previous_text: str,
    current_role: str,
    current_text: str,
) -> list[str]:
    """Extract facts from a conversation turn pair using Gemini."""
    user_content = (
        f"Current Conversation:\n"
        f"{previous_role}: {previous_text}\n"
        f"{current_role}: {current_text}"
    )

    tool = build_gemini_tool()

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=user_content)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[tool],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                        )
                    ),
                ),
            )
            break
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  ERROR calling Gemini after 3 attempts: {e}")
                return []
            wait = 2**attempt
            tqdm.write(f"  Retrying Gemini in {wait}s (attempt {attempt + 1}/3): {e}")
            time.sleep(wait)

    # Parse tool call response
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call and part.function_call.name == "extract_facts":
                facts = part.function_call.args.get("facts", [])
                if isinstance(facts, list):
                    return [str(f) for f in facts]
    return []


# ── LLM Judge via Azure OpenAI ──────────────────────────────
async def judge_fact_relevance(
    client: AsyncAzureOpenAI,
    model: str,
    question: str,
    answer: str,
    fact: str,
) -> bool:
    """Judge whether a single fact is relevant to answering the question."""
    prompt = (
        "You are evaluating whether an extracted fact is relevant to answering a question.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {answer}\n"
        f"Extracted fact: {fact}\n\n"
        "Does this extracted fact contain information that would help answer the question?\n"
        "Be generous: if the fact is about the same topic or event as the question,\n"
        "count it as relevant even if it does not contain the exact answer.\n\n"
        'Return JSON: {"relevant": true} or {"relevant": false}'
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
            return bool(result.get("relevant", False))
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  ERROR judging fact after 3 attempts: {e}")
                return False
            wait = 2**attempt
            await asyncio.sleep(wait)

    return False


async def judge_facts_batch(
    client: AsyncAzureOpenAI,
    model: str,
    question: str,
    answer: str,
    facts: list[str],
) -> list[bool]:
    """Judge all facts for a QA pair in parallel."""
    if not facts:
        return []
    tasks = [judge_fact_relevance(client, model, question, answer, f) for f in facts]
    return list(await asyncio.gather(*tasks))


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fact extraction accuracy evaluation on LoCoMo dataset.")
    parser.add_argument("--limit", type=int, default=10, help="Number of QA pairs to evaluate (default: 10)")
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/extraction_results.json",
        help="Path to save results JSON",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    limit = args.limit
    output_path = Path(args.output)

    check_env()

    # Load extraction prompt
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "backend"
        / "services"
        / "memory"
        / "src"
        / "memory"
        / "prompts"
        / "extraction.txt"
    )
    system_prompt = prompt_path.read_text()

    # Load dataset
    dataset = load_dataset()

    # Collect and sample QA entries
    all_entries = collect_qa_entries(dataset)
    print(f"Total QA entries with evidence (categories 1-4): {len(all_entries)}")

    random.seed(RANDOM_SEED)
    sampled = random.sample(all_entries, min(limit, len(all_entries)))
    print(f"Sampled {len(sampled)} QA pairs (seed={RANDOM_SEED})")

    # Clients
    extraction_model = os.environ.get("MEMORY_EXTRACTION_MODEL", "gemini-2.0-flash")
    judge_model = os.environ["EVAL_LLM_JUDGE_MODEL"]

    gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    azure_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01",
        timeout=30.0,
        max_retries=0,
    )

    # Evaluate
    per_pair_results: list[dict] = []
    total_facts_extracted = 0
    total_facts_relevant = 0

    for entry in tqdm(sampled, desc="Evaluating", unit="pair"):
        conversation = entry["conversation"]
        evidence_dia_ids = entry["evidence"]

        # Extract facts from all evidence turns
        all_extracted_facts: list[str] = []
        for raw_evidence in evidence_dia_ids:
            parsed_ids = parse_evidence_ids(raw_evidence)
            if not parsed_ids:
                tqdm.write(f"  WARNING: No valid dia_ids in evidence string: {raw_evidence!r}")
                continue
            for session_num, turn_index in parsed_ids:
                dia_id = f"D{session_num}:{turn_index}"
                ctx = build_extraction_context(conversation, dia_id)
                if ctx is None:
                    tqdm.write(f"  WARNING: Could not find turn {dia_id}")
                    continue

                previous_role, previous_text, current_role, current_text = ctx
                facts = extract_facts_from_turn(
                    gemini_client,
                    extraction_model,
                    system_prompt,
                    previous_role,
                    previous_text,
                    current_role,
                    current_text,
                )
                all_extracted_facts.extend(facts)

        # Deduplicate facts
        all_extracted_facts = list(dict.fromkeys(all_extracted_facts))

        # Judge relevance
        relevance = await judge_facts_batch(
            azure_client,
            judge_model,
            entry["question"],
            entry["answer"],
            all_extracted_facts,
        )

        relevant_facts = [f for f, r in zip(all_extracted_facts, relevance, strict=True) if r]
        hit = len(relevant_facts) > 0

        total_facts_extracted += len(all_extracted_facts)
        total_facts_relevant += len(relevant_facts)

        per_pair_results.append(
            {
                "sample_id": entry["sample_id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "category": entry["category"],
                "evidence_dia_ids": evidence_dia_ids,
                "extracted_facts": all_extracted_facts,
                "relevant_facts": relevant_facts,
                "hit": hit,
            }
        )

    await azure_client.close()

    # ── Aggregate metrics ────────────────────────────────────
    n_evaluated = len(per_pair_results)
    if n_evaluated == 0:
        print("No QA pairs were successfully evaluated.")
        return

    hits = sum(1 for r in per_pair_results if r["hit"])
    recall = hits / n_evaluated
    precision = total_facts_relevant / total_facts_extracted if total_facts_extracted > 0 else 0.0
    avg_facts = total_facts_extracted / n_evaluated

    # By category
    by_category: dict[int, dict] = {}
    for cat_id, cat_label in CATEGORIES.items():
        cat_results = [r for r in per_pair_results if r["category"] == cat_id]
        if not cat_results:
            continue
        cat_hits = sum(1 for r in cat_results if r["hit"])
        cat_total_extracted = sum(len(r["extracted_facts"]) for r in cat_results)
        cat_total_relevant = sum(len(r["relevant_facts"]) for r in cat_results)
        by_category[cat_id] = {
            "label": cat_label,
            "recall": cat_hits / len(cat_results),
            "precision": cat_total_relevant / cat_total_extracted if cat_total_extracted > 0 else 0.0,
            "count": len(cat_results),
            "total_facts_extracted": cat_total_extracted,
            "total_facts_relevant": cat_total_relevant,
        }

    # ── Print results ────────────────────────────────────────
    print("\nExtraction Accuracy Evaluation")
    print("──────────────────────────────")
    print(f"QA pairs evaluated: {n_evaluated}")
    print(f"Avg facts extracted per turn: {avg_facts:.1f}")
    print(f"\nRecall:    {recall:.1%} (pairs where ≥1 fact was relevant)")
    print(f"Precision: {precision:.1%} (relevant facts / total facts extracted)")

    print("\nBy category:")
    for cat_id in sorted(by_category):
        cat = by_category[cat_id]
        label = f"{cat['label']}:"
        print(f"  {label:<14} Recall={cat['recall']:.1%} Precision={cat['precision']:.1%} ({cat['count']} pairs)")

    # ── Save results ─────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "limit": limit,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        "overall": {
            "recall": recall,
            "precision": precision,
            "avg_facts_per_turn": avg_facts,
            "total_pairs": n_evaluated,
            "total_facts_extracted": total_facts_extracted,
            "total_facts_relevant": total_facts_relevant,
        },
        "by_category": {str(k): v for k, v in by_category.items()},
        "per_pair": per_pair_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
