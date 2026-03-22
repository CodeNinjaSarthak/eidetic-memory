#!/usr/bin/env python3
# /// script
# dependencies = ["google-genai>=1.0.0", "python-dotenv>=1.0.0", "tqdm>=4.66.0"]
# ///
"""
Conflict resolution accuracy evaluation.

Evaluates EvolutionEngine's ability to correctly classify the relationship
between an existing memory fact and a new candidate fact (ADD/UPDATE/DELETE/NOOP).

Usage:
    uv run python eval/eval_conflict_resolution.py
    uv run python eval/eval_conflict_resolution.py --output eval/results/conflict_results.json

Reads .env.development for configuration.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ── sys.path setup for backend imports ───────────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "backend" / "services" / "memory" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "services" / "llm" / "src"))
sys.path.insert(0, str(_repo_root / "backend" / "packages" / "config" / "src"))

from llm.generation.gemini import GeminiService  # noqa: E402
from memory.models.memory import MemoryOperation, MemoryUpdate  # noqa: E402
from memory.pipeline.update import EvolutionEngine  # noqa: E402
from storage.models import MemoryFact  # noqa: E402

# ── Configuration ────────────────────────────────────────────
env_path = _repo_root / ".env.development"
load_dotenv(env_path)

REQUIRED_VARS = ["GOOGLE_API_KEY"]


def check_env() -> None:
    """Verify all required environment variables are set."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in .env.development and re-run.")
        sys.exit(1)


# ── Test case definition ─────────────────────────────────────
@dataclass
class TestCase:
    """A single conflict resolution test scenario."""

    name: str
    existing: str
    candidate: str
    expected: list[MemoryOperation]


TEST_CASES: list[TestCase] = [
    # ── CONTRADICTION / DELETE ───────────────────────────────
    TestCase(
        name="direct_contradiction_preference",
        existing="User likes coffee",
        candidate="User dislikes coffee",
        expected=[MemoryOperation.DELETE, MemoryOperation.UPDATE],
    ),
    TestCase(
        name="direct_contradiction_diet",
        existing="User is vegetarian",
        candidate="User started eating meat",
        expected=[MemoryOperation.DELETE, MemoryOperation.UPDATE],
    ),
    # ── UPDATE ───────────────────────────────────────────────
    TestCase(
        name="job_change",
        existing="User works at Meta",
        candidate="User works at Google",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="location_change",
        existing="User lives in New York",
        candidate="User moved to San Francisco",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="more_specific_fact",
        existing="User likes hiking",
        candidate="User loves hiking with friends on weekends",
        expected=[MemoryOperation.UPDATE, MemoryOperation.NOOP],
    ),
    # ── ADD (no conflict) ────────────────────────────────────
    TestCase(
        name="non_conflicting_preference",
        existing="User likes coffee",
        candidate="User likes tea",
        expected=[MemoryOperation.ADD],
    ),
    TestCase(
        name="unrelated_facts",
        existing="User works at Google",
        candidate="User has a dog named Max",
        expected=[MemoryOperation.ADD],
    ),
    # ── NOOP ─────────────────────────────────────────────────
    TestCase(
        name="redundant_fact_exact",
        existing="User likes coffee",
        candidate="User likes coffee",
        expected=[MemoryOperation.NOOP],
    ),
    TestCase(
        name="redundant_fact_paraphrase",
        existing="User likes coffee",
        candidate="User enjoys coffee",
        expected=[MemoryOperation.NOOP, MemoryOperation.UPDATE],
    ),
    TestCase(
        name="redundant_fact_subset",
        existing="User loves hiking with friends on weekends",
        candidate="User likes hiking",
        expected=[MemoryOperation.NOOP],
    ),
    # ── CONTRADICTION / DELETE (new) ─────────────────────────────
    TestCase(
        name="contradiction_with_new_value",
        existing="User prefers summer over winter",
        candidate="User now prefers winter over summer",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="negation_with_no_replacement",
        existing="User is learning Spanish",
        candidate="User gave up learning Spanish",
        expected=[MemoryOperation.DELETE],
    ),
    TestCase(
        name="negation_hobby",
        existing="User plays guitar every evening",
        candidate="User quit playing guitar",
        expected=[MemoryOperation.DELETE],
    ),
    # ── UPDATE (new) ──────────────────────────────────────────────
    TestCase(
        name="vehicle_change",
        existing="User drives a Honda Civic",
        candidate="User bought a Tesla",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="relationship_status_change",
        existing="User is single",
        candidate="User got married recently",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="education_change",
        existing="User is studying at community college",
        candidate="User transferred to Stanford",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="diet_change_with_new_value",
        existing="User follows a keto diet",
        candidate="User switched to a vegan diet",
        expected=[MemoryOperation.UPDATE],
    ),
    TestCase(
        name="preference_reversal_with_new_value",
        existing="User dislikes spicy food",
        candidate="User has developed a taste for spicy food",
        expected=[MemoryOperation.UPDATE],
    ),
    # ── ADD (new) ─────────────────────────────────────────────────
    TestCase(
        name="new_hobby_no_conflict",
        existing="User likes painting",
        candidate="User recently started learning piano",
        expected=[MemoryOperation.ADD],
    ),
    TestCase(
        name="new_personal_detail",
        existing="User works as a software engineer",
        candidate="User has two younger siblings",
        expected=[MemoryOperation.ADD],
    ),
    TestCase(
        name="new_preference_different_domain",
        existing="User prefers tea over coffee",
        candidate="User enjoys running in the morning",
        expected=[MemoryOperation.ADD],
    ),
    # ── NOOP (new) ────────────────────────────────────────────────
    TestCase(
        name="vaguer_subset_exercise",
        existing="User goes to the gym every morning and does weightlifting",
        candidate="User exercises regularly",
        expected=[MemoryOperation.NOOP],
    ),
    TestCase(
        name="vaguer_subset_diet",
        existing="User follows a strict vegan diet and avoids all animal products",
        candidate="User is vegan",
        expected=[MemoryOperation.NOOP],
    ),
    TestCase(
        name="redundant_fact_synonym",
        existing="User is a software engineer",
        candidate="User works as a developer",
        expected=[MemoryOperation.NOOP, MemoryOperation.UPDATE],
    ),
    TestCase(
        name="redundant_fact_restatement",
        existing="User has a cat named Whiskers",
        candidate="User owns a cat",
        expected=[MemoryOperation.NOOP],
    ),
    TestCase(
        name="redundant_with_different_phrasing",
        existing="User graduated from MIT with a degree in computer science",
        candidate="User studied computer science at MIT",
        expected=[MemoryOperation.NOOP],
    ),
]


# ── Runner ───────────────────────────────────────────────────
async def run_test(
    engine: EvolutionEngine, test: TestCase
) -> dict:
    """Run a single test case and return the result dict."""
    existing_fact = MemoryFact(
        id="fact-0",
        user_id="eval-user",
        content=test.existing,
        embedding=None,
    )

    update: MemoryUpdate = await engine.decide(
        candidate_fact=test.candidate,
        existing_memories=[existing_fact],
    )

    passed = update.operation in test.expected
    expected_str = ", ".join(op.value for op in test.expected)

    if passed:
        print(f"  ✓ {test.name} — {update.operation.value} (expected: {expected_str})")
    else:
        print(f"  ✗ {test.name} — {update.operation.value} (expected: {expected_str})")

    return {
        "name": test.name,
        "existing": test.existing,
        "candidate": test.candidate,
        "expected": [op.value for op in test.expected],
        "actual": update.operation.value,
        "passed": passed,
    }


async def main() -> None:
    """Run all conflict resolution test cases."""
    parser = argparse.ArgumentParser(
        description="Evaluate EvolutionEngine conflict resolution accuracy."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/conflict_results.json",
        help="Path to save results JSON (default: eval/results/conflict_results.json)",
    )
    args = parser.parse_args()

    check_env()

    model = os.environ.get("MEMORY_EXTRACTION_MODEL", "gemini-2.0-flash")
    llm = GeminiService(model=model)
    engine = EvolutionEngine(llm_service=llm)

    print(f"\nRunning {len(TEST_CASES)} conflict resolution tests (model: {model})\n")

    results: list[dict] = []
    for test in tqdm(TEST_CASES, desc="Evaluating", unit="case"):
        result = await run_test(engine, test)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    accuracy = passed / len(results) if results else 0.0

    print("\n  Conflict Resolution Accuracy")
    print("  ─────────────────────────────")
    print(f"  Total:    {len(results)}")
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Accuracy: {accuracy:.1%}")

    failed_cases = [r for r in results if not r["passed"]]
    if failed_cases:
        print("\n  Failed cases:")
        for r in failed_cases:
            expected_str = ", ".join(r["expected"])
            print(f"    - {r['name']}: got {r['actual']} expected {expected_str}")

    # ── Save JSON ────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {"timestamp": datetime.now(UTC).isoformat()},
        "overall": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "accuracy": round(accuracy, 4),
        },
        "results": results,
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
