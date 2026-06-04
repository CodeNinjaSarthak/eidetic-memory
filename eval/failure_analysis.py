"""Failure analysis for improvement_week2_full_v1.json evaluation results."""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "improvement_week2_full_v1.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
}

FAILURE_MODES = [
    "FACT_NOT_RETRIEVED",
    "FACT_NOT_EXTRACTED",
    "WRONG_SPEAKER",
    "STALE_FACT",
    "MULTI_HOP_INCOMPLETE",
    "GENERATION_ERROR",
    "OTHER",
]


def normalize(text) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def classify_failure(pair: dict) -> tuple[str, str]:
    """
    Heuristically classify the failure mode for an incorrect pair.
    Returns (mode, reason).
    """
    question = normalize(pair["question"])
    gold = normalize(pair["gold_answer"])
    generated = normalize(pair["generated_answer"])
    memories = [normalize(m) for m in pair.get("memories_retrieved", [])]
    cat = pair["category"]

    # Extract the subject from the question (first proper noun or name)
    # Check for wrong-speaker: question about person X but memories about someone else
    # Crude heuristic: check if gold tokens appear in any memory
    gold_tokens = set(gold.split())
    significant_gold = {t for t in gold_tokens if len(t) > 3}

    memory_hit = any(
        len(significant_gold & set(m.split())) >= max(1, len(significant_gold) // 3)
        for m in memories
    )

    # Check if no memories were retrieved at all
    if not memories:
        return "FACT_NOT_EXTRACTED", "no memories retrieved at all"

    # Check for multi-hop incomplete: category 3 and partial overlap only
    if cat == 3:
        # If memories contain some relevant info but not the connecting fact
        if memory_hit:
            return "MULTI_HOP_INCOMPLETE", "retrieved partial chain, missing connecting fact"
        else:
            return "FACT_NOT_EXTRACTED", "multi-hop fact chain not stored"

    # Check for WRONG_SPEAKER: memories seem to be retrieved but about different person
    # Heuristic: look for different names in question vs top memories
    question_words = set(question.split())
    # If generated answer contains unrelated person's info
    if not memory_hit and memories:
        # Check if memories contain relevant topic words from question
        question_content_words = {w for w in question_words if len(w) > 4 and w not in
                                   {"what", "when", "where", "which", "whose", "about", "their", "would", "could", "there"}}
        topic_in_memories = any(
            len(question_content_words & set(m.split())) >= 1 for m in memories[:5]
        )
        if not topic_in_memories:
            return "FACT_NOT_EXTRACTED", "fact likely never stored — no topically relevant memories"
        else:
            return "FACT_NOT_RETRIEVED", "relevant topic in memories but correct answer not surfaced"

    # Check for STALE_FACT: temporal questions where date/time mismatch
    if cat == 2:
        date_patterns = [r"\d{4}", r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                         r"\b(week|month|year|day)\b"]
        gold_has_date = any(re.search(p, gold) for p in date_patterns)
        gen_has_date = any(re.search(p, generated) for p in date_patterns)
        if gold_has_date and gen_has_date and gold != generated:
            # Check if any retrieved memory has a date that matches generated (stale)
            gen_date_tokens = {w for w in generated.split() if re.search(r'\d{4}', w) or len(w) > 4}
            stale = any(
                len(gen_date_tokens & set(m.split())) >= 1 for m in memories[:5]
            )
            if stale:
                return "STALE_FACT", "retrieved older fact with different date than gold"
        if memory_hit:
            return "GENERATION_ERROR", "relevant memory retrieved but wrong date/time extracted"

    # Check for GENERATION_ERROR: gold tokens appear in memories but answer is wrong
    if memory_hit:
        # The fact was retrieved but generation failed
        return "GENERATION_ERROR", "relevant fact in retrieved memories but answer still wrong"

    # Default: fact not retrieved
    return "FACT_NOT_RETRIEVED", "gold answer not present in any retrieved memory"


def format_memories(memories: list[str], n: int = 3) -> str:
    if not memories:
        return "  (none)"
    return "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(memories[:n]))


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    pairs = data["per_pair"]
    incorrect = [p for p in pairs if p.get("label") == "WRONG"]

    print("=" * 80)
    print("FAILURE ANALYSIS — improvement_week2_full_v1")
    print(f"Overall accuracy: {data['overall']['accuracy']:.1%}  "
          f"({data['overall']['correct']}/{data['overall']['total']})")
    print(f"Total incorrect: {len(incorrect)}")
    print("=" * 80)

    # Group by category
    by_category: dict[int, list[dict]] = defaultdict(list)
    for p in incorrect:
        by_category[p["category"]].append(p)

    # Per-category analysis
    all_mode_counts: dict[str, dict[str, int]] = {}

    for cat_id in sorted(by_category.keys()):
        cat_name = CATEGORY_NAMES[cat_id]
        cat_pairs = by_category[cat_id]
        total_in_cat = data["by_category"][str(cat_id)]["total"]
        correct_in_cat = data["by_category"][str(cat_id)]["correct"]

        print(f"\n{'='*80}")
        print(f"CATEGORY: {cat_name.upper()}  "
              f"(accuracy {correct_in_cat}/{total_in_cat} = "
              f"{correct_in_cat/total_in_cat:.1%}, {len(cat_pairs)} failures)")
        print("=" * 80)

        # Classify all failures
        classified: list[tuple[dict, str, str]] = []
        for p in cat_pairs:
            mode, reason = classify_failure(p)
            classified.append((p, mode, reason))

        # Count modes
        mode_counts: dict[str, int] = defaultdict(int)
        for _, mode, _ in classified:
            mode_counts[mode] += 1
        all_mode_counts[cat_name] = dict(mode_counts)

        # Pick 8 representative examples — sample across modes
        shown_modes: dict[str, int] = defaultdict(int)
        examples: list[tuple[dict, str]] = []

        # First pass: one per mode
        for p, mode, _reason in classified:
            if shown_modes[mode] == 0:
                examples.append((p, mode))
                shown_modes[mode] += 1

        # Second pass: fill up to 8 by most-common mode
        sorted_by_mode = sorted(classified, key=lambda x: -mode_counts[x[1]])
        for p, mode, _reason in sorted_by_mode:
            if len(examples) >= 8:
                break
            if (p, mode) not in examples:
                examples.append((p, mode))

        examples = examples[:8]

        for i, (p, mode) in enumerate(examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"QUESTION:          {p['question']}")
            print(f"GOLD ANSWER:       {p['gold_answer']}")
            print(f"PREDICTED:         {p['generated_answer']}")
            print("TOP 3 RETRIEVED FACTS:")
            print(format_memories(p.get("memories_retrieved", []), 3))
            print(f"FAILURE MODE:      {mode}")

        # Mode breakdown for this category
        print(f"\n--- Failure mode breakdown for {cat_name} ---")
        for mode in FAILURE_MODES:
            count = mode_counts.get(mode, 0)
            pct = count / len(cat_pairs) * 100 if cat_pairs else 0
            bar = "#" * (count // max(1, len(cat_pairs) // 20))
            print(f"  {mode:<26}  {count:>4}  ({pct:5.1f}%)  {bar}")

    # Cross-category summary
    print(f"\n{'='*80}")
    print("CROSS-CATEGORY FAILURE MODE SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<26}  {'s_hop':>6}  {'temp':>6}  {'m_hop':>6}  {'o_dom':>6}  {'TOTAL':>6}")
    print("-" * 70)
    totals: dict[str, int] = defaultdict(int)
    for mode in FAILURE_MODES:
        row = []
        for cat_name in ["single_hop", "temporal", "multi_hop", "open_domain"]:
            cnt = all_mode_counts.get(cat_name, {}).get(mode, 0)
            totals[mode] += cnt
            row.append(cnt)
        print(f"  {mode:<26}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}  {row[3]:>6}  {totals[mode]:>6}")

    # BM25 analysis
    print(f"\n{'='*80}")
    print("BM25 IMPACT ANALYSIS")
    print("=" * 80)

    bm25_likely = ["FACT_NOT_RETRIEVED", "MULTI_HOP_INCOMPLETE"]
    bm25_unlikely = ["FACT_NOT_EXTRACTED", "GENERATION_ERROR", "STALE_FACT", "WRONG_SPEAKER"]

    bm25_fixable = sum(totals.get(m, 0) for m in bm25_likely)
    not_fixable = sum(totals.get(m, 0) for m in bm25_unlikely)
    total_failures = len(incorrect)

    print(f"\nTotal failures: {total_failures}")
    print(f"\nBM25 LIKELY TO FIX ({bm25_fixable} failures, {bm25_fixable/total_failures:.1%}):")
    for mode in bm25_likely:
        cnt = totals.get(mode, 0)
        print(f"  {mode}: {cnt}")
        if mode == "FACT_NOT_RETRIEVED":
            print("    Reason: BM25 excels at keyword/exact-phrase recall that dense vectors miss.")
            print("    Dense embeddings can fail on rare names, dates, and entity-specific strings.")
            print("    Hybrid retrieval (BM25 + dense) would surface these literal matches.")
        elif mode == "MULTI_HOP_INCOMPLETE":
            print("    Reason: Intermediate facts with specific entity names benefit from BM25.")
            print("    The connecting fact often contains exact tokens (names, IDs) dense search misses.")

    print(f"\nBM25 UNLIKELY TO FIX ({not_fixable} failures, {not_fixable/total_failures:.1%}):")
    for mode in bm25_unlikely:
        cnt = totals.get(mode, 0)
        if cnt == 0:
            continue
        print(f"  {mode}: {cnt}")
        if mode == "FACT_NOT_EXTRACTED":
            print("    Reason: The fact was never stored — no retrieval method can find what isn't there.")
            print("    Fix: improve extraction prompt to capture more facts per turn.")
        elif mode == "GENERATION_ERROR":
            print("    Reason: The right memory was retrieved but the LLM synthesized the wrong answer.")
            print("    Fix: improve generation prompt, add chain-of-thought, or use a stronger model.")
        elif mode == "STALE_FACT":
            print("    Reason: BM25 would retrieve the same stale memory (also matches keywords).")
            print("    Fix: improve memory update logic to overwrite old facts reliably.")
        elif mode == "WRONG_SPEAKER":
            print("    Reason: BM25 also matches on topic, not speaker — same cross-contamination risk.")
            print("    Fix: namespace memories by user_id more strictly in retrieval filters.")

    print(f"\n{'='*80}")
    print("TOP RECOMMENDED FIXES (by failure count)")
    print("=" * 80)
    sorted_modes = sorted(totals.items(), key=lambda x: -x[1])
    for i, (mode, cnt) in enumerate(sorted_modes[:3], 1):
        pct = cnt / total_failures * 100
        print(f"\n#{i}: {mode}  ({cnt} failures, {pct:.1f}%)")
        if mode == "FACT_NOT_RETRIEVED":
            print("  → Add BM25/hybrid retrieval. Dense alone misses exact-phrase matches.")
        elif mode == "FACT_NOT_EXTRACTED":
            print("  → Improve extraction: run multi-pass extraction, lower confidence threshold.")
        elif mode == "GENERATION_ERROR":
            print("  → Add fact-grounding step: ask model to cite the specific memory it used.")
        elif mode == "MULTI_HOP_INCOMPLETE":
            print("  → Add multi-hop retrieval: use first-hop answer to query for second-hop facts.")
        elif mode == "STALE_FACT":
            print("  → Fix update logic: ensure UPDATE operations replace, not append, old facts.")
        elif mode == "WRONG_SPEAKER":
            print("  → Enforce user_id filter strictly; add speaker-tagging to memory payloads.")

    print()


if __name__ == "__main__":
    main()
