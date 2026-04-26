"""Analyze multi-hop (category=3) error patterns in Eidetic Memory eval results.

Error categories (applied in priority order):
  1. wrong_speaker   — retrieved fact contains speaker-confusion noise
                       ("X's name is Y" pattern in retrieved facts)
  2. outdated_fact   — contradictory near-duplicate facts in context
                       (same entity, conflicting state/date)
  3. reasoning_fail  — gold-answer keywords ARE in context, model still fails
  4. missing_fact    — gold-answer keywords absent from context (default)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
INPUT_PATH = RESULTS_DIR / "final_eidetic_memory_all10.json"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "up", "about", "into",
    "through", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while",
    "yes", "no", "likely", "yes,", "no,", "if", "she", "he", "they",
    "their", "her", "his", "it", "its", "we", "you", "i", "me",
    "him", "them", "our", "your", "my", "who", "what", "when",
    "where", "why", "how", "which", "that", "this", "these", "those",
}

# Minimum keyword length to count as meaningful
MIN_KW_LEN = 3

_SPEAKER_CONFUSION_RE = re.compile(
    r"(\w+)'s name is (\w+)", re.IGNORECASE
)

# Time-flavored words that signal a fact could be stale
_TEMPORAL_SIGNALS = re.compile(
    r"\b(recently|last\s+\w+|currently|now|used to|previously|"
    r"\d{4}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_keywords(text: str) -> set[str]:
    """Return meaningful lowercase tokens from text."""
    return {
        w.lower().strip(".,;:!?()'\"")
        for w in text.split()
        if len(w) >= MIN_KW_LEN and w.lower().strip(".,;:!?()'\"") not in STOP_WORDS
    }


def gold_keywords_in_facts(gold: str, facts: list[str]) -> bool:
    """Return True if at least two gold keywords appear in the retrieved facts."""
    keywords = extract_keywords(gold)
    if not keywords:
        return False
    facts_blob = " ".join(facts).lower()
    hits = sum(1 for kw in keywords if kw in facts_blob)
    # Require at least 2 keyword hits so single-word golds don't trivially match
    return hits >= max(1, min(2, len(keywords)))


def has_speaker_confusion(facts: list[str]) -> bool:
    """Detect 'X's name is Y' cross-attribution noise in retrieved facts."""
    for fact in facts:
        if _SPEAKER_CONFUSION_RE.search(fact):
            return True
    return False


def has_outdated_fact(facts: list[str]) -> bool:
    """Detect when near-duplicate facts with conflicting temporal info are present.

    Strategy: for any two facts that share at least 4 keywords (same entity/topic),
    check if they also contain temporal signals — indicating competing versions.
    """
    temporal_facts = [f for f in facts if _TEMPORAL_SIGNALS.search(f)]
    if len(temporal_facts) < 2:
        return False

    kw_sets = [extract_keywords(f) for f in temporal_facts]
    for i in range(len(kw_sets)):
        for j in range(i + 1, len(kw_sets)):
            overlap = kw_sets[i] & kw_sets[j]
            # 4+ shared content words → likely the same fact, different version
            if len(overlap) >= 4:
                return True
    return False


def categorize(pair: dict) -> str:
    """Assign one error category to a wrong multi-hop prediction."""
    facts = pair.get("memories_retrieved", [])
    gold = str(pair.get("gold_answer", ""))
    generated = pair.get("generated_answer", "")

    if has_speaker_confusion(facts):
        return "wrong_speaker"

    if has_outdated_fact(facts):
        return "outdated_fact"

    if gold_keywords_in_facts(gold, facts) and "don't know" not in generated.lower():
        return "reasoning_fail"

    return "missing_fact"


# ── Formatting ─────────────────────────────────────────────────────────────

CATEGORY_META = {
    "wrong_speaker":  "Wrong speaker  — retrieved fact from the wrong speaker",
    "outdated_fact":  "Outdated fact  — older/contradictory version retrieved",
    "reasoning_fail": "Reasoning fail — facts present but answer wrong",
    "missing_fact":   "Missing fact   — relevant fact not in retrieved context",
}

DIVIDER = "─" * 72


def print_summary(counts: Counter, total_wrong: int, total_cat3: int) -> None:
    print()
    print("=" * 72)
    print("  Multi-hop Error Analysis  (category=3)")
    print("=" * 72)
    print(f"  Total multi-hop questions : {total_cat3}")
    print(f"  Wrong predictions         : {total_wrong}")
    print()
    print(f"  {'Error type':<40}  {'Count':>5}  {'Share':>6}")
    print(f"  {DIVIDER[2:]}")
    for key, desc in CATEGORY_META.items():
        n = counts.get(key, 0)
        share = n / total_wrong if total_wrong else 0
        print(f"  {desc:<40}  {n:>5}  {share:>6.1%}")
    print("=" * 72)


def print_examples(wrong: list[dict], n: int = 10) -> None:
    """Print one representative example per category, then fill to n."""
    by_cat: dict[str, list[dict]] = {}
    for p in wrong:
        by_cat.setdefault(p["_error_type"], []).append(p)

    # One example per category first, then round-robin to reach n
    ordered: list[dict] = []
    for key in CATEGORY_META:
        if key in by_cat:
            ordered.append(by_cat[key][0])
    i = 1
    while len(ordered) < n:
        for key in CATEGORY_META:
            if len(ordered) >= n:
                break
            if key in by_cat and i < len(by_cat[key]):
                ordered.append(by_cat[key][i])
        i += 1
        if i > max(len(v) for v in by_cat.values()):
            break

    print(f"\n{'=' * 72}")
    print(f"  Representative Examples (up to {n})")

    for idx, p in enumerate(ordered[:n], 1):
        etype = p["_error_type"]
        print(f"\n{DIVIDER}")
        print(f"  [{idx}] {CATEGORY_META[etype]}")
        print(f"  Conv : {p['conv_id']}")
        print(f"  Q    : {p['question']}")
        print(f"  Gold : {p['gold_answer']}")
        print(f"  Pred : {p['generated_answer']}")
        facts = p.get("memories_retrieved", [])
        print(f"  Facts retrieved: {len(facts)}")
        for f in facts[:4]:
            print(f"    • {f[:100]}")
        if len(facts) > 4:
            print(f"    … (+{len(facts) - 4} more)")

    print(f"\n{'=' * 72}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    with open(INPUT_PATH) as f:
        data = json.load(f)

    all_cat3 = [p for p in data["per_pair"] if p["category"] == 3]
    wrong = [p for p in all_cat3 if p["label"] == "WRONG"]

    for p in wrong:
        p["_error_type"] = categorize(p)

    counts: Counter = Counter(p["_error_type"] for p in wrong)

    print_summary(counts, total_wrong=len(wrong), total_cat3=len(all_cat3))
    print_examples(wrong, n=10)


if __name__ == "__main__":
    main()
