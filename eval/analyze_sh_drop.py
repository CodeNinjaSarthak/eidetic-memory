"""Analyze the single-hop accuracy drop between dev (10 convs) and held-out (4 convs).

Metrics computed per split:
  - Accuracy
  - Avg question length (tokens)
  - Avg retrieved facts that contain the gold answer
  - Proportion of questions where answer IS in context (retrieval OK)
  - Proportion of those where generated answer is WRONG (reasoning failure)
  - Proportion where answer is NOT in context at all (retrieval failure)
  - Proportion of multi-entity questions (ask about two people)
  - Proportion of numeric/date gold answers

Also isolates the "apples-to-apples" comparison: dev accuracy restricted to
the 4 held-out conv IDs, to separate selection effect from run-to-run variance.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path(__file__).parent / "results"
DEV_PATH = RESULTS_DIR / "final_eidetic_memory_all10.json"
HELD_PATH = RESULTS_DIR / "heldout_4convs_eidetic_memory.json"

HELD_CONV_IDS = {"conv-42", "conv-43", "conv-47", "conv-48"}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "but", "or",
    "not", "no", "yes", "if", "as", "it", "its", "this", "that",
    "they", "their", "them", "she", "he", "his", "her", "we",
    "you", "i", "me", "who", "what", "when", "where", "how",
    "which", "both", "also",
}

_NUMERIC_RE = re.compile(r"\b\d{4}\b|\b\d+\b")
_MULTI_ENTITY_RE = re.compile(
    r"\bboth\b|\b(?:and|&)\b.{0,30}\b(?:and|&)\b"
    r"|\b[A-Z][a-z]+\b.{0,40}\band\b.{0,40}\b[A-Z][a-z]+\b",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_keywords(text: str) -> list[str]:
    return [
        w.lower().strip(".,;:!?()'\"")
        for w in text.split()
        if len(w) > 2 and w.lower() not in STOP_WORDS
    ]


def gold_in_facts(gold: str, facts: list[str]) -> bool:
    """True if at least half of meaningful gold keywords appear in the facts."""
    kws = extract_keywords(str(gold))
    if not kws:
        return False
    blob = " ".join(facts).lower()
    hits = sum(1 for kw in kws if kw in blob)
    return hits >= max(1, len(kws) // 2)


def facts_containing_gold(gold: str, facts: list[str]) -> int:
    """Count individual retrieved facts that contain at least one gold keyword."""
    kws = set(extract_keywords(str(gold)))
    if not kws:
        return 0
    return sum(
        1 for f in facts
        if any(kw in f.lower() for kw in kws)
    )


def is_numeric_gold(gold: str) -> bool:
    return bool(_NUMERIC_RE.search(str(gold)))


def is_multi_entity_question(question: str) -> bool:
    """True if question names two distinct people or uses 'both'."""
    if " both " in question.lower():
        return True
    # Two or more capitalized names in the question
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
    proper = [n for n in names if n not in {"What", "Which", "Where", "When", "Who", "How", "Does", "Did", "Is", "Are", "Have", "Has", "Do"}]
    return len(set(proper)) >= 2


# ── Core metrics ───────────────────────────────────────────────────────────

def compute_metrics(pairs: list[dict]) -> dict:
    """Return a metrics dict for a list of single-hop QA pairs."""
    n = len(pairs)
    if n == 0:
        return {}

    correct = [p for p in pairs if p["label"] == "CORRECT"]
    wrong   = [p for p in pairs if p["label"] == "WRONG"]

    # Retrieval quality
    gold_present       = [p for p in pairs if gold_in_facts(str(p["gold_answer"]), p.get("memories_retrieved", []))]
    gold_absent        = [p for p in pairs if not gold_in_facts(str(p["gold_answer"]), p.get("memories_retrieved", []))]

    wrong_with_gold    = [p for p in wrong  if gold_in_facts(str(p["gold_answer"]), p.get("memories_retrieved", []))]
    wrong_without_gold = [p for p in wrong  if not gold_in_facts(str(p["gold_answer"]), p.get("memories_retrieved", []))]

    avg_facts_with_gold = mean(
        facts_containing_gold(str(p["gold_answer"]), p.get("memories_retrieved", []))
        for p in pairs
    )

    return {
        "n": n,
        "accuracy": len(correct) / n,
        "avg_q_len_tokens": mean(len(p["question"].split()) for p in pairs),
        "avg_facts_containing_gold": avg_facts_with_gold,
        "pct_gold_in_context": len(gold_present) / n,
        "pct_gold_absent": len(gold_absent) / n,
        # Of wrong predictions
        "pct_wrong_reasoning_fail": len(wrong_with_gold) / len(wrong) if wrong else 0,
        "pct_wrong_retrieval_fail": len(wrong_without_gold) / len(wrong) if wrong else 0,
        # Question hardness signals
        "pct_multi_entity_q": sum(1 for p in pairs if is_multi_entity_question(p["question"])) / n,
        "pct_numeric_gold": sum(1 for p in pairs if is_numeric_gold(str(p["gold_answer"]))) / n,
    }


# ── Per-conv breakdown ─────────────────────────────────────────────────────

def conv_accuracy(pairs: list[dict], conv_id: str) -> tuple[int, int]:
    subset = [p for p in pairs if p["conv_id"] == conv_id]
    correct = sum(1 for p in subset if p["label"] == "CORRECT")
    return correct, len(subset)


# ── Formatting ─────────────────────────────────────────────────────────────

DIVIDER = "─" * 72


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def fmt_f(v: float) -> str:
    return f"{v:.2f}"


def print_comparison(
    dev_all: dict,
    dev_4convs: dict,
    held: dict,
) -> None:
    rows = [
        ("Questions (n)",           "n",                      str,     False),
        ("Accuracy",                "accuracy",               fmt_pct, True),
        ("Avg question length",     "avg_q_len_tokens",       fmt_f,   True),
        ("Avg facts with gold kw",  "avg_facts_containing_gold", fmt_f, True),
        ("Gold in context",         "pct_gold_in_context",    fmt_pct, True),
        ("Gold absent (retr. fail)","pct_gold_absent",        fmt_pct, True),
        ("Wrong → reasoning fail",  "pct_wrong_reasoning_fail", fmt_pct, True),
        ("Wrong → retrieval fail",  "pct_wrong_retrieval_fail", fmt_pct, True),
        ("Multi-entity questions",  "pct_multi_entity_q",     fmt_pct, True),
        ("Numeric/date gold answer","pct_numeric_gold",        fmt_pct, True),
    ]

    col1 = 30
    col2 = 14

    print()
    print("=" * 72)
    print("  Single-hop Accuracy Drop: Dev vs Held-out")
    print("=" * 72)
    print(f"  {'Metric':<{col1}}  {'Dev (all 10)':>{col2}}  {'Dev (4 convs)':>{col2}}  {'Held-out':>{col2}}")
    print(f"  {DIVIDER[2:]}")

    for label, key, fmt, _flag in rows:
        va = fmt(dev_all[key])   if key in dev_all   else "—"
        vb = fmt(dev_4convs[key]) if key in dev_4convs else "—"
        vc = fmt(held[key])      if key in held      else "—"
        print(f"  {label:<{col1}}  {va:>{col2}}  {vb:>{col2}}  {vc:>{col2}}")

    print("=" * 72)


def print_per_conv(
    dev_pairs: list[dict],
    held_pairs: list[dict],
) -> None:
    print()
    print(f"{'─' * 72}")
    print("  Per-conversation accuracy (single-hop only)")
    print(f"  {'Conv':<10}  {'Dev acc':>8}  {'Dev n':>6}  {'Held acc':>9}  {'Held n':>7}")
    print(f"  {'─' * 60}")
    all_convs = sorted(HELD_CONV_IDS)
    for cid in all_convs:
        dc, dn = conv_accuracy(dev_pairs, cid)
        hc, hn = conv_accuracy(held_pairs, cid)
        dev_str  = f"{dc/dn:.1%}" if dn else "—"
        held_str = f"{hc/hn:.1%}" if hn else "—"
        print(f"  {cid:<10}  {dev_str:>8}  {dn:>6}  {held_str:>9}  {hn:>7}")
    print(f"{'─' * 72}")


def print_question_examples(dev_pairs: list[dict], held_pairs: list[dict]) -> None:
    """Show examples where the same question has different outcomes."""
    dev_map  = {(p["conv_id"], p["question"]): p for p in dev_pairs}
    held_map = {(p["conv_id"], p["question"]): p for p in held_pairs}

    flipped = [
        (k, dev_map[k], held_map[k])
        for k in held_map
        if k in dev_map and dev_map[k]["label"] != held_map[k]["label"]
    ]

    if not flipped:
        print("\n  No questions with different labels between runs.\n")
        return

    print(f"\n{'─' * 72}")
    print(f"  Questions with different labels across runs ({len(flipped)} total)")
    print(f"  (same question, same conv, different generation outcome)")
    for k, dp, hp in flipped[:6]:
        transition = f"{dp['label']} → {hp['label']}"
        print(f"\n  [{transition}]  {k[0]}")
        print(f"  Q    : {k[1]}")
        print(f"  Gold : {dp['gold_answer']}")
        print(f"  Dev  : {dp['generated_answer'][:80]}")
        print(f"  Held : {hp['generated_answer'][:80]}")
    print(f"\n{'─' * 72}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    with open(DEV_PATH) as f:
        dev_data = json.load(f)
    with open(HELD_PATH) as f:
        held_data = json.load(f)

    dev_sh  = [p for p in dev_data["per_pair"]  if p["category"] == 1]
    held_sh = [p for p in held_data["per_pair"] if p["category"] == 1]

    # Apples-to-apples: dev restricted to the same 4 conv IDs
    dev_sh_4convs = [p for p in dev_sh if p["conv_id"] in HELD_CONV_IDS]

    metrics_dev_all    = compute_metrics(dev_sh)
    metrics_dev_4convs = compute_metrics(dev_sh_4convs)
    metrics_held       = compute_metrics(held_sh)

    print_comparison(metrics_dev_all, metrics_dev_4convs, metrics_held)
    print_per_conv(dev_sh, held_sh)
    print_question_examples(dev_sh, held_sh)

    # Verdict
    dev_4_acc  = metrics_dev_4convs["accuracy"]
    held_acc   = metrics_held["accuracy"]
    selection  = metrics_dev_all["accuracy"] - dev_4_acc
    run_var    = dev_4_acc - held_acc

    print()
    print("  Diagnosis")
    print(f"  {'─' * 60}")
    print(f"  Dev (all 10) → Dev (4 held convs)  :  {metrics_dev_all['accuracy']:.1%} → {dev_4_acc:.1%}"
          f"  ({selection:+.1%} selection effect)")
    print(f"  Dev (4 convs) → Held-out (same qs) :  {dev_4_acc:.1%} → {held_acc:.1%}"
          f"  ({run_var:+.1%} run-to-run variance)")
    print()
    dominant = "selection" if abs(selection) > abs(run_var) else "run-to-run variance"
    print(f"  Dominant cause of SH drop: {dominant}")
    print()


if __name__ == "__main__":
    main()
