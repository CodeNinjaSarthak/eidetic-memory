"""Bootstrap 95% CI and paired significance test for eval results."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

RESULTS_DIR = Path(__file__).parent / "results"

EIDETIC_PATH = RESULTS_DIR / "final_eidetic_memory_all10.json"
RAG_PATH = RESULTS_DIR / "rag_baseline_all10.json"
HELDOUT_PATH = RESULTS_DIR / "heldout_4convs_eidetic_memory.json"

N_BOOTSTRAP = 1000
SEED = 42

CATEGORY_LABELS = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain"}


class BootstrapResult(NamedTuple):
    mean: float
    ci_lo: float
    ci_hi: float
    n: int


def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def accuracy(items: list[dict]) -> float:
    if not items:
        return 0.0
    return sum(1 for p in items if p["label"] == "CORRECT") / len(items)


def bootstrap_ci(items: list[dict], rng: random.Random) -> BootstrapResult:
    n = len(items)
    boot_accs = [
        accuracy(rng.choices(items, k=n)) for _ in range(N_BOOTSTRAP)
    ]
    boot_accs.sort()
    lo = boot_accs[int(0.025 * N_BOOTSTRAP)]
    hi = boot_accs[int(0.975 * N_BOOTSTRAP)]
    return BootstrapResult(mean=accuracy(items), ci_lo=lo, ci_hi=hi, n=n)


def bootstrap_ci_by_category(
    items: list[dict], rng: random.Random
) -> dict[int, BootstrapResult]:
    by_cat: dict[int, list[dict]] = {}
    for p in items:
        by_cat.setdefault(p["category"], []).append(p)
    return {cat: bootstrap_ci(cat_items, rng) for cat, cat_items in sorted(by_cat.items())}


def paired_bootstrap_p_value(
    a_items: list[dict],
    b_items: list[dict],
    rng: random.Random,
) -> tuple[float, float]:
    """Paired bootstrap test: H0 = accuracy(A) <= accuracy(B).

    Matches items by (conv_id, question). Returns (observed_diff, p_value)
    where p_value is the fraction of bootstrap samples where diff <= 0.
    """
    key = lambda p: (p["conv_id"], p["question"])
    a_map = {key(p): p for p in a_items}
    b_map = {key(p): p for p in b_items}

    shared_keys = list(a_map.keys() & b_map.keys())
    paired = [(a_map[k], b_map[k]) for k in shared_keys]

    observed_diff = accuracy([a for a, _ in paired]) - accuracy([b for _, b in paired])

    n = len(paired)
    count_no_effect = 0
    for _ in range(N_BOOTSTRAP):
        sample = rng.choices(paired, k=n)
        diff = accuracy([a for a, _ in sample]) - accuracy([b for _, b in sample])
        if diff <= 0:
            count_no_effect += 1

    p_value = count_no_effect / N_BOOTSTRAP
    return observed_diff, p_value


def fmt(r: BootstrapResult) -> str:
    return f"{r.mean:.1%} ± [{r.ci_lo:.1%}, {r.ci_hi:.1%}]  (n={r.n})"


def print_system_table(label: str, items: list[dict], rng: random.Random) -> None:
    overall = bootstrap_ci(items, rng)
    by_cat = bootstrap_ci_by_category(items, rng)

    print(f"\n{'─' * 62}")
    print(f"  {label}")
    print(f"{'─' * 62}")
    print(f"  {'Overall':<14}  {fmt(overall)}")
    for cat_id, cat_label in CATEGORY_LABELS.items():
        r = by_cat.get(cat_id)
        if r is None:
            continue
        print(f"  {cat_label:<14}  {fmt(r)}")


def main() -> None:
    rng = random.Random(SEED)

    eidetic = load_result(EIDETIC_PATH)
    rag = load_result(RAG_PATH)
    heldout = load_result(HELDOUT_PATH)

    print("\n" + "=" * 62)
    print("  Bootstrap 95% CI  (1000 samples, seed=42)")
    print("=" * 62)

    print_system_table("Eidetic Memory — all 10 convs", eidetic["per_pair"], rng)
    print_system_table("RAG Baseline    — all 10 convs", rag["per_pair"], rng)
    print_system_table("Eidetic Memory — held-out 4 convs", heldout["per_pair"], rng)

    # Paired significance test (eidetic vs RAG, shared 10-conv set)
    obs_diff, p_val = paired_bootstrap_p_value(
        eidetic["per_pair"], rag["per_pair"], rng
    )
    print(f"\n{'=' * 62}")
    print("  Paired Bootstrap Significance Test")
    print(f"  Eidetic Memory vs RAG Baseline (10-conv set)")
    print(f"{'─' * 62}")
    print(f"  Observed accuracy difference : {obs_diff:+.1%}")
    print(f"  H0: diff ≤ 0  (Eidetic is not better than RAG)")
    print(f"  p-value (one-sided)          : {p_val:.4f}")
    significance = "significant" if p_val < 0.05 else "NOT significant"
    print(f"  Result (α=0.05)              : {significance}")
    print("=" * 62)


if __name__ == "__main__":
    main()
