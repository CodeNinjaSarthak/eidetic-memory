# EIDETIC MEMORY — RESULTS SNAPSHOT
# Session: June 2026 (final, post-v2: extraction context + prompt optimization)

---

## INGESTION STATUS (all 10 conversations)
conv-26: caroline=272, melanie=264, total=536 facts
conv-30: jon=152, gina=166, total=318 facts
conv-41: john + maria ✅
conv-42: joanna + nate ✅
conv-43: tim + john ✅
conv-44: audrey + andrew ✅
conv-47: james + john ✅
conv-48: deborah + jolene ✅
conv-49: evan + sam ✅
conv-50: calvin + dave ✅
Total: 4431 turns processed, 6720 facts extracted (8 new convs)
Grand total across all 10: ~7574 facts
v2 extraction precision: 83.0% (Wilson 95% CI [74.5, 89.1])
v2 attribution accuracy: 97.0% (Wilson 95% CI [91.5, 99.0])
Speaker misattribution share of errors: 15% (down from 60% in v1)
Evolution engine accuracy: 100% on 26 hand-labeled conflict resolution cases

---

## ABLATION TABLE (conv-26 + conv-30 only, n=233)
| Run                                        | Score  | CI (95%)      |
|--------------------------------------------|--------|---------------|
| No isolation, no rerank                    | 36.5%  | [30.5, 42.9]  |
| + Isolation only                           | 26.6%  | [20.6, 32.6]  |
| + Isolation + RR, no rerank                | 25.8%  | [20.2, 31.8]  |
| + Isolation + RR + cross-encoder           | 55.8%  | [49.8, 62.2]  |
| + Isolation + score-based + cross-encoder  | 55.8%  | [49.4, 61.8]  |
| + NER + two-pass                           | 56.6%  | [50.2, 62.7]  |
| Full pipeline (all 10 convs, n=1540)       | 56.3%  | [54.0, 58.8]  |

Reranker contribution: +29.2 pp (55.8 − 26.6)
Full retrieval-structure gain: +20.1 pp (36.5 → 56.6)
Note: ablation above reflects v1 retrieval architecture only.
v1→v2 incremental gains are tracked separately below.

---

## V1 → V2 INCREMENTAL CONTRIBUTION (full benchmark, n=1540)
| Component                   | Overall | Temporal |
|-----------------------------|---------|----------|
| v1 (retrieval arch only)    | 56.3%   | 64.2%    |
| + Extraction context        | 64.6%   | 72.3%    |
| + Generation prompt fix     | 66.6%   | 75.1%    |

Extraction context covers: rolling conversation summary (every 15 turns),
recent-message window (last 10), speaker attribution via Current Speaker header.
Generation prompt fix: removed 5-6 word output cap, tightened "I don't know"
threshold, added explicit token limits (200 factual / 350 OD / 50 rephrase).
Side effect of prompt fix: two-pass firing rate dropped from 45.4% → 1.7%
of non-OD queries, reducing average LLM calls from ~1.9 → 1.02 per query.

---

## FINAL BENCHMARK RESULTS (all 10 conversations, n=1540)

### Eidetic Memory v1 vs baselines
| Category    | RAG Baseline | Pipeline v2 (no reranker) | Eidetic Memory v1 | Delta v1 vs RAG |
|-------------|-------------|--------------------------|-------------------|-----------------|
| Single-hop  | 31.9%       | 38.7%                    | 40.1%             | +8.2pp          |
| Temporal    | 24.9%       | 57.3%                    | 64.2%             | +39.3pp         |
| Multi-hop   | 22.9%       | 28.1%                    | 40.6%             | +17.7pp         |
| Open-domain | 58.4%       | 47.3%                    | 60.5%             | +2.1pp          |
| **Overall** | **44.4%**   | **46.6%**                | **56.3%**         | **+11.9pp**     |

### Eidetic Memory v2 vs baselines ← CANONICAL
| Category    | RAG Baseline | Eidetic Memory v2 | Delta v2 vs RAG |
|-------------|-------------|-------------------|-----------------|
| Single-hop  | 31.9%       | 50.4%             | +18.5pp         |
| Temporal    | 24.9%       | 75.1%             | +50.2pp         |
| Multi-hop   | 22.9%       | 51.0%             | +28.1pp         |
| Open-domain | 58.4%       | 70.5%             | +12.1pp         |
| **Overall** | **44.4%**   | **66.6%**         | **+22.2pp**     |

Bootstrap 95% CIs (v2): Overall [64.4%, 68.8%], Temporal [69.9%, 80.1%]
Significance vs RAG: p < 0.0001 (one-sided paired bootstrap, 1,000 samples)
Average LLM calls per query: 1.02

### Held-out validation — v1 (conv-47/43/48/42, n=718)
| Category    | Score     |
|-------------|-----------|
| Single-hop  | 26.6%     |
| Temporal    | 68.3%     |
| Multi-hop   | 29.2%     |
| Open-domain | 60.9%     |
| **Overall** | **55.0%** |

Full–held-out gap v1: 1.3 pp (56.3% vs 55.0%)

### Held-out validation — v2 (conv-47/43/48/42, n=718) ← CANONICAL
| Category    | Score     |
|-------------|-----------|
| Single-hop  | 44.0%     |
| Temporal    | 76.8%     |
| Multi-hop   | 39.6%     |
| Open-domain | 69.7%     |
| **Overall** | **65.2%** |

Full–held-out gap v2: 1.4 pp (66.6% vs 65.2%) — consistent with v1, no overfitting.
Temporal gain over RAG positive across all 10 conversations (+17pp to +68pp).

### Key findings (v2)
- Cross-encoder reranker: +29.2 pp on dev set (unchanged from v1)
- Temporal accuracy: +50.2 pp over RAG (24.9% → 75.1%)
- Temporal v2 (75.1%) within noise of Hindsight-20B (76.3%) at fraction of model scale
- Multi-hop v2 (51.0%) exceeds Memobase (46.9%)
- Overall v2 (66.6%) at parity with Mem0 (66.9%) at just 1.02 LLM calls/query
- Open-domain regression from v1 fully resolved (60.5% → 70.5%, above RAG)

---

## SOTA COMPARISON (LoCoMo, LLM-as-judge accuracy)
| System                              | Single-hop | Temporal | Multi-hop | Open-domain | Overall   | Spk.Iso | Rerank |
|-------------------------------------|-----------|---------|----------|------------|-----------|---------|--------|
| RAG baseline (ours)                 | 31.9%     | 24.9%   | 22.9%    | 58.4%      | 44.4%     | ✗       | ✗      |
| Pipeline v2 (ours)                  | 38.7%     | 57.3%   | 28.1%    | 47.3%      | 46.6%     | ✓       | ✗      |
| Eidetic Memory v1                   | 40.1%     | 64.2%   | 40.6%    | 60.5%      | 56.3%     | ✓       | ✓      |
| **Eidetic Memory v2 ← CANONICAL**   | **50.4%** |**75.1%**| **51.0%**| **70.5%**  | **66.6%** | **✓**   | **✓**  |
| Mem0                                | —         | —       | —        | —          | 66.9%     | ✗       | ✗      |
| Memobase                            | 70.92%    | 85.05%  | 46.88%   | 77.17%     | 75.78%    | ✗       | ✗      |
| Hindsight (OSS-20B)                 | 74.11%    | 76.32%  | 64.58%   | 90.96%     | 83.18%    | ✗       | ✓      |
| Hindsight (OSS-120B)                | 76.79%    | 79.44%  | 62.50%   | 93.68%     | 85.67%    | ✗       | ✓      |

Gap to Hindsight-120B: 19.1 pp overall (down from 29.4 pp in v1).
Note: Mem0 2026 algorithm (92.5%) and ByteRover 2.0 (92.2%) use different judge
models and harnesses — not directly comparable without controlled re-evaluation.

---

## REPO STATE
GitHub: CodeNinjaSarthak/eidetic-memory
Branch: main (v1) / improvement/week2 (v2 — all v2 results from this branch)
Tests: 152 passing

## ENV FOR BENCHMARK RUNS
LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-small

## OUTPUT FILES
# v1 (retrieval architecture)
- eval/results/final_eidetic_memory_local_reranker_m3.json  — v1 canonical (56.3%, n=1540)
- eval/results/heldout_local_reranker_m3.json               — v1 held-out (55.0%, n=718)
- eval/results/qa_accuracy_all10.json                       — pipeline v1, all 10 convs
- eval/results/qa_accuracy_all10_v2.json                    — pipeline v2 (category gate)
- eval/results/qa_rag_baseline_all10.json                   — RAG baseline (44.4%)
- eval/results/qa_rag_baseline_conv26.json                  — smoke test, conv-26 only
# v2 (extraction context + prompt fix)
- eval/results/improvement_week2_full_v1.json               — v2 extraction only (64.6%)
- eval/results/prompt_fix_full_v1.json                      — v2 CANONICAL (66.6%, n=1540)
- eval/results/heldout_prompt_fix.json                      — v2 held-out (65.2%, n=718)
- eval/results/prompt_fix_dev.json                          — v2 dev set (69.5%, n=233)

## PAPER CORE CLAIM (final, v2)
"Per-speaker memory isolation with rolling extraction context and cross-encoder
reranking (cross-encoder/ms-marco-MiniLM-L-6-v2, 66M parameters, local CPU,
no API key) achieves 66.6% overall accuracy and 75.1% on temporal reasoning
on LoCoMo (n=1540) — a +50.2 pp gain over RAG on temporal queries and parity
with Mem0 (66.9%). The system averages 1.02 LLM calls per query (two-pass
fires on 1.7% of non-open-domain queries), making it the most call-efficient
system above 65% accuracy with a published call count."