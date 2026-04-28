# EIDETIC MEMORY — RESULTS SNAPSHOT
# Session: April 2026 (final, post-reranker)

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
| + Full pipeline (all 10 convs, n=1540)     | 56.3%  | [53.8, 58.7]* |

*placeholder CI — recompute with `eval/bootstrap_ci.py` before paper submission

Reranker contribution: +29.2 pp (55.8 − 26.6)
Full retrieval-structure gain: +20.1 pp (36.5 → 56.6)

---

## FINAL BENCHMARK RESULTS (all 10 conversations, n=1540)

### Eidetic Memory vs baselines
| Category    | RAG Baseline | Pipeline v2 (no reranker) | Eidetic Memory (cross-encoder) | Delta Eidetic vs RAG |
|-------------|-------------|--------------------------|-------------------------------|----------------------|
| Single-hop  | 31.9%       | 38.7%                    | 40.1%                         | +8.2pp               |
| Temporal    | 24.9%       | 57.3%                    | 64.2%                         | +39.3pp              |
| Multi-hop   | 22.9%       | 28.1%                    | 40.6%                         | +17.7pp              |
| Open-domain | 58.4%       | 47.3%                    | 60.5%                         | +2.1pp               |
| **Overall** | **44.4%**   | **46.6%**                | **56.3%**                     | **+11.9pp**          |

### Held-out validation (conv-47/43/48/42, n=718)
| Category    | Score     |
|-------------|-----------|
| Single-hop  | 26.6%     |
| Temporal    | 68.3%     |
| Multi-hop   | 29.2%     |
| Open-domain | 60.9%     |
| **Overall** | **55.0%** |

### Key findings
- Cross-encoder reranker is the load-bearing component: +29.2 pp on dev set (55.8 vs 26.6)
- Temporal accuracy: +39.3 pp over RAG (64.2% vs 24.9%); +6.9 pp over Pipeline v2
- Open-domain recovers with reranker: +13.2 pp over Pipeline v2 (60.5% vs 47.3%)
- Held-out overall (55.0%) consistent with dev set (56.3%) — no dev-set overfitting

---

## SOTA COMPARISON (LoCoMo, LLM-as-judge accuracy)
| System                              | Single-hop | Temporal | Multi-hop | Open-domain | Overall   |
|-------------------------------------|-----------|---------|----------|------------|-----------|
| RAG baseline (ours)                 | 31.9%     | 24.9%   | 22.9%    | 58.4%      | 44.4%     |
| Pipeline v2 (ours)                  | 38.7%     | 57.3%   | 28.1%    | 47.3%      | 46.6%     |
| Eidetic Memory (with cross-encoder) | 40.1%     | 64.2%   | 40.6%    | 60.5%      | **56.3%** |
| Mem0                                | —         | —       | —        | —          | ~66.9%    |
| Memobase                            | 70.92%    | 85.05%  | 46.88%   | 77.17%     | 75.78%    |
| Hindsight (OSS-20B)                 | 74.11%    | 76.32%  | 64.58%   | 90.96%     | 83.18%    |
| Hindsight (OSS-120B)                | 76.79%    | 79.44%  | 62.50%   | 93.68%     | 85.67%    |

Gap to Hindsight-120B: 29.4 pp overall.

---

## REPO STATE
GitHub: CodeNinjaSarthak/eidetic-memory
Branch: main
Tests: 152 passing

## ENV FOR BENCHMARK RUNS
LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-small

## OUTPUT FILES
- eval/results/final_eidetic_memory_local_reranker_m3.json  — canonical result (Eidetic Memory, cross-encoder, n=1540)
- eval/results/heldout_local_reranker_m3.json               — held-out result (conv-47/43/48/42, n=718)
- eval/results/qa_accuracy_all10.json                       — pipeline v1, all 10 convs
- eval/results/qa_accuracy_all10_v2.json                    — pipeline v2 (category gate), all 10 convs
- eval/results/qa_rag_baseline_all10.json                   — RAG baseline, all 10 convs
- eval/results/qa_rag_baseline_conv26.json                  — smoke test, conv-26 only

## PAPER CORE CLAIM (final)
"Per-speaker memory isolation combined with cross-encoder reranking
(cross-encoder/ms-marco-MiniLM-L-6-v2, 66M parameters, local CPU, no API key)
achieves 56.3% overall accuracy and 64.2% on temporal reasoning on LoCoMo
(n=1540) — a +39.3 pp gain over RAG on temporal queries. The system averages
1.9 LLM calls per query via selective two-pass retrieval (fires on 45.4% of
non-open-domain queries), making it practical at inference time."
