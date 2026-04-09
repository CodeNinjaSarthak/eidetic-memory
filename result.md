# EIDETIC MEMORY — RESULTS SNAPSHOT
# Session: March 26, 2026 (final update)

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
| Run                              | Score  | Details                        |
|----------------------------------|--------|--------------------------------|
| Baseline                         | 14.8%  | Gemini, conv-30 only, n=81     |
| + Per-speaker isolation          | 31.8%  | Multi-namespace retrieval      |
| + Fixed merge (round-robin)      | 52.4%  | zip_longest interleaving       |
| + Named entities + two-pass      | 53.2%  | Current best, n=233            |
| mem0 paper                       | ~70%   | Reference                      |
| Hindsight (OSS-120B)             | 85.67% | SOTA open-source               |
| Hindsight (Gemini-3)             | 89.61% | SOTA overall                   |

Round-robin merge = single largest lever (+20pp)

---

## FINAL BENCHMARK RESULTS (all 10 conversations, n=1540)

### Pipeline v1 vs v2 vs RAG Baseline
| Category    | RAG Baseline | Pipeline v1 | Pipeline v2 (cat≠4 gate) | Delta v2 vs RAG |
|-------------|-------------|------------|--------------------------|-----------------|
| Single-hop  | 31.9%       | 36.5%      | 38.7%                    | +6.8pp          |
| Temporal    | 24.9%       | 56.4%      | 57.3%                    | +32.4pp         |
| Multi-hop   | 22.9%       | 29.2%      | 28.1%                    | +5.2pp          |
| Open-domain | 58.4%       | 48.2%      | 47.3%                    | -11.1pp         |
| **Overall** | **44.4%**   | **46.6%**  | **46.6%**                | **+2.2pp**      |

### Key finding
Open-domain regression is NOT caused by two-pass retrieval.
It is a fundamental extraction pipeline tradeoff:
- Extracted facts excel at structured/factual recall (+32.4pp temporal)
- Extracted facts lose conversational breadth needed for open-domain (-11.1pp)
- The system is a specialist, not a generalist — this is the paper's nuanced claim

---

## SOTA COMPARISON (LoCoMo, LLM-as-judge accuracy)
| System              | Single-hop | Temporal | Multi-hop | Open-domain | Overall |
|---------------------|-----------|---------|----------|------------|---------|
| RAG baseline (ours) | 31.9%     | 24.9%   | 22.9%    | 58.4%      | 44.4%   |
| Pipeline v2 (ours)  | 38.7%     | 57.3%   | 28.1%    | 47.3%      | 46.6%   |
| Mem0                | —         | —       | —        | —          | ~66.9%  |
| Memobase            | 70.92%    | 85.05%  | 46.88%   | 77.17%     | 75.78%  |
| Hindsight (OSS-20B) | 74.11%    | 76.32%  | 64.58%   | 90.96%     | 83.18%  |
| Hindsight (OSS-120B)| 76.79%    | 79.44%  | 62.50%   | 93.68%     | 85.67%  |

---

## OPEN ISSUES
- Open-domain regression (-11.1pp vs RAG) — architectural, not fixable by gating
- Batch UPDATE without id (second code path) — still fires as NOOP, low priority

---

## NEXT STEPS (priority order)
1. Add BM25 + RRF merge to retrieval (~1 day, ~+5pp)
2. Add cross-encoder reranker (~2 hours, ~+3pp)
3. Add temporal date filtering (~1 day, ~+3pp temporal)
4. Test each on --limit 100 before full run to save tokens
Expected after all 3: ~55-60% overall

---

## REPO STATE
GitHub: CodeNinjaSarthak/eidetic-memory
Branch: dev
Tests: 152 passing (142 original + 10 new regression tests)

## ENV FOR BENCHMARK RUNS
LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-small

## OUTPUT FILES
- eval/results/qa_accuracy_all10.json       — pipeline v1, all 10 convs
- eval/results/qa_accuracy_all10_v2.json    — pipeline v2 (category gate), all 10 convs
- eval/results/qa_rag_baseline_all10.json   — RAG baseline, all 10 convs
- eval/results/qa_rag_baseline_conv26.json  — smoke test, conv-26 only

## FILES CREATED/MODIFIED THIS SESSION
- eval/ingest_locomo_production.py (checkpoint + parallelism + retry + LLMError + timeout)
- eval/eval_qa_rag_baseline.py (RAG baseline + resume + content filter)
- eval/eval_qa_accuracy.py (resume + concurrency=5 + content filter + category gate)
- backend/services/llm/src/llm/generation/azure.py (60s timeout)
- backend/services/memory/tests/test_update.py (10 regression tests)
- eval/checkpoints/ (gitignored)
- result.md (this file)

## PAPER CORE CLAIM (revised)
"Per-speaker memory isolation with dual-namespace round-robin retrieval
recovers +32.4pp on temporal reasoning and +6.8pp on single-hop QA in
multi-party conversation memory systems on LoCoMo (n=1540). The extraction
pipeline trades open-domain breadth for structured fact precision, suggesting
memory extraction systems are best suited as specialists for factual and
temporal reasoning tasks."