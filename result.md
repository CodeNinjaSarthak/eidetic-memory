# EIDETIC MEMORY — RESULTS SNAPSHOT
# Session: March 26, 2026 (updated end of session)

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
| mem0 paper                       | ~70%   | Target ceiling                 |

Round-robin merge = single largest lever (+20pp)

---

## FINAL BENCHMARK RESULTS (all 10 conversations, n=1540)

### Pipeline vs RAG Baseline
| Category    | RAG Baseline | Pipeline | Delta    |
|-------------|-------------|----------|----------|
| Single-hop  | 31.9%       | 36.5%    | +4.6pp   |
| Temporal    | 24.9%       | 56.4%    | +31.5pp  |
| Multi-hop   | 22.9%       | 29.2%    | +6.3pp   |
| Open-domain | 58.4%       | 48.2%    | -10.2pp  |
| **Overall** | **44.4%**   | **46.6%**| **+2.2pp**|

### RAG Baseline detail (all 10, n=1540)
Overall: 44.4% (683/1540 CORRECT)
Single-hop:  31.9% (282 pairs)
Temporal:    24.9% (321 pairs)
Multi-hop:   22.9% (96 pairs)
Open-domain: 58.4% (841 pairs)

### Pipeline detail (all 10, n=1540)
Overall: 46.6% (717/1540 CORRECT)
Single-hop:  36.5% (282 pairs)
Temporal:    56.4% (321 pairs)
Multi-hop:   29.2% (96 pairs)
Open-domain: 48.2% (841 pairs)

### RAG Baseline (conv-26 only, n=152) — smoke test
Overall: 39.5% (60/152 CORRECT)
Single-hop:  28.1% (32 pairs)
Temporal:    32.4% (37 pairs)
Multi-hop:   23.1% (13 pairs)
Open-domain: 51.4% (70 pairs)

Retrieval: Hit@20=56% (n=100, locomo_eval collection, Azure embeddings)

---

## KEY FINDINGS
1. Temporal is the dominant win: +31.5pp — pipeline's strongest signal
2. Open-domain regresses: -10.2pp — two-pass hurts easy questions
3. Overall modest at +2.2pp because open-domain is 55% of all pairs (841/1540)
4. Fix needed: per-category gate — disable two-pass for open-domain questions
5. Without the open-domain regression, estimated overall would be ~52-53%

---

## OPEN ISSUES
- Batch UPDATE without id (second code path, non-integer None) — still fires,
  visible in logs as "Batch UPDATE without id, falling back to NOOP"
- Open-domain regression (-10.2pp) from two-pass — per-category gate needed
  (don't trigger second pass for open-domain / category 4 questions)

---

## REPO STATE
GitHub: CodeNinjaSarthak/eidetic-memory
Branch: dev
Checkpoint file: eval/checkpoints/ingestion_progress.json
  completed: [conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50]
Tests: 142 passing + 10 new regression tests = 152 total

---

## ENV FOR BENCHMARK RUNS
LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-small

---

## NEXT STEPS (priority order)
1. Fix open-domain regression — add per-category gate in eval_qa_accuracy.py:
   skip two-pass if category == 4 (open-domain)
   Expected impact: overall ~52-53%, open-domain back to ~58%
2. Run confidence intervals — 3-5 runs with different seeds (~$8)
3. Build paper in Overleaf using ACL/EMNLP 2026 template
4. Target: EMNLP 2026 workshops (check ACL 2026 deadline first)

---

## PAPER CORE CLAIM (updated)
"Per-speaker memory isolation with dual-namespace round-robin retrieval
recovers +31.5pp on temporal reasoning in multi-party conversation memory
systems on the LoCoMo benchmark, with overall +2.2pp over RAG baseline.
Two-pass retrieval improves hard categories (temporal, multi-hop) but
regresses on open-domain questions, suggesting a per-category retrieval
gate as future work."

---

## FILES CREATED THIS SESSION
- eval/ingest_locomo_production.py (checkpoint + parallelism + retry + LLMError fix)
- eval/eval_qa_rag_baseline.py (RAG baseline + resume + content filter)
- eval/eval_qa_accuracy.py (resume + concurrency=5 + content filter)
- backend/services/llm/src/llm/generation/azure.py (60s timeout)
- backend/services/memory/tests/test_update.py (10 regression tests)
- eval/checkpoints/ (gitignored)

---

## OUTPUT FILES
- eval/results/qa_accuracy_all10.json — pipeline eval, all 10 convs
- eval/results/qa_rag_baseline_all10.json — RAG baseline, all 10 convs
- eval/results/qa_rag_baseline_conv26.json — smoke test, conv-26 only