#!/bin/bash
set -o pipefail

echo "Starting three ablation runs in parallel..."
echo "Logs: eval/logs/ablation_run_[1,2,3].log"
mkdir -p eval/logs eval/results

# RUN 1: GPT-4.1, no isolation, no rerank
# This is the new W2 baseline row for Table 3
uv run python eval/eval_qa_accuracy.py \
  --conv-ids conv-26 conv-30 \
  --no-isolation \
  --no-rerank \
  --concurrency 8 \
  --output eval/results/ablation_gpt41_no_isolation_no_rerank.json \
  > eval/logs/ablation_run_1.log 2>&1 &
PID1=$!
echo "Run 1 (no-isolation, no-rerank): PID $PID1"

# RUN 2: GPT-4.1, per-speaker isolation, no rerank
# Reconstructs the 31.8% row with correct model+convs
uv run python eval/eval_qa_accuracy.py \
  --conv-ids conv-26 conv-30 \
  --no-rerank \
  --concurrency 8 \
  --output eval/results/ablation_gpt41_isolation_no_rerank.json \
  > eval/logs/ablation_run_2.log 2>&1 &
PID2=$!
echo "Run 2 (isolation, no-rerank): PID $PID2"

# RUN 3: GPT-4.1, per-speaker isolation, WITH rerank, conv-26+30 only
# Reconstructs the final model on the same subset for apples-to-apples
uv run python eval/eval_qa_accuracy.py \
  --conv-ids conv-26 conv-30 \
  --concurrency 8 \
  --output eval/results/ablation_gpt41_isolation_rerank_2convs.json \
  > eval/logs/ablation_run_3.log 2>&1 &
PID3=$!
echo "Run 3 (isolation, rerank, 2 convs): PID $PID3"

echo ""
echo "All three running. Monitor with:"
echo "  tail -f eval/logs/ablation_run_1.log"
echo "  tail -f eval/logs/ablation_run_2.log"
echo "  tail -f eval/logs/ablation_run_3.log"
echo ""
echo "Waiting for all to complete..."
# Wait for all jobs, collect exit codes independently
FAILED=0
wait $PID1
STATUS1=$?
[ $STATUS1 -eq 0 ] && echo "Run 1 DONE" || { echo "Run 1 FAILED (exit $STATUS1)"; FAILED=1; }

wait $PID2
STATUS2=$?
[ $STATUS2 -eq 0 ] && echo "Run 2 DONE" || { echo "Run 2 FAILED (exit $STATUS2)"; FAILED=1; }

wait $PID3
STATUS3=$?
[ $STATUS3 -eq 0 ] && echo "Run 3 DONE" || { echo "Run 3 FAILED (exit $STATUS3)"; FAILED=1; }

if [ $FAILED -eq 1 ]; then
  echo ""
  echo "WARNING: One or more runs failed. Check logs in eval/logs/"
  echo "Partial results are saved — re-run the failed job to resume."
fi

echo ""
echo "Extracting results..."
for f in \
  eval/results/ablation_gpt41_no_isolation_no_rerank.json \
  eval/results/ablation_gpt41_isolation_no_rerank.json \
  eval/results/ablation_gpt41_isolation_rerank_2convs.json; do
  if [ -f "$f" ]; then
    python3 -c "
import json
d = json.load(open('$f'))
print(f'$f')
print(f'  overall: {d[\"overall\"][\"accuracy\"]:.1%} ({d[\"overall\"][\"correct\"]}/{d[\"overall\"][\"total\"]})')
meta = d.get(\"metadata\", {})
print(f'  no_isolation={meta.get(\"no_isolation\")}, no_rerank={meta.get(\"no_rerank\")}')
cats = d.get(\"by_category\", {})
for k,v in cats.items():
    print(f'  {v[\"label\"]}: {v[\"accuracy\"]:.1%} ({v[\"total\"]} pairs)')
print()
"
  else
    echo "$f — NOT FOUND"
  fi
done
