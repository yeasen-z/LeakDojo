#!/bin/bash
# LeakDojo - Evaluation Examples
# Usage: bash scripts/run_evaluate.sh <RESULT_JSONL_PATH>

RESULT_PATH="${1:?Usage: bash scripts/run_evaluate.sh <RESULT_JSONL_PATH>}"

echo "===== Evaluating: ${RESULT_PATH} ====="
python evaluate.py "${RESULT_PATH}" --num_records 200
