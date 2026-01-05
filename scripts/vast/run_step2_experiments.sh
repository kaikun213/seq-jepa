#!/usr/bin/env bash
# Run Step 2 experiments: Exp A (EMA+Rate) and Exp B (Teacherless)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DONE_PATH="${RUN_DONE_PATH:-$REPO_ROOT/runs/vast_step2.done}"
RUN_FAIL_PATH="${RUN_FAIL_PATH:-$REPO_ROOT/runs/vast_step2.failed}"

cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "========================================"
echo "Step 2 - Exp A: EMA + Rate Loss"
echo "========================================"
set +e
python train.py --config configs/remote/step2_exp_a_ema_rate.yaml
exp_a_exit=$?
set -e

if [[ $exp_a_exit -ne 0 ]]; then
  echo "Exp A failed with exit code $exp_a_exit"
  echo "exp_a_exit=$exp_a_exit" > "$RUN_FAIL_PATH"
  exit "$exp_a_exit"
fi

echo "========================================"
echo "Step 2 - Exp B: Teacherless + Rate"
echo "========================================"
set +e
python train.py --config configs/remote/step2_exp_b_teacherless.yaml
exp_b_exit=$?
set -e

if [[ $exp_b_exit -ne 0 ]]; then
  echo "Exp B failed with exit code $exp_b_exit"
  echo "exp_b_exit=$exp_b_exit" > "$RUN_FAIL_PATH"
  exit "$exp_b_exit"
fi

echo "========================================"
echo "Step 2 experiments completed successfully"
echo "========================================"
touch "$RUN_DONE_PATH"

