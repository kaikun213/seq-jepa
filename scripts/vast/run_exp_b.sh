#!/usr/bin/env bash
# Run Step 2 Exp B: Teacherless + Rate Loss only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DONE_PATH="${RUN_DONE_PATH:-$REPO_ROOT/runs/vast_exp_b.done}"
RUN_FAIL_PATH="${RUN_FAIL_PATH:-$REPO_ROOT/runs/vast_exp_b.failed}"

cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "========================================"
echo "Step 2 - Exp B: Teacherless + Rate Loss"
echo "Config: configs/remote/step2_exp_b_teacherless.yaml"
echo "========================================"
set +e
python train.py --config configs/remote/step2_exp_b_teacherless.yaml
exp_exit=$?
set -e

if [[ $exp_exit -ne 0 ]]; then
  echo "Exp B failed with exit code $exp_exit"
  echo "exp_exit=$exp_exit" > "$RUN_FAIL_PATH"
  exit "$exp_exit"
fi

echo "========================================"
echo "Exp B completed successfully"
echo "========================================"
touch "$RUN_DONE_PATH"

