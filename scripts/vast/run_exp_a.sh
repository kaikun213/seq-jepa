#!/usr/bin/env bash
# Run Step 2 Exp A: EMA + Rate Loss only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DONE_PATH="${RUN_DONE_PATH:-$REPO_ROOT/runs/vast_exp_a.done}"
RUN_FAIL_PATH="${RUN_FAIL_PATH:-$REPO_ROOT/runs/vast_exp_a.failed}"

cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "========================================"
echo "Step 2 - Exp A: EMA + Rate Loss"
echo "Config: configs/remote/step2_exp_a_ema_rate.yaml"
echo "========================================"
set +e
python train.py --config configs/remote/step2_exp_a_ema_rate.yaml
exp_exit=$?
set -e

if [[ $exp_exit -ne 0 ]]; then
  echo "Exp A failed with exit code $exp_exit"
  echo "exp_exit=$exp_exit" > "$RUN_FAIL_PATH"
  exit "$exp_exit"
fi

echo "========================================"
echo "Exp A completed successfully"
echo "========================================"
touch "$RUN_DONE_PATH"

