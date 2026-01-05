#!/usr/bin/env bash
# Run Step 2 Exp A: EMA + Rate Loss only
# With self-destruct after completion
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
else
  echo "========================================"
  echo "Exp A completed successfully"
  echo "========================================"
  touch "$RUN_DONE_PATH"
fi

# Self-destruct: destroy this instance if VAST_API_KEY and VAST_INSTANCE_ID are set
if [[ -n "${VAST_API_KEY:-}" && -n "${VAST_INSTANCE_ID:-}" ]]; then
  echo "Self-destruct enabled. Destroying instance $VAST_INSTANCE_ID in 30 seconds..."
  sleep 30
  pip install vastai --quiet 2>/dev/null || true
  vastai destroy instance "$VAST_INSTANCE_ID" --api-key "$VAST_API_KEY" || echo "Failed to self-destruct"
fi

exit ${exp_exit:-0}
