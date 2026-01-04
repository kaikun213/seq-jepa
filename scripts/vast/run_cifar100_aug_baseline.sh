#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-configs/remote/cifar100_aug_baseline.yaml}"
RUN_DONE_PATH="${RUN_DONE_PATH:-$REPO_ROOT/runs/vast_cifar100_baseline.done}"
RUN_FAIL_PATH="${RUN_FAIL_PATH:-$REPO_ROOT/runs/vast_cifar100_baseline.failed}"

cd "$REPO_ROOT"

echo "Using config: $CONFIG_PATH"
set +e
python train.py --config "$CONFIG_PATH"
exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
  echo "exit_code=$exit_code" > "$RUN_FAIL_PATH"
  exit "$exit_code"
fi

touch "$RUN_DONE_PATH"
