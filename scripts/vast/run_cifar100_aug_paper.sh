#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-configs/remote/cifar100_aug_paper.yaml}"
RUN_DONE_PATH="${RUN_DONE_PATH:-$REPO_ROOT/runs/vast_cifar100_paper.done}"
RUN_FAIL_PATH="${RUN_FAIL_PATH:-$REPO_ROOT/runs/vast_cifar100_paper.failed}"

cd "$REPO_ROOT"

set +e
python train.py --config "$CONFIG_PATH"
train_exit=$?
set -e

if [[ $train_exit -ne 0 ]]; then
  echo "exit_code=$train_exit" > "$RUN_FAIL_PATH"
  exit "$train_exit"
fi

set +e
python scripts/eval/frozen_probe_cifar100_aug.py --config "$CONFIG_PATH"
eval_exit=$?
set -e

if [[ $eval_exit -ne 0 ]]; then
  echo "exit_code=$eval_exit" > "$RUN_FAIL_PATH"
  exit "$eval_exit"
fi

touch "$RUN_DONE_PATH"
