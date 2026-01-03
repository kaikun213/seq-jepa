#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-configs/remote/cifar100_aug_baseline.yaml}"

cd "$REPO_ROOT"

echo "Using config: $CONFIG_PATH"
python train.py --config "$CONFIG_PATH"
