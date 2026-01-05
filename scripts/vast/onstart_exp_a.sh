#!/usr/bin/env bash
# Onstart script for Step 2 Exp A (EMA + Rate)
# With self-destruct capability
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-seq-jepa-streaming}"
# Use public HTTPS URL (repo is public) for reliable cloning
REPO_URL="https://github.com/kaikun213/seq-jepa.git"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Export W&B and Vast credentials to environment
{
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY=${WANDB_API_KEY}"
  fi
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    echo "WANDB_PROJECT=${WANDB_PROJECT}"
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    echo "WANDB_ENTITY=${WANDB_ENTITY}"
  fi
  # Pass through VAST_API_KEY for self-destruct
  if [[ -n "${VAST_API_KEY:-}" ]]; then
    echo "VAST_API_KEY=${VAST_API_KEY}"
  fi
  if [[ -n "${VAST_INSTANCE_ID:-}" ]]; then
    echo "VAST_INSTANCE_ID=${VAST_INSTANCE_ID}"
  fi
} >> /etc/environment

# Source for current shell
source /etc/environment 2>/dev/null || true

# Clone repo if not present, or pull latest
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git fetch origin main
  git reset --hard origin/main
  cd "$WORKDIR"
fi

cd "$REPO_DIR"

# Verify we have the latest code with the run scripts
if [[ ! -f "scripts/vast/run_exp_a.sh" ]]; then
  echo "ERROR: run_exp_a.sh not found. Trying to pull latest..."
  git fetch origin main --depth=1
  git reset --hard origin/main
fi

# Setup Python environment
python -m venv .venv
source .venv/bin/activate

pip install -U pip wheel
REQS_FILE="requirements.txt"
REQS_TMP="/tmp/requirements-remote.txt"
FILTER_PATTERN='^(deepgaze-pytorch|Pillow_SIMD)=='

if grep -Eq "$FILTER_PATTERN" "$REQS_FILE"; then
  grep -Ev "$FILTER_PATTERN" "$REQS_FILE" > "$REQS_TMP"
  pip install -r "$REQS_TMP"
else
  pip install -r "$REQS_FILE"
fi

# Install vastai CLI for self-destruct
pip install vastai --quiet

# Run Exp A
chmod +x scripts/vast/run_exp_a.sh
scripts/vast/run_exp_a.sh
