#!/usr/bin/env bash
# Onstart script for Step 2 Exp A (EMA + Rate)
# With self-destruct capability
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-seq-jepa-streaming}"
REPO_URL_SSH="git@github-private:kaikun213/seq-jepa-streaming.git"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Setup GitHub SSH key if provided
if [[ -n "${GITHUB_SSH_KEY_B64:-}" ]]; then
  mkdir -p ~/.ssh
  echo "$GITHUB_SSH_KEY_B64" | base64 -d > ~/.ssh/id_ed25519
  chmod 600 ~/.ssh/id_ed25519
  ssh-keyscan github.com >> ~/.ssh/known_hosts
  cat > ~/.ssh/config <<'CFG'
Host github-private
  Hostname github.com
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
CFG
  REPO_URL="$REPO_URL_SSH"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
  REPO_URL="https://${GITHUB_TOKEN}@github.com/kaikun213/seq-jepa-streaming.git"
else
  echo "Missing GitHub credentials. Set GITHUB_SSH_KEY_B64 or GITHUB_TOKEN." >&2
  exit 1
fi

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

# Clone repo if not present
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

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
scripts/vast/run_exp_a.sh
