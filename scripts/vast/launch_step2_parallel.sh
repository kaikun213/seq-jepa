#!/usr/bin/env bash
# Launch Step 2 experiments in PARALLEL on Vast.ai
# Creates two separate instances for EMA and Teacherless
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

: "${VAST_API_KEY:?Set VAST_API_KEY in .env or env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env or env}"

# RTX 4090 primary, RTX 3090 backup
GPU_LIST_DEFAULT='["RTX_4090","RTX_3090","A10","RTX_3080"]'
GPU_LIST="${VAST_GPU_LIST:-$GPU_LIST_DEFAULT}"
QUERY_DEFAULT="reliability>0.98 num_gpus==1 gpu_name in ${GPU_LIST} gpu_ram>=12 disk_space>=40 inet_down>=50 rented=any"
QUERY="${VAST_QUERY:-$QUERY_DEFAULT}"
ORDER="${VAST_ORDER:-dph}"
LIMIT="${VAST_LIMIT:-10}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime}"
DISK_GB="${VAST_DISK_GB:-40}"

GITHUB_SSH_KEY_B64=""
if [[ -n "${GITHUB_SSH_KEY_PATH:-}" ]]; then
  if [[ ! -f "$GITHUB_SSH_KEY_PATH" ]]; then
    echo "GITHUB_SSH_KEY_PATH not found: $GITHUB_SSH_KEY_PATH" >&2
    exit 1
  fi
  GITHUB_SSH_KEY_B64="$(base64 < "$GITHUB_SSH_KEY_PATH" | tr -d '\n')"
fi

ENV_ARGS=("-e" "WANDB_API_KEY=${WANDB_API_KEY}")
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  ENV_ARGS+=("-e" "WANDB_PROJECT=${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENV_ARGS+=("-e" "WANDB_ENTITY=${WANDB_ENTITY}")
fi
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  ENV_ARGS+=("-e" "GITHUB_TOKEN=${GITHUB_TOKEN}")
fi
if [[ -n "$GITHUB_SSH_KEY_B64" ]]; then
  ENV_ARGS+=("-e" "GITHUB_SSH_KEY_B64=${GITHUB_SSH_KEY_B64}")
fi

ENV_STRING="${ENV_ARGS[*]}"

if [[ -z "${GITHUB_TOKEN:-}" && -z "$GITHUB_SSH_KEY_B64" ]]; then
  echo "Set GITHUB_SSH_KEY_PATH or GITHUB_TOKEN for repo clone." >&2
  exit 1
fi

echo "========================================="
echo "Launching Step 2 Experiments in PARALLEL"
echo "Exp A: EMA + Rate (30 epochs, paper-aligned)"
echo "Exp B: Teacherless + Rate (30 epochs, paper-aligned)"
echo "Expected runtime: ~30-60 min each on RTX 4090"
echo "========================================="
echo "Searching offers: $QUERY"

# Get multiple offers for parallel launches
OFFER_JSON=$(vastai search offers "$QUERY" --limit "$LIMIT" -o "$ORDER" --raw --api-key "$VAST_API_KEY")

# Parse two best offers
OFFERS=$(OFFER_JSON="$OFFER_JSON" python - <<'PY'
import json, os, sys
raw = os.environ.get("OFFER_JSON", "")
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("")
    sys.exit(0)
if len(data) < 2:
    # Not enough offers for parallel
    if data:
        print(f"{data[0]['id']} {data[0]['id']}")  # Use same offer twice
    sys.exit(0)
print(f"{data[0]['id']} {data[1]['id']}")
PY
)

OFFER_A="${OFFERS%% *}"
OFFER_B="${OFFERS##* }"

if [[ -z "$OFFER_A" || -z "$OFFER_B" ]]; then
  echo "No offers matched query." >&2
  exit 1
fi

echo "Selected offer for Exp A: $OFFER_A"
echo "Selected offer for Exp B: $OFFER_B"

# Launch Exp A
echo ""
echo "Creating instance for Exp A (EMA + Rate)..."
CREATE_A_JSON=$(vastai create instance "$OFFER_A" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --label "seqjepa-step2-exp-a" \
  --ssh --direct \
  --onstart "$REPO_ROOT/scripts/vast/onstart_exp_a.sh" \
  --env "$ENV_STRING" \
  --api-key "$VAST_API_KEY" \
  --raw)

INSTANCE_A=$(CREATE_A_JSON="$CREATE_A_JSON" python - <<'PY'
import json, re, os
raw = os.environ.get("CREATE_A_JSON", "")
instance_id = ""
try:
    data = json.loads(raw.strip())
    instance_id = str(data.get("new_contract", "") or "")
except Exception:
    match = re.search(r"\"new_contract\"\s*:\s*(\d+)", raw) or re.search(
        r"new_contract\s*[:=]\s*(\d+)", raw
    )
    if match:
        instance_id = match.group(1)
print(instance_id)
PY
)

if [[ -z "$INSTANCE_A" ]]; then
  echo "Failed to create instance for Exp A. Raw response:"
  echo "$CREATE_A_JSON"
  exit 1
fi
echo "Created Exp A instance: $INSTANCE_A"

# Launch Exp B
echo ""
echo "Creating instance for Exp B (Teacherless + Rate)..."
CREATE_B_JSON=$(vastai create instance "$OFFER_B" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --label "seqjepa-step2-exp-b" \
  --ssh --direct \
  --onstart "$REPO_ROOT/scripts/vast/onstart_exp_b.sh" \
  --env "$ENV_STRING" \
  --api-key "$VAST_API_KEY" \
  --raw)

INSTANCE_B=$(CREATE_B_JSON="$CREATE_B_JSON" python - <<'PY'
import json, re, os
raw = os.environ.get("CREATE_B_JSON", "")
instance_id = ""
try:
    data = json.loads(raw.strip())
    instance_id = str(data.get("new_contract", "") or "")
except Exception:
    match = re.search(r"\"new_contract\"\s*:\s*(\d+)", raw) or re.search(
        r"new_contract\s*[:=]\s*(\d+)", raw
    )
    if match:
        instance_id = match.group(1)
print(instance_id)
PY
)

if [[ -z "$INSTANCE_B" ]]; then
  echo "Failed to create instance for Exp B. Raw response:"
  echo "$CREATE_B_JSON"
  # Don't exit - Exp A is already running
  echo "WARNING: Exp A is running on instance $INSTANCE_A"
else
  echo "Created Exp B instance: $INSTANCE_B"
fi

echo ""
echo "========================================="
echo "Both experiments launched!"
echo "Exp A (EMA): Instance $INSTANCE_A"
echo "Exp B (Teacherless): Instance ${INSTANCE_B:-FAILED}"
echo ""
echo "Monitor on W&B: https://wandb.ai/kaikun213/seq-jepa-streaming"
echo "Group: step2-remote"
echo "========================================="
echo ""
echo "To check status:"
echo "  vastai show instances --api-key \$VAST_API_KEY"
echo ""
echo "To destroy when done:"
echo "  vastai destroy instance $INSTANCE_A --api-key \$VAST_API_KEY"
if [[ -n "${INSTANCE_B:-}" ]]; then
  echo "  vastai destroy instance $INSTANCE_B --api-key \$VAST_API_KEY"
fi

