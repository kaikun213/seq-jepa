#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

: "${VAST_API_KEY:?Set VAST_API_KEY in .env or env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env or env}"

GPU_LIST_DEFAULT='["RTX_3060","A10","RTX_3070","RTX_3080","RTX_3090"]'
GPU_LIST="${VAST_GPU_LIST:-$GPU_LIST_DEFAULT}"
QUERY_DEFAULT="reliability>0.98 num_gpus==1 gpu_name in ${GPU_LIST} gpu_ram>=12 disk_space>=40 inet_down>=50 rented=any"
QUERY="${VAST_QUERY:-$QUERY_DEFAULT}"
ORDER="${VAST_ORDER:-dph}"
LIMIT="${VAST_LIMIT:-5}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime}"
DISK_GB="${VAST_DISK_GB:-40}"
LABEL="${VAST_LABEL:-seqjepa-cifar100-baseline}"

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

echo "Searching offers: $QUERY"
OFFER_JSON=$(vastai search offers "$QUERY" --limit "$LIMIT" -o "$ORDER" --raw --api-key "$VAST_API_KEY")
OFFER_ID=$(OFFER_JSON="$OFFER_JSON" python - <<'PY'
import json, os, sys
raw = os.environ.get("OFFER_JSON", "")
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("", end="")
    sys.exit(0)
if not data:
    print("", end="")
    sys.exit(0)
print(data[0]["id"])
PY
)

if [[ -z "$OFFER_ID" ]]; then
  echo "No offers matched query." >&2
  exit 1
fi

echo "Selected offer id: $OFFER_ID"

CREATE_JSON=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --label "$LABEL" \
  --ssh --direct \
  --onstart "$REPO_ROOT/scripts/vast/onstart_cifar100_aug_baseline.sh" \
  --env "$ENV_STRING" \
  --api-key "$VAST_API_KEY" \
  --raw)

INSTANCE_ID=$(echo "$CREATE_JSON" | python - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("")
    sys.exit(0)
print(data.get("new_contract", ""))
PY
)

if [[ -z "$INSTANCE_ID" ]]; then
  echo "Failed to create instance. Raw response:"
  echo "$CREATE_JSON"
  exit 1
fi

echo "Created instance: $INSTANCE_ID"

AUTO_DESTROY="${VAST_AUTO_DESTROY:-1}"
POLL_SECS="${VAST_POLL_SECS:-60}"
DONE_PATH="${VAST_DONE_PATH:-/workspace/seq-jepa-streaming/runs/vast_cifar100_baseline.done}"
FAIL_PATH="${VAST_FAIL_PATH:-/workspace/seq-jepa-streaming/runs/vast_cifar100_baseline.failed}"

if [[ "$AUTO_DESTROY" != "0" ]]; then
  echo "Waiting for run completion before destroying instance."
  while true; do
    SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw --api-key "$VAST_API_KEY" | python - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("")
    sys.exit(0)
host = data.get("ssh_host") or ""
port = data.get("ssh_port") or ""
if host and port:
    print(f"{host} {port}")
PY
)
    SSH_HOST="${SSH_INFO%% *}"
    SSH_PORT="${SSH_INFO##* }"

    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" && "$SSH_HOST" != "$SSH_PORT" ]]; then
      if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@${SSH_HOST}" "test -f '$DONE_PATH'"; then
        echo "Run completed."
        break
      fi
      if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@${SSH_HOST}" "test -f '$FAIL_PATH'"; then
        echo "Run failed."
        break
      fi
    fi
    sleep "$POLL_SECS"
  done

  echo "Destroying instance $INSTANCE_ID."
  vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY"
fi
