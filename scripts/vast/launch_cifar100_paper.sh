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

GPU_LIST_DEFAULT='["RTX_4090","A100","A10","RTX_3090"]'
GPU_LIST="${VAST_GPU_LIST:-$GPU_LIST_DEFAULT}"
QUERY_DEFAULT="reliability>0.98 num_gpus==1 gpu_name in ${GPU_LIST} gpu_ram>=24 disk_space>=80 inet_down>=50 rented=any"
QUERY="${VAST_QUERY:-$QUERY_DEFAULT}"
ORDER="${VAST_ORDER:-dph}"
LIMIT="${VAST_LIMIT:-5}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime}"
DISK_GB="${VAST_DISK_GB:-80}"
LABEL="${VAST_LABEL:-seqjepa-cifar100-paper}"

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
  --onstart "$REPO_ROOT/scripts/vast/onstart_cifar100_aug_paper.sh" \
  --env "$ENV_STRING" \
  --api-key "$VAST_API_KEY" \
  --raw)

INSTANCE_ID=$(echo "$CREATE_JSON" | python -c 'import json,sys,re; raw=sys.stdin.read(); instance_id="";\ntry:\n    data=json.loads(raw.strip()); instance_id=str(data.get("new_contract","") or "")\nexcept Exception:\n    match=re.search(r"\"new_contract\"\\s*:\\s*(\\d+)", raw) or re.search(r"new_contract\\s*[:=]\\s*(\\d+)", raw)\n    if match:\n        instance_id=match.group(1)\nprint(instance_id)')

if [[ -z "$INSTANCE_ID" ]]; then
  echo "Failed to create instance. Raw response:"
  echo "$CREATE_JSON"
  exit 1
fi

echo "Created instance: $INSTANCE_ID"

echo "Run is started. Track progress in W&B."

auto_destroy="${VAST_AUTO_DESTROY:-0}"
if [[ "$auto_destroy" == "1" ]]; then
  echo "Auto-destroy enabled; waiting for completion marker."
  DONE_PATH="${VAST_DONE_PATH:-/workspace/seq-jepa-streaming/runs/vast_cifar100_paper.done}"
  FAIL_PATH="${VAST_FAIL_PATH:-/workspace/seq-jepa-streaming/runs/vast_cifar100_paper.failed}"
  while true; do
    SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw --api-key "$VAST_API_KEY" | python -c 'import json,sys; raw=sys.stdin.read().strip();\ntry:\n    data=json.loads(raw)\nexcept json.JSONDecodeError:\n    print(""); sys.exit(0)\nhost=data.get("ssh_host") or ""; port=data.get("ssh_port") or "";\nprint(f"{host} {port}" if host and port else "")'))
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
    sleep "${VAST_POLL_SECS:-300}"
  done

  if [[ "${VAST_COPY_RESULTS:-0}" == "1" ]]; then
    DEST="${VAST_COPY_DEST:-$REPO_ROOT/runs/remote/cifar100-aug-paper}"
    REMOTE_PATH="/workspace/seq-jepa-streaming/runs/remote/cifar100-aug-paper"
    mkdir -p "$DEST"
    scp -r -P "$SSH_PORT" "root@${SSH_HOST}:${REMOTE_PATH}" "$DEST" || true
  fi

  echo "Destroying instance $INSTANCE_ID."
  vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY"
fi
