#!/usr/bin/env bash
set -euo pipefail

CERT_NAME="${CERT_NAME:-Zscaler}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/Documents/cacert}"

BUNDLE_PATH="$(python3 - <<'PY'
try:
    import certifi
    print(certifi.where())
except Exception:
    print("")
PY
)"

if [[ -z "$BUNDLE_PATH" ]]; then
  echo "certifi not found. Activate the target Python env and retry." >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
BACKUP_PATH="$BACKUP_DIR/cacert.pem.$(date +%Y%m%d%H%M%S)"
cp "$BUNDLE_PATH" "$BACKUP_PATH"

CERT_PATH="$BACKUP_DIR/zscaler.pem"
if ! security find-certificate -a -p -c "$CERT_NAME" > "$CERT_PATH"; then
  echo "Failed to export certificate '$CERT_NAME' from keychain." >&2
  echo "Set CERT_NAME to the exact certificate name and retry." >&2
  exit 1
fi

if [[ ! -s "$CERT_PATH" ]]; then
  echo "No certificate data found for '$CERT_NAME'." >&2
  exit 1
fi

if ! grep -q "BEGIN CERTIFICATE" "$CERT_PATH"; then
  echo "Exported certificate does not look like PEM." >&2
  exit 1
fi

cat "$CERT_PATH" >> "$BUNDLE_PATH"

echo "Backup written to: $BACKUP_PATH"
echo "Certifi bundle updated: $BUNDLE_PATH"

env_line_ssl="export SSL_CERT_FILE=\"$BUNDLE_PATH\""
env_line_req="export REQUESTS_CA_BUNDLE=\"$BUNDLE_PATH\""

ZSHRC="$HOME/.zshrc"
if ! rg -q "SSL_CERT_FILE" "$ZSHRC" 2>/dev/null; then
  echo "$env_line_ssl" >> "$ZSHRC"
fi
if ! rg -q "REQUESTS_CA_BUNDLE" "$ZSHRC" 2>/dev/null; then
  echo "$env_line_req" >> "$ZSHRC"
fi

echo "Updated $ZSHRC with SSL_CERT_FILE and REQUESTS_CA_BUNDLE."
