#!/bin/bash
# install-service.sh — install this repo as the ~/qwen-serving deploy tree the
# shipped qwen-serving.service units assume, register the unit, print enable.
#
#   bash scripts/install-service.sh [single|batch]   # default: single
#
# Copies the repo to ~/qwen-serving excluding models/, .env and .git (the big
# and the host-local bits; symlink the model dir into the deploy tree instead),
# installs the chosen qwen-serving.service unit to ~/.config/systemd/user/,
# runs systemctl --user daemon-reload, and prints the enable command —
# enabling is left to the operator. Refuses unless the systemd user bus is
# present (no systemd --user here: containers, WSL without it, plain ssh).
set -e

MODE=${1:-single}
case "$MODE" in single|batch) ;; *)
  echo "install-service: unknown mode '$MODE' (want single|batch)" >&2; exit 1 ;;
esac

systemctl --user show-environment >/dev/null 2>&1 \
  || { echo "install-service: refusing: no systemd user bus (systemctl --user cannot talk to one here)" >&2; exit 1; }

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST=$HOME/qwen-serving
mkdir -p "$DEST"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete --exclude 'models/' --exclude '.env' --exclude '.git' "$SRC/" "$DEST/"
else
  echo "install-service: rsync not found, falling back to cp (no --delete)" >&2
  rm -rf "$DEST"; mkdir -p "$DEST"
  cp -a "$SRC/." "$DEST/"
  rm -rf "$DEST/models" "$DEST/.env" "$DEST/.git"
fi

mkdir -p "$HOME/.config/systemd/user"
cp "$DEST/$MODE/qwen-serving.service" "$HOME/.config/systemd/user/qwen-serving.service"
systemctl --user daemon-reload

echo "install-service: deployed $SRC ($MODE) -> $DEST (models/, .env, .git excluded)"
echo "install-service: unit installed to ~/.config/systemd/user/qwen-serving.service"
echo "enable with: systemctl --user enable --now qwen-serving"
echo "logs: journalctl --user -u qwen-serving -f  (launcher also appends qwen.log in the deploy tree)"
