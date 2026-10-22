#!/usr/bin/env bash
# Offline memory drain (SSH).

set -uo pipefail

echo "[drain] disabling watchdog + nightly timers so nothing restarts Domicile mid-drain"
systemctl --user stop cchat-watchdog.timer || true
systemctl --user stop mem0-nightly.timer  || true

echo "[drain] stopping Domicile (cchat.service; tmux kill-server also kills mem0_server child)"
systemctl --user stop cchat.service || { echo "[drain] FAILED to stop Domicile"; exit 1; }

pkill -f mem0_server.py || true

sleep 3

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
LOG="$HERE/data/mem0-nightly-drain.log"
echo "[drain] running mem0_nightly.py -> $LOG"

if python3 mem0_nightly.py > "$LOG" 2>&1; then
  echo "[drain] nightly exited 0"
else
  echo "[drain] WARNING: nightly exited non-zero; see $LOG"
fi

echo
echo "[drain] ingest-status.json:"
cat "$HERE/data/mem0/ingest-status.json" 2>/dev/null || true
echo
echo "[drain] DONE. Restart Domicile when ready:"
echo "  systemctl --user start cchat.service"
echo "  systemctl --user start cchat-watchdog.timer"
echo "  systemctl --user start mem0-nightly.timer"
