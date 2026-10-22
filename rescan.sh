#!/bin/bash
# Memory rebuild: build | validate | promote | rollback.
set -euo pipefail

CD="${CCHAT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MEM0="$CD/data/mem0"
LIVE="$MEM0/qdrant"
REBUILD="$MEM0/qdrant-rebuild"
FTS_LIVE="$MEM0/mem0-fts.db"
FTS_REBUILD="$MEM0/mem0-fts-rebuild.db"
cd "$CD"

case "${1:-}" in
  build)
    shift
    exec /usr/bin/python3 mem0_rescan.py build "$@"
    ;;

  validate)
    shift
    exec /usr/bin/python3 mem0_rescan.py validate "$@"
    ;;

  promote)
    if [ ! -d "$REBUILD" ]; then
      echo "no $REBUILD -- run ./rescan.sh build first" >&2; exit 1
    fi
    if [ -n "${TMUX:-}" ] && tmux display-message -p '#{socket_path}' 2>/dev/null | grep -q cchat; then
      echo "refusing: you are inside the Domicile tmux. SSH in fresh and rerun." >&2
      exit 1
    fi
    TS=$(date +%Y%m%d-%H%M%S)
    echo ">> stopping nightly timer + Domicile"
    systemctl --user stop mem0-nightly.timer 2>/dev/null || true
    systemctl --user stop mem0-nightly.service 2>/dev/null || true
    systemctl --user stop cchat.service
    for i in $(seq 1 30); do
      if ! fuser "$LIVE"/*.lock >/dev/null 2>&1; then break; fi
      sleep 1
    done
    echo ">> swapping store  (live -> qdrant.pre-rescan.$TS)"
    mv "$LIVE" "$MEM0/qdrant.pre-rescan.$TS"
    mv "$REBUILD" "$LIVE"
    rm -f "$FTS_LIVE" "$FTS_REBUILD"          # keyword mirror -- rebuilds from qdrant
    mv "$MEM0/rescan-checkpoint.json" "$MEM0/rescan-checkpoint.$TS.json" 2>/dev/null || true
    echo ">> restarting Domicile + nightly timer"
    systemctl --user start cchat.service
    systemctl --user start mem0-nightly.timer
    echo ">> done. Old store kept at $MEM0/qdrant.pre-rescan.$TS"
    echo "   (tomorrow's nightly will re-feed the 4 pending logs; guard dedups.)"
    ;;

  rollback)
    PREV=$(ls -1d "$MEM0"/qdrant.pre-rescan.* 2>/dev/null | sort | tail -1 || true)
    if [ -z "$PREV" ]; then echo "no pre-rescan store to roll back to" >&2; exit 1; fi
    if [ -n "${TMUX:-}" ] && tmux display-message -p '#{socket_path}' 2>/dev/null | grep -q cchat; then
      echo "refusing: you are inside the Domicile tmux. SSH in fresh and rerun." >&2
      exit 1
    fi
    TS=$(date +%Y%m%d-%H%M%S)
    echo ">> rolling back to $PREV"
    systemctl --user stop cchat.service
    for i in $(seq 1 30); do
      if ! fuser "$LIVE"/*.lock >/dev/null 2>&1; then break; fi
      sleep 1
    done
    mv "$LIVE" "$MEM0/qdrant.failed-rescan.$TS"
    mv "$PREV" "$LIVE"
    rm -f "$FTS_LIVE"
    systemctl --user start cchat.service
    echo ">> rolled back. Failed store kept at $MEM0/qdrant.failed-rescan.$TS"
    ;;

  *)
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
    ;;
esac
