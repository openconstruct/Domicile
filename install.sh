#!/usr/bin/env bash
# Domicile/cchat installer: venv + dependencies, then the setup wizard.
# Usage (from the cchat folder):  ./install.sh            full install + wizard
#                                 ./install.sh --no-wizard   deps only
#                                 ./install.sh --edit        reopen the editor later
# Debian/Ubuntu assumed for the apt hints; the venv part is distro-neutral.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CCHAT_VENV:-$HOME/.cchat-venv}"
say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--edit" ]]; then
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV; run ./install.sh first"
  cd "$HERE" && exec "$VENV/bin/python" wizard.py --edit
fi

say "Checking system packages"
command -v python3 >/dev/null || die "python3 not found (apt install python3)"
python3 - <<'PY' || die "Python 3.10+ required"
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
python3 -c "import venv, ensurepip" 2>/dev/null || die "venv missing (apt install python3-venv)"
command -v tmux >/dev/null || die "tmux missing (apt install tmux) - needed for the background service"

say "Creating venv at $VENV"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"

say "Installing dependencies (a few minutes on small machines)"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements.txt" -r "$HERE/requirements-mem0.txt"

[[ "${1:-}" == "--no-wizard" ]] && { say "Done. Run the wizard with: $VENV/bin/python $HERE/wizard.py"; exit 0; }

say "Starting setup wizard"
cd "$HERE" && exec "$VENV/bin/python" wizard.py
