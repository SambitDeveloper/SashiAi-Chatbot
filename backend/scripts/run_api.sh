#!/usr/bin/env bash
# Development launcher for the API with auto-reload that actually works on WSL.
#   scripts/run_api.sh            # port 8000
#   PORT=8010 scripts/run_api.sh
# On a Windows drive mounted in WSL (/mnt/c/...) file-change events are not delivered, so `uvicorn --reload` silently
# kept serving code from hours earlier (2026-11-18). Polling fixes that; it is switched on only for /mnt/* paths.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$PWD" == /mnt/* ]]; then
  export WATCHFILES_FORCE_POLLING=true
fi
exec uv run uvicorn app.main:app --reload --reload-dir app --port "${PORT:-8000}"
