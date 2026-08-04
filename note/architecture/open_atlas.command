#!/bin/zsh
set -euo pipefail

ATLAS_ROOT="${0:A:h}"
PORT="${PORT:-8765}"
BASE_URL="http://127.0.0.1:${PORT}"
TARGET_PATH="${ATLAS_PAGE:-/02_frontres_design_inspector.html}"

atlas_is_ready() {
  /usr/bin/curl -fsS "${BASE_URL}/healthz" 2>/dev/null \
    | /usr/bin/grep -q '"service":"mosaic-frontres-atlas"'
}

if atlas_is_ready; then
  print "[Atlas] reuse: ${BASE_URL}${TARGET_PATH}"
  if [[ "$(/usr/bin/uname -s)" == "Darwin" ]]; then
    /usr/bin/open "${BASE_URL}${TARGET_PATH}"
  else
    print "${BASE_URL}${TARGET_PATH}"
  fi
  exit 0
fi

if [[ ! -f "${ATLAS_ROOT}/auxiliary/atlas_app/node_modules/roughjs/bundled/rough.esm.js" ]]; then
  print -u2 "[Atlas] missing roughjs dependency"
  print -u2 "[Atlas] run: npm --prefix '${ATLAS_ROOT}/auxiliary/atlas_app' install"
  exit 1
fi

# The server intentionally owns this terminal in the foreground. Background
# daemons are reaped by some IDE/agent shells and recreate the original
# connection-refused failure. Ctrl-C closes the Atlas explicitly.
if [[ "$(/usr/bin/uname -s)" == "Darwin" ]]; then
  (
    for _attempt in {1..50}; do
      if atlas_is_ready; then
        print "[Atlas] ready: ${BASE_URL}${TARGET_PATH}"
        /usr/bin/open "${BASE_URL}${TARGET_PATH}"
        exit 0
      fi
      /bin/sleep 0.1
    done
    print -u2 "[Atlas] server failed to become ready at ${BASE_URL}"
  ) &
else
  print "[Atlas] open after startup: ${BASE_URL}${TARGET_PATH}"
fi

exec /usr/bin/env PORT="${PORT}" node "${ATLAS_ROOT}/auxiliary/atlas_app/serve_architecture.mjs"
