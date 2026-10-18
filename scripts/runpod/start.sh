#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${OMNISCRIPT_MOCK:-0}" != "1" ]]; then
    python -m scripts.runpod.download_models
fi

exec "$@"