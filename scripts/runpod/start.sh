#!/usr/bin/env bash
set -euo pipefail

printf 'Startup: entrypoint reached\n'

# Retain diagnostics across container exits when persistent storage is attached.
log_dir="${OMNISCRIPT_STARTUP_LOG_DIR:-/runpod-volume/omniscript-logs}"
if [[ -d /runpod-volume || -n "${OMNISCRIPT_STARTUP_LOG_DIR:-}" ]]; then
    if mkdir -p "$log_dir"; then
        log_file="$log_dir/startup-${HOSTNAME:-worker}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
        if touch "$log_file"; then
            exec > >(tee -a "$log_file") 2>&1
            printf 'Startup: log=%s\n' "$log_file"
        else
            printf 'Startup: persistent log unavailable\n' >&2
        fi
    else
        printf 'Startup: persistent log directory unavailable\n' >&2
    fi
fi

trap 'status=$?; printf "Startup: failed at line %s (exit %s)\n" "$LINENO" "$status" >&2; exit "$status"' ERR

cd /app

if [[ "${OMNISCRIPT_MOCK:-0}" != "1" ]]; then
    printf 'Startup: preparing model files\n'
    python -m scripts.runpod.download_models
    printf 'Startup: model files prepared\n'
fi

printf 'Startup: launching application\n'
exec "$@"
