#!/usr/bin/env bash
# Click-run entry point: see scripts/click_run.sh (./run.sh --help).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/click_run.sh" "$@"
