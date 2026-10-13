#!/usr/bin/env bash
# Build the HumanEval sandbox image of the Q200v2 kit and print its immutable id (sha256:...).
# The kit's runner refuses to grade the humaneval family without it (--image-id).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBDIR="$("$HERE/fetch_kit.sh")"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$SUBDIR/docker/q200_sandbox_driver.py" "$CTX/"
cp "$HERE/sandbox/Dockerfile" "$CTX/Dockerfile"
docker build -q -t minicpm-v47-q200-sandbox:1 "$CTX" >/dev/null
docker image inspect --format '{{.Id}}' minicpm-v47-q200-sandbox:1
