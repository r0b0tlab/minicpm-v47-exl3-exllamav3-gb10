#!/usr/bin/env bash
# Q200v2 text-180 (80 GSM8K, 40 HumanEval, 40 IFEval, 20 hard reasoning) through any
# OpenAI-compatible endpoint, with the public kit's own runner (fetched, hash-verified).
#
#   eval/q200/run_q200.sh BASE_URL OUT_DIR [--workers N] [--model NAME] [--limit-per-family N]
#                                          [--manual-evidence FILE] [--run-id ID]
#
# Request settings (the ones the reference runs used): thinking on, chat_template_kwargs
# {"enable_thinking": true, "thinking": true, "reasoning_effort": "low"}, temperature 0 and
# top_p 1 (the runner's own), max_tokens 8192, timeout 3600 s per request.
# HumanEval grading needs the kit's sandbox image: SANDBOX_IMAGE_ID=sha256:... (eval/build_sandbox.sh).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$(dirname "$HERE")"
BASE_URL="${1:?usage: run_q200.sh BASE_URL OUT_DIR [options]}"; OUT="${2:?OUT_DIR}"; shift 2
WORKERS=1; MODEL=""; LIMIT=""; MANUAL=""; RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --limit-per-family) LIMIT="$2"; shift 2 ;;
    --manual-evidence) MANUAL="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
SUBDIR="$("$EVAL/fetch_kit.sh")"
if [[ -z "${SANDBOX_IMAGE_ID:-}" ]]; then
  SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' minicpm-v47-q200-sandbox:1 2>/dev/null || true)"
fi
[[ -n "$SANDBOX_IMAGE_ID" ]] || { echo "no sandbox image: run eval/build_sandbox.sh and set SANDBOX_IMAGE_ID" >&2; exit 1; }
if [[ -z "$MODEL" ]]; then
  MODEL="$(python3 -c 'import json,sys,urllib.request;print(json.load(urllib.request.urlopen(sys.argv[1].rstrip("/")+"/v1/models"))["data"][0]["id"])' "$BASE_URL")"
fi
DATASET="$SUBDIR/artifacts/quality-text-180-v2.jsonl"
if [[ -n "$LIMIT" ]]; then
  # a subset is a sanity check, never a score: the kit's frozen-hash gate is relaxed by running the runner on a copy
  echo "refusing --limit-per-family: the kit's runner only accepts the frozen 180-row set" >&2
  exit 64
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-q200-w${WORKERS}-$STAMP}"
mkdir -p "$OUT"
cd "$OUT"
extra=()
if [[ -n "$MANUAL" ]]; then extra+=(--manual-evidence "$MANUAL"); fi
T0="$(date +%s.%N)"
set +e
PYTHONPATH="$EVAL/shims" python3 "$SUBDIR/scripts/run_quality_set.py" \
  --base-url "${BASE_URL%/}" --model "$MODEL" --run-id "$RUN_ID" --set "$DATASET" \
  --max-tokens 8192 --timeout 3600 --workers "$WORKERS" \
  --image-id "$SANDBOX_IMAGE_ID" --profile-id exllamav3-minicpmv47-gb10 --candidate-id exllamav3-serve \
  --admission-config "$EVAL/shims/admission.json" \
  --chat-template-kwargs '{"enable_thinking":true,"thinking":true,"reasoning_effort":"low"}' \
  "${extra[@]}"
RC=$?
set -e
T1="$(date +%s.%N)"
python3 - "$RUN_ID" "$T0" "$T1" "$WORKERS" "$RC" <<'PY'
import json, sys
run_id, t0, t1, workers, rc = sys.argv[1:6]
json.dump({"run_id": run_id, "workers": int(workers), "wall_seconds": round(float(t1) - float(t0), 1),
           "runner_exit_code": int(rc)}, open("wall-%s.json" % run_id, "w"), indent=1)
PY
echo "runner exit code $RC; rows: $OUT/$RUN_ID.rows.jsonl"
exit "$RC"
