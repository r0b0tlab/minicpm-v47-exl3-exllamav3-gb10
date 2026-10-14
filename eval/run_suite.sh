#!/usr/bin/env bash
# The evaluation suite against ANY OpenAI-compatible endpoint, writing an evidence bundle.
#
#   eval/run_suite.sh BASE_URL LABEL [--lanes smoke,bench,niah,legd,q200] [--workers 1] [--date YYYY-MM-DD]
#                     [--column exllamav3-plain|exllamav3-dspark] [--engine-name N]
#                     [--engine-commit SHA] [--mode TEXT] [--manual-evidence FILE] [--streams 1]
#
# Lanes (same methods as the reference measurements, so the columns compare side by side):
#   smoke  3-request smoke (scripts/smoke.py)
#   bench  decode / prefill / TTFT grid at 1k / 4k / 32k / 128k / 261k, 3 cold repeats, medians (scripts/bench.py)
#   niah   22 token-exact needle questions over a 261,001-token document, greedy, thinking off (scripts/exact_niah.py)
#   legd   the frozen 80-prompt plain-vs-drafted kit: greedy 80 + sampled 30, thinking off, <= 320 tokens
#          (PLAIN_ROWS=path/to/plain.rows.jsonl counts the drafted streams identical to the plain ones)
#   q200   Q200v2 text-180 with the public kit's runner (eval/q200/run_q200.sh); needs docker + SANDBOX_IMAGE_ID
#
# Raw outputs go to .state/eval/<label>-<date>/ (not committed); the bundle goes to evidence/<label>-<date>/ with
# bundle.json, public rows and MANIFEST.sha256. Add the bundle to protocol.json and run
# scripts/render_results.py --write to refresh the tables.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BASE_URL="${1:?usage: run_suite.sh BASE_URL LABEL [options]}"; LABEL="${2:?LABEL}"; shift 2
LANES="smoke,bench,niah,legd,q200"; WORKERS=1; STREAMS=1; DATE="$(date -u +%Y-%m-%d)"
COLUMN="exllamav3-plain"; ENGINE_NAME="ExLlamaV3"; ENGINE_COMMIT=""; MODE="plain decode (no draft)"; MANUAL=""
DEPTHS="1024,4096,32768,131072,261000"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes) LANES="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --streams) STREAMS="$2"; shift 2 ;;
    --date) DATE="$2"; shift 2 ;;
    --column) COLUMN="$2"; shift 2 ;;
    --engine-name) ENGINE_NAME="$2"; shift 2 ;;
    --engine-commit) ENGINE_COMMIT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --manual-evidence) MANUAL="$2"; shift 2 ;;
    --depths) DEPTHS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
if [[ -z "$ENGINE_COMMIT" && "$ENGINE_NAME" == ExLlamaV3 ]]; then
  ENGINE_COMMIT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["exllamav3"]["commit"])' "$ROOT/runtime.lock.json")"
fi
PY="${PYTHON:-python3}"
RAW="$ROOT/.state/eval/$LABEL-$DATE"
BUNDLE="$ROOT/evidence/$LABEL-$DATE"
mkdir -p "$RAW"
has() { [[ ",$LANES," == *",$1,"* ]]; }
say() { printf '[suite %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if has smoke; then say "smoke"; "$PY" "$ROOT/scripts/smoke.py" --base "$BASE_URL" --out "$RAW/smoke.json"; fi
if has bench; then say "bench"; "$PY" "$ROOT/scripts/bench.py" --base "$BASE_URL" --depths "$DEPTHS" --out "$RAW/bench.json"; fi
if has niah;  then say "niah";  "$PY" "$ROOT/scripts/exact_niah.py" --base "$BASE_URL" --out "$RAW/niah.json" || say "niah: not every question exact (recorded)"; fi
if has legd;  then
  say "legd (streams $STREAMS)"
  "$PY" "$ROOT/eval/legd/run_legd.py" --base "$BASE_URL" --out "$RAW" --label "$COLUMN" --streams "$STREAMS"
fi
if has q200;  then
  say "q200 (workers $WORKERS)"
  "$ROOT/eval/q200/run_q200.sh" "$BASE_URL" "$RAW/q200" --workers "$WORKERS" ${MANUAL:+--manual-evidence "$MANUAL"} --run-id "q200-w$WORKERS"
fi

args=(--out "$BUNDLE" --id "$LABEL-$DATE" --column "$COLUMN" --engine-name "$ENGINE_NAME" --engine-commit "$ENGINE_COMMIT"
      --mode "$MODE" --date "$DATE" --q200-workers "$WORKERS")
[[ -f "$RAW/smoke.json" ]] && args+=(--smoke "$RAW/smoke.json")
[[ -f "$RAW/bench.json" ]] && args+=(--bench "$RAW/bench.json")
[[ -f "$RAW/niah.json" ]] && args+=(--niah "$RAW/niah.json")
if [[ -f "$RAW/legd-$COLUMN-s$STREAMS.rows.jsonl" ]]; then
  # PLAIN_ROWS: the plain (no draft) rows of the same kit; counts how many drafted streams are identical to them
  args+=(--legd-run "=$RAW/legd-$COLUMN-s$STREAMS.rows.jsonl${PLAIN_ROWS:+:$PLAIN_ROWS}")
fi
Q200_ROWS="$(compgen -G "$RAW/q200/q200-w$WORKERS.rows.jsonl" || true)"
if [[ -n "$Q200_ROWS" ]]; then
  args+=(--q200-rows "$Q200_ROWS")
  [[ -f "$RAW/q200/wall-q200-w$WORKERS.json" ]] && args+=(--q200-wall "$RAW/q200/wall-q200-w$WORKERS.json")
  [[ -n "$MANUAL" ]] && args+=(--manual-evidence "$MANUAL" --grader-note "hard_reasoning graded item by item against the kit's reference answers (see q200.hard_reasoning.manual.json)")
fi
"$PY" "$HERE/tools/make_bundle.py" "${args[@]}"
say "bundle: evidence/$LABEL-$DATE"
