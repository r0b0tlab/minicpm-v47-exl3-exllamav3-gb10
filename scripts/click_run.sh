#!/usr/bin/env bash
# click_run.sh - one-command, idempotent, resumable start of the EXL3 model pack described by
# config/model.yaml on one NVIDIA GB10 (DGX Spark, aarch64, SM121) with ExLlamaV3.
#
#   ./run.sh                      # preflight -> venv -> torch -> build -> download -> verify -> serve -> smoke
#   ./run.sh --plan               # print what would run and why, change nothing
#   ./run.sh --profile gb10-text-32k --port 8001
#   ./run.sh --stop               # stop the server started by this checkout
#
# Stages (each writes .state/<stage>.json; unchanged completed stages are skipped):
#   preflight  venv  torch  build  download  verify-model  retire-cache  serve  smoke
#
# Environment / flags (flag wins):
#   PROFILE      --profile P      gb10-text-262k (default) | gb10-text-32k | gb10-text-16k-lowmem
#   PORT         --port N         listen port (default 8000)
#   HOST         --host H         bind address (default 127.0.0.1; no authentication, do not expose)
#   MODEL_DIR    --model-dir D    where the pack lives / is downloaded (default ./models/<repo name>)
#   HF_REPO      --hf-repo R      override the pinned repo (then the pack is "unpinned")
#   REVISION     --revision SHA   override the pinned revision
#   NO_DOWNLOAD=1 --no-download   use MODEL_DIR as is
#   SKIP_BUILD=1 --skip-build     do not (re)build the extension
#   SERVE_ONLY=1 --serve-only     only retire-cache, serve, smoke (venv/build/model assumed present)
#   ASSUME_YES=1 --yes            no confirmation prompts
#   DRY_RUN=1    --plan|--explain print the plan only
#   VENV, CACHE_DIR, BUILD_JOBS, DL_WORKERS, READY_TIMEOUT, PYTHON, GUARD=1, CHUNK_SIZE (--chunk-size N), BATCH_SIZE (--batch-size N; 1 is the qualified value, see docs/QUALIFICATION.md),
#   RETIRE_DIRS="dirA dirB" (more model directories whose shard page cache should be retired too)
#   --from S | --only S | --until S | --force | --verify-model | --allow-busy
#   --with-draft (download the DSpark draft pack and serve with drafting; profile gb10-text-262k-dspark does the same)
#   --draft DIR (serve with this draft directory, no download) | --draft-mode adaptive|fixed | --draft-only (download only)
#   --stop | --status | --env-report | --help
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
cd "$ROOT"

STAGES=(preflight venv torch build download verify-model retire-cache serve smoke)

# ---------------------------------------------------------------- defaults
PROFILE="${PROFILE:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MODEL_DIR="${MODEL_DIR:-}"
DRAFT_DIR="${DRAFT_DIR:-}"   # set => enable DSpark drafting with this pack (the server refuses cleanly if unsupported)
HF_REPO="${HF_REPO:-}"
REVISION="${REVISION:-}"
NO_DOWNLOAD="${NO_DOWNLOAD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SERVE_ONLY="${SERVE_ONLY:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
DRY_RUN="${DRY_RUN:-0}"
GUARD="${GUARD:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-}"      # prefill chunk override (default: the profile's chunk_size)
BATCH_SIZE="${BATCH_SIZE:-}"      # max concurrent sequences override (default: the profile's max_batch_size)
ARTIFACTS="target"
WANT_DRAFT=0                      # --with-draft or a drafting profile: download the draft pack and serve with it
DRAFT_MODE="${DRAFT_MODE:-}"      # adaptive | fixed (default: the profile's)
ALLOW_BUSY=0
FORCE=0
FORCE_VERIFY=0
FROM_STAGE=""
ONLY_STAGE=""
UNTIL_STAGE=""
ACTION="run"
STATE="${STATE:-$ROOT/.state}"
VENV="${VENV:-$ROOT/.venv}"
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/minicpm-v47-exl3-gb10}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 8)}"
DL_WORKERS="${DL_WORKERS:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
TORCH_INDEX_EXTRA="https://pypi.org/simple"

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --no-download) NO_DOWNLOAD=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --serve-only) SERVE_ONLY=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --plan|--explain|--dry-run) DRY_RUN=1; shift ;;
    --from) FROM_STAGE="$2"; shift 2 ;;
    --only) ONLY_STAGE="$2"; shift 2 ;;
    --until) UNTIL_STAGE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --verify-model) FORCE_VERIFY=1; shift ;;
    --with-draft) WANT_DRAFT=1; shift ;;
    --draft) DRAFT_DIR="$2"; shift 2 ;;
    --draft-mode) DRAFT_MODE="$2"; shift 2 ;;
    --draft-only) ARTIFACTS="draft"; shift ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    --stop) ACTION=stop; shift ;;
    --status) ACTION=status; shift ;;
    --env-report) ACTION='env'; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 64 ;;
  esac
done

for s in "$FROM_STAGE" "$ONLY_STAGE" "$UNTIL_STAGE"; do
  if [[ -n "$s" && " ${STAGES[*]} " != *" $s "* ]]; then
    python3 "$SCRIPTS/errcode.py" show E_USAGE --detail "unknown stage '$s'" >&2
    exit 64
  fi
done

mkdir -p "$STATE/logs"
# one run.sh per checkout at a time (the lock fd is closed for the background server)
exec 9> "$STATE/run.lock"
if ! flock -n 9; then
  python3 "$SCRIPTS/errcode.py" show E_USAGE --detail "another run.sh of this checkout is active (.state/run.lock)" >&2
  exit 64
fi

# ---------------------------------------------------------------- config (no model constants in shell)
cfgenv() { python3 "$SCRIPTS/cfg.py" "$@"; }
if [[ -z "$PROFILE" ]]; then PROFILE="$(cfgenv profiles | head -n1)"; fi
PROFILE_ENV="$(cfgenv env "$PROFILE" 2>&1)" \
  || { python3 "$SCRIPTS/errcode.py" show E_USAGE --detail "unknown profile '$PROFILE': $PROFILE_ENV" >&2; exit 64; }
eval "$PROFILE_ENV"
MODEL_ARCH="$(cfgenv get model.arch)"
WEIGHTS_GIB="$(cfgenv get model.weights_gib)"
eval "$(cfgenv artifact target | sed 's/^/T_/')"
eval "$(cfgenv artifact draft | sed 's/^/D_/')"
[[ -n "$MODEL_DIR" ]] || MODEL_DIR="$ROOT/models/$T_ART_DIR"
# shellcheck disable=SC2153
DRAFT_PATH="${DRAFT_DIR:-$(dirname "$MODEL_DIR")/$D_ART_DIR}"
# a drafting profile or --with-draft: fetch the draft pack and serve with it attached
if [[ "$P_DRAFT" == 1 ]]; then WANT_DRAFT=1; fi
if [[ "$WANT_DRAFT" == 1 ]]; then
  [[ -n "$DRAFT_DIR" ]] || DRAFT_DIR="$DRAFT_PATH"
  if [[ "$ARTIFACTS" == target ]]; then ARTIFACTS="target draft"; fi
fi
TORCH_PIN="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["torch"]["version"])' "$ROOT/runtime.lock.json")"
TORCH_INDEX="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["torch"]["index_url"])' "$ROOT/runtime.lock.json")"
EXL_REPO="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["exllamav3"]["repo"])' "$ROOT/runtime.lock.json")"
EXL_COMMIT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["exllamav3"]["commit"])' "$ROOT/runtime.lock.json")"
ARCH_LIST="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["arch"]["torch_cuda_arch_list"])' "$ROOT/runtime.lock.json")"

MC_PYTHON="$(python3 "$SCRIPTS/preflight.py" --find-python 2>/dev/null || true)"
MC_PYTHON_VERSION=""
if [[ -n "$MC_PYTHON" ]]; then
  MC_PYTHON_VERSION="$("$MC_PYTHON" -c 'import platform;print(platform.python_version())' 2>/dev/null || true)"
fi

export ROOT STATE VENV MODEL_DIR DRAFT_PATH CACHE_DIR PROFILE HOST PORT HF_REPO REVISION \
  NO_DOWNLOAD SKIP_BUILD SERVE_ONLY SKIP_VERIFY ARTIFACTS MC_PYTHON MC_PYTHON_VERSION
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

CURRENT_STAGE=""
T_BEGIN=$SECONDS
say() { printf '[%4ss] %s\n' "$((SECONDS - T_BEGIN))" "$*"; }

die() {
  local code="$1"; shift
  local detail="${*:-}"
  python3 "$SCRIPTS/errcode.py" show "$code" --stage "$CURRENT_STAGE" --detail "$detail" >&2
  if [[ -n "$CURRENT_STAGE" ]]; then
    python3 "$SCRIPTS/stagectl.py" record "$CURRENT_STAGE" --status failed \
      --duration "$((SECONDS - STAGE_T0))" --detail "$code: $detail" >/dev/null 2>&1 || true
  fi
  bash "$SCRIPTS/env_report.sh" "$STATE/env-report.txt" >/dev/null 2>&1 || true
  echo "  report: .state/env-report.txt (sanitized; safe to paste into an issue)" >&2
  exit 1
}
STAGE_T0=$SECONDS

confirm() {
  # $1 = question. Proceeds without asking when non-interactive or ASSUME_YES.
  if [[ "$ASSUME_YES" == 1 || ! -t 0 ]]; then return 0; fi
  local ans
  read -r -p "$1 [y/N] " ans
  [[ "$ans" == y || "$ans" == Y || "$ans" == yes ]]
}

base_url() { echo "http://$HOST:$PORT"; }
health_ok() { curl -fsS -m 3 "$(base_url)/health" 2>/dev/null | grep -q '"status": *"ok"'; }
server_pid() { if [[ -f "$STATE/server.pid" ]]; then cat "$STATE/server.pid"; fi; }
server_alive() { local p; p="$(server_pid)"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

stop_server() {
  local p; p="$(server_pid)"
  if [[ -z "$p" ]] || ! kill -0 "$p" 2>/dev/null; then
    say "no server of this checkout is running"; rm -f "$STATE/server.pid"; return 0
  fi
  say "stopping server pid $p"
  kill -TERM "$p" 2>/dev/null || true
  local i
  for i in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  if kill -0 "$p" 2>/dev/null; then kill -KILL "$p" 2>/dev/null || true; fi
  local g; g="$(cat "$STATE/guard.pid" 2>/dev/null || true)"
  if [[ -n "$g" ]] && kill -0 "$g" 2>/dev/null; then kill -TERM "$g" 2>/dev/null || true; fi
  rm -f "$STATE/server.pid" "$STATE/guard.pid" "$STATE/serve.json"
  say "stopped"
}

# ---------------------------------------------------------------- non-run actions
case "$ACTION" in
  stop) stop_server; exit 0 ;;
  env) exec bash "$SCRIPTS/env_report.sh" ;;
  status)
    python3 "$SCRIPTS/stagectl.py" show
    if server_alive && health_ok; then
      say "server pid $(server_pid) healthy at $(base_url)"
    else
      say "no healthy server at $(base_url)"
    fi
    exit 0 ;;
esac

DECIDE_ARGS=()
[[ "$FORCE" == 1 ]] && DECIDE_ARGS+=(--force)
[[ -n "$FROM_STAGE" ]] && DECIDE_ARGS+=(--from "$FROM_STAGE")
[[ -n "$ONLY_STAGE" ]] && DECIDE_ARGS+=(--only "$ONLY_STAGE")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "profile $PROFILE  port $PORT  host $HOST"
  echo "model dir : $MODEL_DIR"
  echo "venv      : $VENV"
  echo "cache dir : $CACHE_DIR"
  echo "pins      : ExLlamaV3 ${EXL_COMMIT:0:12} | torch $TORCH_PIN | arch $ARCH_LIST"
  echo "pack      : ${HF_REPO:-$T_ART_REPO} @ ${REVISION:-$T_ART_REV}"
  if [[ -n "$DRAFT_DIR" ]]; then echo "drafting  : on (${DRAFT_MODE:-${P_DRAFT_MODE:-adaptive}}), draft pack $DRAFT_DIR"; else echo "drafting  : off"; fi
  python3 "$SCRIPTS/cfg.py" need "$PROFILE"
  echo
  python3 "$SCRIPTS/stagectl.py" plan "${DECIDE_ARGS[@]}"
  exit 0
fi

# ---------------------------------------------------------------- stage helpers
run_logged() {
  # run_logged LOGNAME cmd...   (output to terminal and .state/logs/LOGNAME.log)
  local log="$STATE/logs/$1.log"; shift
  echo "+ $*" >> "$log"
  "$@" 2>&1 | tee -a "$log"
}

wheel_dir() { python3 "$SCRIPTS/stagectl.py" wheel-dir; }

hf_cli() {
  if [[ -x "$VENV/bin/hf" ]]; then echo "$VENV/bin/hf"; else echo "$VENV/bin/python -m huggingface_hub.cli.hf"; fi
}

STAGE_DETAIL=""
STAGE_EXTRA=""

stage_preflight() {
  local args=(--profile "$PROFILE" --port "$PORT" --bind-host "$HOST" --path "$ROOT" \
              --path "$(dirname "$MODEL_DIR")" --path "$(dirname "$VENV")" --path "$CACHE_DIR" \
              --json-out "$STATE/preflight.json")
  args+=(--disk-needed-gib "$(python3 "$SCRIPTS/stagectl.py" disk-needed)")
  local verdict
  verdict="$(python3 "$SCRIPTS/stagectl.py" decide build "${DECIDE_ARGS[@]}" | cut -f1)"
  if [[ "$verdict" == RUN && -z "$(compgen -G "$(wheel_dir)/exllamav3-*.whl" || true)" ]]; then
    args+=(--need-build)
  fi
  [[ "$ALLOW_BUSY" == 1 ]] && args+=(--allow-busy)
  local rc=0
  python3 "$SCRIPTS/preflight.py" "${args[@]}" || rc=$?
  if [[ $rc -ne 0 ]]; then
    local code detail
    code="$(python3 -c 'import json,sys;d=[c for c in json.load(open(sys.argv[1])) if c["status"]=="fail"];print(d[0]["code"] if d else "E_USAGE")' "$STATE/preflight.json")"
    detail="$(python3 -c 'import json,sys;d=[c for c in json.load(open(sys.argv[1])) if c["status"]=="fail"];print("; ".join(c["code"]+": "+c["detail"] for c in d))' "$STATE/preflight.json")"
    die "$code" "$detail"
  fi
  STAGE_DETAIL="all checks passed for $PROFILE"
}

stage_venv() {
  [[ -n "$MC_PYTHON" ]] || die E_PYTHON "no Python 3.11 - 3.13 found"
  if [[ -x "$VENV/bin/python" ]]; then
    say "[venv] reusing $VENV ($("$VENV/bin/python" --version 2>&1))"
  else
    run_logged venv "$MC_PYTHON" -m venv "$VENV" || die E_VENV "python -m venv failed for $MC_PYTHON"
  fi
  "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || die E_VENV "pip is missing in $VENV"
  STAGE_DETAIL="python $("$VENV/bin/python" -c 'import platform;print(platform.python_version())')"
}

stage_torch() {
  local py="$VENV/bin/python"
  say "[torch] installing torch==$TORCH_PIN from $TORCH_INDEX"
  run_logged torch "$py" -m pip install --index-url "$TORCH_INDEX" --extra-index-url "$TORCH_INDEX_EXTRA" \
    "torch==$TORCH_PIN" || die E_PIP_TORCH "pip install torch==$TORCH_PIN failed"
  say "[torch] installing pinned requirements"
  run_logged torch "$py" -m pip install -r "$ROOT/requirements.txt" || die E_PIP_REQS "pip install -r requirements.txt failed"
  "$py" - "$TORCH_PIN" <<'PY' || die E_PIP_TORCH "installed torch does not match the pin"
import sys, torch
want = sys.argv[1]
assert torch.__version__ == want, (torch.__version__, want)
assert torch.version.cuda and torch.version.cuda.startswith("13"), torch.version.cuda
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
  STAGE_DETAIL="torch $TORCH_PIN"
}

stage_build() {
  local py="$VENV/bin/python" key wdir src t0 hit=0
  key="$(python3 "$SCRIPTS/stagectl.py" build-key)"
  wdir="$(wheel_dir)"
  if [[ -n "$(compgen -G "$wdir/exllamav3-*.whl" || true)" && "$FORCE" != 1 ]]; then
    hit=1
    say "[build] cache hit: $key"
  else
    local nvcc cuda_home
    nvcc="$(python3 "$SCRIPTS/preflight.py" --find-nvcc || true)"
    [[ -n "$nvcc" ]] || die E_NVCC "nvcc not found"
    cuda_home="$(dirname "$(dirname "$nvcc")")"
    src="$CACHE_DIR/src/exllamav3-${EXL_COMMIT:0:12}"
    mkdir -p "$src"
    if [[ ! -d "$src/.git" ]]; then git -C "$src" init -q; fi
    if ! git -C "$src" cat-file -e "${EXL_COMMIT}^{commit}" 2>/dev/null; then
      say "[build] fetching ExLlamaV3 ${EXL_COMMIT:0:12} from $EXL_REPO"
      git -C "$src" remote remove origin 2>/dev/null || true
      git -C "$src" remote add origin "$EXL_REPO"
      run_logged build git -C "$src" fetch --depth 1 origin "$EXL_COMMIT" || die E_CLONE "git fetch $EXL_REPO $EXL_COMMIT failed"
    fi
    git -C "$src" checkout -q --force "$EXL_COMMIT" || die E_CLONE "checkout of $EXL_COMMIT failed"
    [[ "$(git -C "$src" rev-parse HEAD)" == "$EXL_COMMIT" ]] || die E_COMMIT "HEAD is $(git -C "$src" rev-parse HEAD)"
    say "[build] compiling the CUDA extension for sm_${ARCH_LIST/./} with $BUILD_JOBS jobs (one-off; cached afterwards)"
    t0=$SECONDS
    rm -rf "$wdir.tmp"; mkdir -p "$wdir.tmp"
    (
      export TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS="$BUILD_JOBS" CUDA_HOME="$cuda_home"
      export PATH="$VENV/bin:$cuda_home/bin:$PATH"
      cd "$src"
      run_logged build "$py" -m pip wheel --no-build-isolation --no-deps -w "$wdir.tmp" "$src"
    ) || { tail -n 30 "$STATE/logs/build.log" >&2 || true; die E_BUILD "pip wheel failed (see .state/logs/build.log)"; }
    rm -rf "$wdir"; mkdir -p "$(dirname "$wdir")"; mv "$wdir.tmp" "$wdir"
    STAGE_EXTRA="{\"compile_seconds\": $((SECONDS - t0))}"
    say "[build] compiled in $((SECONDS - t0)) s"
  fi
  local whl
  whl="$(compgen -G "$wdir/exllamav3-*.whl" | head -n1)"
  run_logged build "$py" -m pip install --no-deps --force-reinstall "$whl" || die E_BUILD "pip install of the built wheel failed"
  "$py" - "$MODEL_ARCH" <<'PY' || die E_IMPORT "exllamav3 / extension import check failed"
import importlib
import sys
import exllamav3
from exllamav3.architecture.architectures import get_architectures
importlib.import_module("exllamav3.ext")
arch = sys.argv[1]
assert arch in get_architectures(), "architecture %s is not registered in this ExLlamaV3 build" % arch
print("exllamav3 import ok; %s registered" % arch)
PY
  printf '{"key": "%s", "commit": "%s", "wheel": "%s", "cache_hit": %s}\n' "$key" "$EXL_COMMIT" "$(basename "$whl")" \
    "$([[ $hit == 1 ]] && echo true || echo false)" > "$STATE/build-installed.json"
  if [[ -z "$STAGE_EXTRA" ]]; then STAGE_EXTRA="{\"cache_hit\": true}"; fi
  STAGE_DETAIL="ExLlamaV3 ${EXL_COMMIT:0:12} ($([[ $hit == 1 ]] && echo 'cached wheel' || echo 'compiled'))"
}

download_one() {
  local name="$1" repo rev dir
  eval "$(cfgenv artifact "$name" | sed 's/^/A_/')"
  repo="$A_ART_REPO"; rev="$A_ART_REV"; dir="$MODEL_DIR"
  [[ "$name" == draft ]] && dir="$DRAFT_PATH"
  if [[ "$name" == target ]]; then repo="${HF_REPO:-$repo}"; rev="${REVISION:-$rev}"; fi
  say "[download] $repo @ ${rev:0:12} -> $dir (anonymous, $DL_WORKERS workers, resumable)"
  mkdir -p "$dir"
  local attempt rc=1 cli
  cli="$(hf_cli)"
  for attempt in 1 2 3; do
    # shellcheck disable=SC2086
    if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$CACHE_DIR/hf" \
         $cli download "$repo" --revision "$rev" --local-dir "$dir" --max-workers "$DL_WORKERS" \
         2>&1 | tee -a "$STATE/logs/download.log"; then
      rc=0; break
    fi
    say "[download] attempt $attempt failed; retrying in $((attempt * 15)) s"
    sleep $((attempt * 15))
  done
  if [[ $rc -ne 0 ]]; then
    if grep -qE '429|Too Many Requests|rate limit' "$STATE/logs/download.log" 2>/dev/null; then
      die E_DOWNLOAD "Hugging Face rate limit (HTTP 429) for $repo"
    fi
    die E_DOWNLOAD "hf download of $repo failed"
  fi
}

stage_download() {
  local t0=$SECONDS a
  confirm "Download the model files now ($ARTIFACTS; about $WEIGHTS_GIB GiB for the target pack)?" || die E_USAGE "download declined"
  for a in $ARTIFACTS; do download_one "$a"; done
  STAGE_DETAIL="downloaded in $((SECONDS - t0)) s"
  STAGE_EXTRA="{\"download_seconds\": $((SECONDS - t0))}"
}

verify_one() {
  local name="$1" dir="$MODEL_DIR" args=()
  [[ "$name" == draft ]] && dir="$DRAFT_PATH"
  args=(--dir "$dir" --artifact "$name" --cache "$STATE/verify-cache-$name.json")
  [[ "$FORCE_VERIFY" == 1 ]] && args+=(--full)
  if [[ "$name" == target && ( -n "$HF_REPO" || -n "$REVISION" ) ]]; then
    if [[ ( -n "$HF_REPO" && "$HF_REPO" != "$T_ART_REPO" ) || ( -n "$REVISION" && "$REVISION" != "$T_ART_REV" ) ]]; then
      [[ -f "$dir/UPLOAD_MANIFEST.json" ]] || die E_NO_MANIFEST "no pinned manifest for ${HF_REPO:-$T_ART_REPO}@${REVISION:-$T_ART_REV} and no UPLOAD_MANIFEST.json in $dir"
      args+=(--manifest-file "$dir/UPLOAD_MANIFEST.json")
    fi
  fi
  python3 "$SCRIPTS/verify_model.py" "${args[@]}" || die E_VERIFY "$name pack differs from the pinned manifest"
}

stage_verify_model() {
  local t0=$SECONDS a
  for a in $ARTIFACTS; do verify_one "$a"; done
  STAGE_DETAIL="verified in $((SECONDS - t0)) s"
}

stage_retire_cache() {
  local out="$STATE/retire-cache.json" args=() d
  args=(--dir "$MODEL_DIR")
  # other model directories whose shards are filling the page cache (space-separated list)
  for d in ${RETIRE_DIRS:-}; do args+=(--dir "$d"); done
  python3 "$SCRIPTS/retire_page_cache.py" "${args[@]}" --out "$out" \
    --workspace-mib "$P_NONWEIGHT_MIB" \
    || die E_RETIRE "see $out"
  STAGE_DETAIL="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("memfree %d MiB, memavailable %d MiB" % (d["memfree_after_mib"], d["memavailable_after_mib"]))' "$out")"
}

stage_serve() {
  local log="$STATE/logs/serve.log" pid t0=$SECONDS i
  : > "$log"
  say "[serve] starting server: profile $PROFILE, window $P_WINDOW tokens, $(base_url)"
  (
    cd "$ROOT"
    local extra=()
    if [[ -n "$DRAFT_DIR" ]]; then extra+=(--draft "$DRAFT_DIR"); fi
    if [[ -n "$DRAFT_MODE" ]]; then extra+=(--draft-mode "$DRAFT_MODE"); fi
    if [[ -n "$CHUNK_SIZE" ]]; then extra+=(--chunk-size "$CHUNK_SIZE"); fi
    if [[ -n "$BATCH_SIZE" ]]; then extra+=(--max-batch-size "$BATCH_SIZE"); fi
    setsid nohup "$VENV/bin/python" "$SCRIPTS/serve_openai.py" --model-dir "$MODEL_DIR" --profile "$PROFILE" \
      --host "$HOST" --port "$PORT" "${extra[@]}" >> "$log" 2>&1 < /dev/null 9>&- &
    echo $! > "$STATE/server.pid"
  )
  pid="$(server_pid)"
  for ((i = 0; i < READY_TIMEOUT; i += 2)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 30 "$log" >&2 || true
      if grep -q "Insufficient VRAM in split" "$log"; then die E_LOAD_VRAM "see .state/logs/serve.log"; fi
      if grep -qE "DRAFT UNSUPPORTED|DRAFT PROBE FAILED" "$log"; then die E_DRAFT "$(grep -E 'DRAFT UNSUPPORTED|DRAFT PROBE FAILED' "$log" | head -n1 | cut -c1-200)"; fi
      die E_SERVE_EXIT "server pid $pid exited"
    fi
    if health_ok; then break; fi
    if (( i % 30 == 0 && i > 0 )); then say "[serve] still loading ($i s) ..."; fi
    sleep 2
  done
  if ! health_ok; then
    tail -n 30 "$log" >&2 || true
    die E_SERVE_TIMEOUT "not ready after $READY_TIMEOUT s"
  fi
  printf '{"profile": "%s", "port": %s, "host": "%s", "pid": %s}\n' "$PROFILE" "$PORT" "$HOST" "$pid" > "$STATE/serve.json"
  if [[ "$GUARD" == 1 ]]; then
    setsid nohup "$VENV/bin/python" "$SCRIPTS/guard.py" --pid "$pid" --out "$STATE/logs/guard.jsonl" \
      >> "$STATE/logs/guard.log" 2>&1 < /dev/null 9>&- &
    echo $! > "$STATE/guard.pid"
    say "[serve] memory guard armed (GUARD=1), log .state/logs/guard.jsonl"
  fi
  STAGE_DETAIL="ready after $((SECONDS - t0)) s (pid $pid)"
  STAGE_EXTRA="{\"ready_seconds\": $((SECONDS - t0))}"
}

stage_smoke() {
  "$VENV/bin/python" "$SCRIPTS/smoke.py" --base "$(base_url)" --model "$M_SERVED_NAME" \
    --out "$STATE/smoke-result.json" || die E_SMOKE "see output above"
  STAGE_DETAIL="3 requests ok"
  if [[ -n "$DRAFT_DIR" ]]; then
    # drafted requests must work and stay close to plain decoding: how many of N prompts are token-identical
    local parity
    parity="$("$VENV/bin/python" "$SCRIPTS/draft_parity.py" --base "$(base_url)" --n 12 --out "$STATE/draft-parity.json")" \
      || die E_SMOKE "drafted request failed: $parity"
    echo "$parity"
    STAGE_DETAIL="3 requests ok; $(printf '%s' "$parity" | tail -n1)"
  fi
}

run_stage() {
  local st="$1" line action reason fn
  line="$(python3 "$SCRIPTS/stagectl.py" decide "$st" "${DECIDE_ARGS[@]}")" || die E_STATE "stagectl decide $st failed"
  action="${line%%$'\t'*}"; reason="${line#*$'\t'}"
  if [[ "$st" == verify-model && "$FORCE_VERIFY" == 1 && "$action" == SKIP && -z "$ONLY_STAGE$FROM_STAGE" ]]; then
    action=RUN; reason="--verify-model"
  fi
  if [[ "$action" == SKIP ]]; then say "[$st] skip: $reason"; return 0; fi
  say "[$st] run: $reason"
  CURRENT_STAGE="$st"; STAGE_T0=$SECONDS; STAGE_DETAIL=""; STAGE_EXTRA=""
  fn="stage_${st//-/_}"
  "$fn"
  local rec=(--status ok --duration "$((SECONDS - STAGE_T0))" --detail "$STAGE_DETAIL")
  if [[ -n "$STAGE_EXTRA" ]]; then rec+=(--extra-json "$STAGE_EXTRA"); fi
  python3 "$SCRIPTS/stagectl.py" record "$st" "${rec[@]}" >/dev/null
  say "[$st] ok in $((SECONDS - STAGE_T0)) s${STAGE_DETAIL:+ - $STAGE_DETAIL}"
  CURRENT_STAGE=""
}

# ---------------------------------------------------------------- main
say "minicpm-v47-exl3-exllamav3-gb10: profile $PROFILE, window $P_WINDOW tokens, $HOST:$PORT"

# An instance of this checkout that is healthy with the same profile is reused.
if server_alive && health_ok && [[ -f "$STATE/serve.json" ]] \
   && grep -q "\"profile\": \"$PROFILE\"" "$STATE/serve.json" && [[ "$FORCE" != 1 && -z "$ONLY_STAGE" ]]; then
  say "server pid $(server_pid) already serving $PROFILE: reusing it (./run.sh --stop to stop)"
  run_stage smoke
else
  if server_alive; then
    say "a server of this checkout is running with another profile/port: stopping it first"
    stop_server
  fi
  for st in "${STAGES[@]}"; do
    run_stage "$st"
    if [[ -n "$UNTIL_STAGE" && "$st" == "$UNTIL_STAGE" ]]; then say "stopping after --until $UNTIL_STAGE"; break; fi
  done
fi

if [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == serve || "$ONLY_STAGE" == smoke ]] && server_alive && health_ok; then
  cat <<EOF

================================================================================
 Serving $M_SERVED_NAME
   base URL : $(base_url)/v1
   model    : $M_SERVED_NAME   (window $P_WINDOW tokens, thinking on by default)
   stop     : ./run.sh --stop        status: ./run.sh --status
   log      : .state/logs/serve.log

 curl $(base_url)/v1/chat/completions -H 'Content-Type: application/json' -d '{
   "model": "$M_SERVED_NAME",
   "messages": [{"role": "user", "content": "Why is the sky blue? One paragraph."}],
   "max_tokens": 1024}'

 No authentication: keep the default loopback bind, do not expose this port.
================================================================================
EOF
  python3 "$SCRIPTS/stagectl.py" show
fi
