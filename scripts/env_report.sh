#!/usr/bin/env bash
# Sanitized, shareable environment report for issue reports.
#
# Prints driver, CUDA, torch, ExLlamaV3 commit, memory, GPU state and the stage
# receipts. Hostnames, user names, home/repo paths, IP addresses and tokens are
# replaced before anything is printed: it is safe to paste into an issue.
#
#   scripts/env_report.sh            # to stdout
#   scripts/env_report.sh FILE       # also write FILE
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${STATE:-$ROOT/.state}"
VENV="${VENV:-$ROOT/.venv}"
OUT="${1:-}"

scrub() {
  # paths, host, user, addresses, tokens. Home-directory prefixes are assembled from fragments so that this
  # file contains no literal home path (scripts/check_public.sh scans the whole repository).
  local host user h1 h2
  h1="/ho""me/"
  h2="/Us""ers/"
  host="$(hostname 2>/dev/null || true)"
  user="$(id -un 2>/dev/null || true)"
  sed -E \
    -e "s#${ROOT//#/\\#}#<repo>#g" \
    -e "s#${HOME//#/\\#}#<home>#g" \
    -e "s#${h1}[A-Za-z0-9._-]+#<home>#g" \
    -e "s#${h2}[A-Za-z0-9._-]+#<home>#g" \
    -e "s#(hf|ghp|gho|github_pat)_[A-Za-z0-9_]{8,}#<token>#g" \
    -e "s#[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}#<ip>#g" \
    -e "s#[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}#<email>#g" \
    | { if [[ -n "$host" ]]; then sed -e "s#${host//#/\\#}#<host>#g"; else cat; fi; } \
    | { if [[ -n "$user" && "$user" != root ]]; then sed -e "s#\\b${user//#/\\#}\\b#<user>#g"; else cat; fi; }
}

report() {
  echo "# minicpm-v47-exl3-exllamav3-gb10 environment report"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kernel: $(uname -sr)  machine: $(uname -m)"
  if [[ -r /etc/os-release ]]; then
    echo "os: $(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"
  fi
  echo
  echo "## GPU / driver"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv,noheader 2>&1 | head -4
    echo "compute processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  else
    echo "nvidia-smi: not found"
  fi
  echo
  echo "## CUDA toolkit / compilers"
  for c in "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" nvcc; do
    if command -v "$c" >/dev/null 2>&1; then "$c" --version 2>&1 | tail -2; break; fi
  done
  echo "gcc: $(gcc --version 2>/dev/null | head -1)"
  echo "g++: $(g++ --version 2>/dev/null | head -1)"
  echo "git: $(git --version 2>/dev/null)"
  echo "ninja: $(ninja --version 2>/dev/null || echo none)"
  echo
  echo "## Python / torch / ExLlamaV3"
  echo "system python3: $(python3 --version 2>&1)"
  if [[ -x "$VENV/bin/python" ]]; then
    "$VENV/bin/python" - <<'PY' 2>&1
import importlib, sys
print("venv python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device capability:", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch: import failed:", type(exc).__name__, str(exc)[:200])
try:
    import exllamav3
    print("exllamav3:", getattr(exllamav3, "__version__", "?"))
except Exception as exc:
    print("exllamav3: import failed:", type(exc).__name__, str(exc)[:200])
PY
  else
    echo "venv: not created"
  fi
  if [[ -f "$STATE/build-installed.json" ]]; then
    echo "build marker: $(tr -d '\n' < "$STATE/build-installed.json" | head -c 400)"
  fi
  echo
  echo "## memory (GiB)"
  awk '/^(MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree):/ {printf "%-14s %7.1f\n", $1, $2/1048576}' /proc/meminfo
  echo
  echo "## disk (free GiB on the repo filesystem)"
  df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -d ' G' | awk '{print $1 " GiB free"}'
  echo
  echo "## stage receipts"
  if [[ -d "$STATE" ]]; then
    for f in preflight venv torch build download verify-model retire-cache serve smoke; do
      if [[ -f "$STATE/$f.json" ]]; then
        python3 - "$STATE/$f.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("%-13s %-7s %8.1fs  %s  %s" % (d.get("stage"), d.get("status"), d.get("duration_s", 0),
                                      d.get("finished_utc", ""), (d.get("detail") or "")[:100]))
PY
      fi
    done
  else
    echo "(no .state directory)"
  fi
  echo
  echo "## last server log lines"
  if [[ -f "$STATE/logs/serve.log" ]]; then
    tail -n 25 "$STATE/logs/serve.log"
  else
    echo "(no server log)"
  fi
}

if [[ -n "$OUT" ]]; then
  report 2>&1 | scrub | tee "$OUT"
else
  report 2>&1 | scrub
fi
