#!/usr/bin/env bash
# Public-content gate: fails (non-zero) if any shipped text file mentions private engine, host, user or path vocabulary.
#
#   scripts/check_public.sh [DIR]          # DIR defaults to the repository root
#
# The vocabulary below is assembled from fragments at run time so that this file itself contains none of the
# strings it looks for and needs no exclusion. Binary files are skipped; .git, virtualenvs, caches and the
# fetched kit checkout (eval/.kit, not part of this repository) are not scanned.
set -u
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# word-ish patterns (grep -E, case-insensitive)
PATTERNS=(
  "re""xl3"
  "gb""-model"
  "gb""-server"
  "node""[0-9]"
  "/ho""me/"
  "/Us""ers/"
  "r0b0t""dgx"
  "192""\\.168"
  "pro""-serve"
  "\\ba""dam\\b"
  "\\bman""ning\\b"
  "mr""-r0b0t"
  "spark""-[0-9a-f]{4}"
  "hf_[A-Za-z0-9]{20,}"
  "ghp_[A-Za-z0-9]{20,}"
  "-----BEGIN [A-Z ]*PRIVATE KEY"
)
# e-mail addresses: a name, an at-sign and a dotted domain
EMAIL="[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\\.[A-Za-z0-9-]+)*\\.[A-Za-z]{2,}"

hits=0
scan() {
  local pat="$1" label="$2"
  local out
  out="$(grep -rIniE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.kit \
            --exclude-dir=.state --exclude-dir=models --exclude-dir=.cache --exclude-dir=node_modules \
            -e "$pat" "$ROOT" 2>/dev/null | head -n 20 || true)"
  # compressed text (the generated needle document ships as .gz): look inside as well
  while IFS= read -r gz; do
    if zcat "$gz" 2>/dev/null | grep -qiE -e "$pat"; then
      out="${out:+$out
}${gz#"$ROOT"/}: (inside the gzip) matches"
    fi
  done < <(find "$ROOT" \( -name .git -o -name .venv -o -name .kit -o -name .state -o -name models -o -name .cache \) -prune -o -name '*.gz' -print 2>/dev/null)
  if [[ -n "$out" ]]; then
    echo "PUBLIC-CHECK HIT ($label):" >&2
    echo "$out" | sed "s#^$ROOT/##" | cut -c1-200 >&2
    hits=$((hits + 1))
  fi
}

for p in "${PATTERNS[@]}"; do
  scan "$p" "$p"
done
scan "$EMAIL" "e-mail address"

if [[ $hits -ne 0 ]]; then
  echo "check_public: FAILED ($hits pattern(s) matched)" >&2
  exit 1
fi
echo "check_public: OK (no private vocabulary found under $(basename "$ROOT"))"
