#!/usr/bin/env bash
# Fetch the public Q200v2 kit at the pinned commit and verify its MANIFEST.sha256.
# Nothing from the kit is vendored in this repository (datasets included).
#
#   eval/fetch_kit.sh            # -> eval/.kit/<commit12>/ ; prints the subset directory
#   KIT_DIR=/path eval/fetch_kit.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/kit.lock.json"
get() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$LOCK" "$1"; }
REPO="$(get repo)"; COMMIT="$(get commit)"; SUBSET="$(get subset)"; MANIFEST="$(get manifest)"
DEST="${KIT_DIR:-$HERE/.kit/${COMMIT:0:12}}"
SUBDIR="$DEST/$SUBSET"
if [[ ! -f "$SUBDIR/$MANIFEST" ]]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote remove origin 2>/dev/null || true
  git -C "$DEST" remote add origin "$REPO"
  git -C "$DEST" config core.sparseCheckout true
  echo "$SUBSET/" > "$DEST/.git/info/sparse-checkout"
  git -C "$DEST" fetch --depth 1 origin "$COMMIT" >&2
  git -C "$DEST" checkout -q --force FETCH_HEAD
fi
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || { echo "kit checkout is not $COMMIT" >&2; exit 1; }
(cd "$SUBDIR" && sha256sum --quiet -c "$MANIFEST") >&2 || { echo "kit MANIFEST.sha256 verification FAILED" >&2; exit 1; }
want="$(get dataset_sha256)"
have="$(sha256sum "$SUBDIR/$(get dataset)" | cut -d' ' -f1)"
[[ "$want" == "$have" ]] || { echo "dataset hash mismatch: $have" >&2; exit 1; }
echo "$SUBDIR"
