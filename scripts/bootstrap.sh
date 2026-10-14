#!/usr/bin/env bash
# One-liner bootstrap:
#
#   curl -fsSL https://raw.githubusercontent.com/r0b0tlab/minicpm-v47-exl3-exllamav3-gb10/v2.1.0/scripts/bootstrap.sh | bash
#
# Clones this repository at the pinned tag, verifies the sha256 of scripts/click_run.sh and
# scripts/verify_tree.py against the values embedded below BEFORE executing anything from the checkout, runs the
# offline tree check, then hands over to ./run.sh (all arguments are passed on).
#
# Environment: BOOTSTRAP_DIR (target directory), BOOTSTRAP_REF (tag or commit), BOOTSTRAP_NO_RUN=1 (verify only),
# BOOTSTRAP_REPO_URL (a mirror).
set -Eeuo pipefail

REPO_URL="${BOOTSTRAP_REPO_URL:-https://github.com/r0b0tlab/minicpm-v47-exl3-exllamav3-gb10}"
REPO_REF="${BOOTSTRAP_REF:-v2.1.0}"
DEST="${BOOTSTRAP_DIR:-minicpm-v47-exl3-exllamav3-gb10}"
# kept current by scripts/update_bootstrap_hashes.py (scripts/verify_tree.py fails when they are stale)
EXPECT_CLICK_RUN_SHA256="53a116318ea78c45f9a2e1b34d34c9643017bcbfbb3493bc6a2aa21df1e6bdd1"
EXPECT_VERIFY_TREE_SHA256="99cb5ea35ef9c45ab321c0f6ed7cf6634b3240f5e5b204dbc2d91db944104bd1"

die() { echo "bootstrap: $*" >&2; exit 1; }
command -v git >/dev/null 2>&1 || die "git is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

if [[ -e "$DEST/.git" ]]; then
  echo "bootstrap: reusing existing checkout $DEST"
  git -C "$DEST" fetch --depth 1 origin "$REPO_REF" >/dev/null 2>&1 || true
else
  git clone --quiet --depth 1 --branch "$REPO_REF" "$REPO_URL" "$DEST" 2>/dev/null \
    || { git init -q "$DEST" && git -C "$DEST" remote add origin "$REPO_URL" \
         && git -C "$DEST" fetch --quiet --depth 1 origin "$REPO_REF" && git -C "$DEST" checkout -q FETCH_HEAD; } \
    || die "could not fetch $REPO_URL at $REPO_REF"
fi

check() {
  local file="$1" want="$2" have
  have="$(sha256sum "$DEST/$file" | cut -d' ' -f1)"
  if [[ "$have" != "$want" ]]; then
    die "$file has sha256 $have, expected $want: refusing to execute anything from this checkout"
  fi
}
check scripts/click_run.sh "$EXPECT_CLICK_RUN_SHA256"
check scripts/verify_tree.py "$EXPECT_VERIFY_TREE_SHA256"
echo "bootstrap: click_run.sh and verify_tree.py match the embedded hashes"

python3 "$DEST/scripts/verify_tree.py" --root "$DEST" --skip-public \
  || die "tree check failed (see the findings above)"
if [[ "${BOOTSTRAP_NO_RUN:-0}" == 1 ]]; then
  echo "bootstrap: BOOTSTRAP_NO_RUN=1, not starting the server"
  exit 0
fi
cd "$DEST"
exec ./run.sh "$@"
