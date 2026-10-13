#!/usr/bin/env python3
"""Rewrite the two embedded sha256 values of scripts/bootstrap.sh from the files they protect.

Run after any change to scripts/click_run.sh or scripts/verify_tree.py (scripts/verify_tree.py fails when stale).
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {"EXPECT_CLICK_RUN_SHA256": "scripts/click_run.sh", "EXPECT_VERIFY_TREE_SHA256": "scripts/verify_tree.py"}


def digest(rel, root=ROOT):
    with open(os.path.join(root, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(root=ROOT):
    path = os.path.join(root, "scripts", "bootstrap.sh")
    text = open(path).read()
    for var, rel in TARGETS.items():
        text = re.sub(r'^%s="[0-9a-f]{64}"$' % var, '%s="%s"' % (var, digest(rel, root)), text, flags=re.M)
    open(path, "w").write(text)
    print("bootstrap.sh hashes updated")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT)
