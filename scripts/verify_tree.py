#!/usr/bin/env python3
"""Offline conformance check of this repository against protocol.json (docs/PROTOCOL.md). Stdlib only.

  scripts/verify_tree.py [--root DIR] [--json] [--skip-public]

Checks, each with its own exit-code class (the exit status is the lowest failing class):

  10 layout     every required path exists
  11 protocol   protocol.json parses and agrees with config/profiles.yaml
  20 evidence   every bundle listed in protocol.json (and every bundle.json found under evidence/) has its files
                and a MANIFEST.sha256 that matches them byte for byte
  30 pins       runtime.lock.json == config/model.yaml == the commits / revisions / torch version quoted in the docs
                == the engine commit recorded in the bundles
  40 forbidden  scripts/check_public.sh finds no private vocabulary
  50 size       no file above the limit (except the allow-list) and the repository stays small
  60 generated  README / docs tables between GENERATED markers equal what the bundles render
  70 errors     every E_ code used by a script is defined in scripts/errors.tsv
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CLASSES = {10: "layout", 11: "protocol", 20: "evidence", 30: "pins", 40: "forbidden", 50: "size", 60: "generated",
           70: "errors"}


class Report(object):
    def __init__(self):
        self.findings = []

    def add(self, cls, msg):
        self.findings.append((cls, msg))

    def classes(self):
        return sorted({c for c, _ in self.findings})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root, exclude_dirs):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            yield os.path.join(base, f)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def check_layout(root, proto, rep):
    for rel in proto["required_paths"]:
        p = os.path.join(root, rel)
        ok = os.path.isdir(p) if rel.endswith("/") else os.path.isfile(p)
        if not ok:
            rep.add(10, "missing required path: %s" % rel)


def check_protocol(root, proto, rep):
    import cfg
    for key in ("protocol", "version", "required_paths", "profiles", "pins", "evidence_bundles", "generated_blocks",
                "size_limits", "exit_codes"):
        if key not in proto:
            rep.add(11, "protocol.json lacks %r" % key)
    if proto.get("version") != 2:
        rep.add(11, "protocol.json version is %r, expected 2" % proto.get("version"))
    try:
        prof = cfg.load_profiles(root)
    except Exception as exc:  # noqa: BLE001
        rep.add(11, "config/profiles.yaml unreadable: %s" % exc)
        return
    names = list(prof["profiles"])
    if sorted(names) != sorted(proto.get("profiles", {}).get("names", [])):
        rep.add(11, "profile names differ: protocol.json %s vs profiles.yaml %s" % (
            proto.get("profiles", {}).get("names"), names))
    if prof.get("default") != proto.get("profiles", {}).get("default"):
        rep.add(11, "default profile differs between protocol.json and profiles.yaml")


def parse_manifest(path):
    entries = {}
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            m = re.match(r"^([0-9a-f]{64})  (.+)$", line)
            if not m:
                raise ValueError("%s:%d: not 'sha256  path'" % (path, n))
            entries[m.group(2)] = m.group(1)
    return entries


def check_bundle_dir(root, rel, rep):
    d = os.path.join(root, rel)
    if not os.path.isdir(d):
        rep.add(20, "bundle directory missing: %s" % rel)
        return None
    bj = os.path.join(d, "bundle.json")
    mf = os.path.join(d, "MANIFEST.sha256")
    if not os.path.isfile(bj):
        rep.add(20, "%s has no bundle.json" % rel)
        return None
    try:
        bundle = json.loads(read(bj))
    except ValueError as exc:
        rep.add(20, "%s/bundle.json is not JSON: %s" % (rel, exc))
        return None
    for key in ("schema", "id", "column", "engine", "hardware", "date", "results", "caveats"):
        if key not in bundle:
            rep.add(20, "%s/bundle.json lacks %r" % (rel, key))
    if not os.path.isfile(mf):
        rep.add(20, "%s has no MANIFEST.sha256" % rel)
        return bundle
    try:
        entries = parse_manifest(mf)
    except ValueError as exc:
        rep.add(20, str(exc))
        return bundle
    on_disk = {os.path.relpath(p, d) for p in walk_files(d, set())} - {"MANIFEST.sha256"}
    for name, digest in entries.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            rep.add(20, "%s: manifest lists missing file %s" % (rel, name))
        elif sha256_file(p) != digest:
            rep.add(20, "%s: sha256 mismatch for %s" % (rel, name))
    for name in sorted(on_disk - set(entries)):
        rep.add(20, "%s: file %s is not in MANIFEST.sha256" % (rel, name))
    for lane, spec in (bundle.get("lanes") or {}).items():
        for key in ("file", "rows"):
            v = spec.get(key) if isinstance(spec, dict) else None
            for name in ([v] if isinstance(v, str) else (v or [])):
                if isinstance(name, str) and not os.path.isfile(os.path.join(d, name)):
                    rep.add(20, "%s: lane %s cites missing file %s" % (rel, lane, name))
    return bundle


def check_evidence(root, proto, rep):
    listed = {e["dir"] for e in proto.get("evidence_bundles", [])}
    bundles = {}
    for rel in sorted(listed):
        bundles[rel] = check_bundle_dir(root, rel, rep)
    ev = os.path.join(root, "evidence")
    if os.path.isdir(ev):
        for name in sorted(os.listdir(ev)):
            rel = "evidence/" + name
            if os.path.isdir(os.path.join(ev, name)) and os.path.isfile(os.path.join(ev, name, "bundle.json")) \
                    and rel not in listed:
                rep.add(20, "bundle %s exists but is not listed in protocol.json" % rel)
    return bundles


def check_pins(root, proto, bundles, rep):
    import cfg
    lock = json.loads(read(os.path.join(root, "runtime.lock.json")))
    model = cfg.load_model(root)
    for name in ("target", "draft"):
        art = model["artifacts"][name]
        lk = lock["models"][name]
        if art["hf_repo"] != lk["hf_repo"] or art["revision"] != lk["revision"]:
            rep.add(30, "%s: runtime.lock.json (%s@%s) != config/model.yaml (%s@%s)" % (
                name, lk["hf_repo"], lk["revision"][:12], art["hf_repo"], art["revision"][:12]))
    commit = lock["exllamav3"]["commit"]
    # results measured on an earlier pin of the same line stay valid evidence while they say so
    known = [commit] + [c for c in (lock["exllamav3"].get("base_commit"),) if c] + \
        list(lock["exllamav3"].get("bundle_commits", []))
    torch = lock["torch"]["version"]
    req = read(os.path.join(root, "requirements.txt"))
    if re.search(r"^torch\b", req, re.M):
        rep.add(30, "requirements.txt must not pin torch (runtime.lock.json owns it)")
    for rel in proto["pins"].get("readme_quotes", []):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        text = read(p)
        if commit[:7] not in text:
            rep.add(30, "%s does not quote the ExLlamaV3 commit %s" % (rel, commit[:7]))
        for m in re.finditer(r"commit\s+`?([0-9a-f]{7,40})`?", text):
            if not any(k.startswith(m.group(1)) for k in known):
                rep.add(30, "%s quotes commit %s which is not the pinned %s" % (rel, m.group(1), commit[:12]))
        for m in re.finditer(r"torch[ =]+(\d+\.\d+\.\d+)(\+cu\d+)?", text):
            if not torch.startswith(m.group(1)):
                rep.add(30, "%s quotes torch %s, lock pins %s" % (rel, m.group(1), torch))
        for name in ("target", "draft"):
            rev = lock["models"][name]["revision"]
            for m in re.finditer(r"%s\s*@\s*`?([0-9a-f]{7,40})`?" % re.escape(lock["models"][name]["hf_repo"]), text):
                if not rev.startswith(m.group(1)):
                    rep.add(30, "%s quotes %s@%s, lock pins %s" % (rel, name, m.group(1), rev[:12]))
    readme = os.path.join(root, "README.md")
    if os.path.isfile(readme):
        text = read(readme)
        for name in ("target", "draft"):
            if lock["models"][name]["revision"][:12] not in text:
                rep.add(30, "README.md does not quote the %s revision %s" % (name, lock["models"][name]["revision"][:12]))
    bs = os.path.join(root, "scripts", "bootstrap.sh")
    if os.path.isfile(bs):
        text = read(bs)
        for var, rel in (("EXPECT_CLICK_RUN_SHA256", "scripts/click_run.sh"),
                         ("EXPECT_VERIFY_TREE_SHA256", "scripts/verify_tree.py")):
            m = re.search(r'^%s="([0-9a-f]{64})"$' % var, text, re.M)
            have = sha256_file(os.path.join(root, rel))
            if not m or m.group(1) != have:
                rep.add(30, "scripts/bootstrap.sh %s is stale (run scripts/update_bootstrap_hashes.py)" % var)
    for rel, b in bundles.items():
        if not b:
            continue
        if str(b.get("column", "")).startswith("exllamav3"):
            c = (b.get("engine") or {}).get("commit")
            if c and c not in known:
                rep.add(30, "%s was measured on commit %s, lock pins %s" % (rel, c[:12], commit[:12]))


def check_forbidden(root, rep):
    script = os.path.join(root, "scripts", "check_public.sh")
    if not os.path.isfile(script):
        return
    p = subprocess.run(["bash", script, root], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if p.returncode != 0:
        for line in p.stdout.strip().splitlines()[:12]:
            rep.add(40, line)


def check_size(root, proto, rep):
    lim = proto["size_limits"]
    allow = set(lim.get("allow_large", []))
    total = 0
    for p in walk_files(root, set(proto.get("scan_exclude_dirs", []))):
        rel = os.path.relpath(p, root)
        size = os.path.getsize(p)
        total += size
        if size > lim["max_file_bytes"] and rel not in allow:
            rep.add(50, "%s is %d bytes (limit %d, not in allow_large)" % (rel, size, lim["max_file_bytes"]))
    if total > lim["max_repo_bytes"]:
        rep.add(50, "repository is %d bytes (limit %d)" % (total, lim["max_repo_bytes"]))


def check_generated(root, rep):
    import render_results
    bundles = render_results.load_bundles(root)
    proto = render_results.load_protocol(root)
    for rel, names in sorted(proto.get("generated_blocks", {}).items()):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            rep.add(60, "%s is missing" % rel)
            continue
        found, drift = render_results.process(path, bundles, write=False)
        for n in names:
            if n not in found:
                rep.add(60, "%s lacks the GENERATED block %s" % (rel, n))
        for n in drift:
            rep.add(60, "%s: block %s differs from the evidence bundles (run scripts/render_results.py --write)" % (rel, n))


def check_errors(root, rep):
    import errcode
    defined = set(errcode.load(os.path.join(root, "scripts", "errors.tsv")))
    used = {}
    for p in walk_files(os.path.join(root, "scripts"), {"__pycache__"}):
        if not p.endswith((".sh", ".py")) or os.path.basename(p) in ("verify_tree.py", "errcode.py"):
            continue
        for m in re.finditer(r"\b(E_[A-Z0-9_]{2,})\b", read(p)):
            used.setdefault(m.group(1), os.path.relpath(p, root))
    for code, where in sorted(used.items()):
        if code not in defined:
            rep.add(70, "%s used in %s is not defined in scripts/errors.tsv" % (code, where))


def run(root, skip_public=False):
    rep = Report()
    pj = os.path.join(root, "protocol.json")
    if not os.path.isfile(pj):
        rep.add(10, "protocol.json is missing")
        return rep
    try:
        proto = json.loads(read(pj))
    except ValueError as exc:
        rep.add(11, "protocol.json is not valid JSON: %s" % exc)
        return rep
    check_layout(root, proto, rep)
    check_protocol(root, proto, rep)
    bundles = check_evidence(root, proto, rep)
    try:
        check_pins(root, proto, bundles, rep)
    except (OSError, KeyError, ValueError) as exc:
        rep.add(30, "pin check could not run: %s" % exc)
    if not skip_public:
        check_forbidden(root, rep)
    check_size(root, proto, rep)
    try:
        check_generated(root, rep)
    except (OSError, KeyError, ValueError) as exc:
        rep.add(60, "generated-block check could not run: %s" % exc)
    try:
        check_errors(root, rep)
    except (OSError, ValueError) as exc:
        rep.add(70, "error-table check could not run: %s" % exc)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-public", action="store_true", help="skip the scripts/check_public.sh scan (class 40)")
    a = ap.parse_args(argv)
    rep = run(os.path.abspath(a.root), a.skip_public)
    classes = rep.classes()
    if a.json:
        print(json.dumps({"ok": not classes, "classes": classes,
                          "findings": [{"class": c, "name": CLASSES[c], "message": m} for c, m in rep.findings]}, indent=1))
    else:
        for c, m in rep.findings:
            print("[%d %s] %s" % (c, CLASSES[c], m))
        if classes:
            print("verify_tree: FAILED, classes %s" % ", ".join("%d %s" % (c, CLASSES[c]) for c in classes))
        else:
            print("verify_tree: OK (layout, protocol, evidence, pins, public scan, size, generated blocks, error codes)")
    return classes[0] if classes else 0


if __name__ == "__main__":
    sys.exit(main())
