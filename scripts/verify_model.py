#!/usr/bin/env python3
"""Verify a downloaded model directory against the pinned manifest.

The manifest (path, size, sha256) comes from config/model.yaml (artifact
"target" or "draft") or, for an unpinned repo, from the repo's own
UPLOAD_MANIFEST.json (--manifest-file). Size is always checked. sha256 is
recomputed, except that a file whose (size, mtime_ns) match a previous
successful receipt is trusted unless --full is given; the receipt lives next
to the stage receipts (.state/verify-cache.json), never inside the model tree.

exit status: 0 verified, 1 mismatch/missing, 3 usage/config error
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cfg  # noqa: E402

CHUNK = 16 * 1024 * 1024


def sha256_file(path, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_from_upload_manifest(path):
    doc = json.load(open(path))
    return [(f["path"], int(f["size"]), f["sha256"]) for f in sorted(doc["files"], key=lambda f: f["path"])]


def load_cache(path):
    if not path:
        return {}
    try:
        with open(path) as fh:
            doc = json.load(fh)
        return doc.get("files", {}) if doc.get("schema") == 1 else {}
    except (OSError, ValueError):
        return {}


def save_cache(path, files):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"schema": 1, "files": files}, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def verify(directory, entries, cache_path=None, full=False, workers=4, hasher=sha256_file, log=None):
    """Return {"ok": bool, "results": [...], "hashed_bytes": int, "cached_files": int}."""
    cache = {} if full else load_cache(cache_path)
    new_cache = {}
    results = []
    todo = []
    for path, size, sha in entries:
        full_path = os.path.join(directory, path)
        try:
            st = os.stat(full_path)
        except OSError:
            results.append({"path": path, "status": "missing", "size": size})
            continue
        if st.st_size != size:
            results.append({"path": path, "status": "size", "size": size, "actual_size": st.st_size})
            continue
        c = cache.get(path)
        if c and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns and c.get("sha256") == sha:
            results.append({"path": path, "status": "ok", "size": size, "how": "cached"})
            new_cache[path] = c
            continue
        todo.append((path, size, sha, full_path, st))

    hashed = 0

    def work(item):
        path, size, sha, full_path, st = item
        t0 = time.time()
        actual = hasher(full_path)
        return item, actual, time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for item, actual, dt in pool.map(work, todo):
            path, size, sha, full_path, st = item
            hashed += size
            if actual == sha:
                results.append({"path": path, "status": "ok", "size": size, "how": "hashed", "seconds": round(dt, 2)})
                new_cache[path] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
            else:
                results.append({"path": path, "status": "sha256", "size": size, "expected": sha, "actual": actual})
            if log:
                log("  %-48s %s (%.1f GiB in %.1fs)" % (path, "ok" if actual == sha else "SHA256 MISMATCH",
                                                        size / 1024 ** 3, dt))
    results.sort(key=lambda r: r["path"])
    ok = all(r["status"] == "ok" for r in results) and len(results) == len(entries)
    if ok or any(r["status"] == "ok" for r in results):
        # keep receipts only for files that verified, drop stale ones
        save_cache(cache_path, new_cache)
    return {"ok": ok, "results": results, "hashed_bytes": hashed,
            "cached_files": sum(1 for r in results if r.get("how") == "cached"),
            "files": len(entries)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--artifact", default="target")
    ap.add_argument("--manifest-file", help="use this UPLOAD_MANIFEST.json instead of config/model.yaml")
    ap.add_argument("--cache", help="receipt file for (size, mtime) -> sha256")
    ap.add_argument("--full", action="store_true", help="recompute every sha256")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", default=cfg.ROOT)
    args = ap.parse_args(argv)
    try:
        if args.manifest_file:
            entries = manifest_from_upload_manifest(args.manifest_file)
        else:
            entries = cfg.manifest(cfg.load_model(args.root), args.artifact)
    except (cfg.ConfigError, OSError, KeyError, ValueError) as exc:
        print("verify_model: %s" % exc, file=sys.stderr)
        return 3
    t0 = time.time()
    res = verify(args.dir, entries, cache_path=args.cache, full=args.full, workers=args.workers,
                 log=None if args.json else print)
    res["seconds"] = round(time.time() - t0, 1)
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        bad = [r for r in res["results"] if r["status"] != "ok"]
        missing_listed = len(entries) - len(res["results"])
        for r in bad:
            print("  %s: %s" % (r["path"], r["status"]))
        print("verify: %d/%d files ok (%d hashed %.1f GiB, %d from receipt) in %.1fs" % (
            sum(1 for r in res["results"] if r["status"] == "ok"), len(entries),
            len(res["results"]) - res["cached_files"] - len(bad), res["hashed_bytes"] / 1024 ** 3,
            res["cached_files"], res["seconds"]))
        if missing_listed:
            print("verify: %d manifest entries were not evaluated" % missing_listed)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
