"""Evidence-bundle helpers shared by the bundle tools (stdlib only).

A bundle is a directory evidence/<id>/ with bundle.json (method, kit + hash, hardware, engine + commit, date,
caveats, normalised results), the public data files it cites, and MANIFEST.sha256 (sha256sum format, every file
except itself). scripts/render_results.py turns bundle.json into the generated tables of README.md and
docs/RESULTS.md; scripts/verify_tree.py checks the manifests.
"""
import hashlib
import json
import os
import statistics

SCHEMA = "r0b0tlab.evidence-bundle.v1"
FAMILIES = ["gsm8k", "humaneval", "ifeval", "hard_reasoning"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory):
    names = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), directory)
            if rel != "MANIFEST.sha256":
                names.append(rel)
    with open(os.path.join(directory, "MANIFEST.sha256"), "w") as fh:
        for rel in sorted(names):
            fh.write("%s  %s\n" % (sha256_file(os.path.join(directory, rel)), rel))


def sanitize_q200_row(row, manual=None):
    """Public per-row record: no prompt, no response text, no identifiers of the serving stack."""
    grader = row.get("grader") or {}
    passed = row.get("passed")
    graded_by = "kit"
    if row["family"] == "hard_reasoning":
        if manual is not None and row["id"] in manual:
            passed, graded_by = bool(manual[row["id"]]), "manual"
        else:
            passed, graded_by = None, "ungraded"
    elif passed is not True and passed is not False:
        passed, graded_by = None, "ungraded"
    return {
        "id": row["id"], "family": row["family"], "passed": passed, "graded_by": graded_by,
        "finish_reason": row.get("finish_reason"), "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "elapsed_seconds": round(float(row.get("elapsed_seconds") or row.get("elapsed") or 0.0), 3),
        "content_sha256": row.get("content_sha256"),
    }


def q200_results(public_rows, workers, wall_seconds=None):
    fam = {}
    for f in FAMILIES:
        rs = [r for r in public_rows if r["family"] == f]
        fam[f] = {"n": len(rs), "passed": sum(1 for r in rs if r["passed"] is True),
                  "ungraded": sum(1 for r in rs if r["passed"] is None),
                  "failed": sum(1 for r in rs if r["passed"] is False)}
    tokens = sum(r["completion_tokens"] or 0 for r in public_rows)
    sum_elapsed = sum(r["elapsed_seconds"] for r in public_rows)
    per_req = [r["completion_tokens"] / r["elapsed_seconds"] for r in public_rows
               if r["elapsed_seconds"] and r["completion_tokens"]]
    finish = {}
    for r in public_rows:
        finish[r["finish_reason"]] = finish.get(r["finish_reason"], 0) + 1
    out = {
        "workers": workers, "n": len(public_rows), "passed": sum(f["passed"] for f in fam.values()),
        "families": fam, "completion_tokens": tokens, "finish_reasons": finish,
        "sum_request_seconds": round(sum_elapsed, 1),
        "per_request_tok_s_median": round(statistics.median(per_req), 1) if per_req else None,
        "e2e_tok_s": round(tokens / wall_seconds, 1) if wall_seconds else (round(tokens / sum_elapsed, 1) if workers == 1 and sum_elapsed else None),
        "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
    }
    per_family = {}
    for f in FAMILIES:
        rs = [r for r in public_rows if r["family"] == f]
        t = sum(r["completion_tokens"] or 0 for r in rs)
        s = sum(r["elapsed_seconds"] for r in rs)
        per_family[f] = {"completion_tokens": t, "request_seconds": round(s, 1),
                         "tok_s_serial": round(t / s, 1) if (s and workers == 1) else None}
    out["per_family_throughput"] = per_family
    return out


def write_bundle(directory, doc):
    os.makedirs(directory, exist_ok=True)
    doc = dict(doc)
    doc["schema"] = SCHEMA
    with open(os.path.join(directory, "bundle.json"), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    write_manifest(directory)


def _mean_se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, (var / n) ** 0.5, n


def legd_from_rows(rows, compare_rows=None, streams=1):
    """Per mode/class summary of a Leg D rows file (run_legd.py): tok/s mean +- s.e., pooled tok/s, mean tokens,
    stop counts, drafting counters, and (with compare_rows) how many streams equal the other run's, by sha256."""
    other = {}
    for r in compare_rows or []:
        other[(r["mode"], r["name"])] = r["tokens_sha256"]
    out = {}
    for mode in ("greedy", "sampled"):
        for cls in ("gsm8k", "code", "chat", "all"):
            rs = [r for r in rows if r["mode"] == mode and (cls == "all" or r["class"] == cls)]
            if not rs:
                continue
            m, se, n = _mean_se([r.get("tok_s") for r in rs])
            dsum = sum(r.get("decode_s") or 0 for r in rs)
            pooled = sum(r.get("decode_tokens", 0) for r in rs) / dsum if dsum else None
            cell = {"n": n, "tok_s_mean": m, "tok_s_se": se, "tok_s_pooled": pooled,
                    "mean_tokens": sum(r["tokens"] for r in rs) / len(rs),
                    "stopped": sum(1 for r in rs if r.get("stopped")), "streams": streams}
            acc = sum(r.get("accepted", 0) for r in rs)
            rej = sum(r.get("rejected", 0) for r in rs)
            if any("accepted" in r for r in rs):
                cell.update(accepted=acc, rejected=rej, accept_rate=(acc / (acc + rej)) if acc + rej else None,
                            accepted_per_token=acc / max(1, sum(r["tokens"] for r in rs)))
            if other:
                comp = [r for r in rs if (r["mode"], r["name"]) in other]
                cell.update(compared=len(comp),
                            identical_to_plain=sum(1 for r in comp if other[(r["mode"], r["name"])] == r["tokens_sha256"]))
            out["%s/%s" % (mode, cls)] = cell
    return out
