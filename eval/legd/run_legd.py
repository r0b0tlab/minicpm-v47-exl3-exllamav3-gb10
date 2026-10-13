#!/usr/bin/env python3
"""Leg D runner: the frozen 80-prompt kit, plain vs drafted decode, through the real server (stdlib only).

The prompts are token-id lists (40 GSM8K, 20 HumanEval = "code", 20 IFEval = "chat" prompts of the Q200v2 text kit,
rendered with the pack's chat template, thinking OFF, `max_new` 320). Two arms of prompts:

  greedy   all 80, temperature 0
  sampled  the first 10 of every class (30), temperature 1.0 / top_k 20 / top_p 0.95 (the pack's defaults)

and a number of concurrent streams (1 by default; 4 sends four prompts at a time and reports aggregate tokens/s per
group). Every request records the engine's own decode timing (tokens after the first over the decode seconds), the
completion token count, and a sha256 of the generated token ids, so a drafted run can be compared byte-for-byte
with the plain run of the same prompts (--compare).

    eval/legd/run_legd.py --base http://127.0.0.1:8000 --out OUT_DIR --label plain [--streams 1]
    eval/legd/run_legd.py --report OUT_DIR/legd-plain.rows.jsonl [--compare other.rows.jsonl]

Drafted runs use the same command against a server started with a DSpark draft (--draft / DRAFT_DIR).
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
import oai_client as oc  # noqa: E402

DEFAULT_PROMPTS = os.path.join(HERE, "legd_prompts.json")
CLASSES = ["gsm8k", "code", "chat"]
SAMPLED = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}


def rows_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_prompts(path):
    doc = json.load(open(path))
    if rows_hash(doc["rows"]) != doc["rows_sha256"]:
        raise SystemExit("legd prompt file does not match its rows_sha256: refusing to run")
    return doc


def plan(doc, arms, sample_n):
    out = []
    per_class = {}
    for i, r in enumerate(doc["rows"]):
        k = per_class.setdefault(r["class"], 0)
        per_class[r["class"]] += 1
        if "greedy" in arms:
            out.append({"mode": "greedy", "idx": i, "row": r})
        if "sampled" in arms and k < sample_n:
            out.append({"mode": "sampled", "idx": i, "row": r})
    return out


def one(base, model, item, max_new):
    row = item["row"]
    body = {"model": model, "prompt": row["ids"], "max_tokens": min(max_new, row["max_new"]), "return_token_ids": True}
    if item["mode"] == "greedy":
        body["temperature"] = 0
    else:
        body.update(SAMPLED)
        body["seed"] = 1000 + item["idx"]
    t0 = time.perf_counter()
    r = oc.post_json(base, "/v1/completions", body, timeout=3600)
    wall = time.perf_counter() - t0
    tm = r.get("timings", {})
    ids = r["choices"][0].get("token_ids") or []
    n = r["usage"]["completion_tokens"]
    det = r["usage"].get("completion_tokens_details", {})
    rec = {
        "mode": item["mode"], "name": row["name"], "class": row["class"], "prompt_len": row["prompt_len"],
        "tokens": n, "decode_tokens": max(0, n - 1), "decode_s": tm.get("decode_s"), "prefill_s": tm.get("prefill_s"),
        "tok_s": tm.get("decode_tok_s"), "wall_s": round(wall, 3),
        "tokens_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "stopped": r["choices"][0]["finish_reason"] == "stop",
    }
    if "accepted_prediction_tokens" in det:
        rec["accepted"] = det["accepted_prediction_tokens"]
        rec["rejected"] = det["rejected_prediction_tokens"]
    return rec


def run(base, model, doc, arms, streams, sample_n, label, log=print):
    card = oc.first_model(base)
    model = model or card["id"]
    items = plan(doc, arms, sample_n)
    rows = []
    t_start = time.perf_counter()
    max_new = int(doc.get("max_new", 320))
    # warm-up
    oc.post_json(base, "/v1/completions", {"model": model, "prompt": doc["rows"][0]["ids"], "max_tokens": 8,
                                           "temperature": 0})
    groups = []
    for mode in ("greedy", "sampled"):
        sel = [it for it in items if it["mode"] == mode]
        for g in range(0, len(sel), streams):
            groups.append(sel[g: g + streams])
    for gi, grp in enumerate(groups):
        t0 = time.perf_counter()
        if streams == 1:
            recs = [one(base, model, grp[0], max_new)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(grp)) as pool:
                recs = list(pool.map(lambda it: one(base, model, it, max_new), grp))
        wall = time.perf_counter() - t0
        group_tokens = sum(r["tokens"] for r in recs)
        for r in recs:
            r.update(streams=streams, group=gi, group_tokens=group_tokens, group_wall_s=round(wall, 3), label=label)
            rows.append(r)
        if gi % 10 == 0:
            log("  group %d/%d: %s %s %.1f tok/s" % (gi + 1, len(groups), recs[0]["mode"], recs[0]["name"],
                                                    recs[0]["tok_s"] or 0))
    return rows, round(time.perf_counter() - t_start, 1)


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n), n


def fmt(m, se, nd=1):
    if m != m:
        return "-"
    return "%.*f" % (nd, m) if se != se else "%.*f +-%.*f" % (nd, m, nd, se)


def report(rows, compare=None):
    lines, summary = [], {}
    other = {}
    if compare:
        for r in compare:
            other[(r["mode"], r["name"])] = r["tokens_sha256"]
    for mode in ("greedy", "sampled"):
        mr = [r for r in rows if r["mode"] == mode]
        if not mr:
            continue
        streams = mr[0].get("streams", 1)
        lines += ["", "### %s (%d stream%s)" % (mode, streams, "" if streams == 1 else "s"), "",
                  "| class | n | tok/s mean +-se | tok/s pooled | mean tokens | stopped |" + (
                      " accepted/(acc+rej) |" if any("accepted" in r for r in mr) else ""),
                  "|---|---:|---:|---:|---:|---:|" + ("---:|" if any("accepted" in r for r in mr) else "")]
        for cls in CLASSES + ["all"]:
            rs = [r for r in mr if cls == "all" or r["class"] == cls]
            if not rs:
                continue
            m, se, n = mean_se([r["tok_s"] for r in rs])
            if streams > 1:
                # aggregate over groups: tokens of the group / its wall seconds (prefill included)
                seen, agg = set(), []
                for r in rs:
                    if r["group"] not in seen:
                        seen.add(r["group"])
                        agg.append(r["group_tokens"] / r["group_wall_s"])
                m, se, n = mean_se(agg)
            dsum = sum(r["decode_s"] or 0 for r in rs)
            pooled = sum(r["decode_tokens"] for r in rs) / dsum if dsum else float("nan")
            line = "| %s | %d | %s | %.1f | %.0f | %d/%d |" % (cls, n, fmt(m, se), pooled,
                                                                 sum(r["tokens"] for r in rs) / len(rs),
                                                                 sum(1 for r in rs if r["stopped"]), len(rs))
            if any("accepted" in r for r in mr):
                acc = sum(r.get("accepted", 0) for r in rs)
                rej = sum(r.get("rejected", 0) for r in rs)
                line += " %.3f |" % (acc / (acc + rej)) if acc + rej else " - |"
            lines.append(line)
            summary["%s/%s" % (mode, cls)] = {"n": n, "tok_s_mean": m, "tok_s_se": se, "tok_s_pooled": pooled,
                                              "streams": streams}
    if other:
        diff = [r["name"] for r in rows if (r["mode"], r["name"]) in other and other[(r["mode"], r["name"])] != r["tokens_sha256"]]
        common = sum(1 for r in rows if (r["mode"], r["name"]) in other)
        lines += ["", "token-for-token comparison with the other run: %d of %d prompts identical (%s)" % (
            common - len(diff), common, "differing: " + ", ".join(diff[:10]) if diff else "all identical")]
        summary["compare"] = {"common": common, "identical": common - len(diff), "differing": diff}
    return "\n".join(lines), summary


def read_rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--label", default="plain")
    ap.add_argument("--arms", default="greedy,sampled")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--sample-n", type=int, default=10, help="sampled arm: first N prompts per class")
    ap.add_argument("--report", help="only render the report of this rows.jsonl")
    ap.add_argument("--compare", help="rows.jsonl of another run to compare token ids against")
    a = ap.parse_args(argv)
    cmp_rows = read_rows(a.compare) if a.compare else None
    if a.report:
        text, _ = report(read_rows(a.report), cmp_rows)
        print(text)
        return 0
    if not a.out:
        ap.error("--out is required")
    doc = load_prompts(a.prompts)
    arms = [x for x in a.arms.split(",") if x]
    print("Leg D (%s): %d prompts, arms %s, %d stream(s)" % (a.label, len(doc["rows"]), arms, a.streams))
    rows, wall = run(a.base, a.model, doc, arms, a.streams, a.sample_n, a.label)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "legd-%s-s%d.rows.jsonl" % (a.label, a.streams))
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    text, summary = report(rows, cmp_rows)
    summary["wall_seconds"] = wall
    summary["rows_sha256"] = doc["rows_sha256"]
    with open(os.path.join(a.out, "legd-%s-s%d.summary.json" % (a.label, a.streams)), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(text)
    print("LEGD_OK rows=%d wall=%.0fs -> %s" % (len(rows), wall, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
