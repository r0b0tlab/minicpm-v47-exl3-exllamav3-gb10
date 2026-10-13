#!/usr/bin/env python3
"""Token-exact needle retrieval over the shipped 261,001-token document (stdlib only).

For every (cut, needle) question of eval/niah/niah_261k.json.gz the prompt is the document's first `cut` token ids
followed by the question suffix ids (thinking off), sent greedily to /v1/completions as an id list, so there is no
tokenizer or template ambiguity on the client side. A question is EXACT when the generated text, stripped, equals the
seven-digit passcode. Cuts are asked in ascending order so the server's paged prefix cache carries each prefix
forward; the first question of every cut therefore costs only the new tokens, and the running sum of those prefill
times is the cold prefill time of the cut (reported as cold_prefill_cumulative_s).

    scripts/exact_niah.py --base http://127.0.0.1:8000 --out niah.json [--cuts 4096,32000] [--max-tokens 16]

Exit status 0 when every asked question is EXACT, 1 otherwise (a wrong answer is a result, not an error: the JSON
says whose passcode was returned instead).
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oai_client as oc  # noqa: E402

DEFAULT_DOC = os.path.join(os.path.dirname(HERE), "eval", "niah", "niah_261k.json.gz")


def load_doc(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def classify(text, answer, needles):
    t = text.strip()
    if t == answer:
        return "exact", None
    for i, n in enumerate(needles):
        if n["value"] in t:
            return "other_needle", i
    digits = re.findall(r"\d+", t)
    return ("wrong_digits" if digits else "no_digits"), None


def run(base, model, doc, cuts=None, max_tokens=16, log=print):
    card = oc.first_model(base)
    window = int(card.get("max_model_len", 0))
    model = model or card["id"]
    ids = doc["ids"]
    needles = doc["needles"]
    rows, cum_prefill, seen_cut = [], 0.0, set()
    mem0 = _metrics(base)
    for q in sorted(doc["questions"], key=lambda q: (q["cut"], q["needle"])):
        if cuts and q["cut"] not in cuts:
            continue
        prompt = ids[: q["cut"]] + q["suffix_ids"]
        row = {"cut": q["cut"], "needle": q["needle"], "depth": q["depth"], "prompt_tokens_sent": len(prompt)}
        if len(prompt) + max_tokens + 2 > window:
            row.update(status="skipped", reason="needs %d tokens, window is %d" % (len(prompt) + max_tokens + 2, window))
            rows.append(row)
            log("  cut %7d needle %d depth %.3f  SKIPPED (window)" % (q["cut"], q["needle"], q["depth"]))
            continue
        t0 = time.perf_counter()
        r = oc.post_json(base, "/v1/completions", {
            "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "return_token_ids": True}, timeout=7200)
        wall = time.perf_counter() - t0
        text = r["choices"][0]["text"]
        verdict, other = classify(text, q["answer"], needles)
        tm = r.get("timings", {})
        first_at_cut = q["cut"] not in seen_cut
        if first_at_cut:
            seen_cut.add(q["cut"])
            cum_prefill += tm.get("prefill_s", 0.0)
        row.update(status="ok", verdict=verdict, exact=verdict == "exact", answer=q["answer"], generated=text.strip()[:40],
                   other_needle=other, finish_reason=r["choices"][0]["finish_reason"],
                   prompt_tokens=r["usage"]["prompt_tokens"], completion_tokens=r["usage"]["completion_tokens"],
                   cached_tokens=tm.get("cached_tokens"), prefill_s=tm.get("prefill_s"), wall_s=round(wall, 2),
                   first_question_at_cut=first_at_cut,
                   cold_prefill_cumulative_s=round(cum_prefill, 2) if first_at_cut else None,
                   usage_matches=r["usage"]["prompt_tokens"] == len(prompt))
        rows.append(row)
        log("  cut %7d needle %d depth %.3f  %-12s prefill %6.2fs cached %6s  -> %r" % (
            q["cut"], q["needle"], q["depth"], verdict.upper(), tm.get("prefill_s", 0), tm.get("cached_tokens"),
            text.strip()[:24]))
    asked = [r for r in rows if r["status"] == "ok"]
    summary = {
        "window": window, "asked": len(asked), "skipped": len(rows) - len(asked),
        "exact": sum(1 for r in asked if r["exact"]),
        "wrong": [{"cut": r["cut"], "needle": r["needle"], "verdict": r["verdict"], "other_needle": r["other_needle"]}
                  for r in asked if not r["exact"]],
        "all_usage_matches": all(r["usage_matches"] for r in asked),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in asked), default=0),
        "memory_before": mem0, "memory_after": _metrics(base),
    }
    return {"summary": summary, "rows": rows, "model": model, "doc": {
        "tokens": doc["tokens"], "ids_sha256": doc.get("ids_sha256"), "needle_positions": [n["pos"] for n in needles],
        "cuts": doc["cuts"], "prose": doc.get("prose")}}


def _metrics(base):
    try:
        m = oc.get_json(base, "/metrics")
        return {k: m.get(k) for k in ("mem_available_gib", "torch_allocated_gib", "torch_reserved_gib",
                                      "torch_max_reserved_gib")}
    except Exception:  # noqa: BLE001
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--doc", default=DEFAULT_DOC)
    ap.add_argument("--cuts", default="", help="comma list of cuts to ask (default: all)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = load_doc(a.doc)
    cuts = [int(x) for x in a.cuts.split(",") if x] or None
    print("NIAH: %d questions in %s, window check against the server" % (len(doc["questions"]), os.path.basename(a.doc)))
    res = run(a.base, a.model, doc, cuts, a.max_tokens)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    s = res["summary"]
    print("NIAH: %d of %d exact (%d skipped); wrong: %s" % (s["exact"], s["asked"], s["skipped"], s["wrong"] or "none"))
    return 0 if s["asked"] and s["exact"] == s["asked"] else 1


if __name__ == "__main__":
    sys.exit(main())
