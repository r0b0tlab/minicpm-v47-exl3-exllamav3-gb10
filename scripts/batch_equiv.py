#!/usr/bin/env python3
"""Is batching correct on this pack? Compare serial and concurrent greedy decoding token for token (stdlib only).

Sends the same prompts twice to /v1/completions as token ids, greedy: first one at a time, then `--streams` at a
time against a server started with max_batch_size >= streams. A batched run is "equivalent" for a prompt when its
generated token ids equal the serial ones; the report gives how many prompts are identical, and for the others the
position of the first divergence and the number of tokens that agree before it.

    scripts/batch_equiv.py --base http://127.0.0.1:8000 --streams 4 --prompts 12 --max-tokens 256 --out batch.json

A MoE model with a recurrent (gated delta net) state can differ between batch sizes by fp16 rounding even when
every kernel is correct; an exact match on a decently sized set is the evidence this repository requires before it
offers a batched profile, and a divergence is reported rather than hidden.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oai_client as oc  # noqa: E402

DEFAULT_PROMPTS = os.path.join(os.path.dirname(HERE), "eval", "legd", "legd_prompts.json")


def gen(base, model, ids, max_tokens):
    r = oc.post_json(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": max_tokens,
                                               "temperature": 0, "return_token_ids": True}, timeout=3600)
    return r["choices"][0]["token_ids"], r.get("timings", {})


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def run(base, model, prompts, streams, max_tokens, log=print, serial_repeat=False):
    serial, concurrent_ = [], [None] * len(prompts)
    t0 = time.perf_counter()
    for p in prompts:
        serial.append(gen(base, model, p["ids"], max_tokens)[0])
    t_serial = time.perf_counter() - t0
    serial_again = None
    if serial_repeat:
        # is the serial arm itself deterministic? (same prompts, one at a time, again)
        serial_again = [gen(base, model, p["ids"], max_tokens)[0] for p in prompts]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        futs = {pool.submit(gen, base, model, p["ids"], max_tokens): i for i, p in enumerate(prompts)}
        for f in concurrent.futures.as_completed(futs):
            concurrent_[futs[f]] = f.result()[0]
    t_conc = time.perf_counter() - t0
    rows = []
    for i, p in enumerate(prompts):
        div = first_divergence(serial[i], concurrent_[i])
        rows.append({"name": p["name"], "tokens": len(serial[i]), "identical": div is None, "first_divergence": div,
                     "agree_prefix": len(serial[i]) if div is None else div})
        log("  %-12s %4d tokens  %s" % (p["name"], len(serial[i]), "identical" if div is None else "DIVERGES at %d" % div))
    total = sum(r["tokens"] for r in rows)
    extra = {}
    if serial_again is not None:
        extra["serial_vs_serial_identical"] = sum(1 for a, b in zip(serial, serial_again) if a == b)
    return {**extra, "streams": streams, "prompts": len(prompts), "max_tokens": max_tokens,
            "identical": sum(1 for r in rows if r["identical"]), "tokens": total,
            "serial_seconds": round(t_serial, 1), "concurrent_seconds": round(t_conc, 1),
            "serial_tok_s": round(total / t_serial, 1), "concurrent_tok_s": round(total / t_conc, 1), "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompts-file", default=DEFAULT_PROMPTS)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--serial-repeat", action="store_true", help="also run the serial arm twice")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    card = oc.first_model(a.base)
    model = a.model or card["id"]
    if card.get("max_batch_size", 1) < a.streams:
        print("note: the server reports max_batch_size %s < %d streams; the concurrent arm will be serialised"
              % (card.get("max_batch_size"), a.streams))
    doc = json.load(open(a.prompts_file))
    # spread over the three classes: every k-th prompt
    rows = doc["rows"]
    step = max(1, len(rows) // a.prompts)
    prompts = rows[::step][: a.prompts]
    res = run(a.base, model, prompts, a.streams, a.max_tokens, serial_repeat=a.serial_repeat)
    res["max_batch_size"] = card.get("max_batch_size")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("BATCH_EQUIV identical %d/%d, %d tokens; serial %.1f tok/s, %d streams %.1f tok/s%s" % (
        res["identical"], res["prompts"], res["tokens"], res["serial_tok_s"], a.streams, res["concurrent_tok_s"],
        "; serial vs serial identical %d/%d" % (res["serial_vs_serial_identical"], res["prompts"])
        if "serial_vs_serial_identical" in res else ""))
    return 0 if res["identical"] == res["prompts"] else 1


if __name__ == "__main__":
    sys.exit(main())
