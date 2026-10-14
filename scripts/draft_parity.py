#!/usr/bin/env python3
"""Drafted-vs-plain token parity probe (stdlib only).

Sends N prompts of the frozen Leg D kit (greedy, up to 320 new tokens, thinking off) to a server that has the DSpark
drafter attached and compares the generated token ids, by sha256, with the plain (no-draft) streams recorded in
evidence/exllamav3-plain-*/legd-*.rows.jsonl for the same prompts.

Speculative decoding verifies every draft with the target, so the output is the target's own; but the multi-row
verify kernels and the single-row decode kernel of this model differ by fp16 noise, so near ties can resolve the
other way and a stream can diverge from plain decoding without anything being wrong. This probe says how many
streams are identical; it fails only when requests fail.

    scripts/draft_parity.py --base http://127.0.0.1:8000 [--n 12] [--reference ROWS.jsonl] [--out FILE]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import oai_client as oc  # noqa: E402


def default_reference():
    pats = sorted(glob.glob(os.path.join(ROOT, "evidence", "exllamav3-plain-*", "legd-*-s1.rows.jsonl")))
    return pats[-1] if pats else None


def load_reference(path):
    ref = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("mode") == "greedy":
                ref[r["name"]] = r["tokens_sha256"]
    return ref


def pick(rows, n):
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def run(base, rows, ref, model=None):
    model = model or oc.first_model(base)["id"]
    res = []
    for r in rows:
        resp = oc.post_json(base, "/v1/completions", {"model": model, "prompt": r["ids"], "max_tokens": r["max_new"],
                                                      "temperature": 0, "return_token_ids": True}, timeout=1800)
        ids = resp["choices"][0].get("token_ids") or []
        sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        det = resp["usage"].get("completion_tokens_details", {})
        res.append({"name": r["name"], "tokens": resp["usage"]["completion_tokens"], "identical": ref.get(r["name"]) == sha,
                    "have_reference": r["name"] in ref, "accepted": det.get("accepted_prediction_tokens"),
                    "rejected": det.get("rejected_prediction_tokens"),
                    "decode_tok_s": resp.get("timings", {}).get("decode_tok_s")})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--prompts", default=os.path.join(ROOT, "eval", "legd", "legd_prompts.json"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ref_path = a.reference or default_reference()
    if not ref_path or not os.path.isfile(ref_path):
        print("DRAFT_PARITY no plain reference rows found; drafted requests only")
        ref = {}
    else:
        ref = load_reference(ref_path)
    rows = pick(json.load(open(a.prompts))["rows"], a.n)
    res = run(a.base, rows, ref, a.model)
    comparable = [r for r in res if r["have_reference"]]
    same = sum(1 for r in comparable if r["identical"])
    acc = sum(r["accepted"] or 0 for r in res)
    rej = sum(r["rejected"] or 0 for r in res)
    out = {"prompts": len(res), "comparable": len(comparable), "identical": same, "accepted": acc, "rejected": rej,
           "mean_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in res) / max(1, len(res)), 1), "rows": res}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    print("DRAFT_PARITY %d of %d prompts identical to the plain streams (greedy, drafted decode %.1f tok/s mean, "
          "%d drafts accepted, %d rejected)" % (same, len(comparable), out["mean_decode_tok_s"], acc, rej))
    return 0


if __name__ == "__main__":
    sys.exit(main())
