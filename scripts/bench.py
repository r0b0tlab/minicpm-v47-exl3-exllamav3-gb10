#!/usr/bin/env python3
"""Decode / prefill / time-to-first-token grid through the real server (stdlib only).

For every depth (default 1,024 / 4,096 / 32,768 / 131,072 / 261,000 tokens) and repeat:

  * a prompt of exactly `depth` token ids is cut from the shipped needle document at a different offset and
    prefixed with random ids, so the paged prefix cache cannot reuse an earlier repeat (every repeat is a COLD
    prefill; usage.prompt_tokens_details.cached_tokens is recorded to prove it)
  * the request asks for `decode_tokens` new tokens with ignore_eos, greedy; the server's own timings give
    prefill seconds (time to first token inside the engine) and decode tokens/s over the remaining tokens
  * the same request is streamed once to record the client-side time to first token

Reported per depth: median over repeats of decode tok/s, prefill tok/s, TTFT; plus the TTFT of one repeat of an
identical prompt (prefix-cache hit) as a separate row. Memory (MemAvailable, torch allocated/reserved) is read from
/metrics before the grid and after every depth.

    scripts/bench.py --base http://127.0.0.1:8000 --out bench.json [--depths 1024,4096] [--repeats 3]
"""
import argparse
import gzip
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oai_client as oc  # noqa: E402

DEFAULT_DOC = os.path.join(os.path.dirname(HERE), "eval", "niah", "niah_261k.json.gz")
DEFAULT_DEPTHS = [1024, 4096, 32768, 131072, 261000]


def load_ids(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["ids"]


def make_prompt(ids, depth, repeat, seed=17):
    """`depth` ids: 16 random 'salt' ids (defeat prefix reuse) + a slice of the document at a repeat-specific offset."""
    rng = random.Random("%s-%s-%s" % (seed, depth, repeat))
    salt = [rng.randrange(1000, 60000) for _ in range(16)]
    body = depth - len(salt)
    span = max(1, len(ids) - body)
    offset = (repeat * 7919 + depth * 31) % span
    return salt + ids[offset: offset + body]


def metrics(base):
    try:
        m = oc.get_json(base, "/metrics")
        return {k: m.get(k) for k in ("mem_available_gib", "torch_allocated_gib", "torch_reserved_gib",
                                      "torch_max_reserved_gib")}
    except Exception:  # noqa: BLE001
        return {}


def one_run(base, model, prompt, decode_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": decode_tokens, "temperature": 0, "ignore_eos": True}
    t0 = time.perf_counter()
    r = oc.post_json(base, "/v1/completions", body, timeout=7200)
    wall = time.perf_counter() - t0
    tm = r.get("timings", {})
    n = r["usage"]["completion_tokens"]
    return {
        "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": n,
        "cached_tokens": tm.get("cached_tokens"), "prefill_s": tm.get("prefill_s"), "decode_s": tm.get("decode_s"),
        "decode_tok_s": tm.get("decode_tok_s"),
        "prefill_tok_s": (r["usage"]["prompt_tokens"] / tm["prefill_s"]) if tm.get("prefill_s") else None,
        "wall_s": round(wall, 3),
    }


def stream_ttft(base, model, prompt):
    body = {"model": model, "prompt": prompt, "max_tokens": 2, "temperature": 0, "ignore_eos": True}
    t_first = None
    t0 = time.perf_counter()
    for t, chunk in oc.post_stream(base, "/v1/completions", body, timeout=7200):
        ch = chunk.get("choices") or []
        if ch and ch[0].get("text") and t_first is None:
            t_first = t
    return t_first


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run(base, model, ids, depths, repeats, decode_tokens, log=print):
    card = oc.first_model(base)
    model = model or card["id"]
    window = int(card.get("max_model_len", 0))
    out = {"model": model, "window": window, "chunk_size": card.get("chunk_size"), "kv_bits": card.get("kv_bits"),
           "max_batch_size": card.get("max_batch_size"), "decode_tokens": decode_tokens, "repeats": repeats,
           "memory_before": metrics(base), "depths": []}
    # warm-up: a tiny request so the first measured prefill does not pay one-off kernel selection
    oc.post_json(base, "/v1/completions", {"model": model, "prompt": make_prompt(ids, 512, 99), "max_tokens": 8,
                                           "temperature": 0, "ignore_eos": True})
    for depth in depths:
        if depth + decode_tokens + 2 > window:
            log("  depth %7d: skipped (window %d)" % (depth, window))
            out["depths"].append({"depth": depth, "status": "skipped", "reason": "window %d" % window})
            continue
        runs = []
        for rep in range(repeats):
            prompt = make_prompt(ids, depth, rep)
            runs.append(one_run(base, model, prompt, decode_tokens))
            r = runs[-1]
            log("  depth %7d rep %d: prefill %7.2fs (%6.0f tok/s) decode %6.2f tok/s cached %s" % (
                depth, rep, r["prefill_s"] or 0, r["prefill_tok_s"] or 0, r["decode_tok_s"] or 0, r["cached_tokens"]))
        # prefix-cache hit: the last repeat's prompt again
        hit = one_run(base, model, make_prompt(ids, depth, repeats - 1), 4)
        ttft_client = stream_ttft(base, model, make_prompt(ids, depth, repeats + 5))  # one more cold prompt, streamed
        row = {
            "depth": depth, "status": "ok", "runs": runs,
            "decode_tok_s_median": med([r["decode_tok_s"] for r in runs]),
            "prefill_tok_s_median": med([r["prefill_tok_s"] for r in runs]),
            "ttft_cold_s_median": med([r["prefill_s"] for r in runs]),
            "ttft_cold_client_s": round(ttft_client, 3) if ttft_client else None,
            "ttft_cache_hit_s": hit["prefill_s"], "cache_hit_cached_tokens": hit["cached_tokens"],
            "all_cold": all((r["cached_tokens"] or 0) <= 512 for r in runs),
            "memory_after": metrics(base),
        }
        out["depths"].append(row)
        log("  depth %7d: median decode %.2f tok/s, prefill %.0f tok/s, cold TTFT %.2fs, cache-hit TTFT %.2fs" % (
            depth, row["decode_tok_s_median"] or 0, row["prefill_tok_s_median"] or 0, row["ttft_cold_s_median"] or 0,
            row["ttft_cache_hit_s"] or 0))
    out["memory_after"] = metrics(base)
    return out


def table(res):
    lines = ["| depth (tokens) | decode tok/s | prefill tok/s | cold TTFT (s) | cache-hit TTFT (s) |",
             "|---:|---:|---:|---:|---:|"]
    for d in res["depths"]:
        if d.get("status") != "ok":
            lines.append("| %d | - | - | - | - |" % d["depth"])
            continue
        lines.append("| %d | %.1f | %.0f | %.2f | %.2f |" % (d["depth"], d["decode_tok_s_median"] or 0,
                                                            d["prefill_tok_s_median"] or 0, d["ttft_cold_s_median"] or 0,
                                                            d["ttft_cache_hit_s"] or 0))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--doc", default=DEFAULT_DOC)
    ap.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ids = load_ids(a.doc)
    depths = [int(x) for x in a.depths.split(",") if x]
    res = run(a.base, a.model, ids, depths, a.repeats, a.decode_tokens)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(table(res))
    print("BENCH_OK depths=%d repeats=%d" % (len(depths), a.repeats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
