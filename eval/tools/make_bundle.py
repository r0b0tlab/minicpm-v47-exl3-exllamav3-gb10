#!/usr/bin/env python3
"""Assemble an evidence bundle from the lane outputs of eval/run_suite.sh (stdlib only).

    eval/tools/make_bundle.py --out evidence/exllamav3-plain-20261013 --id exllamav3-plain-20261013 \
        --column exllamav3-plain --engine-name ExLlamaV3 --engine-commit 4d083d25... --mode "plain decode (no draft)" \
        --q200-rows RUN/q200.rows.jsonl --q200-wall RUN/wall.json --bench RUN/bench.json --niah RUN/niah.json \
        --legd-summary RUN/legd.summary.json --smoke RUN/smoke.json
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundlelib as bl  # noqa: E402


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--column", required=True, choices=("exllamav3-plain", "exllamav3-dspark", "reference-engine"))
    ap.add_argument("--title", default="")
    ap.add_argument("--engine-name", default="ExLlamaV3")
    ap.add_argument("--engine-commit", default="")
    ap.add_argument("--mode", default="plain decode (no draft)")
    ap.add_argument("--hardware", default="1x NVIDIA GB10 (DGX Spark, aarch64, SM121), 128 GB unified memory")
    ap.add_argument("--date", required=True)
    ap.add_argument("--profile", default="gb10-text-262k")
    ap.add_argument("--q200-rows")
    ap.add_argument("--q200-wall")
    ap.add_argument("--q200-workers", type=int, default=1)
    ap.add_argument("--q200-extra", action="append", default=[],
                    help="WORKERS:ROWS[:WALL[:MANUAL]] another Q200v2 run (for example 4 requests in flight)")
    ap.add_argument("--manual-evidence", help="the kit-format manual evidence file used for the hard_reasoning grades")
    ap.add_argument("--grader-note", default="")
    ap.add_argument("--bench")
    ap.add_argument("--niah")
    ap.add_argument("--legd-summary")
    ap.add_argument("--legd-rows", action="append", default=[])
    ap.add_argument("--legd-run", action="append", default=[],
                    help="NAME=ROWS[:COMPARE_ROWS]: a Leg D rows file; NAME '' is the primary run, otherwise the key "
                         "legd_<NAME> (plain_ctl, fixed15); COMPARE_ROWS are the plain streams to count identical ones")
    ap.add_argument("--json-result", action="append", default=[],
                    help="KEY=PATH: copy a JSON file into the bundle and expose it as results[KEY]")
    ap.add_argument("--smoke")
    ap.add_argument("--memory", help="a /metrics JSON captured after load")
    ap.add_argument("--clickrun", help="JSON with the clean-room click-run stage timings (stage -> {seconds, note})")
    ap.add_argument("--extra-json", action="append", default=[], help="NAME=PATH of a JSON file to copy into the bundle")
    ap.add_argument("--caveat", action="append", default=[])
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    lock = load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "runtime.lock.json"))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    results, lanes = {}, {}

    if a.q200_rows:
        manual = None
        if a.manual_evidence:
            me = load(a.manual_evidence)
            manual = {r["id"]: r["passed"] for r in me["rows"]}
        rows = [bl.sanitize_q200_row(json.loads(l), manual) for l in open(a.q200_rows) if l.strip()]
        wall = load(a.q200_wall)["wall_seconds"] if a.q200_wall else None
        name = "q200.rows.public.jsonl"
        with open(os.path.join(out, name), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        results["q200"] = bl.q200_results(rows, a.q200_workers, wall)
        lanes["q200"] = {"rows": name, "kit": "Q200v2 text-180 (r0b0bench subsets/q200v2)",
                         "dataset_sha256": "66a75701cbeea69f212e1c8be92aab9efaf3fa4d7af3c6911c8f7864a17d8d14",
                         "request": {"chat_template_kwargs": {"enable_thinking": True, "thinking": True,
                                                              "reasoning_effort": "low"},
                                     "temperature": 0, "top_p": 1, "max_tokens": 8192,
                                     "workers": a.q200_workers},
                         "hard_reasoning_grading": a.grader_note or "ungraded"}
        if a.manual_evidence:
            shutil.copy(a.manual_evidence, os.path.join(out, "q200.hard_reasoning.manual.json"))
    for spec in a.q200_extra:
        parts = spec.split(":")
        workers, rows_path = int(parts[0]), parts[1]
        wall_path = parts[2] if len(parts) > 2 and parts[2] else None
        manual_path = parts[3] if len(parts) > 3 and parts[3] else None
        manual = None
        if manual_path:
            manual = {r["id"]: r["passed"] for r in load(manual_path)["rows"]}
            shutil.copy(manual_path, os.path.join(out, "q200-w%d.hard_reasoning.manual.json" % workers))
        rows = [bl.sanitize_q200_row(json.loads(l), manual) for l in open(rows_path) if l.strip()]
        name = "q200-w%d.rows.public.jsonl" % workers
        with open(os.path.join(out, name), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        wall = load(wall_path)["wall_seconds"] if wall_path else None
        results["q200_w%d" % workers] = bl.q200_results(rows, workers, wall)
        lanes["q200_w%d" % workers] = {"rows": name}
    if a.bench:
        b = load(a.bench)
        shutil.copy(a.bench, os.path.join(out, "bench.json"))
        results["bench"] = {
            "chunk_size": b.get("chunk_size"), "kv_bits": b.get("kv_bits"), "repeats": b.get("repeats"),
            "decode_tokens": b.get("decode_tokens"),
            "depths": {str(d["depth"]): {"decode_tok_s": d.get("decode_tok_s_median"),
                                         "prefill_tok_s": d.get("prefill_tok_s_median"),
                                         "ttft_cold_s": d.get("ttft_cold_s_median"),
                                         "ttft_cache_hit_s": d.get("ttft_cache_hit_s")}
                       for d in b["depths"] if d.get("status") == "ok"},
            "memory_before": b.get("memory_before"), "memory_after": b.get("memory_after")}
        lanes["bench"] = {"file": "bench.json"}
    if a.niah:
        n = load(a.niah)
        shutil.copy(a.niah, os.path.join(out, "niah.json"))
        s = n["summary"]
        results["niah"] = {"asked": s["asked"], "exact": s["exact"], "skipped": s["skipped"], "wrong": s["wrong"],
                           "max_prompt_tokens": s["max_prompt_tokens"], "window": s["window"],
                           "cold_prefill_s": {str(r["cut"]): r["cold_prefill_cumulative_s"] for r in n["rows"]
                                              if r.get("first_question_at_cut")}}
        lanes["niah"] = {"file": "niah.json"}
    if a.legd_summary:
        shutil.copy(a.legd_summary, os.path.join(out, "legd.summary.json"))
        results["legd"] = load(a.legd_summary)
        for i, p in enumerate(a.legd_rows):
            shutil.copy(p, os.path.join(out, os.path.basename(p)))
        lanes["legd"] = {"file": "legd.summary.json", "rows": [os.path.basename(p) for p in a.legd_rows]}
    for spec in a.legd_run:
        name, _, rest = spec.partition("=")
        rows_path, _, cmp_path = rest.partition(":")
        rows = [json.loads(l) for l in open(rows_path) if l.strip()]
        cmp_rows = [json.loads(l) for l in open(cmp_path) if l.strip()] if cmp_path else None
        key = "legd" + ("_" + name if name else "")
        results[key] = bl.legd_from_rows(rows, cmp_rows)
        fname = "legd%s.rows.jsonl" % ("-" + name if name else "")
        shutil.copy(rows_path, os.path.join(out, fname))
        lanes[key] = {"rows": fname}
    for spec in a.json_result:
        key, path = spec.split("=", 1)
        results[key] = load(path)
        shutil.copy(path, os.path.join(out, key + ".json"))
        lanes[key] = {"file": key + ".json"}
    if a.smoke:
        shutil.copy(a.smoke, os.path.join(out, "smoke.json"))
        lanes["smoke"] = {"file": "smoke.json"}
    if a.memory:
        m = load(a.memory)
        results["memory"] = {k: m.get(k) for k in ("mem_available_gib", "mem_available_before_load_gib",
                                                   "mem_available_after_load_gib", "torch_allocated_gib",
                                                   "torch_reserved_gib", "torch_max_reserved_gib", "load_seconds",
                                                   "window", "profile") if m.get(k) is not None}
    if a.clickrun:
        cr = load(a.clickrun)
        shutil.copy(a.clickrun, os.path.join(out, "clickrun.json"))
        results["clickrun"] = cr
        lanes["clickrun"] = {"file": "clickrun.json"}
    for spec in a.extra_json:
        name, path = spec.split("=", 1)
        shutil.copy(path, os.path.join(out, name + ".json"))
        lanes[name] = {"file": name + ".json"}
    doc = {
        "id": a.id, "column": a.column, "title": a.title or a.id,
        "engine": {"name": a.engine_name, "commit": a.engine_commit, "mode": a.mode},
        "hardware": a.hardware, "date": a.date, "profile": a.profile,
        "model": {"repo": lock["models"]["target"]["hf_repo"], "revision": lock["models"]["target"]["revision"]},
        "reproducible_with_this_repo": a.column != "reference-engine",
        "caveats": a.caveat, "note": a.note, "lanes": lanes, "results": results,
    }
    bl.write_bundle(out, doc)
    print("bundle written: %s (%d result groups)" % (out, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
