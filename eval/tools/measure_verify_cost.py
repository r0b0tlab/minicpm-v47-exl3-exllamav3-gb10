#!/usr/bin/env python3
"""Measure the cost grid of adaptive verification on this machine: one target verify pass over 1 .. 16 rows and the
draft side of a round, wall clock, one stream (needs the GPU; hold your lock).

The grid is what `Qwen3DSparkModel.set_verify_cost` consumes (config/profiles.yaml: draft.verify_cost_ms,
draft.draft_cost_ms). The measurement harness is the fork's tools/dspark/spec_check.py (profile task); this wrapper
loads it from the checkout that click_run.sh fetched and runs it at one or more context lengths in a single process.

    python eval/tools/measure_verify_cost.py --spec-check CACHE_DIR/src/exllamav3-<commit12>/tools/dspark/spec_check.py \\
        --target models/<pack> --draft models/<draft> --contexts 300 --out verify_cost.json
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(os.path.dirname(HERE), "legd", "legd_prompts.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-check", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--contexts", default="300",
                    help="committed tokens when the verify pass is timed (the harness addresses 1,024 tokens: 300 is "
                         "the default; larger values need a larger block table than spec_check builds)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--runs", type=int, default=3, help="repeat each grid this many times (the stored grid is the median)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("spec_check", a.spec_check)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    import torch
    from exllamav3 import Cache, Config, Model

    ids = []
    for r in json.load(open(PROMPTS))["rows"]:
        ids += r["ids"]
    contexts = [int(x) for x in a.contexts.split(",")]
    need = max(contexts) + 40
    while len(ids) < need:
        ids = ids + ids
    t0 = time.time()
    target = Model.from_config(Config.from_directory(a.target))
    draft = Model.from_config(Config.from_directory(a.draft))
    pages = (max(contexts) + 40 + 255) // 256 + 4
    cache = Cache(target, max_num_tokens=max(20 * 256, pages * 256), max_batch_size=5, max_history=15)
    dcache = Cache(draft, max_num_tokens=max(20 * 256, pages * 256))
    target.load(progressbar=False)
    draft.load(progressbar=False)
    draft.attach_to(target)          # a generator does this; the profile task drives the models directly
    print("loaded in %.0f s" % (time.time() - t0), flush=True)
    models = SimpleNamespace(target=target, draft=draft, profile=None, profile_ids=ids,
                             rig=sc.Rig(target, draft, cache, dcache))
    out = {"contexts": {}, "reps": a.reps, "runs": a.runs,
           "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    for ctx in contexts:
        grids, drafts, updates = [], [], []
        for _ in range(a.runs):
            models.profile = None
            p = sc.ensure_profile(models, context=ctx, reps=a.reps)
            grids.append(p["verify_ms"])
            drafts.append(p["draft_ms"])
            updates.append(p["draft_kv_update_ms"])
        med = [statistics.median(g[i] for g in grids) for i in range(16)]
        out["contexts"][str(ctx)] = {"verify_ms": [round(x, 2) for x in med],
                                     "draft_ms": round(statistics.median(drafts), 2),
                                     "draft_kv_update_ms": round(statistics.median(updates), 2),
                                     "runs_verify_ms": [[round(x, 2) for x in g] for g in grids]}
        print("context %d: verify ms by rows 1..16 %s; draft %.1f + update %.1f ms" % (
            ctx, out["contexts"][str(ctx)]["verify_ms"], statistics.median(drafts), statistics.median(updates)), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
