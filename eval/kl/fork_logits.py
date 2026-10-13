#!/usr/bin/env python3
"""Logits of a model directory (an EXL3 pack, or the bf16 HF checkpoint = the fork's fp16 "floor") through the exllamav3 fork's own forward,
for the ids sets of make_kl_ids.py. Text path only (Embedding -> 40 TransformerBlocks -> RMSNorm -> lm_head), module by module like the
fork's eval/model_diff.py: each module is loaded, run on every micro-batch, unloaded, so the 70 GB bf16 checkpoint streams through a
GB10 next to a running conversion and a 20 GB pack needs no more than its largest layer.

Candidates are STREAMED: the logits of a micro-batch go straight into the sinks and are dropped.
    --store {full,topk,none}   full: logits.f16 (row-centred fp16, 7.1 GB for Leg A) + statistics (a LogitSet kl_compare.py reads);
                               topk: statistics + top-256 only (no full file); none
    --compare-ref REFDIR       per-position metrics against the HF reference set computed on the fly (kl_metrics.compare_chunk; exact when
                               the reference has logits.f16 or final_hidden.bf16 (+ --lm-head from the checkpoint), lumped top-256 else);
                               written to <out-dir>/metrics.npz. A 100 x 1024 candidate needs ~10 MB, not 50 GB.
    --dump-layers 0,1,3        the fork's input and output of those blocks for the first --dump-rows rows (fp32 [rows, T, 2048]: the
                               fork's residual stream is fp32) and dump.json: the input of `bf16_ref_logits.py --pack-check`

Width and range checks (the shim pads lm_head to 248,192 and trims it back): every logit row must be 248,144 wide; a 248,192-wide output
is sliced and recorded; any argmax id >= 248,144 is an error.

    # correctness runs share the GPU: -s; a conversion holds the lock exclusively (-w bounds the wait)
    flock -s -w 3600 <your-gpu-lock> python -I eval/kl/fork_logits.py \\
        --model <bf16-checkpoint-dir> --ids ids/legA.safetensors --out-dir floor-legA --store full
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("EXLLAMAV3_DIR"):
    sys.path.insert(0, os.environ["EXLLAMAV3_DIR"])
import numpy as np
import torch

import common
import kl_metrics as km


def git_head(path):
    """HEAD commit of a checkout by reading files (no git process): None if unknown."""
    try:
        g = os.path.join(path, ".git")
        if os.path.isfile(g):
            g = open(g).read().split("gitdir:")[1].strip()
        head = open(os.path.join(g, "HEAD")).read().strip()
        if head.startswith("ref:"):
            ref = head.split()[1]
            for base in (g, os.path.dirname(os.path.dirname(g))):   # worktree gitdir -> common dir
                p = os.path.join(base, ref)
                if os.path.exists(p):
                    return open(p).read().strip()
            packed = os.path.join(os.path.dirname(os.path.dirname(g)), "packed-refs")
            if os.path.exists(packed):
                for line in open(packed):
                    if line.strip().endswith(ref):
                        return line.split()[0]
            return None
        return head
    except Exception:
        return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="pack directory (exl3) or the bf16 HF checkpoint directory")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--rows", default="all")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", help="set name (default: the out-dir's basename)")
    ap.add_argument("--kind", choices=["cand", "floor"], default="cand")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mb-rows", type=int, default=4, help="rows per forward (activation memory)")
    ap.add_argument("--store", choices=["full", "topk", "none"], default="full")
    ap.add_argument("--compare-ref", metavar="REFDIR")
    ap.add_argument("--ref-mode", choices=["auto", "full", "hidden", "topk"], default="auto")
    ap.add_argument("--lm-head-from", default=common.MODEL_DIR, help="checkpoint directory holding lm_head.weight (hidden-state reference mode)")
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--dump-rows", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512, help="positions per sink chunk")
    ap.add_argument("--min-mem-gb", type=float, default=15.0)
    args = ap.parse_args()

    from exllamav3 import Config, Model
    from exllamav3.util.memory import free_mem

    t_start = time.time()
    device = torch.device(args.device)
    args.model = os.path.expanduser(args.model)
    ids_all, man = common.load_ids_set(os.path.expanduser(args.ids))
    sel = common.parse_rows(args.rows, ids_all.shape[0])
    ids = ids_all[sel]
    R, T = ids.shape
    ids_sha = common.ids_sha256(ids)
    names = [s["name"] for s in man.get("samples", [])]
    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    # reference for streaming comparison
    ref_src = None
    arrays = None
    if args.compare_ref:
        ref_set = common.LogitSet(os.path.expanduser(args.compare_ref))
        if ref_set.meta.get("ids_sha256") != ids_sha:
            sys.exit(f"{args.compare_ref}: reference made for different ids/rows (sha {ref_set.meta.get('ids_sha256')} vs {ids_sha})")
        mode = args.ref_mode
        lm_head = None
        if mode in ("auto", "hidden") and not ref_set.has("logits") and ref_set.has("final_hidden"):
            from safetensors import safe_open

            wm = json.load(open(os.path.join(args.lm_head_from, "model.safetensors.index.json")))["weight_map"]
            with safe_open(os.path.join(args.lm_head_from, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
                lm_head = f.get_tensor("lm_head.weight")
            mode = "hidden"
        ref_src = km.SetSource(ref_set, device=device, mode=mode, lm_head=lm_head)
        arrays = km.fresh_arrays(R, T)
        print(f"streaming comparison against {ref_set.path} (mode {ref_src.mode})", flush=True)

    ls = None
    if args.store != "none":
        want = ["rowmax", "lse", "tail_lse", "entropy", "argmax", "tgt_logit", "topk_val", "topk_idx"] + (["logits"] if args.store == "full" else [])
        meta = {"ids_sha256": ids_sha, "rows_selected": sel, "row_names": [names[r] for r in sel] if names else None, "tokens": T,
                "producer": {"tool": "fork_logits.py", "model_dir": os.path.realpath(args.model),
                             "fork_dir": os.environ.get("MINICPM_FORK_DIR", common.FORK_DIR), "fork_head": None,
                             "forward": "exllamav3 fork, module-by-module, fp16 activations"},
                "centered_logits": True}
        ls = common.LogitSet(out, create=True, rows=R, tokens=T, meta=meta, name=args.name or os.path.basename(out.rstrip("/")),
                             kind=args.kind, want=want)

    dump_layers = [int(x) for x in args.dump_layers.split(",") if x != ""]
    dumps = {}
    if dump_layers:
        dump_rows = min(args.dump_rows, R)

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    if ls is not None:
        ls.meta["producer"]["fork_head"] = git_head(os.environ.get("MINICPM_FORK_DIR", common.FORK_DIR))
        ls.meta["producer"]["arch"] = type(config).__name__
        ls.meta["producer"]["model_class"] = type(model).__name__
        ls.save_meta()
    print(f"{type(config).__name__} / {type(model).__name__}: {len(model.modules)} modules; rows {R} x {T}; device {device}", flush=True)

    mbs = [(a, min(a + args.mb_rows, R)) for a in range(0, R, args.mb_rows)]
    params = [{} for _ in mbs]
    states = []
    for (a, b), p in zip(mbs, params):
        x = torch.from_numpy(ids[a:b]).long()
        states.append(model.prepare_inputs(x, p))
    first_block = getattr(model, "first_block_idx", 1)
    n_mod = len(model.modules)
    seen_width = None
    argmax_max = -1
    finite = True
    per_module = []

    for idx, module in enumerate(model.modules):
        if common.mem_available_gb() < args.min_mem_gb:
            sys.exit(f"MemAvailable {common.mem_available_gb():.1f} GB below --min-mem-gb {args.min_mem_gb} before module {idx} ({module.key})")
        t0 = time.time()
        config.stc.begin_deferred_load()
        module.load(device if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()
        t_load = time.time() - t0
        t1 = time.time()
        is_head = idx == n_mod - 1
        layer_no = idx - first_block
        for b, ((a, bb), p) in enumerate(zip(mbs, params)):
            p["dev_cache"] = None
            s = module.prepare_for_device(states[b], p)
            dumping = bool(dump_layers) and layer_no in dump_layers and 0 <= layer_no < config.num_hidden_layers and a < dump_rows
            if dumping:
                # the block adds into its input in place: take the copy before the forward. The fork's residual stream is fp32.
                n = min(bb, dump_rows) - a
                d = dumps.setdefault(layer_no, {"in": [], "out": []})
                d["in"].append(s[:n].detach().float().cpu().numpy().copy())
            y = module.forward(s, p).clone()
            if dumping:
                d["out"].append(y[:n].detach().float().cpu().numpy())
            if not is_head:
                states[b] = y
                continue
            # logits of this micro-batch: [rows, T, V or padded V]
            width = y.shape[-1]
            seen_width = width
            if width != common.V:
                if width < common.V:
                    sys.exit(f"logits are {width} wide, expected {common.V}")
                y = y[..., :common.V]
            for j in range(bb - a):
                r = a + j
                for lo in range(0, T, args.chunk):
                    hi = min(lo + args.chunk, T)
                    x = y[j, lo:hi].float()
                    if not torch.isfinite(x).all():
                        finite = False
                    am = int(x.argmax(-1).max())
                    argmax_max = max(argmax_max, am)
                    tgt = km.next_token_targets(ids[r], lo, hi, device)
                    if ls is not None:
                        km.store_chunk(ls, r, lo, x, tgt, full=args.store == "full", topk=True)
                    if ref_src is not None:
                        m = km.compare_chunk(ref_src, r, lo, hi, x, tgt)    # the reference set has the same rows in the same order
                        km.put(arrays, r, lo, m)
            states[b] = None
            del y
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_module.append({"idx": idx, "key": module.key, "load_s": round(t_load, 2), "run_s": round(time.time() - t1, 2)})
        module.unload()
        config.stc.close()
        free_mem()
        if idx % 8 == 0 or is_head:
            print(f"  module {idx:3d}/{n_mod - 1} {module.key:48s} load {t_load:5.1f}s run {time.time() - t1:5.1f}s", flush=True)

    if ls is not None:
        ls.flush()
    if arrays is not None:
        npz = os.path.join(out, "metrics.npz")
        np.savez_compressed(npz, meta=json.dumps({"ids_sha256": ids_sha, "reference": args.compare_ref, "rows_selected": sel}), **arrays)
        km_s = km.summarize(arrays, row_names=[names[r] for r in sel] if names else None)
        print(f"KL(ref||fork) mean {km_s['kl_ref_to_q']['mean']:.5f} median {km_s['kl_ref_to_q']['median']:.5f} p99 {km_s['kl_ref_to_q']['p99']:.4f} "
              f"max {km_s['kl_ref_to_q']['max']:.3f}; KL(fork||ref) mean {km_s['kl_q_to_ref']['mean']:.5f}; top-1 {100 * km_s['top1_agreement']:.2f}%",
              flush=True)
    if dumps:
        rows_d = min(args.dump_rows, R)
        info = {"rows": rows_d, "tokens": T, "ids_sha256": ids_sha, "rows_selected": sel[:rows_d], "dtype": "<f4", "layers": {}}
        for L, d in dumps.items():
            for k in ("in", "out"):
                arr = np.concatenate(d[k], axis=0)
                fn = f"layer{L:02d}_{k}.f32"
                arr.astype("<f4").tofile(os.path.join(out, fn))
                d[k] = fn
            info["layers"][str(L)] = {"in": d["in"], "out": d["out"]}
        common.atomic_json(os.path.join(out, "dump.json"), info)
        print(f"dumped layers {sorted(dumps)} for {rows_d} row(s)", flush=True)
    summary = {"rows": R, "tokens": T, "logit_width_seen": seen_width, "vocab": common.V, "argmax_max": argmax_max,
               "argmax_in_vocab": bool(argmax_max < common.V), "finite": finite, "wall_s": round(time.time() - t_start, 1)}
    if ls is not None:
        ls.save_meta(complete=True, **summary)
    common.atomic_json(os.path.join(out, "run.json"), {"summary": summary, "modules": per_module,
                                                      "args": {k: v for k, v in vars(args).items()}})
    print(json.dumps(summary), flush=True)
    if argmax_max >= common.V or not finite:
        sys.exit(2)


if __name__ == "__main__":
    main()
