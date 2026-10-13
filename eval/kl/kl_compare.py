#!/usr/bin/env python3
"""KL of EXL3 candidates (and of the fork's own bf16 run, the noise floor) against the HF bf16 reference, MiniCPM-V-4.7 text legs.

Inputs are LogitSets (common.LogitSet; written by bf16_ref_logits.py and fork_logits.py) over one ids set:

    --ref DIR        the HF bf16 reference (full logits.f16, or final_hidden.bf16 + lm_head, or the top-256 store)
    --cand NAME=DIR  candidate sets: logits.f16 (compared here, chunk by chunk) or metrics.npz (fork_logits.py --compare-ref wrote them)
    --floor NAME=DIR the fork's fp16 forward on the SAME bf16 weights: KL(ref || floor) is the numerics floor of the fork (the reference's own rounding noise and the fork-vs-HF line); each candidate is also compared
                     against it (KL(floor || cand): the quantisation distance with the fork-vs-HF numerics removed)
    --alt-ref NAME=DIR  a second reference arm, e.g. bf16_ref_logits.py --residual fp32 (the bf16 reference's own rounding noise); every
                     candidate and the floor are compared against it too

Per set and direction: mean / median / p95 / p99 / max of KL(ref || q) and KL(q || ref), top-1 agreement, top-k overlap k = 1..5 (and
exact ordered agreement), ppl of both, the same by reference entropy, per row (with its kind), per position. `--json` writes the
kl.json block (schema minicpm-v47-kl-v1) for the model card; scalars only.

Exact vs lumped: a reference with full logits (or the hidden-state route) gives exact KL in both directions; a top-256-only reference
gives the lumped lower bound (kl_metrics.py docstring). `--check-lumped` measures lumped / exact on the rows where both exist.

    python -I eval/kl/kl_compare.py --ids ids/legA.safetensors --ref ref-legA --floor floor=floor-legA \\
        --cand q4=cand-legA-4.00bpw --json kl_legA.json
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import common
import kl_metrics as km


def parse_named(specs):
    out = []
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"expected NAME=DIR, got {s!r}")
        n, d = s.split("=", 1)
        out.append((n, os.path.expanduser(d)))
    return out


def load_lm_head(model_dir):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx["lm_head.weight"]), framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight")


def load_metrics_npz(path):
    z = np.load(path)
    return {k: z[k] for k in km.PER_POSITION_KEYS if k in z.files}, (json.loads(str(z["meta"])) if "meta" in z.files else {})


def save_metrics_npz(path, arrays, meta):
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def compare_sets(ref_src, cand_set, ids, rows_idx, device, tag, ch=km.CH):
    """Per-position metrics of the LogitSet `cand_set` (logits.f16) against `ref_src`: arrays {key: [len(rows_idx), T]}."""
    cand = km.SetSource(cand_set, device=device, mode="full")
    n_rows = len(rows_idx)
    T = ids.shape[1]
    arr = km.fresh_arrays(n_rows, T)
    t0 = time.time()
    for j, r in enumerate(rows_idx):
        for lo in range(0, T, ch):
            hi = min(lo + ch, T)
            q = cand.logits(r, lo, hi)
            m = km.compare_chunk(ref_src, r, lo, hi, q, km.next_token_targets(ids[r], lo, hi, device))
            km.put(arr, j, lo, m)
        if (j + 1) % 2 == 0 or j + 1 == n_rows:
            print(f"    [{tag}] row {j + 1}/{n_rows}  {time.time() - t0:.0f}s", flush=True)
    return arr


def card_line(name, s, ref="ref"):
    k, kr = s["kl_ref_to_q"], s["kl_q_to_ref"]
    ppl = s.get("ppl", {})
    return (f"{name:28s} KL({ref}||q) mean {k['mean']:.5f} med {k['median']:.5f} p95 {k['p95']:.4f} p99 {k['p99']:.4f} max {k['max']:.3f} | "
            f"KL(q||{ref}) mean {kr['mean']:.5f} | top-1 {100 * s['top1_agreement']:.2f}% top-5 ov {s['topk_overlap']['5']:.3f} | "
            f"ppl ratio {ppl.get('ratio_q_over_ref', float('nan')):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-mode", choices=["auto", "full", "hidden", "topk"], default="auto")
    ap.add_argument("--model-dir", default=common.MODEL_DIR, help="for --ref-mode hidden: where lm_head.weight is")
    ap.add_argument("--lm-head", help="a torch file with the lm_head weight [V, 2048] (default: read from --model-dir)")
    ap.add_argument("--cand", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--floor", metavar="NAME=DIR")
    ap.add_argument("--alt-ref", metavar="NAME=DIR")
    ap.add_argument("--rows", default="all", help="rows of the reference set to use")
    ap.add_argument("--device", default="cpu", help="cpu (default; no GPU lock needed) or cuda (then run under flock -s)")
    ap.add_argument("--chunk", type=int, default=km.CH)
    ap.add_argument("--check-lumped", action="store_true", help="lumped (top-256) vs exact KL on the first candidate")
    ap.add_argument("--json", help="write the kl.json block here")
    ap.add_argument("--npz-dir", help="write the per-position metric arrays of every comparison here (reusable, ~1 MB per 14 rows)")
    ap.add_argument("--leg", default=None, help="label stored in the json (A, B or C)")
    ap.add_argument("--mask-npz", help="Leg C scaffold: an .npz of bool [rows, T] arrays over the reference's rows; key `mask` restricts "
                    "the headline statistics (e.g. the continuation positions after the prompt), every other key is reported as an "
                    "extra scope (e.g. `first8`: the first 8 continuation positions) under entry['scopes']")
    args = ap.parse_args()

    ids_all, man = common.load_ids_set(args.ids)
    ref_set = common.LogitSet(os.path.expanduser(args.ref))
    if ref_set.meta.get("ids_sha256") and man.get("ids_sha256"):
        rows_sel = ref_set.meta.get("rows_selected", list(range(ref_set.rows)))
        assert common.ids_sha256(ids_all[rows_sel]) == ref_set.meta["ids_sha256"], "the reference was made for a different ids set / rows"
    else:
        rows_sel = ref_set.meta.get("rows_selected", list(range(ref_set.rows)))
    ids = ids_all[rows_sel]                                       # [rows of the reference, T]
    T = ids.shape[1]
    sel = common.parse_rows(args.rows, ids.shape[0])
    names = [s["name"] for s in man.get("samples", [])]
    kinds = [s["kind"] for s in man.get("samples", [])]
    row_names = [names[r] if names else f"row{r}" for r in rows_sel]
    row_kinds = [kinds[r] if kinds else None for r in rows_sel]
    device = args.device

    lm_head = None
    mode = args.ref_mode
    if mode in ("hidden", "auto") and not ref_set.has("logits") and ref_set.has("final_hidden"):
        lm_head = torch.load(args.lm_head) if args.lm_head else load_lm_head(os.path.expanduser(args.model_dir))
        mode = "hidden"
    ref_src = km.SetSource(ref_set, device=device, mode=mode, lm_head=lm_head)
    exactness = {"full": "exact (stored fp16 logits)", "hidden": "exact (recomputed from the stored bf16 hidden state and lm_head)",
                 "topk": "lumped top-256 lower bound"}[ref_src.mode]
    print(f"reference {ref_set.path}: {ref_set.rows} rows x {T}, mode {ref_src.mode} -> {exactness}", flush=True)

    result = {"schema": "minicpm-v47-kl-v1", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "leg": args.leg or man.get("leg"),
              "ids": {"file": os.path.abspath(os.path.expanduser(args.ids)), "name": man.get("name"), "ids_sha256": man.get("ids_sha256"),
                      "rows": len(sel), "tokens_per_row": T, "vocab": common.V, "complete": man.get("complete")},
              "reference": {"dir": ref_set.path, "kind": "HF transformers bf16, layer-streamed",
                            "producer": ref_set.meta.get("producer"), "kl_computation": exactness},
              "rows": [{"row": int(rows_sel[r]), "name": row_names[r], "kind": row_kinds[r]} for r in sel],
              "candidates": {}}
    if args.npz_dir:
        os.makedirs(os.path.expanduser(args.npz_dir), exist_ok=True)

    scopes = {}
    if args.mask_npz:
        z = np.load(os.path.expanduser(args.mask_npz))
        scopes = {k: z[k].astype(bool) for k in z.files}
        for k, v in scopes.items():
            assert v.shape == (ids.shape[0], T), (k, v.shape)
    head_mask = scopes.pop("mask", None)

    def summarize(arr, with_scopes=False):
        s = km.summarize(arr, row_names=row_names, row_kinds=row_kinds if all(row_kinds) else None, rows=sel, mask=head_mask)
        if with_scopes and scopes:
            s["scopes"] = {k: km.summarize(arr, rows=sel, mask=(v if head_mask is None else v & head_mask)) for k, v in scopes.items()}
        return s

    def run_pair(name, set_dir, ref, tag):
        """metrics of set_dir against `ref` (SetSource) -> arrays"""
        npz = os.path.join(set_dir, "metrics.npz")
        ls = common.LogitSet(set_dir)
        if ls.has("logits"):
            return compare_sets(ref, ls, ids, list(range(ids.shape[0])), device, f"{name} {tag}", args.chunk)
        if os.path.exists(npz) and ref is ref_src:
            arr, meta = load_metrics_npz(npz)
            if meta.get("ids_sha256") not in (None, ref_set.meta.get("ids_sha256")):
                raise SystemExit(f"{npz}: made for different ids")
            return arr
        raise SystemExit(f"{set_dir}: no logits.f16" + (" and no metrics.npz against this reference" if ref is ref_src else ""))

    def maybe_save(tag, arr):
        if args.npz_dir:
            save_metrics_npz(os.path.join(os.path.expanduser(args.npz_dir), f"{tag}.npz"), arr,
                             {"ids_sha256": ref_set.meta.get("ids_sha256"), "reference": ref_set.path, "tag": tag})

    floor_arr = None
    floor_src = None
    if args.floor:
        fname, fdir = parse_named([args.floor])[0]
        print(f"floor {fname}", flush=True)
        floor_arr = run_pair(fname, fdir, ref_src, "vs ref")
        maybe_save(f"{fname}_vs_ref", floor_arr)
        result["floor"] = {"name": fname, "dir": fdir, "vs_ref": summarize(floor_arr)}
        print("  " + card_line(f"{fname} (floor)", result["floor"]["vs_ref"]), flush=True)
        fset = common.LogitSet(fdir)
        if fset.has("logits"):
            floor_src = km.SetSource(fset, device=device, mode="full")
    alt_src = None
    if args.alt_ref:
        aname, adir = parse_named([args.alt_ref])[0]
        print(f"alt reference {aname}", flush=True)
        a_arr = run_pair(aname, adir, ref_src, "vs ref")
        maybe_save(f"{aname}_vs_ref", a_arr)
        result["alt_reference"] = {"name": aname, "dir": adir, "vs_ref": summarize(a_arr)}
        print("  " + card_line(f"{aname} (alt ref vs ref)", result["alt_reference"]["vs_ref"]), flush=True)
        aset = common.LogitSet(adir)
        if aset.has("logits"):
            alt_src = km.SetSource(aset, device=device, mode="full")

    for name, d in parse_named(args.cand):
        print(f"candidate {name}", flush=True)
        entry = {"dir": d}
        try:
            entry["producer"] = common.LogitSet(d).meta.get("producer")
        except Exception:
            pass
        arr = run_pair(name, d, ref_src, "vs ref")
        maybe_save(f"{name}_vs_ref", arr)
        entry["vs_ref"] = summarize(arr, with_scopes=True)
        print("  " + card_line(name, entry["vs_ref"]), flush=True)
        if floor_arr is not None:
            entry["excess_over_floor_kl_ref_to_q"] = km.paired_diff(
                arr["kl_pq"][sel], floor_arr["kl_pq"][sel], None if head_mask is None else head_mask[sel])
        if floor_src is not None:
            ls = common.LogitSet(d)
            if ls.has("logits"):
                f_arr = compare_sets(floor_src, ls, ids, list(range(ids.shape[0])), device, f"{name} vs floor")
                maybe_save(f"{name}_vs_floor", f_arr)
                entry["vs_floor"] = summarize(f_arr)
                print("  " + card_line(name, entry["vs_floor"], ref=args.floor.split("=")[0]), flush=True)
        if alt_src is not None:
            ls = common.LogitSet(d)
            if ls.has("logits"):
                a_arr = compare_sets(alt_src, ls, ids, list(range(ids.shape[0])), device, f"{name} vs alt")
                entry["vs_alt_reference"] = summarize(a_arr)
        result["candidates"][name] = entry

    if args.check_lumped and args.cand and ref_set.has("topk_val") and ref_set.has("tail_lse"):
        name, d = parse_named(args.cand)[0]
        ls = common.LogitSet(d)
        if ls.has("logits") and ref_src.exact:
            cand = km.SetSource(ls, device=device, mode="full")
            ex = {"kl_pq": [], "kl_qp": []}
            lu = {"kl_pq": [], "kl_qp": []}
            for r in sel:
                for lo in range(0, T, args.chunk):
                    hi = min(lo + args.chunk, T)
                    q = cand.logits(r, lo, hi)
                    tgt = km.next_token_targets(ids[r], lo, hi, device)
                    me = km.position_metrics(ref_src.logits(r, lo, hi), q, tgt)
                    t = ref_src.topk(r, lo, hi)
                    ml = km.lumped_metrics(t["val"], t["idx"], t["lse"], t["tail_lse"], t["ent"], t["tgt_logit"], q, tgt)
                    for k in ex:
                        ex[k].append(me[k].cpu().numpy())
                        lu[k].append(ml[k].cpu().numpy())
            chk = {}
            for k in ex:
                e, l = np.concatenate(ex[k]), np.concatenate(lu[k])
                chk[k] = {"exact_mean": float(e.mean()), "lumped_mean": float(l.mean()), "ratio_mean": float(l.mean() / max(e.mean(), 1e-300)),
                          "worst_position_ratio_p1": float(np.percentile(l / np.maximum(e, 1e-12), 1)),
                          # the bound is exact in real arithmetic; the exact side reads fp16-rounded logits, the lumped side fp32 stats
                          "lumped_exceeds_exact_by_more_than_1e-4": int((l > e + 1e-4).sum())}
            result["lumped_vs_exact"] = {"candidate": name, "topk": common.TOPK, "rows": len(sel), **chk}
            print("lumped vs exact:", json.dumps(result["lumped_vs_exact"]), flush=True)

    # compact card table
    card = {}
    if "floor" in result:
        card[result["floor"]["name"] + " (floor)"] = result["floor"]["vs_ref"]
    for n, e in result["candidates"].items():
        card[n] = e["vs_ref"]
    result["card"] = {n: {"kl_ref_to_q": s["kl_ref_to_q"], "kl_q_to_ref": s["kl_q_to_ref"], "top1_agreement": s["top1_agreement"],
                          "topk_overlap": s["topk_overlap"], "ppl_ratio_q_over_ref": s.get("ppl", {}).get("ratio_q_over_ref"),
                          "row_mean_kl_median": s.get("row_mean_kl_median"), "mean_kl_excluding_worst_row": s.get("mean_kl_excluding_worst_row"),
                          "worst_row": s.get("worst_row"), "positions": s["positions"]} for n, s in card.items()}
    print()
    for n, s in card.items():
        print(card_line(n, s))
    if args.json:
        common.atomic_json(os.path.expanduser(args.json), result)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
