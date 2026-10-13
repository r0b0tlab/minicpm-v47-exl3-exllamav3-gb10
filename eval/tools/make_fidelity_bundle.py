#!/usr/bin/env python3
"""Evidence bundle for the quantisation-fidelity measurements published with the pack (kl.json, rel_l2_scan.json,
calibration.json). These were measured through the same ExLlamaV3 forward this repository serves with.

    make_fidelity_bundle.py --out evidence/fidelity-20261013 --kl KL.json --scan SCAN.json --calibration CAL.json --date 2026-10-13
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundlelib as bl  # noqa: E402


def leg(d, ref):
    m = d[ref]
    return {"kl_ref_to_pack_mean": m["kl_ref_to_q"]["mean"], "kl_ref_to_pack_median": m["kl_ref_to_q"]["median"],
            "kl_ref_to_pack_p95": m["kl_ref_to_q"].get("p95"), "kl_ref_to_pack_p99": m["kl_ref_to_q"].get("p99"),
            "kl_ref_to_pack_max": m["kl_ref_to_q"]["max"], "kl_pack_to_ref_mean": m["kl_q_to_ref"]["mean"],
            "kl_excluding_worst_row": m["mean_kl_excluding_worst_row"], "top1_agreement": m["top1_agreement"],
            "ppl_ratio": m["ppl"]["ratio_q_over_ref"], "positions": m["positions"], "rows": m["rows"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kl", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--date", required=True)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    kl = json.load(open(a.kl))
    scan = json.load(open(a.scan))
    for src, name in ((a.kl, "kl.json"), (a.scan, "rel_l2_scan.json"), (a.calibration, "calibration.json")):
        shutil.copy(src, os.path.join(out, name))
    fid = {}
    for name, leg_doc in kl["legs"].items():
        fid["set_" + name] = {"ids_sha256": leg_doc["ids"]["ids_sha256"], "rows": leg_doc["ids"]["rows"],
                              "tokens_per_row": leg_doc["ids"]["tokens_per_row"],
                              "vs_fork_bf16": leg(leg_doc, "vs_fork_bf16"), "vs_hf_bf16": leg(leg_doc, "vs_hf_bf16"),
                              "numerics_floor_fork_vs_hf": leg(leg_doc, "numerics_floor_fork_bf16_vs_hf_bf16")}
    y = kl["yardstick"]["metrics"]
    fid["yardstick_4bpw_published_exl3"] = {"kl_ref_to_pack_mean": y["kl_ref_to_q"]["mean"],
                                           "kl_ref_to_pack_median": y["kl_ref_to_q"]["median"],
                                           "kl_excluding_worst_row": y["mean_kl_excluding_worst_row"],
                                           "top1_agreement": y["top1_agreement"],
                                           "label": kl["yardstick"]["label"], "bpw": 4.0}
    fid["fp32_floor"] = {k: {"label": v["label"], "kl_mean": v["kl_mean"], "kl_excl_worst_row": v["kl_excl_worst_row"],
                             "top1": v["top1"]} for k, v in kl["fp32_floor"].items()}
    s = scan.get("summary", {})
    fid["weight_scan"] = {"summary": s}
    doc = {
        "id": os.path.basename(out), "column": "fidelity",
        "title": "Quantisation fidelity of the 6.08 bpw pack (text), measured through the ExLlamaV3 forward",
        "engine": {"name": "ExLlamaV3", "commit": "4d083d25c1a3e195ce1305623ffa1eef84b834fe",
                   "mode": "fp16 activations; reference = the bf16 checkpoint through the same implementation"},
        "hardware": "1x NVIDIA GB10 (DGX Spark)", "date": a.date, "profile": "n/a",
        "model": {"repo": "r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw", "revision": None},
        "reproducible_with_this_repo": True,
        "caveats": ["Text only. The id sets are not shipped (they contain excerpts of non-public sources); their "
                    "sha256 are in the files. Calibration was text-only.",
                    "These numbers apply to ExLlamaV3 serving: they were measured through the same forward."],
        "note": "", "lanes": {"kl": {"file": "kl.json"}, "scan": {"file": "rel_l2_scan.json"},
                               "calibration": {"file": "calibration.json"}},
        "results": {"fidelity": fid},
    }
    bl.write_bundle(out, doc)
    print("fidelity bundle written to %s" % out)


if __name__ == "__main__":
    main()
