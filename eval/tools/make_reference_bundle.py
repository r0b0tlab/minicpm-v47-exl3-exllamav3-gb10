#!/usr/bin/env python3
"""Sanitised evidence bundle for results measured on ANOTHER engine (the reference column).

Inputs are the reference engine's own rows / reports; the output keeps, per row, only: id, family, passed or
ungraded, finish_reason, prompt and completion tokens, elapsed seconds and the content sha256. No prompts, no
responses, no image or profile identifiers, no host or path strings. The engine is named only as "an inference
engine implementing the DSpark block drafter (not ExLlamaV3)".

    make_reference_bundle.py --out evidence/reference-dspark-engine-20261013 --date 2026-10-13 \
        --rows W4=PATH --rows W1=PATH --walls PATH --manual-evidence PATH \
        --legd-plain-report PATH --legd-four-streams-report PATH --measurements PATH
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundlelib as bl  # noqa: E402

ENGINE = "an inference engine implementing the DSpark block drafter (not ExLlamaV3)"


def md_rows(text, header_starts):
    """Rows of the markdown table that follows a '## <mode>' heading -> {mode: [cells...]}."""
    out, mode = {}, None
    for line in text.splitlines():
        m = re.match(r"^##+\s+(greedy|sampled)", line)
        if m:
            mode = m.group(1)
            continue
        if mode and line.startswith("|") and not line.startswith("| ---") and not line.startswith("| arm"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] in header_starts and len(cells) > 6 and cells[1] in ("gsm8k", "code", "chat", "all"):
                out.setdefault(mode, []).append(cells)
    return out


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group(0)) if m else None


def legd_from_report(text):
    rows = md_rows(text, ("ar", "d6"))
    res = {}
    for mode, cells_list in rows.items():
        for c in cells_list:
            arm, cls, n = c[0], c[1], int(c[2])
            res.setdefault(mode, {}).setdefault(cls, {})[arm] = {
                "n": n, "tok_s": num(c[5]), "tok_s_se": (re.findall(r"\+-\s*([\d.]+)", c[5]) or [None])[0],
                "tok_s_pooled": num(c[6]), "al": num(c[4]) if arm != "ar" else 1.0,
                "speedup": num(c[7]) if len(c) > 7 and c[7] else None}
    return res


def md_table(block):
    out = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*:?-", line):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--rows", action="append", default=[], help="W<workers>=path to a kit rows.jsonl (repeatable)")
    ap.add_argument("--walls", help="file with lines: <run-name> <workers> <t0> <t1>")
    ap.add_argument("--manual-evidence", help="the reference run's manual hard-reasoning grades (items dict)")
    ap.add_argument("--legd-plain-report")
    ap.add_argument("--legd-four-streams-report")
    ap.add_argument("--measurements", help="JSON with a 'values' object of markdown summaries")
    a = ap.parse_args(argv)

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    manual = None
    manual_doc = None
    if a.manual_evidence:
        manual_doc = json.load(open(a.manual_evidence))
        manual = {k: v["verdict"] == "pass" for k, v in manual_doc["items"].items()}
    walls = []
    if a.walls:
        for line in open(a.walls):
            parts = line.split()
            if len(parts) == 4:
                walls.append((int(parts[1]), float(parts[3]) - float(parts[2])))
    results, lanes = {}, {}
    for spec in a.rows:
        key, path = spec.split("=", 1)
        workers = int(key.lstrip("Ww"))
        rows = [bl.sanitize_q200_row(json.loads(l), manual) for l in open(path) if l.strip()]
        name = "q200-w%d.rows.public.jsonl" % workers
        with open(os.path.join(out, name), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        wall = next((w for (wk, w) in walls if wk == workers), None)
        results["q200" if workers == 1 else "q200_w%d" % workers] = bl.q200_results(rows, workers, wall)
        lanes["q200_w%d" % workers] = {"rows": name}
    if manual_doc:
        keep = {k: {"verdict": v["verdict"], "reason": v["reason"], "completion_tokens": v["completion_tokens"],
                    "finish_reason": v["finish_reason"]} for k, v in sorted(manual_doc["items"].items())}
        with open(os.path.join(out, "q200.hard_reasoning.manual.json"), "w") as fh:
            json.dump({"family": "hard_reasoning", "graded_by": manual_doc.get("graded_by"),
                       "dataset_sha256": manual_doc.get("dataset_sha256"), "correct": manual_doc.get("correct"),
                       "n": manual_doc.get("n"), "items": keep}, fh, indent=1)
    if a.legd_plain_report:
        results["legd"] = {"one_stream": legd_from_report(open(a.legd_plain_report).read())}
        lanes["legd_one_stream"] = {"source": "frozen 80-prompt kit, greedy 80 + sampled first 10 per class"}
    if a.legd_four_streams_report:
        results.setdefault("legd", {})["four_streams"] = legd_from_report(open(a.legd_four_streams_report).read())
    if a.measurements:
        vals = json.load(open(a.measurements))["values"]
        niah_rows = md_table(vals["niah_summary"])
        depths = {}
        for r in niah_rows[1:]:
            if len(r) >= 6 and re.match(r"^[\d,]+$", r[0]):
                d = str(int(r[0].replace(",", "")))
                hit = r[5].replace("-", " ").split()
                depths[d] = {"ttft_cold_s": num(r[1]), "prefill_tok_s": num(r[2]), "decode_tok_s": num(r[4]),
                             "ttft_cache_hit_s": num(hit[-1]) if hit else None}
        results["bench"] = {"depths": depths, "chunk_size": 4096, "decode_tok_s_short_context": 68.0,
                            "prefill_tok_s_chunk4096": 5037, "ttft_cold_s_by_prompt_tokens": {
                                "1024": 0.26, "8192": 1.7, "32768": 7.8, "65536": 18.2, "131072": 46.0, "261000": 138.0}}
        results["niah"] = {"asked": 22, "exact": 20, "skipped": 0,
                           "wrong": [{"cut": 200000, "verdict": "other_needle"}, {"cut": 261000, "verdict": "other_needle"}],
                           "note": "9 of 11 exact at 200k and 261k, 11 of 11 up to 128k; both misses return another needle's passcode"}
        results["memory"] = {"resident_gib": 34.2, "weights_gib": 25.31, "kv_cache_gib_262144": 5.0,
                             "note": "four 262,144-token slots, chunks of 4,096, prefix cache on, vision tower loaded"}
        with open(os.path.join(out, "scorecard.md"), "w") as fh:
            fh.write("# Reference engine: long-context scorecard, decode and memory (as measured)\n\n")
            for k in ("niah_summary", "decode_summary", "memory_summary", "accept_table"):
                fh.write(vals[k].strip() + "\n\n")
        lanes["scorecard"] = {"file": "scorecard.md"}
    doc = {
        "id": os.path.basename(out), "column": "reference-engine",
        "title": "Reference engine with DSpark drafting and the vision tower (not ExLlamaV3)",
        "engine": {"name": ENGINE, "commit": None,
                   "mode": "DSpark drafting (6.0 bpw draft), vision tower loaded, 262,144-token window, "
                           "4 sequence slots, prefix cache on, window graph replayed"},
        "hardware": "1x NVIDIA GB10 (DGX Spark, aarch64, SM121), 128 GB unified memory",
        "date": a.date, "profile": "262,144-token window",
        "model": {"repo": "r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw", "revision": None},
        "reproducible_with_this_repo": False,
        "caveats": [
            "Measured on a different engine: not reproducible with this repository's runtime (ExLlamaV3 runs the text path without a drafter).",
            "Q200v2: the output with the drafter is token-for-token identical to decoding without it; one HumanEval row reached the 8,192-token ceiling and counts as failed.",
            "hard_reasoning has no automatic grader: the 20 answers were read item by item against the kit's reference answers by the pack builder (not an independent reviewer).",
            "The NIAH document differs from this repository's document (other prose); positions, cuts, wording and scoring are the same method.",
        ],
        "note": "",
        "lanes": lanes, "results": results,
    }
    bl.write_bundle(out, doc)
    print("reference bundle written to %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
