#!/usr/bin/env python3
"""Render the generated result tables of README.md / docs/*.md from the evidence bundles (stdlib only).

Numbers in this repository's prose tables are never typed by hand: each table sits between

    <!-- BEGIN GENERATED:<name> -->
    ...
    <!-- END GENERATED:<name> -->

and is produced here from evidence/<bundle>/bundle.json (the bundle list and the file -> block map are in
protocol.json). tests/test_render_results.py and scripts/verify_tree.py fail when a file drifts from its bundles.

  render_results.py --write     rewrite the blocks in place
  render_results.py --check     exit 1 if any block differs from what the bundles render
  render_results.py --print NAME
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

COLUMNS = [
    ("exllamav3-plain", "ExLlamaV3, plain decode (this repo's runtime)"),
    ("exllamav3-dspark", "ExLlamaV3 + DSpark drafting"),
]
PENDING = "pending"
FAMILY_LABEL = {"gsm8k": "GSM8K-style math (80)", "humaneval": "HumanEval-style code (40)",
                "ifeval": "IFEval-style instruction following (40)", "hard_reasoning": "Hard reasoning (20, AI-graded, see notes)"}
DEPTH_TARGETS = [("~4k", 4096), ("~32k", 32768), ("~128k", 131072), ("~261k", 261000)]
DRAFT_COMMAND = ("DRAFT_DIR=models/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw ./run.sh --with-draft --yes && "
                 "eval/run_suite.sh http://127.0.0.1:8000 exllamav3-dspark")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_protocol(root=ROOT):
    return load_json(os.path.join(root, "protocol.json"))


def load_bundles(root=ROOT):
    """column -> bundle.json (the entry marked primary, else the latest date)."""
    proto = load_protocol(root)
    by_col = {}
    for entry in proto.get("evidence_bundles", []):
        b = load_json(os.path.join(root, entry["dir"], "bundle.json"))
        b["_dir"] = entry["dir"]
        b["_root"] = root
        by_col.setdefault(b.get("column"), []).append((entry.get("primary", False), b.get("date", ""), b))
    out = {}
    for col, items in by_col.items():
        items.sort(key=lambda t: (t[0], t[1]))
        out[col] = items[-1][2]
    return out


# ---------------------------------------------------------------- cell helpers

def fnum(x, nd=1, unit=""):
    if x is None:
        return "-"
    if isinstance(x, float) and abs(x - round(x)) < 1e-9 and nd == 0:
        x = int(round(x))
    return ("{:,.%df}" % nd).format(x) + unit


def frac(passed, n):
    if passed is None or not n:
        return "-"
    return "%d/%d (%.1f%%)" % (passed, n, 100.0 * passed / n)


def depth_value(bundle, target, field):
    if not bundle:
        return None
    depths = (bundle.get("results", {}).get("bench") or {}).get("depths") or {}
    best, best_err = None, 0.08
    for key, row in depths.items():
        err = abs(int(key) - target) / float(target)
        if err < best_err:
            best, best_err = row, err
    return best.get(field) if best else None


def q200(bundle, key="q200"):
    return (bundle or {}).get("results", {}).get(key)


def legd_value(bundle, kind, mode, cls, field="tok_s_mean"):
    """Leg D number of one kind of decode: plain, drafted (adaptive verification) or fixed15 (always 15 drafts).

    This repository's bundles hold
    results.legd (primary run), results.legd_plain_ctl (plain control on the drafting build) and
    results.legd_fixed15, keyed "<mode>/<class>"."""
    if not bundle:
        return None
    res = bundle.get("results", {}).get("legd") or {}
    col = bundle.get("column")
    if col == "exllamav3-plain":
        src = res if kind == "plain" else None
    elif col == "exllamav3-dspark":
        src = {"plain": bundle["results"].get("legd_plain_ctl"), "drafted": res,
               "fixed15": bundle["results"].get("legd_fixed15")}.get(kind)
    else:
        src = None
    cell = (src or {}).get("%s/%s" % (mode, cls))
    return cell.get(field) if cell else None


def legd_cell(bundle, mode, cls, arm):
    """Back-compatible accessor used by older callers: arm 'ar' = plain, anything else = drafted."""
    return legd_value(bundle, "plain" if arm == "ar" else "drafted", mode, cls)


def legd_streams(bundle):
    res = ((bundle or {}).get("results", {}).get("legd") or {})
    if "one_stream" in res:
        return 1
    for k, v in res.items():
        if isinstance(v, dict) and "streams" in v:
            return v["streams"]
    return None


# ---------------------------------------------------------------- blocks

def row_cells(bundles, fn):
    cells = []
    for col, _label in COLUMNS:
        b = bundles.get(col)
        if b is None:
            cells.append(PENDING if col == "exllamav3-dspark" else "-")
            continue
        try:
            cells.append(fn(b))
        except (KeyError, TypeError, ZeroDivisionError):
            cells.append("-")
    return cells


def table(headers, rows, aligns=None):
    aligns = aligns or (["---"] + ["---:"] * (len(headers) - 1))
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def block_results_summary(bundles):
    rows = []

    def add(label, fn):
        rows.append([label] + row_cells(bundles, fn))

    add("**Q200v2 text-180**, accuracy", lambda b: frac(q200(b)["passed"], q200(b)["n"]))
    for fam in ("gsm8k", "humaneval", "ifeval", "hard_reasoning"):
        add("&nbsp;&nbsp;" + FAMILY_LABEL[fam], lambda b, fam=fam: frac(q200(b)["families"][fam]["passed"],
                                                                         q200(b)["families"][fam]["n"]))
    add("Q200v2 end-to-end tok/s, 1 request in flight",
        lambda b: fnum(q200(b)["e2e_tok_s"]))
    add("Q200v2 end-to-end tok/s, 4 requests in flight (ExLlamaV3: experimental batch 4)",
        lambda b: fnum(q200(b, "q200_w4")["e2e_tok_s"]) if q200(b, "q200_w4") else "not run (batch 1)")
    for label, target in DEPTH_TARGETS:
        add("Decode tok/s at %s context" % label, lambda b, t=target: fnum(depth_value(b, t, "decode_tok_s")))
    add("Prefill tok/s at ~32k", lambda b: fnum(depth_value(b, 32768, "prefill_tok_s"), 0))
    add("Cold time to first token at ~32k (s)", lambda b: fnum(depth_value(b, 32768, "ttft_cold_s"), 1))
    add("Cold time to first token at ~261k (s)", lambda b: fnum(depth_value(b, 261000, "ttft_cold_s"), 1))
    add("Needle retrieval (22 questions, token-exact)",
        lambda b: "%d/%d exact" % (b["results"]["niah"]["exact"], b["results"]["niah"]["asked"]))
    for kind, kind_label in (("plain", "plain"), ("drafted", "drafted, adaptive verification"),
                             ("fixed15", "drafted, always 15 drafts")):
        for cls, label in (("gsm8k", "GSM8K-style"), ("code", "code"), ("chat", "chat")):
            def cell(b, k=kind, c=cls):
                v = legd_value(b, k, "greedy", c)
                if v is not None:
                    return fnum(v)
                return "n/a" if b.get("column") == "exllamav3-plain" else "-"
            add("Leg D decode tok/s, %s, %s" % (label, kind_label), cell)

    def identical(b):
        c = ((b.get("results", {}).get("legd") or {}).get("greedy/all") or {})
        if b.get("column") == "exllamav3-dspark" and c.get("compared"):
            return "%d/%d" % (c["identical_to_plain"], c["compared"])
        return "n/a"
    add("Leg D greedy streams identical to plain decoding", identical)

    def resident(b):
        m = b["results"]["memory"]
        if "resident_gib" in m:
            return fnum(m["resident_gib"], 1, " GiB")
        return "%s GiB allocated" % fnum(m.get("torch_allocated_gib"), 1)
    add("GPU-side resident memory", resident)
    text = table(["metric"] + [c[1] for c in COLUMNS], rows)
    drafted = bundles.get("exllamav3-dspark")
    if drafted is None:
        text += ("\n\n_ExLlamaV3 + DSpark: pending. The fork's drafting path is being implemented; once a build "
                 "supports it this column is filled by_ `" + DRAFT_COMMAND + "`.")
    return text


def block_q200_table(bundles):
    rows = []
    for fam in ("gsm8k", "humaneval", "ifeval", "hard_reasoning"):
        cells = row_cells(bundles, lambda b, fam=fam: frac(q200(b)["families"][fam]["passed"], q200(b)["families"][fam]["n"]))
        rows.append([FAMILY_LABEL[fam]] + cells)
    rows.append(["**all 180**"] + row_cells(bundles, lambda b: frac(q200(b)["passed"], q200(b)["n"])))
    rows.append(["completion tokens"] + row_cells(bundles, lambda b: fnum(q200(b)["completion_tokens"], 0)))
    rows.append(["wall seconds, 1 request in flight"] + row_cells(bundles, lambda b: fnum(q200(b)["wall_seconds"], 0)))
    rows.append(["end-to-end tok/s, 1 request in flight"] + row_cells(bundles, lambda b: fnum(q200(b)["e2e_tok_s"])))
    rows.append(["median per-request tok/s"] + row_cells(bundles, lambda b: fnum(q200(b)["per_request_tok_s_median"])))
    for fam in ("gsm8k", "humaneval", "ifeval", "hard_reasoning"):
        rows.append(["e2e tok/s, %s" % fam] + row_cells(
            bundles, lambda b, fam=fam: fnum(q200(b)["per_family_throughput"][fam]["tok_s_serial"])))
    rows.append(["finish reasons"] + row_cells(bundles, lambda b: ", ".join(
        "%s %d" % (k, v) for k, v in sorted(q200(b)["finish_reasons"].items()))))
    rows.append(["4 requests in flight: pass count"] + row_cells(
        bundles, lambda b: frac(q200(b, "q200_w4")["passed"], q200(b, "q200_w4")["n"]) if q200(b, "q200_w4") else "not run"))
    rows.append(["4 requests in flight: families (gsm8k / humaneval / ifeval / hard)"] + row_cells(
        bundles, lambda b: " / ".join(str(q200(b, "q200_w4")["families"][f]["passed"]) for f in
                                      ("gsm8k", "humaneval", "ifeval", "hard_reasoning")) if q200(b, "q200_w4") else "-"))
    rows.append(["4 requests in flight: wall seconds"] + row_cells(
        bundles, lambda b: fnum(q200(b, "q200_w4")["wall_seconds"], 0) if q200(b, "q200_w4") else "not run"))
    rows.append(["4 requests in flight: end-to-end tok/s"] + row_cells(
        bundles, lambda b: fnum(q200(b, "q200_w4")["e2e_tok_s"]) if q200(b, "q200_w4") else "not run"))
    return table(["Q200v2 text-180"] + [c[1] for c in COLUMNS], rows)


def block_bench_table(bundles):
    depths = []
    for b in bundles.values():
        for d in ((b.get("results", {}).get("bench") or {}).get("depths") or {}):
            depths.append(int(d))
    wanted = sorted({int(round(d, -2)) if d > 1000 else d for d in depths})
    keys = [("1024", 1024), ("4096", 4096), ("32768", 32768), ("131072", 131072), ("261000", 261000)]
    rows = []
    for label, target in keys:
        for field, name in (("decode_tok_s", "decode tok/s"), ("prefill_tok_s", "prefill tok/s"),
                            ("ttft_cold_s", "cold TTFT (s)"), ("ttft_cache_hit_s", "prefix-cache-hit TTFT (s)")):
            vals = row_cells(bundles, lambda b, t=target, f=field: fnum(
                depth_value(b, t, f), 2 if "ttft" in f else (0 if "prefill" in f else 1)))
            if any(v not in ("-", PENDING) for v in vals):
                rows.append(["%s tokens: %s" % (f"{target:,}", name)] + vals)
    del wanted
    return table(["context depth: measure"] + [c[1] for c in COLUMNS], rows)


def block_niah_table(bundles):
    rows = []
    rows.append(["questions asked"] + row_cells(bundles, lambda b: str(b["results"]["niah"]["asked"])))
    rows.append(["exact"] + row_cells(bundles, lambda b: str(b["results"]["niah"]["exact"])))
    rows.append(["misses"] + row_cells(bundles, lambda b: "; ".join(
        "%s@%s -> %s" % ("needle %s" % w.get("needle", "?") if "needle" in w else "cut", w.get("cut"), w.get("verdict"))
        for w in b["results"]["niah"]["wrong"]) or "none"))
    return table(["needle-in-a-haystack, 261,001-token document"] + [c[1] for c in COLUMNS], rows)


def block_legd_table(bundles):
    rows = []
    kinds = (("plain", "plain"), ("drafted", "drafted, adaptive verification"), ("fixed15", "drafted, always 15 drafts"))
    for mode in ("greedy", "sampled"):
        for cls in ("gsm8k", "code", "chat", "all"):
            for kind, label in kinds:
                def cell(b, m=mode, c=cls, k=kind):
                    v = legd_value(b, k, m, c)
                    if v is not None:
                        return fnum(v)
                    return "n/a" if b.get("column") == "exllamav3-plain" else "-"
                rows.append(["%s, %s, %s" % (mode, cls, label)] + row_cells(bundles, cell))
    d = bundles.get("exllamav3-dspark")
    text = table(["Leg D decode tok/s, one stream, thinking off, up to 320 tokens"] + [c[1] for c in COLUMNS], rows)
    if d:
        extra = []
        for kind, label in (("drafted", "adaptive verification"), ("fixed15", "always 15 drafts")):
            for cls in ("gsm8k", "code", "chat", "all"):
                src = d["results"].get("legd" if kind == "drafted" else "legd_fixed15") or {}
                c = src.get("greedy/%s" % cls)
                if not c:
                    continue
                plain = (d["results"].get("legd_plain_ctl") or {}).get("greedy/%s" % cls) or {}
                speed = (c["tok_s_mean"] / plain["tok_s_mean"]) if plain.get("tok_s_mean") else None
                extra.append([label, cls, fnum(c["tok_s_mean"]), "%.2fx" % speed if speed else "-",
                              "%.1f%%" % (100 * c["accept_rate"]) if c.get("accept_rate") is not None else "-",
                              "%s/%s" % (c.get("identical_to_plain", "-"), c.get("compared", "-"))])
        if extra:
            text += "\n\nExLlamaV3 + DSpark, greedy arm. The speed-up is against the plain run of 2026-10-13 (a different session; no same-session control was run), and '-' means that run is not in the bundle:\n\n" + table(
                ["verification", "class", "tok/s", "speed-up", "drafts accepted", "streams identical to plain"], extra,
                ["---", "---", "---:", "---:", "---:", "---:"])
    return text


def block_memory_table(bundles):
    def cell(b):
        m = b["results"]["memory"]
        parts = []
        for k, label in (("resident_gib", "resident"), ("weights_gib", "weights"),
                         ("mem_available_before_load_gib", "MemAvailable before load"),
                         ("mem_available_after_load_gib", "MemAvailable after load"),
                         ("torch_allocated_gib", "allocated"),
                         ("torch_reserved_gib", "reserved"), ("torch_max_reserved_gib", "max reserved (after the grid)"),
                         ("mem_available_gib", "MemAvailable after the grid")):
            if m.get(k) is not None:
                parts.append("%s %s GiB" % (label, fnum(m[k], 1)))
        return "; ".join(parts) or "-"
    return table(["memory"] + [c[1] for c in COLUMNS], [["after load"] + row_cells(bundles, cell)])


def block_fidelity_kl(bundles):
    b = bundles.get("fidelity")
    if not b:
        return "_no fidelity bundle_"
    f = b["results"]["fidelity"]
    rows = []
    for name in ("set_A", "set_B"):
        s = f[name]
        v = s["vs_fork_bf16"]
        rows.append(["Set %s: %d x %d tokens" % (name[-1], s["rows"], s["tokens_per_row"]), fnum(v["kl_ref_to_pack_mean"], 6),
                     fnum(v["kl_ref_to_pack_median"], 6), fnum(v["kl_ref_to_pack_p99"], 4),
                     fnum(v["kl_excluding_worst_row"], 6), "%.2f%%" % (100 * v["top1_agreement"]),
                     fnum(v["ppl_ratio"], 4)])
    y = f["yardstick_4bpw_published_exl3"]
    rows.append(["yardstick on Set A: " + y["label"] + ", 4.00 bpw", fnum(y["kl_ref_to_pack_mean"], 6),
                 fnum(y["kl_ref_to_pack_median"], 6), "-", fnum(y["kl_excluding_worst_row"], 6),
                 "%.2f%%" % (100 * y["top1_agreement"]), "-"])
    return table(["KL to the bf16 model, same implementation", "mean", "median", "p99", "mean without worst row",
                  "top-1 agreement", "ppl ratio"], rows)


def _lane_file(bundle, lane):
    spec = (bundle.get("lanes") or {}).get(lane)
    if not spec or not spec.get("file"):
        return None
    path = os.path.join(bundle["_root"], bundle["_dir"], spec["file"])
    return load_json(path) if os.path.isfile(path) else None


def block_depthscan_table(bundles):
    b = bundles.get("exllamav3-plain")
    d = _lane_file(b, "depthscan") if b else None
    if not d:
        return "_no depth scan recorded_"
    by = {}
    for r in d["scan"]:
        by.setdefault(r["depth"], []).append(r["decode_tok_s"])
    rows = [[f"{k:,}", " / ".join("%.1f" % v for v in vs), "%.1f" % (sum(vs) / len(vs))] for k, vs in sorted(by.items())]
    text = table(["prompt tokens (cold prefill, 128 new tokens)", "decode tok/s of each run", "mean"], rows,
                 ["---:", "---", "---:"])
    lg = d.get("long_generation") or []
    if lg:
        text += "\n\nLong generation from a 64-token prompt (context grows while decoding): " + "; ".join(
            "%s new tokens at %.1f tok/s on average" % (f"{e['new_tokens']:,}", e["decode_tok_s"]) for e in lg) + "."
    return text


def block_chunk_table(bundles):
    b = bundles.get("exllamav3-plain")
    d = _lane_file(b, "chunk_ab") if b else None
    if not d:
        return "_no chunk-size comparison recorded_"
    rows = []
    for c in ("2048", "4096"):
        e = d.get(c)
        if e:
            rows.append([c, fnum(e["prefill_tok_s_median"], 0), fnum(e["ttft_cold_s_median"], 2),
                         fnum(e["decode_tok_s_median"], 1), str(e["repeats"])])
    text = table(["prefill chunk (tokens)", "prefill tok/s at %s tokens" % f"{d['2048']['depth']:,}", "cold TTFT (s)",
                  "decode tok/s", "repeats"], rows)
    text += "\n\nRule: %s. Chosen: %s." % (d.get("rule", ""), d.get("chosen"))
    return text


def block_warmup_table(bundles):
    b = bundles.get("exllamav3-plain")
    d = _lane_file(b, "warmup_4k") if b else None
    if not d:
        return "_no warm-up series recorded_"
    rows = [[e["label"], " / ".join("%.1f" % v for v in e["decode_tok_s"]), "%.1f" % e["decode_tok_s_median"],
             " / ".join("{:,}".format(v) for v in e["prefill_tok_s"])] for e in d["sets"]]
    return d["description"] + "\n\n" + table(["series", "decode tok/s per request", "median", "prefill tok/s per request"],
                                               rows, ["---", "---", "---:", "---"])


def block_draft_table(bundles):
    d = bundles.get("exllamav3-dspark")
    if not d:
        return "_no drafted bundle_"
    dr = d["results"].get("draft") or {}
    vc = dr.get("verify_cost_ms") or []
    text = ("Drafting settings of the measured runs: mode %s, %s drafts per round, Markov walk over the top %s candidates, "
            "draft pack %s." % (dr.get("mode"), dr.get("num_draft_tokens"), dr.get("markov_top_k"), dr.get("draft_pack", "")))
    if vc:
        rows = [[str(i + 1), "%.1f" % v] for i, v in enumerate(vc)]
        half = (len(rows) + 1) // 2
        grid = [rows[i] + (rows[i + half] if i + half < len(rows) else ["", ""]) for i in range(half)]
        text += "\n\nMeasured cost model (ms; one verify pass over n rows, 300 committed tokens; draft side of a round %.1f ms):\n\n" % (
            dr.get("draft_cost_ms") or 0) + table(["rows", "ms", "rows", "ms"], grid, ["---:", "---:", "---:", "---:"])
    return text


def block_clickrun_table(bundles):
    b = bundles.get("exllamav3-plain")
    cr = ((b or {}).get("results") or {}).get("clickrun")
    if not cr:
        return "_no click-run timings recorded_"
    rows = []
    for st in ("preflight", "venv", "torch", "build", "download", "verify-model", "retire-cache", "serve", "smoke"):
        e = cr.get("stages", {}).get(st)
        if e:
            rows.append([st, fnum(e.get("seconds"), 0), e.get("note", "")])
    text = table(["stage", "seconds", "note"], rows, ["---", "---:", "---"])
    if cr.get("conditions"):
        text += "\n\n" + cr["conditions"]
    sc = cr.get("second_checkout")
    if sc:
        rows2 = []
        for st in ("preflight", "venv", "torch", "build", "download", "verify-model"):
            e = sc["stages"].get(st)
            if e:
                rows2.append([st, fnum(e.get("seconds"), 0), e.get("note", "")])
        text += "\n\n" + sc["description"] + " Total to a verified, ready tree: %s s.\n\n" % sc.get("total_seconds") + \
            table(["stage (second checkout)", "seconds", "note"], rows2, ["---", "---:", "---"])
    return text


def block_error_codes(_bundles):
    import errcode
    return errcode.markdown()


BLOCKS = {
    "results-summary": block_results_summary,
    "q200-table": block_q200_table,
    "bench-table": block_bench_table,
    "niah-table": block_niah_table,
    "legd-table": block_legd_table,
    "memory-table": block_memory_table,
    "clickrun-table": block_clickrun_table,
    "draft-table": block_draft_table,
    "depthscan-table": block_depthscan_table,
    "chunk-table": block_chunk_table,
    "warmup-table": block_warmup_table,
    "fidelity-kl": block_fidelity_kl,
    "error-codes": block_error_codes,
}

PATTERN = re.compile(r"(<!-- BEGIN GENERATED:(?P<name>[A-Za-z0-9_-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED:(?P=name) -->)",
                     re.S)


def render_block(name, bundles):
    if name not in BLOCKS:
        raise KeyError("unknown generated block %r" % name)
    return BLOCKS[name](bundles).rstrip("\n")


def process(path, bundles, write):
    """-> (names_found, drifted_names)"""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    found, drift = [], []

    def sub(m):
        name = m.group("name")
        found.append(name)
        new = render_block(name, bundles) + "\n"
        if m.group("body") != new:
            drift.append(name)
        return m.group(1) + (new if write else m.group("body")) + m.group(4)

    out = PATTERN.sub(sub, text)
    if write and out != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
    return found, drift


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", dest="print_name")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    bundles = load_bundles(a.root)
    if a.print_name:
        print(render_block(a.print_name, bundles))
        return 0
    proto = load_protocol(a.root)
    bad = 0
    for rel, names in sorted(proto.get("generated_blocks", {}).items()):
        path = os.path.join(a.root, rel)
        if not os.path.isfile(path):
            print("MISSING FILE %s" % rel)
            bad += 1
            continue
        found, drift = process(path, bundles, write=a.write)
        for n in names:
            if n not in found:
                print("MISSING BLOCK %s in %s" % (n, rel))
                bad += 1
        if drift and not a.write:
            print("DRIFT in %s: %s" % (rel, ", ".join(drift)))
            bad += 1
        elif drift:
            print("updated %s: %s" % (rel, ", ".join(drift)))
    if a.check and bad:
        print("render_results: %d problem(s); run scripts/render_results.py --write" % bad)
        return 1
    if a.check:
        print("render_results: all generated blocks match the evidence bundles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
