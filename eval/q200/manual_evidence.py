#!/usr/bin/env python3
"""Manual grading of the hard_reasoning family (no automatic grader exists for it).

  manual_evidence.py dump  --rows RUN.rows.jsonl --set quality-text-180-v2.jsonl [--id hard-03]
        print each hard_reasoning item: the task, the kit's reference answer and the model's final answer
  manual_evidence.py build --rows RUN.rows.jsonl --summary RUN.summary.json --grades grades.json --reviewer TEXT --out manual.json
        grades.json: {"hard-00": {"passed": true, "rationale": "..."}, ...} (all 20 ids)
        writes the kit's manual-evidence file, bound to the sha256 of every response; pass it to
        eval/q200/run_q200.sh --manual-evidence manual.json --run-id <same run id> to regrade without regenerating

The kit's schema requires method "independent_manual_review"; who actually graded is stated in `reviewer`.
"""
import argparse
import hashlib
import json
import sys


def rows_by_id(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--rows", required=True)
    d.add_argument("--set", required=True)
    d.add_argument("--id")
    b = sub.add_parser("build")
    b.add_argument("--rows", required=True)
    b.add_argument("--summary", required=True)
    b.add_argument("--grades", required=True)
    b.add_argument("--reviewer", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = rows_by_id(a.rows)
    if a.cmd == "dump":
        ds = {}
        with open(a.set) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    ds[r["id"]] = r
        for rid in sorted(k for k in ds if ds[k]["family"] == "hard_reasoning"):
            if a.id and rid != a.id:
                continue
            r = rows.get(rid)
            print("=" * 100)
            print("ID %s   finish=%s completion_tokens=%s" % (rid, r and r.get("finish_reason"), r and r.get("completion_tokens")))
            print("TASK:\n" + ds[rid]["prompt"])
            print("REFERENCE:\n" + str(ds[rid].get("reference")))
            print("ANSWER:\n" + ((r or {}).get("content") or "<empty>"))
        return 0
    grades = json.load(open(a.grades))
    summary = json.load(open(a.summary))
    hard = sorted(k for k, v in rows.items() if v["family"] == "hard_reasoning")
    if set(grades) != set(hard):
        print("grades must cover exactly the %d hard_reasoning ids; missing %s, extra %s" % (
            len(hard), sorted(set(hard) - set(grades)), sorted(set(grades) - set(hard))), file=sys.stderr)
        return 2
    doc = {
        "schema": "r0b0tlab.qwen38.manual_evidence.v1",
        "dataset_sha256": summary["dataset_sha256"],
        "run_identity_sha256": summary["run_identity_sha256"],
        "reviewer": a.reviewer,
        "method": "independent_manual_review",
        "rows": [{"id": k, "content_sha256": hashlib.sha256((rows[k].get("content") or "").encode("utf-8")).hexdigest(),
                  "passed": bool(grades[k]["passed"]), "rationale": grades[k]["rationale"]} for k in hard],
    }
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote %s: %d/%d passed" % (a.out, sum(1 for r in doc["rows"] if r["passed"]), len(doc["rows"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
