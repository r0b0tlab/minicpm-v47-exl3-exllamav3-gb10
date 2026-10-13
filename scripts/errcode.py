#!/usr/bin/env python3
"""Error-code table: one source (scripts/errors.tsv) for click_run.sh and the docs.

  errcode.py show CODE [--stage S] [--detail TEXT]   # failure block printed by click_run.sh
  errcode.py list                                    # codes, one per line
  errcode.py markdown                                # table for docs/TROUBLESHOOTING.md
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "errors.tsv")


def load(path=TSV):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError("errors.tsv: expected 4 tab-separated columns: %r" % line)
            rows[parts[0]] = {"code": parts[0], "stage": parts[1], "cause": parts[2], "remedy": parts[3]}
    return rows


def failure_block(code, stage="", detail="", rows=None):
    rows = rows or load()
    r = rows.get(code) or {"code": code, "stage": stage, "cause": "unclassified failure",
                           "remedy": "see docs/TROUBLESHOOTING.md and run scripts/env_report.sh"}
    lines = ["", "FAILED  stage=%s  code=%s" % (stage or r["stage"], code)]
    if detail:
        lines.append("  detail: %s" % detail)
    lines.append("  cause : %s" % r["cause"])
    lines.append("  fix   : %s" % r["remedy"])
    lines.append("  docs  : docs/TROUBLESHOOTING.md#%s" % code.lower())
    return "\n".join(lines)


def markdown(rows=None):
    rows = rows or load()
    out = ["| Code | Stage | Cause | Fix |", "|---|---|---|---|"]
    for code, r in rows.items():
        esc = lambda s: s.replace("|", "\\|")  # noqa: E731
        out.append("| <a id=\"%s\"></a>`%s` | %s | %s | %s |" % (code.lower(), code, r["stage"], esc(r["cause"]),
                                                              esc(r["remedy"])))
    return "\n".join(out)


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd = argv[0]
    if cmd == "show":
        stage = detail = ""
        rest = argv[2:]
        while rest:
            if rest[0] == "--stage" and len(rest) > 1:
                stage, rest = rest[1], rest[2:]
            elif rest[0] == "--detail" and len(rest) > 1:
                detail, rest = rest[1], rest[2:]
            else:
                rest = rest[1:]
        print(failure_block(argv[1], stage, detail))
    elif cmd == "list":
        print("\n".join(load()))
    elif cmd == "markdown":
        print(markdown())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
