#!/usr/bin/env python3
"""Build the long-context needle document (token ids, no text) used by the NIAH lane.

Method (same as the reference scorecard): one document of max(cuts)+1 tokens, a chat user turn, a preamble, then
natural prose paragraphs with six "needle" sentences spliced in at fixed absolute token positions, each stating a
seven-digit passcode for a made-up codeword. A question about a needle is the first `cut` tokens of the document
followed by the question and the assistant turn (thinking off); the cuts are 4,096 .. 261,000 tokens, the needles sit
at about 2k / 20k / 50k / 100k / 180k / 250k tokens, so a question exists for every (cut, needle) pair whose needle is
wholly inside the prefix (22 questions).

The prose source is public and deterministic enough to rebuild anywhere: the Python standard library's pydoc topics
and module docstrings, plus the docstrings of the torch package pinned in runtime.lock.json (interleaved paragraph by
paragraph, no paragraph repeated). The reference scorecard used other
prose (repository documents, pydoc topics, Ansible READMEs), so the two documents differ while positions, cuts,
question wording and scoring are the same. The generated document ships in this repository
(niah_261k.json.gz); rebuilding it needs only a tokenizer.json and a Python 3.12 standard library.

    .venv/bin/python eval/niah/make_niah.py --tokenizer MODEL_DIR --out eval/niah/niah_261k.json.gz \\
        --docstrings '.venv/lib/python3.*/site-packages/torch/**/*.py'  
"""
import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys

CODEWORDS = [
    "zephyrine", "quillon", "marrowfen", "ostrander", "pellucid", "thistlewick", "brackenmoor", "caldrith", "embervale",
    "fenwick-lark", "gossamer-pike", "harrowgate", "ivorymoss", "juniper-slate", "kestrelhaven", "lanternwood",
]
DEFAULT_CUTS = [4096, 16384, 32000, 64000, 128000, 200000, 261000]
DEFAULT_NEEDLES = [2000, 20000, 50000, 100000, 180000, 250000]
PREAMBLE = ("<|im_start|>user\nBelow is a long document. A few sentences in it state a special passcode for a codeword. "
            "Read the document carefully and answer the question that follows it.\n\n")
QUESTION = ("\n\nQuestion: What is the special passcode for the codeword {codeword}? "
            "Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def usable(block, min_chars=120):
    b = " ".join(block.split())
    if len(b) < min_chars or "|" in b or "```" in b or ">>>" in b or b.count("`") > 6 or re.search(r"\d{5,}", b):
        return None
    symbols = sum(1 for c in b if not (c.isalnum() or c.isspace() or c in ".,;:'\"()-_/"))
    if symbols > 0.08 * len(b):
        return None
    if sum(1 for w in b.split() if w.isalpha()) < 0.6 * len(b.split()):
        return None
    return b


def source_pydoc():
    import pydoc_data.topics as topics
    out = []
    for key in sorted(topics.topics):
        for block in re.split(r"\n\s*\n", topics.topics[key]):
            out.append(block)
    return out


def source_stdlib_docstrings():
    root = os.path.dirname(os.__file__)
    out = []
    skip = ("/test/", "/tests/", "/idlelib/", "/site-packages/", "/lib2to3/", "/turtledemo/", "/ensurepip/", "/__pycache__/")
    paths = [p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True) if not any(x in p for x in skip)]
    for path in sorted(paths):
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    out.extend(re.split(r"\n\s*\n", doc))
    return out


def source_docstrings_glob(pattern):
    out = []
    for path in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)):
        if "/tests/" in path or "/test/" in path:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    out.extend(re.split(r"\n\s*\n", doc))
    return out


def source_glob(pattern):
    out = []
    for path in sorted(glob.glob(os.path.expanduser(pattern))):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                out.extend(re.split(r"\n\s*\n", fh.read()))
        except OSError:
            continue
    return out


def paragraphs(sources):
    per, seen = [], set()
    for blocks in sources:
        keep = []
        for block in blocks:
            b = usable(block)
            if b and b not in seen:
                seen.add(b)
                keep.append(b)
        per.append(keep)
    merged, i = [], 0
    while any(i < len(s) for s in per):
        for s in per:
            if i < len(s):
                merged.append(s[i])
        i += 1
    return merged


def build(tok_enc, paras, cuts, needle_positions, seed):
    total = max(cuts) + 1
    rng = random.Random(seed)
    words = CODEWORDS[:]
    rng.shuffle(words)
    needles = []
    for k, pos in enumerate(needle_positions):
        value = "".join(str(rng.randrange(10)) for _ in range(7))
        if value[0] == "0":
            value = "7" + value[1:]
        needles.append({"pos_asked": pos, "codeword": words[k % len(words)], "value": value})
    ids = list(tok_enc(PREAMBLE))
    pending = sorted(range(len(needles)), key=lambda k: needles[k]["pos_asked"])
    p = 0
    while len(ids) < total:
        while pending and len(ids) >= needles[pending[0]]["pos_asked"]:
            k = pending.pop(0)
            sentence = "The special passcode for the codeword %s is %s.\n\n" % (needles[k]["codeword"], needles[k]["value"])
            needles[k]["pos"] = len(ids)
            ids.extend(tok_enc(sentence))
        if p >= len(paras):
            raise SystemExit("prose ran out after %d paragraphs (%d of %d tokens); add --source globs" % (p, len(ids), total))
        ids.extend(tok_enc(paras[p] + "\n\n"))
        p += 1
    ids = ids[:total]
    for n in needles:
        if "pos" not in n or n["pos"] + 40 > total:
            raise SystemExit("needle at %d did not fit in %d tokens" % (n["pos_asked"], total))
        n["value_ids"] = tok_enc(n["value"])
        assert len(n["value_ids"]) == 7, n
    questions = []
    for cut in cuts:
        for k, n in enumerate(needles):
            if n["pos"] + 40 >= cut:
                continue
            questions.append({"cut": cut, "needle": k, "depth": round(n["pos"] / cut, 4),
                              "suffix_ids": tok_enc(QUESTION.format(codeword=n["codeword"])),
                              "answer": n["value"]})
    return ids, needles, questions, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="model directory holding tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[], help="extra prose file glob (repeatable)")
    ap.add_argument("--docstrings", action="append", default=[],
                    help="recursive glob of .py files whose docstrings are prose sources (repeatable)")
    ap.add_argument("--no-stdlib", action="store_true")
    ap.add_argument("--cuts", default=",".join(map(str, DEFAULT_CUTS)))
    ap.add_argument("--needles", default=",".join(map(str, DEFAULT_NEEDLES)))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok_path = os.path.join(os.path.expanduser(a.tokenizer), "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    with open(tok_path, "rb") as fh:
        tok_sha = hashlib.sha256(fh.read()).hexdigest()

    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    sources = []
    if not a.no_stdlib:
        sources += [source_pydoc(), source_stdlib_docstrings()]
    sources += [source_docstrings_glob(g) for g in a.docstrings]
    sources += [source_glob(g) for g in a.source]
    paras = paragraphs(sources)
    cuts = [int(x) for x in a.cuts.split(",")]
    positions = [int(x) for x in a.needles.split(",")]
    ids, needles, questions, used = build(enc, paras, cuts, positions, a.seed)
    doc = {
        "schema": "minicpm-v47-exl3-gb10.niah.v1",
        "tokens": len(ids), "cuts": cuts, "seed": a.seed,
        "needles": needles, "questions": questions,
        "eos_ids": [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")],
        "prose": "Python %s standard library (pydoc topics and module docstrings)%s" % (
            ".".join(sys.version.split()[0].split(".")[:2]),
            " plus the docstrings of %d user-supplied glob(s)" % len(a.docstrings) if a.docstrings else ""),
        "paragraphs_used": used, "paragraphs_available": len(paras),
        "tokenizer_sha256": tok_sha,
        "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "ids": ids,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print("%d tokens, %d/%d paragraphs, needles at %s, %d questions -> %s (%d bytes)" % (
        len(ids), used, len(paras), [n["pos"] for n in needles], len(questions), a.out, os.path.getsize(a.out)))


if __name__ == "__main__":
    main()
