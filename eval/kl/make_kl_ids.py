#!/usr/bin/env python3
"""Build a token-id set for the KL measurement from text files you provide (rows of 1,024 tokens, no BOS).

The id sets behind the published fidelity numbers contain excerpts of non-public sources and are therefore not
shipped (their sha256 are in evidence/fidelity-*/kl.json). This tool makes an equivalent set from public material:

    python eval/kl/make_kl_ids.py --tokenizer MODEL_DIR --out ids --name mySet \\
        --kind python='/usr/lib/python3.12/*.py' --kind prose='docs/*.md' --rows-per-kind 4 --seed 4747

Each --kind NAME=GLOB contributes `--rows-per-kind` windows of exactly 1,024 tokens (seeded choice of file and offset,
encoded with the model's tokenizer.json, add_special_tokens=False). Output: <out>/<name>.safetensors (input_ids int32
[rows, 1024]) and <out>/<name>.manifest.json (per row: name, kind, source file sha256, offset; tokenizer sha256, seed,
ids sha256), the format fork_logits.py and kl_compare.py read. Deterministic for the same files and seed.
"""
import argparse
import glob
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="model directory with tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ids")
    ap.add_argument("--kind", action="append", required=True, help="NAME=GLOB (repeatable)")
    ap.add_argument("--rows-per-kind", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4747)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok_path = os.path.join(os.path.expanduser(a.tokenizer), "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    rng = random.Random(a.seed)
    rows, samples = [], []
    for spec in a.kind:
        kind, _, pattern = spec.partition("=")
        files = sorted(glob.glob(os.path.expanduser(pattern)))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            sys.exit("no files for kind %s: %s" % (kind, pattern))
        got = 0
        attempts = 0
        while got < a.rows_per_kind and attempts < 200 * a.rows_per_kind:
            attempts += 1
            f = rng.choice(files)
            data = open(f, encoding="utf-8", errors="ignore").read()
            ids = tok.encode(data, add_special_tokens=False).ids
            if len(ids) < common.T:
                continue
            off = rng.randrange(0, len(ids) - common.T + 1)
            rows.append(ids[off: off + common.T])
            samples.append({"row": len(rows) - 1, "name": "%s_%d" % (kind, got), "kind": kind,
                            "source_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(), "offset": off})
            got += 1
        if got < a.rows_per_kind:
            sys.exit("kind %s: only %d usable windows (files too short)" % (kind, got))
    with open(tok_path, "rb") as fh:
        tok_sha = hashlib.sha256(fh.read()).hexdigest()
    manifest = {"tokenizer_sha256": tok_sha, "seed": a.seed, "samples": samples,
                "tokenization": "tokenizers.Tokenizer.encode(text, add_special_tokens=False), no BOS",
                "description": "rows of %d tokens from user-provided files" % common.T}
    p_ids, p_man = common.save_ids_set(a.out, a.name, np.array(rows, dtype=np.int32), manifest)
    print("%d rows -> %s" % (len(rows), p_ids))


if __name__ == "__main__":
    main()
