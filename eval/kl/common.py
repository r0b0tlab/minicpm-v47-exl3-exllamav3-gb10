"""Shared helpers of the MiniCPM-V-4.7-35B-A3B text KL tooling (`eval/kl/`).

Everything here is dependency-light (numpy + the standard library; torch / safetensors are imported lazily) so the CPU unit tests and
`make_kl_ids.py` run anywhere the venv python does. Scripts run under `python -I`, which puts neither the working directory nor the
script directory on `sys.path`: each script inserts its own directory first, then `import common`.

Contents
    constants        V (vocabulary 248144, the checkpoint's `vocab_size`), T (1,024 tokens per row), HIDDEN (2,048)
    ids sets         save_ids_set / load_ids_set: one `<name>.safetensors` (`input_ids` int32 [rows, 1024]) + `<name>.manifest.json`
    logit sets       the directory format shared by bf16_ref_logits.py, fork_logits.py and kl_compare.py (see LogitSet)
    misc             sha256, atomic json, row-spec parsing, safetensors header reading, MemAvailable
"""
import hashlib
import json
import os
import struct

import numpy as np

V = 248144
T = 1024
HIDDEN = 2048
TOPK = 256
SCHEMA_IDS = "minicpm-v47-kl-ids-v1"
SCHEMA_SET = "minicpm-v47-kl-set-v1"

MODEL_DIR = os.environ.get("MINICPM_BF16_DIR", "")   # the bf16 source checkpoint (not shipped; 70 GB)
FORK_DIR = os.environ.get("EXLLAMAV3_DIR", "")      # an ExLlamaV3 checkout (optional if installed in the venv)
FORK_PY = os.environ.get("EXLLAMAV3_PYTHON", "python3")


# ---------------------------------------------------------------------------------------------------------------- small helpers


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj, indent=1):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def parse_rows(spec, n):
    """'0-13', '0,3,5-7' or 'all' -> sorted list of row indices below n."""
    if spec in (None, "", "all"):
        return list(range(n))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    bad = [r for r in out if r < 0 or r >= n]
    if bad:
        raise ValueError(f"row(s) {bad} outside 0..{n - 1}")
    return out


def mem_available_gb():
    """MemAvailable from /proc/meminfo in GB (GB10: unified memory, page cache counts as used)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("inf")


def safetensors_header(path):
    """{name: (dtype string, shape, abs start, abs end)} of one safetensors file, from the header only."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if 8 + n > size:
            raise ValueError(f"{path}: truncated header")
        raw = json.loads(f.read(n))
    out = {}
    for k, v in raw.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = (v["dtype"], v["shape"], 8 + n + a, 8 + n + b)
        if 8 + n + b > size:
            raise ValueError(f"{path}: truncated data for {k}")
    return out


# ---------------------------------------------------------------------------------------------------------------- ids sets


def ids_sha256(ids):
    """sha256 of the little-endian int32 bytes of an [rows, tokens] ids array (the identity of a set)."""
    return sha256_bytes(np.ascontiguousarray(np.asarray(ids, dtype="<i4")).tobytes())


def save_ids_set(out_dir, name, ids, manifest, tokens=T):
    """Write `<out_dir>/<name>.safetensors` (`input_ids` int32 [rows, T]; placeholder rows are all -1) and `<name>.manifest.json`.
    `manifest` is the caller's dict (samples, sources, ...): schema, shape and sha256 are added here. Returns (ids_path, manifest_path).
    `tokens` is 1,024 for every real set; the tests use short rows."""
    from safetensors.numpy import save_file

    ids = np.ascontiguousarray(np.asarray(ids, dtype="<i4"))
    assert ids.ndim == 2 and ids.shape[1] == tokens, ids.shape
    valid = np.array([bool((row >= 0).all()) for row in ids])
    if valid.any():
        assert int(ids[valid].max()) < V, f"token id {int(ids[valid].max())} >= vocab {V}"
    os.makedirs(out_dir, exist_ok=True)
    p_ids = os.path.join(out_dir, f"{name}.safetensors")
    tmp = p_ids + ".tmp"
    # one metadata key only: safetensors serialises the metadata map in hash order, which differs between calls with two or more keys
    save_file({"input_ids": ids}, tmp, metadata={"schema": SCHEMA_IDS})
    os.replace(tmp, p_ids)
    m = dict(manifest)
    m.update({
        "schema": SCHEMA_IDS,
        "name": name,
        "file": os.path.basename(p_ids),
        "rows": int(ids.shape[0]),
        "tokens": int(ids.shape[1]),
        "vocab": V,
        "dtype": "int32",
        "pending_rows": [int(i) for i in np.nonzero(~valid)[0]],
        "complete": bool(valid.all()),
        "ids_sha256": ids_sha256(ids),
        "file_sha256": sha256_file(p_ids),
    })
    p_man = os.path.join(out_dir, f"{name}.manifest.json")
    atomic_json(p_man, m)
    return p_ids, p_man


def load_ids_set(path, require_complete=True):
    """Load an ids set written by save_ids_set. `path` is the `.safetensors` file, its `.manifest.json`, or a directory holding exactly
    one `*.safetensors`. Returns (ids int64 [rows, T], manifest dict). Placeholder rows (-1) raise unless require_complete=False."""
    from safetensors.numpy import load_file

    path = os.path.expanduser(path)
    if os.path.isdir(path):
        cand = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if len(cand) != 1:
            raise ValueError(f"{path}: expected one .safetensors ids file, found {cand}")
        path = os.path.join(path, cand[0])
    if path.endswith(".manifest.json"):
        path = path[: -len(".manifest.json")] + ".safetensors"
    ids = load_file(path)["input_ids"]
    man_path = path[: -len(".safetensors")] + ".manifest.json"
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    if manifest.get("ids_sha256") and manifest["ids_sha256"] != ids_sha256(ids):
        raise ValueError(f"{path}: ids differ from the manifest's sha256 (file edited after make_kl_ids.py?)")
    pending = [int(i) for i in np.nonzero((ids < 0).any(axis=1))[0]]
    if pending and require_complete:
        raise ValueError(f"{path}: {len(pending)} placeholder row(s) {pending[:8]}{'...' if len(pending) > 8 else ''}; "
                         "re-run make_kl_ids.py with --trace to fill the self-sampled rows")
    return ids.astype(np.int64), manifest


# ---------------------------------------------------------------------------------------------------------------- logit sets


class LogitSet:
    """A directory of per-position outputs for one ids set. Written by bf16_ref_logits.py (the reference), fork_logits.py (the floor
    and the candidates) and read by kl_compare.py. All files are raw little-endian arrays, `rows` x `tokens` leading dimensions:

        set.json          schema, name, kind (ref|floor|cand), rows, tokens, vocab, ids_sha256, producer, which files exist
        logits.f16        [rows, T, V] fp16, ROW-CENTERED: stored = raw_logit - rowmax (raw = stored + rowmax). Optional.
        rowmax.f32        [rows, T] fp32, the max raw logit of each position (shift of logits.f16)
        lse.f32           [rows, T] fp32, logsumexp of the raw logits, from the fp32 logits before the fp16 cast
        tail_lse.f32      [rows, T] fp32, logsumexp of the raw logits OUTSIDE the stored top-K (exact tail mass for lumped KL)
        entropy.f32       [rows, T] fp32, entropy of softmax(raw) in nats
        argmax.i32        [rows, T] int32
        tgt_logit.f32     [rows, T] fp32, raw logit of the NEXT token of the row (NaN at the last position)
        topk_val.f32      [rows, T, K] fp32 raw logits of the K=256 largest entries, descending
        topk_idx.i32      [rows, T, K] int32 their ids
        final_hidden.bf16 [rows, T, 2048] bf16 (as int16 bits): the post-`norm` hidden state (reference only): the exact full-vocabulary
                          reference logits are recomputable from it with the bf16 lm_head (kl_compare.py --ref-lm-head)
        metrics.npz       per-position metrics of a candidate against the reference (fork_logits.py --compare-ref): no logits stored

    Row r, position p of every file refers to the same token: ids[r, p] is the input, the logits predict ids[r, p + 1]."""

    FILES = {
        "logits": ("logits.f16", "<f2", (V,)),
        "rowmax": ("rowmax.f32", "<f4", ()),
        "lse": ("lse.f32", "<f4", ()),
        "tail_lse": ("tail_lse.f32", "<f4", ()),
        "entropy": ("entropy.f32", "<f4", ()),
        "argmax": ("argmax.i32", "<i4", ()),
        "tgt_logit": ("tgt_logit.f32", "<f4", ()),
        "topk_val": ("topk_val.f32", "<f4", (TOPK,)),
        "topk_idx": ("topk_idx.i32", "<i4", (TOPK,)),
        "final_hidden": ("final_hidden.bf16", "<i2", (HIDDEN,)),
    }

    def __init__(self, path, create=False, rows=None, tokens=T, meta=None, name=None, kind="cand", want=()):
        self.path = os.path.expanduser(path)
        self.meta_path = os.path.join(self.path, "set.json")
        if create:
            os.makedirs(self.path, exist_ok=True)
            if os.path.exists(self.meta_path):
                self.meta = json.load(open(self.meta_path))
                if self.meta.get("rows") != rows or self.meta.get("ids_sha256") != (meta or {}).get("ids_sha256"):
                    raise ValueError(f"{self.path}: existing set was made for different ids/rows; use another --out-dir or --restart")
            else:
                self.meta = {"schema": SCHEMA_SET, "name": name or os.path.basename(self.path.rstrip("/")), "kind": kind, "rows": int(rows),
                             "tokens": int(tokens), "vocab": V, "files": [], **(meta or {})}
            self.meta["files"] = sorted(set(self.meta.get("files", [])) | set(want))
            atomic_json(self.meta_path, self.meta)
        else:
            self.meta = json.load(open(self.meta_path))
            if self.meta.get("schema") != SCHEMA_SET:
                raise ValueError(f"{self.meta_path}: schema {self.meta.get('schema')}")
        self.rows = self.meta["rows"]
        self.tokens = self.meta["tokens"]
        self._maps = {}

    def has(self, what):
        return os.path.exists(os.path.join(self.path, self.FILES[what][0])) and what in self.meta.get("files", [])

    def array(self, what, mode="r"):
        """numpy memmap of one file, shape [rows, tokens, *tail]. mode 'w+' creates it (preallocated, sparse)."""
        fname, dt, tail = self.FILES[what]
        shape = (self.rows, self.tokens) + tail
        key = (what, mode)
        if key not in self._maps:
            p = os.path.join(self.path, fname)
            if mode == "r":
                self._maps[key] = np.memmap(p, dtype=dt, mode="r", shape=shape)
            else:
                self._maps[key] = np.memmap(p, dtype=dt, mode="r+" if os.path.exists(p) else "w+", shape=shape)
        return self._maps[key]

    def flush(self):
        for m in self._maps.values():
            if hasattr(m, "flush"):
                m.flush()

    def save_meta(self, **update):
        self.meta.update(update)
        atomic_json(self.meta_path, self.meta)
