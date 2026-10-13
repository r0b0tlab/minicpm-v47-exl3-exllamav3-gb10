#!/usr/bin/env python3
"""Fail-closed model-shard page-cache retirement for GB10 unified memory.

On GB10 (128 GB unified LPDDR5X) torch's free-memory reading counts clean file
cache, and the ExLlamaV3 loader sizes its split from that reading. With a warm
shard cache the load fails with

    RuntimeError: Insufficient VRAM in split for model and cache

even though MemAvailable looks healthy. Calling posix_fadvise(POSIX_FADV_DONTNEED)
on ONLY the read-only shard files of the model about to be loaded fixes it (we
reproduced both the failure and the fix). This tool does exactly that, reads the
residency back with mincore(2), and refuses to continue if pages remain or if
physical MemFree is below the model + workspace budget.

It never drops host-wide caches and never writes to the model files.
"""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

MIB = 1048576
PAGE = os.sysconf("SC_PAGE_SIZE")
CACHE_LIMIT = 64 * MIB          # residency tolerated after DONTNEED
WORKSPACE_RESERVE_MIB = 4096    # default MemFree reserve on top of the model bytes

LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_longlong)
LIBC.mincore.restype = ctypes.c_int
LIBC.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
LIBC.munmap.restype = ctypes.c_int
LIBC.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def meminfo_mib(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024
    raise RuntimeError(key + " missing from /proc/meminfo")


def memfree_mib():
    return meminfo_mib("MemFree")


def resident_bytes(path):
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("empty shard: %s" % path)
    pages = (size + PAGE - 1) // PAGE
    bitmap = ctypes.create_string_buffer(pages)
    with path.open("rb", buffering=0) as handle:
        address = LIBC.mmap(None, size, 1, 1, handle.fileno(), 0)
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap", str(path))
        try:
            if LIBC.mincore(address, size, bitmap):
                raise OSError(ctypes.get_errno(), "mincore", str(path))
            count = sum(value & 1 for value in bitmap.raw[:pages])
        finally:
            if LIBC.munmap(address, size):
                raise OSError(ctypes.get_errno(), "munmap", str(path))
    return min(size, count * PAGE)


def retire(root, expected=None, pattern="*.safetensors"):
    root = Path(root).absolute()
    if not root.is_dir() or root != root.resolve(strict=True):
        raise ValueError("unsafe model root: %s" % root)
    files = sorted(root.glob(pattern))
    if not files:
        raise ValueError("no %s files in %s" % (pattern, root))
    if expected is not None and len(files) != expected:
        raise ValueError("expected %d shards in %s; got %d" % (expected, root, len(files)))
    for path in files:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("unsafe shard: %s" % path)
    before = sum(resident_bytes(p) for p in files)
    for path in files:
        with path.open("rb", buffering=0) as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = sum(resident_bytes(p) for p in files)
    return {"root": "<model-dir>", "files": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "before_bytes": before, "after_bytes": after}


def admit_budget(total_bytes, remaining_bytes, free_mib, workspace_mib=WORKSPACE_RESERVE_MIB):
    if remaining_bytes > CACHE_LIMIT:
        raise ValueError("cache residency remains %d MiB (> %d MiB)" % (remaining_bytes // MIB, CACHE_LIMIT // MIB))
    required = (total_bytes + MIB - 1) // MIB + workspace_mib
    if free_mib < required:
        raise ValueError("physical MemFree %d MiB is below the %d MiB model + workspace budget" % (free_mib, required))
    return {"required_memfree_mib": required, "remaining_cached_mib": remaining_bytes // MIB}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, type=Path, help="model directory (repeatable)")
    ap.add_argument("--out", type=Path, help="write a JSON receipt here")
    ap.add_argument("--workspace-mib", type=int, default=WORKSPACE_RESERVE_MIB)
    args = ap.parse_args(argv)
    before_free, before_avail = memfree_mib(), meminfo_mib("MemAvailable")
    targets = [retire(d) for d in args.dir]
    after_free, after_avail = memfree_mib(), meminfo_mib("MemAvailable")
    total = sum(t["total_bytes"] for t in targets)
    remaining = sum(t["after_bytes"] for t in targets)
    receipt = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": targets,
               "memfree_before_mib": before_free, "memfree_after_mib": after_free,
               "memavailable_before_mib": before_avail, "memavailable_after_mib": after_avail}
    try:
        receipt["budget"] = admit_budget(total, remaining, after_free, args.workspace_mib)
        receipt["status"] = "PASS"
    except ValueError as exc:
        receipt["status"] = "BLOCKED"
        receipt["reason"] = str(exc)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "PASS":
        print("CACHE_RETIRE_BLOCKED %s" % receipt["reason"], file=sys.stderr)
        return 1
    print("CACHE_RETIRE_OK files=%d cached_before_mib=%d cached_after_mib=%d memfree_mib=%d memavailable_mib=%d"
          % (sum(t["files"] for t in targets), sum(t["before_bytes"] for t in targets) // MIB,
             remaining // MIB, after_free, after_avail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
