# Memory on GB10 (unified memory)

A GB10 has 128 GB of LPDDR5X shared by the CPU and the GPU. There is no separate VRAM to fit things into: the
weights, the KV cache, the recurrent state, the CUDA workspaces, the OS page cache and every other process come out
of one pool. This page derives the budget the preflight checks and explains why the pack's page cache has to be
retired before loading.

## Budget (per profile; `scripts/cfg.py need <profile>`)

| item | how it is computed | 262k | 32k | 16k low-mem | 262k + DSpark draft |
|---|---|---:|---:|---:|---:|
| weights (pack shards on disk) | sum of shard sizes in `config/model.yaml` | 26.35 GiB | 26.35 | 26.35 | 26.35 |
| KV cache, fp16 | 20,480 B/token x window (10 full-attention layers x 2 x 2 KV heads x 256 x 2 B) | 5.00 GiB | 0.62 | 0.31 | 5.00 |
| recurrent state (gated delta net) | 64,389,120 B per sequence slot, independent of context | 0.06 GiB | 0.06 | 0.06 | 0.06 |
| host recurrent-state checkpoint store | profile `recurrent_cache_gib` (prefix reuse) | 1.00 GiB | 1.00 | 0.25 | 1.00 |
| draft weights | the draft pack on disk (451 MB, 6.0 bpw) | - | - | - | 0.42 GiB |
| draft K/V cache | 10,240 B/token x window (`draft_kv_bytes_per_token`) | - | - | - | 2.50 GiB |
| recurrent snapshots for rewind | 16 x 64,389,120 B (one per drafted position, `max_history`) | - | - | - | 0.96 GiB |
| draft workspace | profile `draft.workspace_gib` | - | - | - | 0.50 GiB |
| workspace | profile `workspace_gib` (activations, MoE staging, CUDA context, allocator slack) | 4.00 GiB | 3.00 | 2.00 | 4.00 |
| **resident total** | sum | **36.41 GiB** | 31.04 | 28.98 | **40.79 GiB** |
| headroom that must stay free | profile `headroom_gib` | 8.00 GiB | 8.00 | 6.00 | 8.00 |
| **`MemAvailable` required** | | **44.41 GiB** | 39.04 | 34.98 | **48.79 GiB** |

The vision tower, window merger and projector (1.1 GB, bf16) are in the shards but are not loaded by ExLlamaV3, so
the loaded weights are about 1 GiB below the on-disk figure; the arithmetic keeps the larger number on purpose.
Measured values from the evaluation runs are in [RESULTS.md](RESULTS.md) (memory table).

KV cache with a quantised cache (`kv_bits` 2 to 8) is `bits x 4 + 2` bytes per 32 elements instead of 64:
about 1.4 GiB at 4 bit and 2.7 GiB at 8 bit for the 262,144-token window. The shipped profiles use the fp16 cache
because the validated measurements were taken with it.

## Why the page cache of the shards must be retired

Loading reads 26 GiB of shards through the page cache. Linux counts clean page cache in `MemAvailable`, but the
CUDA free-memory reading that ExLlamaV3's loader uses on this platform does not treat it as free. When the cache is
already large (a box that has been serving other models, or a second start of the same server) the loader
computes its split from a free figure that is smaller than weights plus KV cache and stops with
`RuntimeError: Insufficient VRAM in split for model and cache`, while `free` shows tens of GiB available.

The `retire-cache` stage (`scripts/retire_page_cache.py`) fixes exactly this and nothing else:

1. it opens only the pack's `*.safetensors` files, read-only, and refuses symlinks and empty files;
2. it asks `mincore(2)` how many of their pages are resident, calls
   `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on each, and reads the residency back;
3. it fails closed if more than 64 MiB stays resident, or if physical `MemFree` is below the model bytes plus the
   profile's non-weight budget;
4. it writes a receipt (`.state/retire-cache.json`) with before/after residency and `MemFree` / `MemAvailable`.

It never drops host-wide caches and never touches model bytes. It runs immediately before every server start.
`--dir` is repeatable (and `RETIRE_DIRS="dirA dirB" ./run.sh` passes more): if other model directories are what
fills the cache on your box, name them too.

What was and was not observed. The failure and the fix were reproduced on our development hosts before this
repository existed (a warm shard cache, `Insufficient VRAM in split for model and cache` at about 59 GiB
`MemAvailable`, gone after retiring only those shard pages). In the clean-room run recorded in
[RESULTS.md](RESULTS.md) the stage released 25.9 GiB of shard cache before the load and the load succeeded; we did
not push this box into the failing state on purpose (exhausting unified memory on a GB10 can hang the machine).

## Other users of the pool

- Do not run other GPU jobs next to the server; the preflight refuses to start with another compute process on the
  GPU (`--allow-busy` overrides, at your risk).
- `GUARD=1 ./run.sh` starts `scripts/guard.py`, an optional bounded-lease watchdog that stops the server if
  `MemAvailable` stays below a floor (default 6 GiB for 3 consecutive 2 s samples) or cannot be read. It is off by
  default and only ever signals the server's own PID.

## Measured (profile `gb10-text-262k`, evaluation runs)

| point | value |
|---|---|
| shard page cache before `retire-cache` | 25.3 GiB (25,931 MiB) resident, 0 MiB after |
| `MemAvailable` after retire, before load | 113.2 GiB |
| load of weights + 262,144-token fp16 KV cache | 13 to 16 s |
| torch allocated after load | 29.8 GiB (reserved 29.8 GiB) |
| `MemAvailable` after load | 79.8 GiB: **33.4 GiB resident** (the budget above estimates 36.4 GiB) |
| CUDA free-memory reading after load | "113.6 of 121.7 GiB in use": 33.8 GiB more than `MemAvailable` implies, the page cache it counts as used |
| after the depth grid up to a 261,000-token prefill | allocated 29.9 GiB, reserved max 30.2 GiB, `MemAvailable` 78.2 GiB |

Nothing grows while serving: the KV cache is allocated for the whole window at load, so a 261,000-token request
moves `MemAvailable` by about 1 GiB.
