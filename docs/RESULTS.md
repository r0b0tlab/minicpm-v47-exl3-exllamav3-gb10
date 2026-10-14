# Results

Measured record of this repository's runtime (ExLlamaV3, plain decode and with DSpark drafting) and of the reference engine, with method and
limits. Tables between GENERATED markers are rendered from `evidence/*/bundle.json` by
`scripts/render_results.py`; everything else is prose around them.

## Setup

- Hardware: one NVIDIA GB10 (DGX Spark, aarch64, SM121), 128 GB unified memory, driver 580.173.02, CUDA toolkit 13.0.88.
- Software: the plain-decode column was measured on ExLlamaV3 commit `4d083d25c1a3` (branch `minicpm-v4.7`); the
  drafted column on commit `5ddae7d12448` (branch `minicpm-v4.7-dspark`, which adds the drafter and a validation in
  the generator and leaves the plain decode path unchanged; the plain comparison is the 2026-10-13 plain run, not a
  same-session control). torch 2.14.0+cu130, Python 3.12.
- Pack: `r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw` @ `8f9f07872b09`, profile `gb10-text-262k` (262,144-token
  window, fp16 KV cache, prefill chunk 2,048, batch size 1, one request at a time, thinking on by default).
- Exclusive GPU for every lane; the server was started by `./run.sh` and stopped after each lane.

## Summary table

<!-- BEGIN GENERATED:results-summary -->
| metric | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| **Q200v2 text-180**, accuracy | 167/180 (92.8%) | 169/180 (93.9%) | 168/180 (93.3%) |
| &nbsp;&nbsp;GSM8K-style math (80) | 74/80 (92.5%) | 75/80 (93.8%) | 76/80 (95.0%) |
| &nbsp;&nbsp;HumanEval-style code (40) | 39/40 (97.5%) | 39/40 (97.5%) | 38/40 (95.0%) |
| &nbsp;&nbsp;IFEval-style instruction following (40) | 35/40 (87.5%) | 35/40 (87.5%) | 35/40 (87.5%) |
| &nbsp;&nbsp;Hard reasoning (20, AI-graded, see notes) | 19/20 (95.0%) | 20/20 (100.0%) | 19/20 (95.0%) |
| Q200v2 end-to-end tok/s, 1 request in flight (reference column: with drafting) | 58.7 | 109.8 | 135.0 |
| Q200v2 end-to-end tok/s, 4 requests in flight (ExLlamaV3: experimental batch 4; reference: with drafting) | 94.0 | not run (batch 1) | 183.8 |
| Decode tok/s at ~4k context | 47.0 | 70.7 | 66.8 |
| Decode tok/s at ~32k context | 48.2 | 48.7 | 54.0 |
| Decode tok/s at ~128k context | 34.7 | 24.1 | 32.8 |
| Decode tok/s at ~261k context | 24.7 | 15.9 | 21.5 |
| Prefill tok/s at ~32k | 2,889 | 2,839 | 4,211 |
| Cold time to first token at ~32k (s) | 11.3 | 11.5 | 7.6 |
| Cold time to first token at ~261k (s) | 226.8 | 227.1 | 137.5 |
| Needle retrieval (22 questions, token-exact) | 19/22 exact | 19/22 exact | 20/22 exact |
| Leg D decode tok/s, GSM8K-style, plain | 60.1 | - | 67.1 |
| Leg D decode tok/s, code, plain | 59.8 | - | 68.2 |
| Leg D decode tok/s, chat, plain | 59.4 | - | 68.4 |
| Leg D decode tok/s, GSM8K-style, drafted, adaptive verification | n/a | 177.8 | 227.3 |
| Leg D decode tok/s, code, drafted, adaptive verification | n/a | 220.5 | 280.4 |
| Leg D decode tok/s, chat, drafted, adaptive verification | n/a | 69.0 | 84.7 |
| Leg D decode tok/s, GSM8K-style, drafted, always 15 drafts | n/a | 162.1 | - |
| Leg D decode tok/s, code, drafted, always 15 drafts | n/a | 207.3 | - |
| Leg D decode tok/s, chat, drafted, always 15 drafts | n/a | 41.0 | - |
| Leg D greedy streams identical to plain decoding | n/a | 59/80 | 80/80 (bit-identical by construction and checked) |
| GPU-side resident memory | 29.9 GiB allocated | 33.8 GiB allocated | 34.2 GiB |
<!-- END GENERATED:results-summary -->

## DSpark drafting

<!-- BEGIN GENERATED:draft-table -->
Drafting settings of the measured runs: mode adaptive, 15 drafts per round, Markov walk over the top 64 candidates, draft pack r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw.

Measured cost model (ms; one verify pass over n rows, 300 committed tokens; draft side of a round 3.6 ms):

| rows | ms | rows | ms |
|---:|---:|---:|---:|
| 1 | 16.2 | 9 | 36.7 |
| 2 | 20.4 | 10 | 39.1 |
| 3 | 23.2 | 11 | 41.4 |
| 4 | 24.5 | 12 | 42.9 |
| 5 | 26.7 | 13 | 44.9 |
| 6 | 28.9 | 14 | 45.3 |
| 7 | 30.7 | 15 | 46.0 |
| 8 | 33.2 | 16 | 46.9 |
<!-- END GENERATED:draft-table -->

## Q200v2 text-180

<!-- BEGIN GENERATED:q200-table -->
| Q200v2 text-180 | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| GSM8K-style math (80) | 74/80 (92.5%) | 75/80 (93.8%) | 76/80 (95.0%) |
| HumanEval-style code (40) | 39/40 (97.5%) | 39/40 (97.5%) | 38/40 (95.0%) |
| IFEval-style instruction following (40) | 35/40 (87.5%) | 35/40 (87.5%) | 35/40 (87.5%) |
| Hard reasoning (20, AI-graded, see notes) | 19/20 (95.0%) | 20/20 (100.0%) | 19/20 (95.0%) |
| **all 180** | 167/180 (92.8%) | 169/180 (93.9%) | 168/180 (93.3%) |
| completion tokens | 157,491 | 155,181 | 161,694 |
| wall seconds, 1 request in flight | 2,683 | 1,413 | 1,198 |
| end-to-end tok/s, 1 request in flight | 58.7 | 109.8 | 135.0 |
| median per-request tok/s | 59.6 | 124.8 | 153.2 |
| e2e tok/s, gsm8k | 59.7 | 126.8 | 156.4 |
| e2e tok/s, humaneval | 59.2 | 111.3 | 141.6 |
| e2e tok/s, ifeval | 59.5 | 93.2 | 115.5 |
| e2e tok/s, hard_reasoning | 59.7 | 130.0 | 154.0 |
| finish reasons | length 1, stop 179 | length 1, stop 179 | length 1, stop 179 |
| 4 requests in flight: pass count | 168/180 (93.3%) | not run | 168/180 (93.3%) |
| 4 requests in flight: families (gsm8k / humaneval / ifeval / hard) | 75 / 39 / 34 / 20 | - | 76 / 38 / 35 / 19 |
| 4 requests in flight: wall seconds | 1,681 | not run | 880 |
| 4 requests in flight: end-to-end tok/s | 94.0 | not run | 183.8 |
<!-- END GENERATED:q200-table -->

### Row-level agreement with the reference engine

Same kit, same prompts, greedy decoding on both engines. "Byte-identical answer" counts rows whose final answer text
is identical (sha256 of `message.content`); different engines run different fp16/bf16 kernels, so long
chain-of-thought generations drift apart even when both are correct.

<!-- BEGIN GENERATED:agreement-table -->
| family | rows | both pass | only ExLlamaV3 plain | only reference engine | neither | byte-identical answer |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 80 | 74 | 0 | 2 | 4 | 38 |
| humaneval | 40 | 38 | 1 | 0 | 1 | 24 |
| ifeval | 40 | 34 | 1 | 1 | 4 | 2 |
| hard_reasoning | 20 | 18 | 1 | 1 | 0 | 2 |
| all | 180 | 164 | 3 | 4 | 9 | 66 |
<!-- END GENERATED:agreement-table -->

## Decode, prefill and time to first token by context depth

<!-- BEGIN GENERATED:bench-table -->
| context depth: measure | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| 1,024 tokens: decode tok/s | 59.5 | 65.6 | - |
| 1,024 tokens: prefill tok/s | 1,096 | 1,115 | - |
| 1,024 tokens: cold TTFT (s) | 0.93 | 0.92 | - |
| 1,024 tokens: prefix-cache-hit TTFT (s) | 0.16 | 0.15 | - |
| 4,096 tokens: decode tok/s | 47.0 | 70.7 | 66.8 |
| 4,096 tokens: prefill tok/s | 2,077 | 2,035 | 4,551 |
| 4,096 tokens: cold TTFT (s) | 1.97 | 2.01 | 0.90 |
| 4,096 tokens: prefix-cache-hit TTFT (s) | 0.16 | 0.16 | 0.07 |
| 32,768 tokens: decode tok/s | 48.2 | 48.7 | 54.0 |
| 32,768 tokens: prefill tok/s | 2,889 | 2,839 | 4,211 |
| 32,768 tokens: cold TTFT (s) | 11.34 | 11.54 | 7.60 |
| 32,768 tokens: prefix-cache-hit TTFT (s) | 0.19 | 0.19 | 0.10 |
| 131,072 tokens: decode tok/s | 34.7 | 24.1 | 32.8 |
| 131,072 tokens: prefill tok/s | 1,764 | 1,764 | 2,777 |
| 131,072 tokens: cold TTFT (s) | 74.29 | 74.29 | 46.20 |
| 131,072 tokens: prefix-cache-hit TTFT (s) | 0.31 | 0.31 | 0.19 |
| 261,000 tokens: decode tok/s | 24.7 | 15.9 | 21.5 |
| 261,000 tokens: prefill tok/s | 1,151 | 1,149 | 1,901 |
| 261,000 tokens: cold TTFT (s) | 226.79 | 227.11 | 137.50 |
| 261,000 tokens: prefix-cache-hit TTFT (s) | 0.35 | 0.35 | 0.31 |
<!-- END GENERATED:bench-table -->

### Decode speed at a depth depends on how long the server has been up

The grid above is medians of three cold repeats per depth on a freshly started server, each depth visited once in
ascending order. Extra runs show that the first requests a new server sees at a given prompt length decode and
prefill more slowly than the same requests later, and that decode tokens per second at a fixed prompt depth varies by
up to about 20 % between depths in the 2k to 32k range (for example 46 tok/s at 2,048 tokens against 59 at 3,072).
A 6,000-token generation from a short prompt averages 58.9 tok/s and the 180 Q200v2 rows average 59.5. We did not
find the cause (kernel selection or lazily initialised kernels for new shapes are the suspects); the series below
makes the warm-up visible. Read the 2k to 32k decode figures as a band of about 46 to 60 tok/s, not as a curve; the
warm steady state at 4,096 tokens is 55 to 57 tok/s.

<!-- BEGIN GENERATED:warmup-table -->
One freshly started server, requests in this order: 5 cold prefills at 4,096 tokens, 3 at 1,024, 5 at 4,096 again (128 new tokens each).

| series | decode tok/s per request | median | prefill tok/s per request |
|---|---|---:|---|
| 4,096 tokens, first visit after server start | 47.6 / 46.3 / 45.9 / 47.2 / 56.0 | 47.2 | 2,061 / 1,893 / 2,003 / 2,096 / 2,113 |
| 1,024 tokens | 59.7 / 59.7 / 59.6 | 59.7 | 1,678 / 1,606 / 1,577 |
| 4,096 tokens, after the other requests | 55.8 / 55.9 / 54.2 / 57.0 / 56.7 | 55.9 | 2,834 / 2,910 / 3,096 / 2,975 / 3,071 |
<!-- END GENERATED:warmup-table -->

The depth scan (two cold repeats per depth, same server, after the grid's server had been replaced by a new one):

<!-- BEGIN GENERATED:depthscan-table -->
| prompt tokens (cold prefill, 128 new tokens) | decode tok/s of each run | mean |
|---:|---|---:|
| 512 | 59.5 / 59.4 | 59.5 |
| 1,024 | 59.4 / 58.9 | 59.2 |
| 1,536 | 59.2 / 59.0 | 59.1 |
| 2,048 | 46.5 / 45.5 | 46.0 |
| 3,072 | 59.0 / 58.8 | 58.9 |
| 4,096 | 55.9 / 53.5 | 54.7 |
| 6,144 | 54.7 / 57.4 | 56.0 |
| 8,192 | 56.3 / 56.1 | 56.2 |
| 16,384 | 53.8 / 54.6 | 54.2 |

Long generation from a 64-token prompt (context grows while decoding): 2,048 new tokens at 59.3 tok/s on average; 6,000 new tokens at 58.9 tok/s on average.
<!-- END GENERATED:depthscan-table -->

## Needle retrieval

<!-- BEGIN GENERATED:niah-table -->
| needle-in-a-haystack, 261,001-token document | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| questions asked | 22 | 22 | 22 |
| exact | 19 | 19 | 20 |
| misses | needle 3@200000 -> other_needle; needle 3@261000 -> other_needle; needle 4@261000 -> other_needle | needle 3@200000 -> other_needle; needle 3@261000 -> other_needle; needle 4@261000 -> other_needle | cut@200000 -> other_needle; cut@261000 -> other_needle |
<!-- END GENERATED:niah-table -->

## Leg D: plain vs drafted decode

<!-- BEGIN GENERATED:legd-table -->
| Leg D decode tok/s, one stream, thinking off, up to 320 tokens | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| greedy, gsm8k, plain | 60.1 | - | 67.1 |
| greedy, gsm8k, drafted, adaptive verification | n/a | 177.8 | 227.3 |
| greedy, gsm8k, drafted, always 15 drafts | n/a | 162.1 | - |
| greedy, code, plain | 59.8 | - | 68.2 |
| greedy, code, drafted, adaptive verification | n/a | 220.5 | 280.4 |
| greedy, code, drafted, always 15 drafts | n/a | 207.3 | - |
| greedy, chat, plain | 59.4 | - | 68.4 |
| greedy, chat, drafted, adaptive verification | n/a | 69.0 | 84.7 |
| greedy, chat, drafted, always 15 drafts | n/a | 41.0 | - |
| greedy, all, plain | 59.9 | - | 67.7 |
| greedy, all, drafted, adaptive verification | n/a | 161.3 | 204.9 |
| greedy, all, drafted, always 15 drafts | n/a | 143.1 | - |
| sampled, gsm8k, plain | 60.0 | - | 68.3 |
| sampled, gsm8k, drafted, adaptive verification | n/a | 165.4 | 198.2 |
| sampled, gsm8k, drafted, always 15 drafts | n/a | 144.1 | - |
| sampled, code, plain | 59.7 | - | 68.3 |
| sampled, code, drafted, adaptive verification | n/a | 225.8 | 271.2 |
| sampled, code, drafted, always 15 drafts | n/a | 213.5 | - |
| sampled, chat, plain | 59.4 | - | 68.4 |
| sampled, chat, drafted, adaptive verification | n/a | 70.9 | 80.7 |
| sampled, chat, drafted, always 15 drafts | n/a | 42.8 | - |
| sampled, all, plain | 59.7 | - | 68.3 |
| sampled, all, drafted, adaptive verification | n/a | 154.0 | 183.4 |
| sampled, all, drafted, always 15 drafts | n/a | 133.5 | - |

ExLlamaV3 + DSpark, greedy arm. The speed-up is against the plain run of 2026-10-13 (a different session; no same-session control was run), and '-' means that run is not in the bundle:

| verification | class | tok/s | speed-up | drafts accepted | streams identical to plain |
|---|---|---:|---:|---:|---:|
| adaptive verification | gsm8k | 177.8 | - | 73.3% | 33/40 |
| adaptive verification | code | 220.5 | - | 82.0% | 20/20 |
| adaptive verification | chat | 69.0 | - | 49.4% | 6/20 |
| adaptive verification | all | 161.3 | - | 70.3% | 59/80 |
| always 15 drafts | gsm8k | 162.1 | - | 48.8% | 33/40 |
| always 15 drafts | code | 207.3 | - | 66.1% | 19/20 |
| always 15 drafts | chat | 41.0 | - | 7.4% | 8/20 |
| always 15 drafts | all | 143.1 | - | 28.4% | 60/80 |
<!-- END GENERATED:legd-table -->

## Prefill chunk size

The profile's chunk size was chosen by measurement before the grid (fresh server per setting, two cold repeats):

<!-- BEGIN GENERATED:chunk-table -->
| prefill chunk (tokens) | prefill tok/s at 32,768 tokens | cold TTFT (s) | decode tok/s | repeats |
|---|---:|---:|---:|---:|
| 2048 | 2,659 | 12.32 | 41.3 | 2 |
| 4096 | 2,778 | 11.80 | 42.0 | 2 |

Rule: 4096 is adopted only if its median prefill tokens/s at 32,768 tokens is more than 5 % above 2048. Chosen: 2048.
<!-- END GENERATED:chunk-table -->

## Click-run, clean-room timings

A fresh clone in an empty directory, a new venv, a cold ExLlamaV3 build from the public branch and a cold cache
directory. The target pack was hard-linked from a local copy (28 GB are not downloaded twice for a test); the
download path itself was exercised on the draft repository (451 MB) and, for the target, on the small files that were
missing, then every shard was verified against the pinned manifest.

<!-- BEGIN GENERATED:clickrun-table -->
| stage | seconds | note |
|---|---:|---|
| preflight | 1 | 11 checks: aarch64, driver 580.173.02, GB10 12.1, idle GPU, toolchain, Python 3.12.3, nvcc 13.0.88, disk, MemAvailable, port |
| venv | 2 | python -m venv with the system Python 3.12.3 |
| torch | 100 | torch 2.14.0+cu130 aarch64 wheel and the pinned requirements |
| build | 126 | ExLlamaV3 4d083d25c1a3 fetched by commit, compiled for sm_121 with 20 jobs, wheel cached by (commit, torch, Python, arch) |
| download | 34 | draft repository (451 MB, anonymous, 8 workers) including one HTTP 429 retried after 15 s; target: 1 s for the 12 small files missing next to the hard-linked shards |
| verify-model | 4 | 28 files, 26.4 GiB hashed, sizes and sha256 against config/model.yaml (page cache warm); a second run with unchanged files re-hashes nothing |
| retire-cache | 2 | 25.9 GiB of shard pages retired (residency after: 0 MiB), MemFree 71.6 GiB |
| serve | 16 | load of the 26 GiB pack and a 262,144-token fp16 KV cache, /health ready |
| smoke | 9 | 3 requests: text, thinking on/off with stream parity, long-prompt retrieval, image part 400 |

Cold first run; the stages total about 5 minutes plus the pack download, which is bounded by the link (28.3 GB). Sum of the listed stage durations, excluding the target download: 294 s.

A second fresh checkout (new directory, new venv) on the same machine with the shared cache directory: the compiled wheel is reused, the pack directory already holds the verified shards. Total to a verified, ready tree: 51 s.

| stage (second checkout) | seconds | note |
|---|---:|---|
| preflight | 1 | nvcc not required (cached wheel) |
| venv | 2 |  |
| torch | 37 | torch wheel and requirements from pip's wheel cache |
| build | 4 | cache hit on 4d083d25c1a3-torch2.14.0cu130-py3.12-sm121: pip install of the cached wheel, import check |
| download | 0 | files already present |
| verify-model | 7 | 28 files, 26.4 GiB hashed (page cache warm) |
<!-- END GENERATED:clickrun-table -->

## Memory

<!-- BEGIN GENERATED:memory-table -->
| memory | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting | Reference engine: DSpark drafting + vision (not reproducible here) |
|---|---:|---:|---:|
| after load | MemAvailable before load 113.2 GiB; MemAvailable after load 79.8 GiB; allocated 29.9 GiB; reserved 30.2 GiB; max reserved (after the grid) 30.2 GiB; MemAvailable after the grid 78.2 GiB | allocated 33.8 GiB; reserved 34.2 GiB; max reserved (after the grid) 34.2 GiB; MemAvailable after the grid 73.2 GiB | resident 34.2 GiB; weights 25.3 GiB |
<!-- END GENERATED:memory-table -->

## Fidelity

<!-- BEGIN GENERATED:fidelity-kl -->
| KL to the bf16 model, same implementation | mean | median | p99 | mean without worst row | top-1 agreement | ppl ratio |
|---|---:|---:|---:|---:|---:|---:|
| Set A: 14 x 1024 tokens | 0.008622 | 0.000565 | 0.0673 | 0.004525 | 97.50% | 1.0041 |
| Set B: 100 x 1024 tokens | 0.004045 | 0.000096 | 0.0440 | 0.003534 | 97.97% | 0.9998 |
| yardstick on Set A: Qwen3.5-35B-A3B, published EXL3 (turboderp), 4.00 bpw | 0.036731 | 0.004592 | - | 0.016337 | 94.69% | - |
<!-- END GENERATED:fidelity-kl -->

## Methods

| lane | method |
|---|---|
| Q200v2 text-180 | the public kit's runner (`subsets/q200v2`, fetched by commit and hash-verified), one request in flight, `temperature` 0, `top_p` 1, `max_tokens` 8,192, `chat_template_kwargs` `{"enable_thinking": true, "thinking": true, "reasoning_effort": "low"}`; GSM8K, HumanEval (kit sandbox) and IFEval graded by the kit, `hard_reasoning` read item by item against the kit's reference answers; wall time includes the kit's inline grading |
| decode / prefill / TTFT | `scripts/bench.py`: prompts of exactly the stated depth cut from the needle document at a different offset behind 16 random ids (no prefix reuse; `cached_tokens` recorded), 128 new tokens with `ignore_eos`, greedy; the engine's own timings (prefill seconds, decode seconds after the first token); median of three cold repeats, plus one repeat of the identical prompt for the prefix-cache-hit TTFT |
| needle retrieval | `scripts/exact_niah.py` on the 261,001-token document: six seven-digit passcodes at about 2k / 20k / 50k / 100k / 180k / 250k tokens, 22 questions over prefixes of 4,096 to 261,000 tokens (a question exists where the needle is wholly inside the prefix), prompt = document prefix ids + question ids (thinking off), greedy, 16 new tokens, EXACT when the stripped answer equals the passcode; cold prefill time per cut is the running sum of the first question of each cut (cuts ascend, the paged prefix cache carries the previous cut) |
| Leg D | `eval/legd/run_legd.py`: 40 GSM8K-style, 20 code, 20 chat prompts as token ids (chat template, thinking off), up to 320 new tokens; greedy on all 80, sampled (T 1.0, top-k 20, top-p 0.95) on the first 10 of each class; tokens per second after the first token from the engine's decode timing; per class mean +- s.e. over prompts and pooled |
| fidelity | KL of the pack against the bf16 model through the same ExLlamaV3 forward (measured when the pack was built; `evidence/fidelity-*`) |

## Limits

- One GB10, one pass per lane, greedy decoding. The Q200v2 kit has 180 rows, so one row is 0.56 percentage points; a
  second run was not made, and ExLlamaV3's fp16 kernels are not bit-stable across load histories.
- The reference-engine column was measured on another engine, with drafting and with the vision tower loaded; it is
  quoted for orientation, never as something this runtime does. Its NIAH document used other prose.
- `hard_reasoning` verdicts: the plain run's were made by the builder's assistant, the drafted run's by a blind AI subagent of that assistant; neither is an independent human review, and the kit's schema names the method
  `independent_manual_review` regardless.
- Throughput is single-stream. Batch size 1 is the only qualified configuration of this runtime (see
  QUALIFICATION.md for the batching experiment).
- Memory figures are the engine's allocator numbers and `MemAvailable` after load, not a power or thermal study.
- No soak test (hours of continuous serving) was run.
