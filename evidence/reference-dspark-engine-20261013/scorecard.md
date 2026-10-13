# Reference engine: long-context scorecard, decode and memory (as measured)

One 261,001-token document (repository documents, Python's pydoc topics and Ansible READMEs; no paragraph repeated) with six seven-digit passcode needles at tokens 2,002 / 20,081 / 50,132 / 100,243 / 180,036 / 250,028; 22 questions about the needles inside prefixes of 4,096 to 261,000 tokens, answered greedily with thinking off, prefill in chunks of 4,096, one stream, one NVIDIA GB10 (128 GB unified memory), the 6.08 bpw EXL3 target.

| context | cold time to first token (s) | prefill tok/s | decode ms / step | decode tok/s | after a prefix-cache hit: first token (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.9 | 4,551 | 14.97 | 66.8 | 0.07 |
| 32,000 | 7.6 | 4,211 | 18.51 | 54.0 | 0.09 - 0.10 |
| 128,000 | 46.2 | 2,777 | 30.44 | 32.8 | 0.15 - 0.19 |
| 261,000 | 137.5 | 1,901 | 46.48 | 21.5 | 0.24 - 0.31 |

**20 of 22 questions exact** (exact at every depth up to 128k, 11 of 11; 9 of 11 at 200k and 261k). Both misses return the passcode of another needle of the same document (never a corrupted digit string) and are reproduced token for token by an independent EXL3 implementation of the same weights (same four prompts, same replies; centred cosine of the two engines' 248,144 logits at the answer position 0.997 to 0.998), so they are the quantised model's slips between two codewords at depth, not an arithmetic artefact. A question prefilled cold at 261,000 tokens has bit-equal candidates to the cached run's.

One NVIDIA GB10 (273 GB/s nominal LPDDR5X, 235 GB/s read at best on the box; 125 TFLOP/s measured on the bf16 tensor path), the 6.08 bpw EXL3 target, one stream, window graph replayed, candidates read back.

- **Decode, no drafter:** 66.8 tok/s at a 4,096-token context (14.97 ms a token), 68.0 at short context in the 80-prompt kit; the bytes a token must read (2,601 MB of weights, summed from the pack's own tensor sizes: head 382 MB, non-expert tensors 1,460 MB, the top-8 routed experts of each layer 760 MB; plus 189 MB of recurrent state and 20,480 B of KV a context token) put the roofline at 95 tok/s on the nominal bandwidth: **70 %** of it, **82 %** of what the memory reads at on this box; the ratio is 64 to 70 % of the nominal roofline at every depth to 261k tokens (21.5 tok/s there).
- **Prefill:** 5,037 tok/s in chunks of 4,096 tokens (3,049 tok/s in chunks of 512, 69 % of the weight-stream roofline); 4,551 tok/s over a 4k to 16k prompt, 3,597 tok/s at 64k and 1,901 tok/s over a whole 261k prompt as the attention core grows with the context.
- **Time to first token, cold, one stream:** 0.26 s at 1,024 tokens, 1.7 s at 8,192, 7.8 s at 32,768, 18.2 s at 65,536, 46 s at 128k, 138 s at 261k; with a prefix-cache hit on all but the last question tokens: 0.07 to 0.31 s at any context.
- **With the drafter:** 206 tok/s mean over the 80-prompt kit (228 on GSM8K-style math, 282 on code, 85 on chat); a prefill with the drafter attached costs 1.8 % more (3,740 against 3,809 tok/s at 8,192 tokens, chunks of 1,024).

The 6.08 bpw EXL3 target on one GB10 (unified memory), with four 262,144-token slots, chunks of 4,096, the prefix cache on:

- **Weights:** 25.31 GiB (27.2 GB) loaded (25.95 GiB of arena blocks); the 6.0 bpw drafter adds 430 MiB (451 MB file) and the vision tower 1.04 GiB.
- **KV cache:** 20,480 B a token (10 full-attention layers, 2 KV heads of 256, bf16 K and V) = 5.0 GiB for 262,144 tokens; the drafter's cache adds 10,240 B a token (7.5 MiB a 256-token page with it).
- **Recurrent state:** 64,389,120 B (61.4 MiB) a sequence for the 30 gated delta net layers, whatever the length; a prefix-cache snapshot is the same 61.4 MiB plus pages shared by reference.
- **Scratch:** a working set of 1.78 GiB at chunks of 4,096 (317 MiB scratch, 1,447 MiB workspaces of which 0.6 GiB is the grouped-expert prefill workspace, 60.5 MiB of window records) inside a 5 GiB reserve; the drafter's scratch is 42 MiB (one stream, 4k context) to 583 MiB (eight streams, 262k) and its tap buffer 10 to 80 MiB.
- **Resident total** in that configuration: 34.2 GiB of the box's memory (111.8 GiB available before the boot, 77.6 after); nothing grows while serving.

Decoding with an inference engine implementing the DSpark block drafter (the 6.08 bpw EXL3 target with its 6.0 bpw drafter, 15 drafted tokens a round, one verify window per round, output token-for-token identical to decoding without the drafter), one stream on one NVIDIA GB10; 80 prompts of a fixed kit (40 GSM8K, 20 HumanEval, 20 IFEval; chat template, thinking off, up to 320 new tokens), the first 10 of each class again sampled. Speed-ups are paired prompt by prompt against the same engine without the drafter (a repeat of the baseline arm is the baseline).

| prompts | n | tokens / verify round (AL) | tok/s with the drafter | tok/s without | speed-up (paired, ± SE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **greedy** | | | | | |
| GSM8K-style math | 40 | 8.36 ± 0.31 | 228.2 ± 5.8 | 68.0 | **3.35x** ± 0.09 |
| HumanEval-style code | 20 | 11.29 ± 0.34 | 282.1 ± 6.0 | 68.0 | **4.15x** ± 0.09 |
| IFEval-style chat | 20 | 2.04 ± 0.11 | 84.8 ± 3.8 | 68.1 | **1.24x** ± 0.06 |
| all prompts | 80 | 7.51 ± 0.42 | 205.8 ± 8.9 | 68.0 | **3.03x** ± 0.13 |
| **sampled, temperature 1.0, top-k 20, top-p 0.95** | | | | | |
| GSM8K-style math | 10 | 7.40 ± 0.58 | 199.1 ± 10.8 | 68.0 | **2.93x** ± 0.16 |
| HumanEval-style code | 10 | 11.61 ± 0.33 | 272.7 ± 7.1 | 67.9 | **4.01x** ± 0.10 |
| IFEval-style chat | 10 | 2.10 ± 0.19 | 80.7 ± 5.2 | 68.0 | **1.19x** ± 0.08 |
| all prompts | 30 | 7.03 ± 0.76 | 184.2 ± 15.4 | 68.0 | **2.71x** ± 0.23 |
| **greedy, four prompts of a class decoded at once** | | | | | |
| aggregate of 4 streams | 80 | | 318 | 138 | **2.29x** ± 0.09 |

**Drafter weight bit width** (greedy; the same prompts; paired Δ AL is taken prompt by prompt against the bf16 drafter, tok/s under the engine's verify-budget policy):

| drafter weights | file | AL (all, full 16-row windows) | paired Δ AL against the bf16 drafter (± SE) | tok/s (all) | speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.0 bpw | 353 MB | 7.69 ± 0.42 | -0.245 ± 0.086 | 200.7 | 2.95x |
| 4.0 bpw | 386 MB | 7.86 ± 0.43 | -0.071 ± 0.048 | 204.3 | 3.00x |
| 5.0 bpw | 418 MB | 7.90 ± 0.43 | -0.034 ± 0.037 | 204.7 | 3.01x |
| **6.0 bpw (shipped)** | 451 MB | 7.94 ± 0.43 | +0.011 ± 0.027 | 205.8 | 3.03x |
| 8.0 bpw | 517 MB | 7.94 ± 0.43 | +0.008 ± 0.022 | 202.9 | 2.98x |
| bf16 (reference) | 778 MB | 7.93 ± 0.43 | 0 (reference) | 196.4 | 2.89x |

The acceptance rule for a smaller drafter (paired Δ AL ≥ −0.05 and no loss of speed) is met down to 5.0 bpw (418 MB) and missed at 4.0 bpw; the 6.0 bpw drafter (451 MB) is 33 MB larger than the smallest that passes, has the same speed, and is the one the pair's gates and sweeps were run with. The drafter's accepted-token curve is flat for GSM8K-style and code text (0.85 to 0.97 conditional acceptance at every depth) and falls to about 0.5 at the first positions for open instruction-following prose, where the engine verifies windows of two or three rows.

