# Qualification

What was exercised on which pins, what passed, and what is explicitly **not** qualified. The numbers behind each
line are in [RESULTS.md](RESULTS.md) and in the hash-listed bundles under `evidence/`.

Pins: ExLlamaV3 `4d083d25c1a3` (branch `minicpm-v4.7`), torch 2.14.0+cu130, CUDA toolkit 13.0.88, driver
580.173.02, Python 3.12.3, one NVIDIA GB10 (aarch64, SM121), pack `8f9f07872b09`, profile `gb10-text-262k`
(262,144-token window, fp16 KV, chunk 2,048, batch size 1).

## Gates

| gate | what | result |
|---|---|---|
| G1 clean room | fresh clone, new venv, cold build of the public branch, anonymous download path, manifest verification, retire-cache, serve, smoke | passed: cold first run 294 s of stages without the pack download (venv 2 s, torch 100 s, extension build 126 s, draft download 34 s with one HTTP 429 retried, target verification 4 s, load 16 s, smoke 9 s); a second checkout reached a verified tree in 51 s |
| G2 smoke | text with thinking off, thinking on with stream = non-stream parity, alias key `thinking` and unknown kwargs, 3,102-token retrieval, image part answered 400, server alive after the error | passed (9 of 9 checks) |
| G3 tokenisation | server `prompt_tokens` equals the Hugging Face `tokenizers` count for the 80 Leg D prompts decoded to text | passed: 80 of 80 prompts, server count = HF count |
| G4 Q200v2 text-180 | the public kit's runner, thinking on, effort low, 8,192 tokens, one request in flight | completed: 167/180 (92.8 %), one row at the 8,192-token ceiling; 59.5 tok/s on average, 44.7 minutes; the reference engine's 168/180 is within one row |
| G5 needle retrieval | 22 token-exact questions over a 261,001-token document, up to a 261,000-token prefix | 19 of 22 exact; the three misses return the passcode of the next needle at 200k and 261k (the reference scorecard missed 2 of 22 on a different document); max prompt 261,034 tokens, all counts matched |
| G6 depth grid | decode, prefill and TTFT at 1k / 4k / 32k / 128k / 261k, three cold repeats, medians | completed; decode 59.5 / 47.0 / 48.2 / 34.7 / 24.7 tok/s at 1k / 4k / 32k / 128k / 261k, cold TTFT 0.9 / 2.0 / 11.3 / 74.3 / 226.8 s; no allocation failure, no pad id generated, no accounting mismatch |
| G7 Leg D plain | 80 greedy + 30 sampled prompts, one stream | completed: 59.9 tok/s over the 80 greedy prompts (60.1 / 59.8 / 59.4 by class), 59.7 over the 30 sampled; 77 of the 110 runs ended on a stop token and 33 at the 320-token cap, as the kit is designed |
| G8 failure isolation | bad JSON, bad parameters, vision parts, over-long prompts answered with 4xx and the server keeps serving (unit tests + smoke) | passed |
| G9 offline suite | 123 unit tests, `verify_tree.py`, `check_public.sh`, shellcheck, CI | passed |

## Batching

Batch size 1 (FIFO) is the qualified setting. A batched mode exists (`BATCH_SIZE=4 ./run.sh`, profile
`gb10-text-262k-b4`) and was exercised, but it did **not** pass the exactness bar this repository sets
(`scripts/batch_equiv.py`: the same prompts decoded one at a time and four at a time, greedy, token for token):

| experiment | result |
|---|---|
| serial decoding, twice | identical, 12 of 12 prompts (decoding is deterministic) |
| 4 streams against serial, run 1 | 10 of 12 prompts identical (two diverge at token 128 and 180) |
| 4 streams against serial, run 2 | 7 of 12 prompts identical (diverging at tokens 123 to 181); the same prompt can match in one run and diverge in the other, so the divergence depends on how requests happen to be batched |
| aggregate decode, 12 prompts x 256 tokens | 59 tok/s serial, 118 tok/s with 4 streams |
| Q200v2 text-180 with 4 workers on the batch-4 server | 168/180, all 180 rows finished with a stop; 1,681 s against 2,683 s serial (1.6x end to end, grading included); 90 of 180 answers byte-identical to the serial run |

Reading: batching is functionally sound (accuracy and finish reasons match the serial run, no errors, no padding id,
no accounting mismatch) and doubles aggregate decode throughput, but it is not bit-reproducible against serial
decoding: with a different batch composition the fp16 kernels round differently and a greedy continuation flips at
a near-tie. We have no evidence of a state-handling bug (a recurrent-state fault would show as garbage, not as a late
flip), but we also have not proven its absence, so the default stays at batch size 1 and the batched profile is
labelled experimental.

## Not qualified

- Concurrency above one request as a *qualified* setting (see Batching: exercised, accuracy-neutral, not token-exact).
- DSpark drafting: the pinned ExLlamaV3 build converts and loads the draft but cannot run it; the server refuses
  `--draft` with a clear message. The "ExLlamaV3 + DSpark" column of the results is a placeholder until a build
  supports it.
- Image, video and audio input (text path only).
- Long-run stability: no soak test; no thermal or power study.
- Quantised KV cache (`kv_bits` below 16): the profiles that exist use fp16.
- Other drivers, CUDA toolkits, torch builds or hardware than the pins above.
- `tool_choice` forcing (no constrained decoding) and `n > 1`.
- Anything on the internet-facing side: no authentication, no TLS, no rate limiting.
