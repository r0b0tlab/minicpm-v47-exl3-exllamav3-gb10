# Changelog

## 2.1.0 - 2026-10-14

- DSpark drafting on ExLlamaV3: loader branch `minicpm-v4.7-dspark` (commit `5ddae7d12448`, on `minicpm-v4.7` at `4d083d25c1a3`), profile `gb10-text-262k-dspark`, `./run.sh --with-draft`, `--draft-mode adaptive|fixed`, and a drafted-vs-plain parity probe in the smoke stage.
- Evaluation: fixed-15 drafting lane, identical-stream counts against the plain run, the manual-evidence builder for the hard-reasoning family, and a drafted evidence bundle (`evidence/exllamav3-drafted-20261014`).
- Measured (ExLlamaV3, one GB10, batch 1): Q200v2 text-180 169/180 with drafting against 167/180 plain; Leg D greedy decode 161 tok/s with adaptive drafting and 143 with fixed 15 drafts, against 60 plain; chat is slower at fixed 15 (41 against 59 tok/s), which is why adaptive is the default.
- Limits, stated in the README: drafted output is not bit-identical to plain decoding (59 of 80 greedy streams identical with adaptive drafting); decode with the draft attached is slower than plain at 128k and 261k (24.1 and 15.9 against 34.7 and 24.7 tok/s), not yet explained; the hard-reasoning grades were produced by an AI assistant, not an independent human reviewer.
- Results tables regenerated from the bundles; the reference-engine column is unchanged and dated.
- The ExLlamaV3 fork is now `r0b0tlab/exllamav3-r0b0t`.
- Bootstrap default ref is `v2.1.0`.

## 2.0.0 - 2026-10-13

- First public release: ExLlamaV3 plain decoding of MiniCPM-V-4.7-35B-A3B on one GB10, the click-run entry point, the OpenAI-compatible server, the evaluation suite, and the evidence bundles for plain decoding and the reference engine.
