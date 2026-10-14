# MiniCPM-V-4.7-35B-A3B EXL3 on one GB10: ExLlamaV3 click-run

Serve our EXL3 pack of [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)
(35B parameters, 3B active, 262,144-token window) with [ExLlamaV3](https://github.com/turboderp-org/exllamav3) on a
single NVIDIA GB10 / DGX Spark (aarch64, SM121) with **one command**: it checks the machine, builds the CUDA
extension from a pinned commit, downloads and verifies the pack, retires the shard page cache that breaks loading on
unified memory, starts an OpenAI-compatible server and smoke-tests it. The repository also ships the evaluation
suite and the measured record, laid out by the r0b0tlab model-repo protocol ([docs/PROTOCOL.md](docs/PROTOCOL.md)).

- Pack (ours, Apache-2.0): [r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw](https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw), revision `8f9f07872b09`, 28.3 GB in 8 shards
- DSpark draft (ours, Apache-2.0): [r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw](https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw), revision `34c6b7758cbe`, 451 MB (run by ExLlamaV3 on the `minicpm-v4.7-dspark` loader branch, see below)
- Loader: [r0b0tlab/exllamav3-r0b0t, branch `minicpm-v4.7-dspark`](https://github.com/r0b0tlab/exllamav3-r0b0t/tree/minicpm-v4.7-dspark), commit `5ddae7d12448` (on top of branch `minicpm-v4.7` @ `4d083d25c1a3`: ExLlamaV3 v1.5.0 base, aarch64 / sm_121 build fixes, the MiniCPM-V-4.7 text architecture, a DSpark draft quantisation shim; the dspark branch adds DSpark drafting); torch 2.14.0+cu130

## What runs where

| | ExLlamaV3 plain decode | ExLlamaV3 + DSpark drafting |
|---|---|---|
| runtime | this repository, profile `gb10-text-262k` | this repository, profile `gb10-text-262k-dspark` (`./run.sh --with-draft`) |
| input | **text only** (image / video parts are a clear HTTP 400) | **text only** |
| decoding | autoregressive | DSpark block drafter: 15 drafts per round, Markov walk over the top 64 candidates, adaptive verification |
| output vs plain decoding | the reference | **not bit-identical**: 59 of 80 greedy streams identical to plain on the 80-prompt kit with adaptive drafting (21 differ; the divergence point and cause were not measured) |
| numbers in this README | measured here | measured here |

ExLlamaV3 stores the vision tower (bf16, byte-identical to the source) but does not run it. The drafted profile
attaches the 451 MB DSpark draft (`r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw`) through the
`minicpm-v4.7-dspark` branch of the loader fork; builds without the drafting path make the server refuse to start with
an explanation. Every draft is verified by the target model. Greedy outputs still differ from plain decoding on some
prompts (21 of 80 in the measured run); the divergence point and the cause were not measured. Adaptive verification is the
default because verifying 15 drafts on low-acceptance text (chat) is *slower* than plain decoding; it verifies the
number of drafts that pays off under a measured cost model of the round.

## Quickstart (click-run)

```bash
git clone https://github.com/r0b0tlab/minicpm-v47-exl3-exllamav3-gb10 && cd minicpm-v47-exl3-exllamav3-gb10
./run.sh            # preflight, venv, torch, build, download, verify, retire page cache, serve, smoke
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minicpm-v-4.7-35b-a3b-exl3-6.08bpw","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":1024}'
```

or, from nothing: `curl -fsSL https://raw.githubusercontent.com/r0b0tlab/minicpm-v47-exl3-exllamav3-gb10/v2.1.2/scripts/bootstrap.sh | bash`
(clones the pinned tag, verifies `click_run.sh` and `verify_tree.py` against embedded hashes before running anything).

Useful variations: `./run.sh --plan` (what would run and why), `PROFILE=gb10-text-32k PORT=8001 ./run.sh`,
`./run.sh --status`, `./run.sh --stop`, `./run.sh --from retire-cache` (restart the server), `--force`,
`--verify-model` (re-hash every shard), `--with-draft` (also download the DSpark draft pack), `GUARD=1` (optional
host-memory watchdog). Flags and environment variables are listed by `./run.sh --help`.

For drafting add the draft pack and attach it: `./run.sh --with-draft` (or `PROFILE=gb10-text-262k-dspark ./run.sh`);
the 451 MB draft is downloaded, verified and attached, and the smoke stage ends with a drafted-vs-plain parity probe.

A cold first run on a GB10 with a fast link: venv 2 s, torch wheel and requirements about 100 s, extension build
about 2 min, pack download bounded by the link (28.3 GB), verification 4 s, load 16 s. Re-runs skip every completed
stage whose inputs are unchanged (`.state/<stage>.json`), and the compiled extension is cached by
(commit, torch, Python, arch), so a second checkout installs it in seconds. Timings of the clean-room run are in
[docs/RESULTS.md](docs/RESULTS.md).

## What you get

| | supported | notes |
|---|---|---|
| `POST /v1/chat/completions` | yes | streaming (SSE) and non-streaming; `reasoning_content` split at `</think>`; thinking on by default; `chat_template_kwargs` honoured (`enable_thinking`, alias key `thinking`, unknown keys ignored) |
| `POST /v1/completions` | yes | string or token-id prompts; `return_token_ids`, `ignore_eos`, `min_tokens` extensions |
| tool calls | yes | the model's Qwen-style XML calls are parsed into OpenAI `tool_calls` (string / number / boolean / JSON types by schema); unparsable output stays in `content`; `tool_choice` is advisory (no constrained decoding) |
| usage accounting | yes | engine token counts (the stop token counts as generated); `timings` block with prefill / decode seconds |
| sampling | yes | defaults from the pack: temperature 1.0, top-k 20, top-p 0.95; `temperature`, `top_p`, `top_k`, `min_p`, penalties, `seed`, `stop` |
| window | yes | up to 262,144 tokens (profile `gb10-text-262k`), `max_tokens` capped by the window, prompts that do not fit are a 400 |
| prefix cache | yes | ExLlamaV3's paged prefix cache; `usage.prompt_tokens_details.cached_tokens` |
| concurrency | experimental | one request at a time (FIFO) is the qualified setting; `BATCH_SIZE=4 ./run.sh` (profile `gb10-text-262k-b4`) batches up to four requests: aggregate decode about 2x, accuracy unchanged on Q200v2, but greedy output is not token-identical to serial decoding ([docs/QUALIFICATION.md](docs/QUALIFICATION.md)) |
| image / video / audio input | no | HTTP 400 with an explanation (needs an engine with the vision tower) |
| DSpark drafting | yes | profile `gb10-text-262k-dspark`; adaptive verification by default; output differs from plain decoding on some prompts (21 of 80 greedy prompts in the measured run), not bit for bit; builds without the drafting path refuse to start |
| authentication, TLS | no | loopback by default, see Security |

Endpoints besides the two above: `GET /health`, `GET /v1/models` (`max_model_len` from the profile), `GET /metrics`
(counters, last decode tok/s, memory), `POST /v1/chat/completions/render` (rendered prompt and exact token ids).

## Results

Methods are the ones of the reference measurements so the columns compare side by side: Q200v2 text-180 with the
public kit's own runner (thinking on, effort low, 8,192 tokens, one request in flight), a 261,001-token needle
document with six seven-digit passcodes and 22 token-exact questions (greedy, thinking off), decode and prefill
at 1k / 4k / 32k / 128k / 261k tokens (medians of three cold repeats), and the frozen 80-prompt plain-vs-drafted kit.
Everything below the line is generated from the hash-listed bundles in [`evidence/`](evidence/).

<!-- BEGIN GENERATED:results-summary -->
| metric | ExLlamaV3, plain decode (this repo's runtime) | ExLlamaV3 + DSpark drafting |
|---|---:|---:|
| **Q200v2 text-180**, accuracy | 167/180 (92.8%) | 169/180 (93.9%) |
| &nbsp;&nbsp;GSM8K-style math (80) | 74/80 (92.5%) | 75/80 (93.8%) |
| &nbsp;&nbsp;HumanEval-style code (40) | 39/40 (97.5%) | 39/40 (97.5%) |
| &nbsp;&nbsp;IFEval-style instruction following (40) | 35/40 (87.5%) | 35/40 (87.5%) |
| &nbsp;&nbsp;Hard reasoning (20, AI-graded, see notes) | 19/20 (95.0%) | 20/20 (100.0%) |
| Q200v2 end-to-end tok/s, 1 request in flight | 58.7 | 109.8 |
| Q200v2 end-to-end tok/s, 4 requests in flight (ExLlamaV3: experimental batch 4) | 94.0 | not run (batch 1) |
| Decode tok/s at ~4k context | 47.0 | 70.7 |
| Decode tok/s at ~32k context | 48.2 | 48.7 |
| Decode tok/s at ~128k context | 34.7 | 24.1 |
| Decode tok/s at ~261k context | 24.7 | 15.9 |
| Prefill tok/s at ~32k | 2,889 | 2,839 |
| Cold time to first token at ~32k (s) | 11.3 | 11.5 |
| Cold time to first token at ~261k (s) | 226.8 | 227.1 |
| Needle retrieval (22 questions, token-exact) | 19/22 exact | 19/22 exact |
| Leg D decode tok/s, GSM8K-style, plain | 60.1 | - |
| Leg D decode tok/s, code, plain | 59.8 | - |
| Leg D decode tok/s, chat, plain | 59.4 | - |
| Leg D decode tok/s, GSM8K-style, drafted, adaptive verification | n/a | 177.8 |
| Leg D decode tok/s, code, drafted, adaptive verification | n/a | 220.5 |
| Leg D decode tok/s, chat, drafted, adaptive verification | n/a | 69.0 |
| Leg D decode tok/s, GSM8K-style, drafted, always 15 drafts | n/a | 162.1 |
| Leg D decode tok/s, code, drafted, always 15 drafts | n/a | 207.3 |
| Leg D decode tok/s, chat, drafted, always 15 drafts | n/a | 41.0 |
| Leg D greedy streams identical to plain decoding | n/a | 59/80 |
| GPU-side resident memory | 29.9 GiB allocated | 33.8 GiB allocated |
<!-- END GENERATED:results-summary -->

Reading the table:

- Every number in this README was measured on this repository's runtime, on one GB10: plain autoregressive decode (profile `gb10-text-262k`) and DSpark drafting (profile `gb10-text-262k-dspark`), text path, batch size 1.
- Accuracy (Q200v2, same kit, same settings, greedy): 167 of 180 rows pass with plain decoding and 169 of 180 with drafting. `hard_reasoning` has no automatic grader: the 20 answers of each run were read item by item against the kit's reference answers, by the builder's assistant for the plain run and by a blind AI subagent for the drafted run; verdicts and rationales are in the bundles. The plain run has one row at the 8,192-token ceiling (`hard-13`, no answer, counted as failed); the drafted run has one (`humaneval-032`, no answer, counted as failed).
- Decode tok/s at a fixed prompt depth between 2k and 32k tokens varies by up to 20 % between depths, and the first
  requests a freshly started server sees decode and prefill more slowly than later ones (4,096 tokens: 47 tok/s
  on the first visit in the grid, 55 to 57 once warm; see "Decode speed at a depth depends on how long the server
  has been up" in [docs/RESULTS.md](docs/RESULTS.md)). Read that range as 46 to 60 tok/s, not as a curve.
- **Drafting (ExLlamaV3 + DSpark, adaptive verification, the default).** Leg D greedy decode averages 161 tok/s
  against 60 plain (code 221 against 60, GSM8K-style 178 against 60, chat 69 against 59). With 15 drafts fixed in
  every round, chat drops to 41 tok/s (0.69x plain), which is why adaptive verification is the default. Greedy
  streams match plain on 59 of 80 prompts with adaptive drafting (60 of 80 with fixed 15). Q200v2 at batch 1
  runs at 110 tok/s end to end against 59 plain, and scores 169/180 against 167/180 plain. That difference is
  within what one run per row can show; it is not evidence that drafting improves accuracy.
- **Drafting at long context is not faster.** Decode with the draft attached matches plain at 32k (48.7 against
  48.2 tok/s), is faster at 1k (65.6 against 59.5 tok/s) and at 4k (70.7 against 47.0 on the first visit, 55 to 57 once warm), and is slower at 128k (24.1 against 34.7 tok/s) and 261k (15.9 against
  24.7 tok/s). We have not explained this yet. The cause is not established. Do not read the drafted column as a long-context
  speedup.
- **Grading of the drafted hard-reasoning family.** The 20 answers were graded item by item against the kit's
  reference answers by a blind Sonnet 5.5 subagent of the builder's assistant (given the rows, the kit's reference
  answers and the kit's dump tool; not the plain grades). Its verdicts agree with the plain grades on 19 of 20; the
  one disagreement (`hard-13`) is a budget difference: the plain run ran out of tokens without an answer, and the
  drafted run answered 2/3, which the maintainers' assistant re-checked against the reference answer. The `independent_manual_review` method name in the evidence
  file is the kit's schema field; the grader is an AI assistant, not an independent human reviewer.
- Full method, limits and per-lane detail: [docs/RESULTS.md](docs/RESULTS.md); what is and is not qualified:
  [docs/QUALIFICATION.md](docs/QUALIFICATION.md).

### Fidelity (text)

The pack's divergence from the bf16 model, measured through the same ExLlamaV3 forward this repository serves with
(so these numbers apply to ExLlamaV3 serving). Lower is better; top-1 is next-token agreement.

<!-- BEGIN GENERATED:fidelity-kl -->
| KL to the bf16 model, same implementation | mean | median | p99 | mean without worst row | top-1 agreement | ppl ratio |
|---|---:|---:|---:|---:|---:|---:|
| Set A: 14 x 1024 tokens | 0.008622 | 0.000565 | 0.0673 | 0.004525 | 97.50% | 1.0041 |
| Set B: 100 x 1024 tokens | 0.004045 | 0.000096 | 0.0440 | 0.003534 | 97.97% | 0.9998 |
| yardstick on Set A: Qwen3.5-35B-A3B, published EXL3 (turboderp), 4.00 bpw | 0.036731 | 0.004592 | - | 0.016337 | 94.69% | - |
<!-- END GENERATED:fidelity-kl -->

Calibration was text-only; image-prompt fidelity (measured on the engine with the vision tower, not on ExLlamaV3)
is in the pack's model card.

## Memory budget

One GB10 pool, `MemAvailable` required before start (from `config/model.yaml` and `config/profiles.yaml`;
derivation and the page-cache story in [docs/MEMORY.md](docs/MEMORY.md)):

| profile | window | KV cache | resident (estimate) | `MemAvailable` required |
|---|---:|---:|---:|---:|
| `gb10-text-262k` (default) | 262,144 | 5.00 GiB | 36.4 GiB | 44.4 GiB |
| `gb10-text-32k` | 32,768 | 0.62 GiB | 31.0 GiB | 39.0 GiB |
| `gb10-text-16k-lowmem` | 16,384 | 0.31 GiB | 29.0 GiB | 35.0 GiB |

On GB10 the loader's free-memory reading counts the shards' page cache as used, so loading can fail with
`Insufficient VRAM in split for model and cache` while `MemAvailable` looks fine. `./run.sh` retires **only** the
pack's shard pages (`posix_fadvise(DONTNEED)`, residency read back with `mincore`) right before every start; never a
host-wide cache drop. If you start the server by hand, run `scripts/retire_page_cache.py --dir <model dir>` first.

## Profiles

`config/profiles.yaml`: `gb10-text-262k` (default, full window, fp16 KV, chunk 2,048), `gb10-text-32k`,
`gb10-text-16k-lowmem` (smallest footprint, 1,024-token chunks), `gb10-text-262k-dspark` (DSpark drafting attached),
`gb10-text-262k-b4` (experimental, up to 4 sequences batched). A profile is data: window, KV bits, batch size,
chunk size, host recurrent-cache store, workspace and headroom. `scripts/cfg.py need <profile>` prints its budget.

## How this repository is organised

```
run.sh                   one-command start (wrapper of scripts/click_run.sh)
README.md  LICENSE  THIRD_PARTY_NOTICES.md
runtime.lock.json        exact pins: engine commit, torch wheel, CUDA/Python/driver floors, model repos + revisions
protocol.json            the layout protocol, machine-readable (required paths, bundles, generated blocks, limits)
requirements.txt         pinned Python requirements (torch is pinned in runtime.lock.json)
config/                  model.yaml (facts + per-file manifest), profiles.yaml
scripts/                 click_run.sh, serve_openai.py, preflight / stagectl / verify_model / retire_page_cache,
                         smoke / bench / exact_niah, verify_tree / render_results / check_public / env_report,
                         guard (optional), bootstrap, errors.tsv
eval/                    the evaluation suite: kit fetch + shims, Q200v2 wrapper, Leg D, NIAH document, run_suite.sh
evidence/                sanitised bundles: bundle.json + public rows + MANIFEST.sha256 each
docs/                    PROTOCOL, RESULTS (full record with limits), QUALIFICATION, MEMORY, TROUBLESHOOTING
tests/                   offline unit tests (no GPU, no network)
.github/workflows/ci.yml tests + tree check + public check on every push
```

The protocol in one paragraph: every top-level path has a defined meaning; the runtime, the model repos and their
revisions are pinned in one place and cross-checked against the config and the documents; `click_run.sh` is a
sequence of named stages that write receipts and skip what is unchanged; every failure has a code, a cause and a
fix; every number in the prose is rendered from a hash-listed evidence bundle; and a public-vocabulary scan and a
size limit gate the tree. Details, exit classes, and how another model repository adopts it:
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Verify

```bash
python3 -m unittest discover -s tests -t . -q      # offline unit tests (131), no GPU
python3 scripts/verify_tree.py                      # layout, pins, evidence manifests, size, generated blocks
bash scripts/check_public.sh                        # no private vocabulary in any shipped file
(cd evidence/<bundle> && sha256sum -c MANIFEST.sha256)
./run.sh --only verify-model --verify-model         # shard sizes and sha256 against config/model.yaml
```

## Reproduce the measurements

```bash
./run.sh                                             # server up on 127.0.0.1:8000
eval/build_sandbox.sh                                # HumanEval sandbox image of the kit (docker)
SANDBOX_IMAGE_ID=sha256:... eval/run_suite.sh http://127.0.0.1:8000 exllamav3-plain
python3 scripts/render_results.py --write            # refresh the generated tables
```

`eval/run_suite.sh` runs against any OpenAI-compatible endpoint, lane by lane (`--lanes smoke,bench,niah,legd,q200`),
and writes `evidence/<label>-<date>/`. The Q200v2 kit is fetched by commit from
[r0b0tlab/r0b0bench](https://github.com/r0b0tlab/r0b0bench) and verified against its `MANIFEST.sha256`; nothing of it
is vendored. See [eval/README.md](eval/README.md) and [scripts/q200_notes.md](scripts/q200_notes.md). Exclusive use
of the GPU is assumed: do not run the lanes next to another model.

## Troubleshooting

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) has the error-code table (what the driver prints) and the usual
suspects: page cache and `Insufficient VRAM in split`, `nvcc` missing, build time and memory, port in use, Hugging
Face rate limits, a shard that does not verify, `400` on image input. `./run.sh --env-report` prints a sanitised
report (no host names, paths, addresses or tokens) for issues.

## Security

The server has **no authentication and no TLS**. It binds `127.0.0.1` by default; do not expose the port. If you
bind another address (`HOST=0.0.0.0`) put it behind a reverse proxy with authentication. Requests run on the
machine's only GPU, so one client can starve the others. The download is anonymous (`HF_TOKEN` is unset for it) and
every file is checked against a pinned size and sha256 before it is used. Model files are loaded with safetensors
(no pickle). `scripts/bootstrap.sh` verifies the scripts it is about to run against embedded hashes.

## Licences

- This repository's code: MIT (`LICENSE`).
- ExLlamaV3 (and our fork): MIT. The model and our quantised derivatives: Apache-2.0 (inherited from OpenBMB's
  MiniCPM-V-4.7-35B-A3B). PyTorch and the Python dependencies keep their own licences.
  See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Links

- Pack: https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw
- DSpark draft: https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw
- Loader branches: https://github.com/r0b0tlab/exllamav3-r0b0t/tree/minicpm-v4.7 (plain decode) and https://github.com/r0b0tlab/exllamav3-r0b0t/tree/minicpm-v4.7-dspark (drafting)
- Evaluation kit: https://github.com/r0b0tlab/r0b0bench (subset `subsets/q200v2`)
- Protocol v1 sibling: https://github.com/r0b0tlab/glm53-flash-exl3-exllamav3-gb10
