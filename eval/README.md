# Evaluation suite

Same methods as the reference measurements, so results from different engines sit in one table.

| lane | what | tool | needs |
|---|---|---|---|
| smoke | text, thinking on/off + stream parity, alias key, long prompt, image part 400 | `scripts/smoke.py` | server |
| bench | decode tok/s, prefill tok/s, TTFT (cold and cache-hit) at 1k / 4k / 32k / 128k / 261k; 3 cold repeats, medians | `scripts/bench.py` | server |
| niah | 22 token-exact needle questions in a 261,001-token document (cuts 4,096 .. 261,000), greedy, thinking off | `scripts/exact_niah.py` | server |
| legd | the frozen 80-prompt plain-vs-drafted kit (greedy 80, sampled 30), one or four streams | `eval/legd/run_legd.py` | server |
| q200 | Q200v2 text-180 with the public kit's runner | `eval/q200/run_q200.sh` | server, docker |
| batch | serial vs 4-streams greedy token equivalence (is batching exact?) | `scripts/batch_equiv.py` | server started with batch size 4 |
| tokens | the server's prompt token count equals the Hugging Face `tokenizers` count (80 prompts) | `scripts/token_parity.py` | server, tokenizers |

`eval/run_suite.sh BASE_URL LABEL` runs them and writes `evidence/<LABEL>-<date>/` (bundle.json, public rows,
MANIFEST.sha256). `eval/tools/make_bundle.py` builds a bundle from lane outputs; `make_fidelity_bundle.py` sanitises results measured elsewhere.

Contents: `kit.lock.json` + `fetch_kit.sh` (pinned public kit, hash-verified), `shims/` (two modules the kit imports
but does not ship), `sandbox/` + `build_sandbox.sh` (HumanEval sandbox image), `legd/` (80 prompts as token ids),
`niah/` (document generator and the generated 261,001-token ids), `kl/` (fidelity tools, see its README).

Notes on the NIAH document: its prose is the Python 3.12 standard library's pydoc topics and docstrings plus the
docstrings of the pinned torch package, so it can be rebuilt from public material; the reference scorecard used
other prose. Needle positions (about 2k / 20k / 50k / 100k / 180k / 250k tokens), cuts, question wording and
scoring are the same. The first question of each cut costs only the new tokens (the paged prefix cache carries the
earlier cut forward); the running sum of those prefill times is the cold prefill time reported per cut.
