# The model-repo protocol, v2

This repository follows the r0b0tlab "file-tree protocol": a fixed layout, one pinned runtime, one command to
serve, and every number in the prose traceable to a hash-listed evidence bundle. v1 is the layout of
[`r0b0tlab/glm53-flash-exl3-exllamav3-gb10`](https://github.com/r0b0tlab/glm53-flash-exl3-exllamav3-gb10); v2 keeps
its invariants and makes them machine-checkable. `protocol.json` is the machine-readable form of this page and
`scripts/verify_tree.py` is its checker (run by `tests/` and by CI on every push, no GPU needed).

## What every top-level path means

| path | required | meaning |
|---|---|---|
| `README.md` | yes | what and why; 3-line quickstart; what runs where; results (generated tables); layout; verification; troubleshooting pointers; security; licences; links |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | yes | our code's licence; the upstream licences and pins we build on |
| `runtime.lock.json` | yes | the exact pins: engine commit, torch wheel, CUDA major, Python range, driver floor, model repos with revisions |
| `protocol.json` | yes | this protocol, machine-readable: required paths, profiles, pins, bundles, generated blocks, size limits, exit classes |
| `requirements.txt` | yes | pinned Python requirements (everything except torch, which `runtime.lock.json` owns) |
| `run.sh`, `scripts/click_run.sh` | yes | the one-command start: preflight, venv, torch, build, download, verify, serve, smoke |
| `config/model.yaml` | yes | every model fact the scripts need: repos, revisions, per-file manifest (size, sha256), arch string, KV bytes per token, recurrent state bytes, weights size, chat-template facts, sampling defaults, stop ids |
| `config/profiles.yaml` | yes | serving profiles (window, KV bits, batch, chunk, host cache, workspace): the preflight memory arithmetic and `/v1/models` read these |
| `scripts/` | yes | driver, server, clients, verification, rendering; `errors.tsv` is the single error-code table |
| `eval/` | optional | the evaluation suite: pinned public kits, wrappers, data, `run_suite.sh` |
| `evidence/` | yes | sanitised public bundles: `bundle.json`, public rows, `MANIFEST.sha256` per bundle |
| `docs/` | yes | `PROTOCOL.md`, `RESULTS.md` (full measured record with limits), `QUALIFICATION.md`, `MEMORY.md`, `TROUBLESHOOTING.md` |
| `tests/` | yes | offline unit tests (no GPU, no network) |
| `.github/workflows/ci.yml` | yes | tests + `verify_tree.py` + `check_public.sh` on every push |

## Invariants

1. **One pin, three views.** `runtime.lock.json`, `config/model.yaml` and the commits / revisions / torch version
   quoted in the README must agree; evidence bundles of the same engine must name the pinned commit
   (`verify_tree.py`, class 30). The pin may name the branch the runtime is built from and its base commit
   (`base_commit`, `bundle_commits` in `runtime.lock.json`): a bundle measured on the base commit, for example a
   plain-decode run that predates the drafting branch, is accepted as long as the drafting branch leaves that
   path unchanged, and says so in its own bundle.
2. **Models are data.** No model constant (repo name, shard count, vocabulary, window, KV size, stop id) is
   hard-coded in shell or in the click-run driver; it comes from `config/model.yaml` and `config/profiles.yaml`.
3. **Stages, receipts, resumability.** `click_run.sh` is a fixed sequence of named stages. A stage that ran writes
   `.state/<stage>.json` (inputs, inputs hash, versions, duration, outcome). A re-run skips a completed stage whose
   inputs are unchanged and whose outputs are still on disk. `--plan` says what would run and why.
4. **Every failure has a code and a fix.** A failure prints the stage, an `E_...` code, the cause and the exact
   remediation line, from the single table `scripts/errors.tsv`, and writes a sanitised environment report.
5. **Numbers are generated.** Result tables between `<!-- BEGIN GENERATED:name -->` markers are rendered from the
   bundles by `scripts/render_results.py`; a test fails when a document drifts from its bundles.
6. **Evidence is hash-listed and sanitised.** A bundle holds `bundle.json` (method, kit and its hash, hardware,
   engine and commit, date, caveats, normalised results), the public rows it cites, and `MANIFEST.sha256`
   (`sha256sum` format, every file except itself). Rows keep ids, verdicts, token counts, timings and content
   hashes, never prompts, responses, hostnames or paths.
7. **Public means scanned.** `scripts/check_public.sh` fails on private vocabulary; it is part of the tree check.
8. **Small.** No file above 1 MB unless allow-listed in `protocol.json`; the whole repository stays under 5 MB.
   Weights, logs and datasets are never vendored: kits are fetched by commit and verified by hash.
9. **Honest columns.** A result states which engine produced it. A column that is not reproducible with the
   repository's runtime says so (`reproducible_with_this_repo` in the bundle, and the column header).

## Exit classes of `scripts/verify_tree.py`

| code | class | meaning |
|---:|---|---|
| 0 | ok | |
| 10 | layout | a required path is missing |
| 11 | protocol | `protocol.json` malformed or disagrees with `config/profiles.yaml` |
| 20 | evidence | a bundle is missing a file, has an unlisted file, or a sha256 differs |
| 30 | pins | lock, config and documents (or bundles) disagree |
| 40 | forbidden | private vocabulary in a shipped file |
| 50 | size | a file above the limit (not allow-listed) or the repository above 5 MB |
| 60 | generated | a generated table differs from what the bundles render |
| 70 | errors | an `E_` code used by a script is not defined in `scripts/errors.tsv` |

The exit status is the lowest failing class; all findings are printed (`--json` for machines).

## Stage receipts

```
.state/<stage>.json   {schema, stage, status, inputs, inputs_hash, versions, duration_s, finished_utc, detail, extra}
stages                preflight  venv  torch  build  download  verify-model  retire-cache  serve  smoke
volatile (always run) preflight  retire-cache  serve  smoke
flags                 --plan/--explain  --from S  --only S  --until S  --force  --verify-model  --stop  --status
```

Inputs per stage: `venv` the interpreter and range; `torch` the torch pin and the requirements hash; `build` the
engine commit, torch, Python, architecture (the compiled wheel is cached under that key, so a second checkout or
venv installs it in seconds); `download` / `verify-model` the manifest digest and (for verification) the size and
mtime of every file, so unchanged shards are not re-hashed (`--verify-model` forces a full re-hash).

## Adopting the protocol for another model repo

1. Copy this tree. Replace `config/model.yaml` (use `scripts/cfg.py`'s manifest format; the per-file manifest can be
   produced from the HF repo's upload manifest) and `config/profiles.yaml`.
2. Edit `runtime.lock.json`: engine commit, torch, model repos and revisions.
3. Keep `scripts/` as is. Model-specific behaviour lives in the server's chat-template facts and in `eval/` data.
4. Run `eval/run_suite.sh` against your server, put the bundle in `evidence/`, list it in `protocol.json`, run
   `scripts/render_results.py --write`, write the prose around the generated blocks.
5. `scripts/verify_tree.py` must exit 0 before you publish.

## v1 to v2 changes

| | v1 (GLM repo) | v2 (this repo) |
|---|---|---|
| layout | convention in a README | `protocol.json` + `scripts/verify_tree.py` + CI |
| start | container image, build script, several launchers | `./run.sh`: staged, idempotent, resumable, `--plan` |
| failures | free text | codes, causes, fixes in one table; sanitised env report |
| model facts | in scripts and YAML templates | `config/model.yaml` + `config/profiles.yaml`; scripts are model-agnostic |
| build | rebuilt per image | wheel cached by (commit, torch, Python, arch) |
| download | n/a (bind mounts) | anonymous, parallel, resumable, manifest-verified, size+mtime receipts |
| results | hand-written tables | rendered from hash-listed bundles; drift fails a test |
| evidence | `MANIFEST.sha256` per bundle | + `bundle.json` (method, kit hash, engine, commit, caveats, results) |
| columns | one engine | plain / drafted / reference columns, each labelled and reproducibility-marked |
| public check | review by hand | `scripts/check_public.sh` + tree check in CI |
| host guard | required, privileged journal | optional `GUARD=1` MemAvailable floor with a bounded lease |
| bootstrap | clone by hand | `scripts/bootstrap.sh`: pinned tag, hashes verified before anything runs |
