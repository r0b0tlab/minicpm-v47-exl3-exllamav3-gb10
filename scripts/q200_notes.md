# Running the Q200v2 kit against this server

The kit is the public subset `subsets/q200v2` of https://github.com/r0b0tlab/r0b0bench (pinned commit in
`eval/kit.lock.json`). Nothing of it is vendored: `eval/fetch_kit.sh` fetches it and verifies its
`MANIFEST.sha256` and the dataset hash. `eval/q200/run_q200.sh` wraps its runner.

```bash
./run.sh --yes                                      # server on 127.0.0.1:8000
eval/build_sandbox.sh                                # prints the sandbox image id (docker)
SANDBOX_IMAGE_ID=sha256:... eval/q200/run_q200.sh http://127.0.0.1:8000 out/q200 --workers 1
```

## Request settings

The runner sends, per row: `temperature: 0`, `top_p: 1`, `max_tokens: 8192`, `stream: false`, and
`chat_template_kwargs: {"enable_thinking": true, "thinking": true, "reasoning_effort": "low"}`. It insists on the
`thinking` key (its `validate_native_thinking`), so a server must **accept the alias `thinking`**: this server maps
it onto `enable_thinking` and ignores other unknown template kwargs instead of answering 400. No shim is needed.
The runner requires a non-empty `message.content`, a positive `usage.prompt_tokens` / `completion_tokens` and
`finish_reason: "stop"` (a row that hits the 8,192-token ceiling is a failure, not an error).

## What the wrapper adds

- `eval/shims/admission_control.py` and `eval/shims/niah_common.py`: the runner imports two modules the public kit
  does not ship; these stand-ins grant every request and load JSON strictly. The lease recorded on each row says
  so ("grants every request, gates nothing").
- `--workers N` (kit limit 4): N requests in flight. This server batches nothing (batch size 1), so more than one
  worker only queues requests; the number is recorded in the bundle.
- `--run-id`, `--manual-evidence`: re-run the same run id with a manual-evidence file to regrade without
  regenerating.

## Grading

GSM8K (flexible exact answer), HumanEval (executed in the kit's docker sandbox, no network, read-only, 256 MB) and
IFEval (rule checks) are graded by the kit. `hard_reasoning` has no automatic grader: the run reports 20 rows as
ungraded until a manual-evidence file is supplied. `eval/q200/manual_evidence.py dump` prints each task with the
kit's reference answer and the model's final answer; `build` turns a grades file into the kit's manual-evidence
format, bound to the sha256 of every response. The kit's schema names the method `independent_manual_review`;
the `reviewer` field says who actually graded. In this repository's bundles the grader is the person or agent that
built the run, which is **not** an independent review.

## Throughput numbers

`eval/tools/make_bundle.py` derives end-to-end throughput as total completion tokens divided by the wall seconds of
the run, per-request tokens per second from each row's `elapsed_seconds`, and per-family throughput as the family's
completion tokens over the sum of its rows' elapsed seconds (valid when one request is in flight).
