# Third-party notices

This repository's own code is MIT licensed (`LICENSE`). It builds on, downloads or references the following
projects. No upstream source code is redistributed here except where stated; the runtime is built and installed
from the pinned revisions in `runtime.lock.json` and `requirements.txt` on the user's machine.

## ExLlamaV3 (build dependency and runtime)

- Upstream: https://github.com/turboderp-org/exllamav3 (turboderp), MIT, Copyright (c) 2025 Turboderp
- Fork used: https://github.com/r0b0tlab/exllamav3-r0b0t, branch `minicpm-v4.7-dspark`, commit
  `5ddae7d124482aa9e705c3a4aedf02cf8d09911b`, on top of branch `minicpm-v4.7` commit
  `4d083d25c1a3e195ce1305623ffa1eef84b834fe` (ExLlamaV3 v1.5.0 base, aarch64 / sm_121 build fixes, the
  MiniCPM-V-4.7 text architecture, a DSpark draft quantisation shim; the dspark branch adds DSpark block
  drafting, whose semantics follow the Apache-2.0 vLLM reference implementation). MIT.
- `scripts/serve_openai.py` imports it; `scripts/click_run.sh` clones and compiles it.

## Model: MiniCPM-V-4.7-35B-A3B

- Upstream model: https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B (OpenBMB), Apache License 2.0.
- Quantised derivative served here: https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw,
  Apache License 2.0 (inherited); its `NOTICE` states the modifications.
- Companion DSpark draft: https://huggingface.co/r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw (Apache-2.0),
  derived from `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`. Downloaded only with `--with-draft`.
- `tests/fixtures/chat_template.jinja` is an unmodified copy of the model's chat template (Apache-2.0), used to
  check template rendering offline.
- The language backbone follows the Qwen3.5/3.6 mixture-of-experts architecture (Qwen team, Apache-2.0).

## Python packages installed into the venv (not redistributed)

| package | licence |
|---|---|
| PyTorch (`torch` 2.14.0+cu130) | BSD-3-Clause |
| NumPy, Jinja2, tokenizers, safetensors, Pillow, PyYAML, pydantic, rich, marisa-trie, llguidance, huggingface_hub | BSD / MIT / Apache-2.0 (see each project) |

NVIDIA CUDA components (nvcc, runtime libraries from the `nvidia-*` wheels) are used under NVIDIA's licence terms.

## Evaluation material

- The Q200v2 kit (https://github.com/r0b0tlab/r0b0bench, MIT) is fetched by commit and verified by hash at run
  time; none of its datasets or code is vendored. `eval/shims/` are small stand-ins for two modules the public kit
  imports but does not ship.
- `eval/legd/legd_prompts.json` holds token ids of 80 prompts drawn from that kit's text set.
- `eval/niah/niah_261k.json.gz` holds token ids (no text) of a document built from the docstrings and pydoc topics
  of the Python 3.12 standard library (PSF licence) and of the pinned PyTorch package (BSD-3-Clause).

