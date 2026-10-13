# KL / fidelity tools

Path-free copies of the tools behind the published fidelity numbers (`evidence/fidelity-*`). They need the **bf16
source checkpoint** (about 70 GB, not shipped) and a GPU; the measured results are shipped as evidence, so running
these is optional.

| tool | what |
|---|---|
| `make_kl_ids.py` | token-id sets (rows of 1,024 tokens) from text files you provide |
| `bf16_ref_logits.py` | the Hugging Face bf16 reference (layer-streamed, text path) for an id set |
| `fork_logits.py` | logits of a pack, or of the bf16 checkpoint ("floor"), through the ExLlamaV3 forward |
| `kl_compare.py` | KL(ref ‖ q) and KL(q ‖ ref), top-k agreement, perplexity, per kind / row / position; `--json` |
| `kl_metrics.py`, `common.py` | shared metrics and set formats |

Environment: `MINICPM_BF16_DIR` (bf16 checkpoint), `EXLLAMAV3_DIR` (ExLlamaV3 checkout, if not installed in the venv).

```bash
python eval/kl/make_kl_ids.py --tokenizer "$MINICPM_BF16_DIR" --out ids --name mySet --kind python='/usr/lib/python3.12/*.py' --kind prose='docs/*.md'
python eval/kl/bf16_ref_logits.py --ids ids/mySet.safetensors --out-dir ref   # HF bf16 reference
python eval/kl/fork_logits.py --model models/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw --ids ids/mySet.safetensors --out-dir cand --store full
python eval/kl/kl_compare.py --ids ids/mySet.safetensors --ref ref --cand pack=cand --json kl.json
```

The id sets of the published numbers (Set A: 14 x 1,024 tokens, Set B: 100 x 1,024) are described in
`evidence/fidelity-*/kl.json` (sha256 of each set, row kinds); the sets themselves contain excerpts of non-public
sources and are not shipped. A set built here will give numbers of the same kind, not the same digits.
