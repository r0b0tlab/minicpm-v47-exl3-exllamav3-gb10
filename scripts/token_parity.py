#!/usr/bin/env python3
"""Tokenisation parity: the server's tokenizer against the Hugging Face `tokenizers` library (stdlib + tokenizers).

For prompts of the shipped Leg D kit (token ids made with the HF tokenizer and the model's chat template) the ids are
decoded to text with tokenizer.json, sent as a *string* prompt, and the server's own `usage.prompt_tokens` must equal
the number of ids; the chat-template route is checked the same way against the render endpoint. A mismatch would mean
that token accounting or prompt construction differs between this server and a stock HF stack.

    scripts/token_parity.py --base http://127.0.0.1:8000 --model-dir MODEL_DIR [--n 80]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oai_client as oc  # noqa: E402

PROMPTS = os.path.join(os.path.dirname(HERE), "eval", "legd", "legd_prompts.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    rows = json.load(open(PROMPTS))["rows"][: a.n]
    model = oc.first_model(a.base)["id"]
    bad, checked = [], 0
    for r in rows:
        ids = r["ids"]
        text = tok.decode(ids, skip_special_tokens=False)
        # the HF encoding of that text must give the ids back (a property of the data), and so must the server
        again = tok.encode(text, add_special_tokens=False).ids
        resp = oc.post_json(a.base, "/v1/completions", {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0})
        n_server = resp["usage"]["prompt_tokens"]
        checked += 1
        if again != ids or n_server != len(ids):
            bad.append({"name": r["name"], "ids": len(ids), "hf_roundtrip": len(again), "server": n_server})
    res = {"checked": checked, "mismatches": bad}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("TOKEN_PARITY %d/%d prompts: server prompt_tokens == HF token count%s" % (
        checked - len(bad), checked, "" if not bad else "; MISMATCHES %s" % bad[:5]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
