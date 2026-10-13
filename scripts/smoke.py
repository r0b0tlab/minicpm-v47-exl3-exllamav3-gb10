#!/usr/bin/env python3
"""Three-request smoke test against a running server (stdlib only).

  1. text, thinking off: arithmetic answer in content, no reasoning_content
  2. thinking on (the default): reasoning_content present, final answer in content,
     finish_reason stop; the same request streamed must reproduce the same text
  3. long-ish prompt: a passcode planted in about 3,000 tokens of filler is retrieved,
     usage.prompt_tokens equals the server's own render of the prompt

Exit status 0 only if every check passes. Writes a JSON summary with --out.
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oai_client as oc  # noqa: E402


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" - " + detail) if detail else ""))
    return bool(ok)


def run(base, model):
    results, ok = {}, True
    card = oc.first_model(base)
    model = model or card["id"]
    ok &= check("GET /v1/models", card.get("max_model_len", 0) >= 4096, "id=%s max_model_len=%s" % (card["id"], card.get("max_model_len")))
    results["max_model_len"] = card.get("max_model_len")

    # 1. text, thinking off
    r = oc.post_json(base, "/v1/chat/completions", {
        "model": model, "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is six times seven? Reply with the number only."}],
        "chat_template_kwargs": {"enable_thinking": False}})
    msg = r["choices"][0]["message"]
    ok &= check("text (thinking off)", "42" in (msg.get("content") or "") and r["choices"][0]["finish_reason"] == "stop"
                and not msg.get("reasoning_content"), repr((msg.get("content") or "")[:60]))
    results["text_tok_s"] = r.get("timings", {}).get("decode_tok_s")

    # 2. thinking on (default)
    payload = {"model": model, "temperature": 0, "max_tokens": 1500,
               "messages": [{"role": "user", "content": "What is 17 + 25? Show your reasoning, then give the final answer."}]}
    r = oc.post_json(base, "/v1/chat/completions", payload)
    msg = r["choices"][0]["message"]
    has_r = bool(msg.get("reasoning_content"))
    ok &= check("thinking on (default)", has_r and "42" in (msg.get("content") or "") and
                r["choices"][0]["finish_reason"] == "stop",
                "reasoning %d chars, content %r" % (len(msg.get("reasoning_content") or ""), (msg.get("content") or "")[:40]))
    got_r, got_c = [], []
    for _t, ch in oc.post_stream(base, "/v1/chat/completions", payload):
        for c in ch.get("choices", []):
            d = c.get("delta", {})
            if d.get("reasoning_content"):
                got_r.append(d["reasoning_content"])
            if d.get("content"):
                got_c.append(d["content"])
    ok &= check("streamed == non-streamed (greedy)", "".join(got_r) == (msg.get("reasoning_content") or "")
                and "".join(got_c) == (msg.get("content") or ""))
    # alias key
    r2 = oc.post_json(base, "/v1/chat/completions", {
        "model": model, "temperature": 0, "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say OK."}],
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low", "some_unknown_kwarg": 1}})
    ok &= check("alias key 'thinking' + unknown kwargs accepted",
                not r2["choices"][0]["message"].get("reasoning_content"))

    # 3. long-ish prompt
    filler = ("The harbour clock struck nine while the ferry loaded crates of oranges and lamp oil. " * 190)
    needle = "The vault passcode for the codeword saffron-heron is 4817263. "
    doc = filler[: len(filler) // 2] + needle + filler[len(filler) // 2:]
    body = {"model": model, "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "user", "content": doc + "\n\nWhat is the vault passcode for the codeword saffron-heron? "
                          "Answer with the number only."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    rr = oc.post_json(base, "/v1/chat/completions", body)
    render = oc.post_json(base, "/v1/chat/completions/render", body)
    content = rr["choices"][0]["message"].get("content") or ""
    ok &= check("long prompt retrieval (%d tokens)" % rr["usage"]["prompt_tokens"], "4817263" in content,
                "answer %r in %.1fs" % (content[:30], time.time() - t0))
    ok &= check("usage.prompt_tokens == render", rr["usage"]["prompt_tokens"] == render["prompt_tokens"],
                "%d vs %d" % (rr["usage"]["prompt_tokens"], render["prompt_tokens"]))
    results["long_prompt_tokens"] = rr["usage"]["prompt_tokens"]

    # error isolation: a vision part is a clean 400 and the server lives
    try:
        oc.post_json(base, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
        ok &= check("image part rejected with 400", False, "no error")
    except oc.ApiError as exc:
        ok &= check("image part rejected with 400", exc.status == 400 and "vision" in json.dumps(exc.body).lower(),
                    str(exc)[:100])
    ok &= check("server alive after error", oc.get_json(base, "/health").get("status") == "ok")
    return ok, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out")
    ap.add_argument("--wait", type=int, default=0, help="wait this many seconds for /health first")
    a = ap.parse_args()
    if a.wait:
        oc.wait_ready(a.base, a.wait)
    ok, results = run(a.base, a.model)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"ok": ok, "results": results, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=1)
    print("SMOKE_OK" if ok else "SMOKE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
