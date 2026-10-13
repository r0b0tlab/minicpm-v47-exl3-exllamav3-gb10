"""Eval lanes (NIAH client, bench grid, Leg D runner) against a scripted OpenAI-compatible endpoint."""
import gzip
import json
import os
import sys
import unittest

from tests import helpers
import exact_niah
import bench

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "eval", "legd"))
import run_legd  # noqa: E402

DOC = os.path.join(ROOT, "eval", "niah", "niah_261k.json.gz")


def completion(text, ids=None, prompt_tokens=10, n=4, cached=0, prefill=0.5, decode=0.4):
    return {"id": "c", "object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": "stop",
                                                                  "token_ids": ids or list(range(n))}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n},
            "timings": {"prefill_s": prefill, "decode_s": decode, "decode_tok_s": (n - 1) / decode,
                        "cached_tokens": cached}}


class ShippedDocTest(unittest.TestCase):
    def test_document_shape(self):
        with gzip.open(DOC, "rt") as fh:
            d = json.load(fh)
        self.assertEqual(d["tokens"], 261001)
        self.assertEqual(len(d["ids"]), 261001)
        self.assertEqual(len(d["needles"]), 6)
        self.assertEqual(len(d["questions"]), 22)
        self.assertEqual(d["cuts"], [4096, 16384, 32000, 64000, 128000, 200000, 261000])
        self.assertLess(max(d["ids"]), 248144)
        for n in d["needles"]:
            self.assertEqual(len(n["value"]), 7)
            self.assertTrue(n["value"].isdigit())
        for q in d["questions"]:
            n = d["needles"][q["needle"]]
            self.assertLess(n["pos"] + 40, q["cut"])        # the needle is wholly inside the prefix
            self.assertEqual(q["answer"], n["value"])


class ShippedDocPrivacyTest(unittest.TestCase):
    def test_document_metadata_has_no_paths(self):
        import re
        with gzip.open(DOC, "rt") as fh:
            d = json.load(fh)
        meta = json.dumps({k: v for k, v in d.items() if k not in ("ids", "questions", "needles")})
        words = ["/ho" + "me/", "/Us" + "ers/", "\\.ve" + "nv", "r0b0" + "t"]
        self.assertIsNone(re.search("|".join(words), meta), meta)


class NiahClientTest(unittest.TestCase):
    def setUp(self):
        self.doc = exact_niah.load_doc(DOC)
        self.questions = self.doc["questions"]

    def handler(self, wrong=()):
        def h(path, body):
            prompt = body["prompt"]
            for q in self.questions:
                tail = q["suffix_ids"]
                if prompt[-len(tail):] == tail and len(prompt) == q["cut"] + len(tail):
                    ans = q["answer"]
                    if (q["cut"], q["needle"]) in wrong:
                        ans = self.doc["needles"][(q["needle"] + 1) % 6]["value"]
                    return completion(ans + "\n", prompt_tokens=len(prompt), cached=max(0, q["cut"] - 2000))
            return completion("???", prompt_tokens=len(prompt))
        return h

    def test_all_exact(self):
        mock = helpers.ScriptedCompletions(self.handler())
        try:
            res = exact_niah.run(mock.base, None, self.doc, cuts=[4096, 32000], log=lambda *_: None)
        finally:
            mock.close()
        s = res["summary"]
        self.assertEqual((s["asked"], s["exact"], s["skipped"]), (3, 3, 0))
        self.assertTrue(s["all_usage_matches"])
        first = [r for r in res["rows"] if r["first_question_at_cut"]]
        self.assertEqual(len(first), 2)
        self.assertEqual(first[-1]["cold_prefill_cumulative_s"], 1.0)

    def test_wrong_answer_names_the_other_needle_and_window_skips(self):
        mock = helpers.ScriptedCompletions(self.handler(wrong={(32000, 1)}),
                                           models={"id": "m", "max_model_len": 40000})
        try:
            res = exact_niah.run(mock.base, None, self.doc, cuts=[32000, 64000], log=lambda *_: None)
        finally:
            mock.close()
        s = res["summary"]
        self.assertEqual((s["asked"], s["exact"], s["skipped"]), (2, 1, 3))
        self.assertEqual(s["wrong"][0]["verdict"], "other_needle")
        self.assertEqual(s["wrong"][0]["other_needle"], 2)
        self.assertEqual(exact_niah.classify("1234567", "7654321", self.doc["needles"])[0], "wrong_digits")
        self.assertEqual(exact_niah.classify("abc", "7654321", self.doc["needles"])[0], "no_digits")


class BenchTest(unittest.TestCase):
    def test_prompts_are_cold_and_exact_length(self):
        ids = list(range(1000, 1000 + 20000))
        seen = set()
        for rep in range(3):
            p = bench.make_prompt(ids, 4096, rep)
            self.assertEqual(len(p), 4096)
            seen.add(tuple(p[:16]))
        self.assertEqual(len(seen), 3)            # distinct salts: no prefix reuse
        self.assertNotEqual(bench.make_prompt(ids, 4096, 0)[16:200], bench.make_prompt(ids, 4096, 1)[16:200])

    def test_run_medians_and_skip(self):
        n = {"i": 0}

        def h(path, body):
            n["i"] += 1
            toks = len(body["prompt"])
            return completion("x", prompt_tokens=toks, n=body.get("max_tokens", 4), prefill=toks / 1000.0,
                              decode=(body.get("max_tokens", 4) - 1) / 50.0, cached=0)
        mock = helpers.ScriptedCompletions(h, models={"id": "m", "max_model_len": 10000, "chunk_size": 2048})
        try:
            res = bench.run(mock.base, None, list(range(1000, 31000)), [1024, 4096, 50000], 3, 16, log=lambda *_: None)
        finally:
            mock.close()
        d0, d1, d2 = res["depths"]
        self.assertEqual(d0["status"], "ok")
        self.assertAlmostEqual(d0["decode_tok_s_median"], 50.0, places=1)
        self.assertAlmostEqual(d0["prefill_tok_s_median"], 1000.0, places=0)
        self.assertTrue(d1["all_cold"])
        self.assertEqual(d2["status"], "skipped")
        self.assertIn("| 4096 |", bench.table(res))


class LegdTest(unittest.TestCase):
    def test_prompt_file_hash_and_plan(self):
        doc = run_legd.load_prompts(run_legd.DEFAULT_PROMPTS)
        self.assertEqual(len(doc["rows"]), 80)
        self.assertEqual(doc["quota"], {"gsm8k": 40, "code": 20, "chat": 20})
        items = run_legd.plan(doc, ["greedy", "sampled"], 10)
        self.assertEqual(sum(1 for i in items if i["mode"] == "greedy"), 80)
        self.assertEqual(sum(1 for i in items if i["mode"] == "sampled"), 30)

    def test_run_and_compare(self):
        doc = run_legd.load_prompts(run_legd.DEFAULT_PROMPTS)
        doc = dict(doc, rows=doc["rows"][:3] + doc["rows"][40:42])

        def h(path, body):
            return completion("t", ids=[1, 2, 3, body["prompt"][0] % 7], n=4)
        mock = helpers.ScriptedCompletions(h)
        try:
            rows, wall = run_legd.run(mock.base, None, doc, ["greedy", "sampled"], 1, 2, "plain", log=lambda *_: None)
            rows4, _ = run_legd.run(mock.base, None, doc, ["greedy"], 4, 2, "plain4", log=lambda *_: None)
        finally:
            mock.close()
        self.assertEqual(len(rows), 5 + 4)
        text, summary = run_legd.report(rows, compare=rows4)
        self.assertIn("greedy/all", summary)
        self.assertEqual(summary["compare"]["differing"], [])
        self.assertIn("| gsm8k |", text)
        sampled = [r for r in rows if r["mode"] == "sampled"]
        self.assertEqual(len(sampled), 4)
        self.assertEqual(rows4[0]["streams"], 4)


if __name__ == "__main__":
    unittest.main()
