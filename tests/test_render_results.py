import json
import os
import shutil
import tempfile
import unittest

from tests import helpers
import render_results as rr

ROOT = helpers.ROOT


class RenderResultsTest(unittest.TestCase):
    def test_documents_match_bundles(self):
        bundles = rr.load_bundles(ROOT)
        proto = rr.load_protocol(ROOT)
        for rel, names in proto["generated_blocks"].items():
            found, drift = rr.process(os.path.join(ROOT, rel), bundles, write=False)
            self.assertEqual(sorted(found), sorted(names), rel)
            self.assertEqual(drift, [], rel)

    def test_all_blocks_render(self):
        bundles = rr.load_bundles(ROOT)
        for name in rr.BLOCKS:
            self.assertTrue(rr.render_block(name, bundles).strip(), name)

    def test_summary_has_three_labelled_columns_and_pending_placeholder(self):
        text = rr.render_block("results-summary", rr.load_bundles(ROOT))
        header = text.splitlines()[0]
        for col in ("ExLlamaV3, plain decode", "ExLlamaV3 + DSpark drafting", "Reference engine"):
            self.assertIn(col, header)
        self.assertIn("not reproducible here", header)

    def test_write_then_check_roundtrip_and_detects_edits(self):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        root = os.path.join(td, "r")
        shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", ".venv", ".state", ".kit", "models", "__pycache__"))
        p = os.path.join(root, "docs", "RESULTS.md")
        text = open(p).read()
        open(p, "w").write(text.replace("<!-- BEGIN GENERATED:niah-table -->\n", "<!-- BEGIN GENERATED:niah-table -->\nSTALE\n"))
        self.assertEqual(rr.main(["--root", root, "--check"]), 1)
        self.assertEqual(rr.main(["--root", root, "--write"]), 0)
        self.assertEqual(rr.main(["--root", root, "--check"]), 0)

    def test_a_drafted_bundle_fills_the_pending_column(self):
        b = rr.load_bundles(ROOT)
        before = rr.render_block("results-summary", b)
        self.assertIn("pending", before)
        b["exllamav3-dspark"] = {
            "column": "exllamav3-dspark", "date": "2026-11-01", "results": {
                "q200": {"passed": 170, "n": 180, "families": {f: {"passed": 1, "n": 2} for f in
                                                              ("gsm8k", "humaneval", "ifeval", "hard_reasoning")},
                         "e2e_tok_s": 120.5, "per_request_tok_s_median": 100.0, "completion_tokens": 1,
                         "wall_seconds": 1.0, "finish_reasons": {"stop": 180},
                         "per_family_throughput": {f: {"tok_s_serial": 1.0} for f in
                                                   ("gsm8k", "humaneval", "ifeval", "hard_reasoning")}},
                "legd": {"greedy/gsm8k": {"tok_s_mean": 210.0, "streams": 1}},
                "memory": {"torch_allocated_gib": 31.0}}}
        after = rr.render_block("results-summary", b)
        self.assertIn("170/180", after)
        self.assertIn("210.0", after)
        self.assertNotIn("_ExLlamaV3 + DSpark: pending", after)

    def test_number_change_in_a_bundle_changes_the_table(self):
        b = rr.load_bundles(ROOT)
        before = rr.render_block("q200-table", b)
        b["reference-engine"]["results"]["q200"]["families"]["gsm8k"]["passed"] = 1
        self.assertNotEqual(rr.render_block("q200-table", b), before)


if __name__ == "__main__":
    unittest.main()
