"""eval/run_suite.sh end to end against a scripted server: lanes -> raw outputs -> bundle -> manifest -> tree check."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from tests import helpers
import serve_openai as so
import verify_tree

ROOT = helpers.ROOT


class RunSuiteTest(unittest.TestCase):
    def test_bench_niah_legd_produce_a_valid_bundle(self):
        eng = helpers.FakeEngine(lambda req: (["12", "34"], "stop"), window=4096)
        srv, base = helpers.start_http(so.Handler, eng)
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        self.addCleanup(srv.shutdown)
        root = os.path.join(td, "repo")
        shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", ".venv", ".state", ".kit", "models",
                                                                  "__pycache__", ".state-test"))
        p = subprocess.run(
            ["bash", os.path.join(root, "eval", "run_suite.sh"), base, "unit-test", "--lanes", "bench,niah,legd",
             "--depths", "1024", "--date", "2026-01-02"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=root)
        self.assertEqual(p.returncode, 0, p.stdout)
        bundle_dir = os.path.join(root, "evidence", "unit-test-2026-01-02")
        b = json.load(open(os.path.join(bundle_dir, "bundle.json")))
        self.assertEqual(b["schema"], "r0b0tlab.evidence-bundle.v1")
        self.assertEqual(b["column"], "exllamav3-plain")
        self.assertTrue(b["reproducible_with_this_repo"])
        lock = json.load(open(os.path.join(ROOT, "runtime.lock.json")))
        self.assertEqual(b["engine"]["commit"], lock["exllamav3"]["commit"])
        self.assertIn("1024", b["results"]["bench"]["depths"])
        self.assertIn("legd", b["results"])
        self.assertIn("niah", b["results"])
        self.assertEqual(b["results"]["niah"]["asked"], 0)      # window 4096: every question skipped, recorded
        # manifest covers every file and the tree checker accepts the bundle
        rep = verify_tree.Report()
        verify_tree.check_bundle_dir(root, "evidence/unit-test-2026-01-02", rep)
        self.assertEqual(rep.findings, [])
        manifest = open(os.path.join(bundle_dir, "MANIFEST.sha256")).read()
        for name in ("bundle.json", "bench.json", "niah.json", "legd.rows.jsonl"):
            self.assertIn("  " + name, manifest)


if __name__ == "__main__":
    unittest.main()
