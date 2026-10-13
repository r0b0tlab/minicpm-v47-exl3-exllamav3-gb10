import os
import re
import subprocess
import sys
import unittest

from tests import helpers
import errcode

ROOT = helpers.ROOT


class ErrcodeTest(unittest.TestCase):
    def test_table_is_well_formed_and_codes_unique(self):
        rows = errcode.load()
        self.assertGreater(len(rows), 20)
        for code, r in rows.items():
            self.assertRegex(code, r"^E_[A-Z0-9_]+$")
            self.assertTrue(r["cause"] and r["remedy"], code)

    def test_every_code_used_in_scripts_is_defined(self):
        defined = set(errcode.load())
        used = set()
        for base, _d, files in os.walk(os.path.join(ROOT, "scripts")):
            for f in files:
                if f.endswith((".sh", ".py")) and f not in ("errcode.py",):
                    used |= set(re.findall(r"\b(E_[A-Z0-9_]{2,})\b", open(os.path.join(base, f)).read()))
        self.assertEqual(sorted(used - defined), [])

    def test_failure_block_has_stage_cause_and_fix(self):
        text = errcode.failure_block("E_NVCC", "preflight", "nvcc not found")
        for needle in ("stage=preflight", "code=E_NVCC", "cause", "fix", "TROUBLESHOOTING.md#e_nvcc"):
            self.assertIn(needle, text)

    def test_troubleshooting_table_has_every_code(self):
        text = open(os.path.join(ROOT, "docs", "TROUBLESHOOTING.md")).read()
        for code in errcode.load():
            self.assertIn("`%s`" % code, text)


class ClickRunTest(unittest.TestCase):
    def run_sh(self, *args, env=None):
        e = dict(os.environ)
        e.update(env or {})
        e["STATE"] = e.get("STATE") or os.path.join(ROOT, ".state-test")
        return subprocess.run(["bash", os.path.join(ROOT, "run.sh")] + list(args), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, env=e, cwd=ROOT)

    def test_scripts_are_syntactically_valid(self):
        for rel in ("run.sh", "scripts/click_run.sh", "scripts/env_report.sh", "scripts/check_public.sh",
                    "scripts/bootstrap.sh", "eval/run_suite.sh", "eval/fetch_kit.sh"):
            p = subprocess.run(["bash", "-n", os.path.join(ROOT, rel)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
            self.assertEqual(p.returncode, 0, rel + p.stdout)

    def test_plan_prints_every_stage_without_side_effects(self):
        p = self.run_sh("--plan", "--profile", "gb10-text-32k", "--port", "8123",
                        env={"STATE": "/tmp/mc-plan-state", "VENV": "/tmp/mc-plan-venv"})
        self.assertEqual(p.returncode, 0, p.stdout)
        for st in ("preflight", "venv", "torch", "build", "download", "verify-model", "retire-cache", "serve", "smoke"):
            self.assertRegex(p.stdout, r"\n  %s\s+(RUN|SKIP)" % st)
        self.assertIn("gb10-text-32k", p.stdout)
        self.assertFalse(os.path.exists("/tmp/mc-plan-venv"))

    def test_dry_run_env_and_from_only_explain(self):
        p = self.run_sh(env={"DRY_RUN": "1", "STATE": "/tmp/mc-plan-state2", "VENV": "/tmp/mc-plan-venv2"})
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("no receipt", p.stdout)
        p = self.run_sh("--plan", "--only", "download", env={"STATE": "/tmp/mc-plan-state2"})
        self.assertRegex(p.stdout, r"download\s+RUN\s+--only download")
        self.assertRegex(p.stdout, r"venv\s+SKIP\s+--only download")
        p = self.run_sh("--plan", "--from", "serve", env={"STATE": "/tmp/mc-plan-state2"})
        self.assertRegex(p.stdout, r"build\s+SKIP\s+before --from serve")

    def test_unknown_stage_and_profile_are_usage_errors(self):
        p = self.run_sh("--plan", "--only", "nonsense")
        self.assertEqual(p.returncode, 64)
        self.assertIn("E_USAGE", p.stdout)
        p = self.run_sh("--plan", "--profile", "does-not-exist")
        self.assertEqual(p.returncode, 64)

    def test_no_model_constants_in_the_driver(self):
        text = open(os.path.join(ROOT, "scripts", "click_run.sh")).read()
        for forbidden in ("MiniCPM", "minicpm-v-4.7", "262144", "248144", "20480", "8f9f0787"):
            self.assertNotIn(forbidden, text, forbidden)


if __name__ == "__main__":
    unittest.main()
