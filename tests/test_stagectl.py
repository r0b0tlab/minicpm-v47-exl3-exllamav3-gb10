import json
import os
import shutil
import tempfile
import unittest

from tests import helpers
import stagectl

ROOT = helpers.ROOT


def make_ctx(tmp, **env):
    e = {"ROOT": ROOT, "STATE": os.path.join(tmp, "state"), "VENV": os.path.join(tmp, "venv"),
         "MODEL_DIR": os.path.join(tmp, "model"), "CACHE_DIR": os.path.join(tmp, "cache"), "PROFILE": "gb10-text-262k",
         "MC_PYTHON_VERSION": "3.12.3"}
    e.update(env)
    return stagectl.Ctx(e)


class StageCtlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_fresh_state_runs_everything(self):
        ctx = make_ctx(self.tmp)
        plan = stagectl.plan(ctx)
        self.assertEqual([r["stage"] for r in plan], stagectl.STAGES)
        self.assertTrue(all(r["action"] == "RUN" for r in plan))

    def test_completed_stage_is_skipped_until_inputs_change(self):
        ctx = make_ctx(self.tmp)
        os.makedirs(os.path.join(ctx.venv, "bin"))
        open(os.path.join(ctx.venv, "bin", "python"), "w").close()
        stagectl.record(ctx, "venv", "ok", 2.0)
        a, why = stagectl.decide(ctx, "venv")
        self.assertEqual(a, "SKIP", why)
        # outputs removed -> invalid
        os.remove(os.path.join(ctx.venv, "bin", "python"))
        a, why = stagectl.decide(ctx, "venv")
        self.assertEqual((a, "outputs invalid" in why), ("RUN", True))

    def test_failed_receipt_runs_again_and_input_change_is_explained(self):
        ctx = make_ctx(self.tmp)
        stagectl.record(ctx, "torch", "failed", 1.0, detail="E_PIP_TORCH")
        a, why = stagectl.decide(ctx, "torch")
        self.assertEqual((a, why), ("RUN", "previous attempt failed"))
        stagectl.record(ctx, "build", "ok", 3.0)
        ctx2 = make_ctx(self.tmp, MC_PYTHON_VERSION="3.11.9")
        # python version feeds the build key and the build inputs
        a, why = stagectl.decide(ctx2, "build")
        self.assertEqual(a, "RUN")
        self.assertIn("inputs changed", why)

    def test_from_only_force_and_flags(self):
        ctx = make_ctx(self.tmp)
        self.assertEqual(stagectl.decide(ctx, "venv", only="download"), ("SKIP", "--only download"))
        self.assertEqual(stagectl.decide(ctx, "download", only="download")[0], "RUN")
        self.assertEqual(stagectl.decide(ctx, "venv", from_stage="download")[0], "SKIP")
        self.assertEqual(stagectl.decide(ctx, "download", from_stage="download")[0], "RUN")
        for flag, stage in (("NO_DOWNLOAD", "download"), ("SKIP_BUILD", "build"), ("SERVE_ONLY", "torch"),
                            ("SKIP_VERIFY", "verify-model")):
            c = make_ctx(self.tmp, **{flag: "1"})
            a, why = stagectl.decide(c, stage)
            self.assertEqual(a, "SKIP", (flag, why))
            self.assertIn(flag, why)
        with self.assertRaises(ValueError):
            stagectl.decide(ctx, "nope")

    def test_volatile_stages_always_run(self):
        ctx = make_ctx(self.tmp)
        for st in ("preflight", "retire-cache", "serve", "smoke"):
            stagectl.record(ctx, st, "ok", 1.0)
            self.assertEqual(stagectl.decide(ctx, st)[0], "RUN")

    def test_receipt_contents(self):
        ctx = make_ctx(self.tmp)
        doc = stagectl.record(ctx, "build", "ok", 126.04, detail="compiled", extra={"compile_seconds": 120})
        on_disk = json.load(open(os.path.join(ctx.state, "build.json")))
        self.assertEqual(doc, on_disk)
        for key in ("inputs", "inputs_hash", "versions", "duration_s", "finished_utc", "status", "stage", "extra"):
            self.assertIn(key, on_disk)
        self.assertEqual(on_disk["duration_s"], 126.0)
        self.assertEqual(on_disk["versions"]["exllamav3_commit"], ctx.exl_commit())

    def test_build_key_has_commit_torch_python_arch(self):
        ctx = make_ctx(self.tmp)
        key = ctx.build_key()
        self.assertTrue(key.startswith(ctx.exl_commit()[:12]))
        self.assertIn("torch2.14.0cu130", key)
        self.assertIn("py3.12", key)
        self.assertTrue(key.endswith("sm121"))

    def test_download_valid_checks_sizes(self):
        ctx = make_ctx(self.tmp)
        ok, why = stagectl.valid_for(ctx, "download")
        self.assertFalse(ok)
        self.assertIn("missing", why)

    def test_disk_needed_shrinks_with_progress(self):
        ctx = make_ctx(self.tmp)
        fresh = stagectl.disk_needed_gib(ctx)
        self.assertGreaterEqual(fresh, 60)
        ctx2 = make_ctx(self.tmp, NO_DOWNLOAD="1", SKIP_BUILD="1")
        os.makedirs(os.path.join(ctx2.venv, "bin"))
        open(os.path.join(ctx2.venv, "bin", "python"), "w").close()
        self.assertEqual(stagectl.disk_needed_gib(ctx2), 0)

    def test_artifact_override_makes_pack_unpinned(self):
        ctx = make_ctx(self.tmp, HF_REPO="someone/else")
        art = ctx.artifact_for("target")
        self.assertTrue(art.get("unpinned"))
        self.assertEqual(art["files"], {})


if __name__ == "__main__":
    unittest.main()
