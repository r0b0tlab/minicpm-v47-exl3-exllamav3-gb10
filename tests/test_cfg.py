import json
import os
import unittest

from tests import helpers
import cfg

ROOT = helpers.ROOT
GIB = 1024 ** 3


class CfgTest(unittest.TestCase):
    def setUp(self):
        self.model = cfg.load_model(ROOT)
        self.profiles = cfg.load_profiles(ROOT)

    def test_default_profile_and_names(self):
        self.assertEqual(self.profiles["default"], "gb10-text-262k")
        for n in ("gb10-text-262k", "gb10-text-262k-b4", "gb10-text-32k", "gb10-text-16k-lowmem"):
            self.assertIn(n, self.profiles["profiles"])

    def test_memory_budget_262k(self):
        m = self.model["model"]
        prof = cfg.get_profile(self.profiles, "gb10-text-262k")
        b = cfg.memory_budget(m, prof)
        # 20,480 B per token * 262,144 tokens = exactly 5 GiB of fp16 KV
        self.assertAlmostEqual(b["kv_cache_gib"], 5.0, places=6)
        self.assertAlmostEqual(b["weights_gib"], m["weights_bytes"] / GIB, places=6)
        self.assertAlmostEqual(b["recurrent_state_gib"], 64389120 / GIB, places=6)
        self.assertGreater(b["need_memavailable_gib"], b["resident_gib"])

    def test_quantised_kv_is_smaller_and_monotonic(self):
        m = self.model["model"]
        sizes = [cfg.kv_bytes_per_token(m, b) for b in (2, 4, 8, 16)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(cfg.kv_bytes_per_token(m, 16), 20480)

    def test_profiles_validate(self):
        with self.assertRaises(cfg.ConfigError):
            cfg.get_profile(self.profiles, "nope")
        bad = {"profiles": {"x": {"window_tokens": 1000}}}
        with self.assertRaises(cfg.ConfigError):
            cfg.get_profile(bad, "x")

    def test_manifest_matches_declared_totals(self):
        for name in ("target", "draft"):
            art = cfg.artifact(self.model, name)
            total = sum(size for _p, size, _s in cfg.manifest(self.model, name))
            self.assertEqual(total, art["total_bytes"], name)

    def test_weights_bytes_is_sum_of_shards(self):
        shards = sum(s for p, s, _ in cfg.manifest(self.model, "target") if p.endswith(".safetensors"))
        self.assertEqual(shards, self.model["model"]["weights_bytes"])

    def test_runtime_lock_matches_model_yaml(self):
        lock = json.load(open(os.path.join(ROOT, "runtime.lock.json")))
        for name in ("target", "draft"):
            art = cfg.artifact(self.model, name)
            self.assertEqual(lock["models"][name]["hf_repo"], art["hf_repo"])
            self.assertEqual(lock["models"][name]["revision"], art["revision"])


if __name__ == "__main__":
    unittest.main()
