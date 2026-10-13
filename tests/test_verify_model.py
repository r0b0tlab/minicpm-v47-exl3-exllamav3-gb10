import hashlib
import os
import shutil
import tempfile
import unittest

from tests import helpers  # noqa: F401
import verify_model


class VerifyModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.files = {"a.bin": b"alpha" * 100, "sub/b.json": b"{}", "c.safetensors": b"x" * 5000}
        self.entries = []
        for rel, data in self.files.items():
            p = os.path.join(self.tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
            self.entries.append((rel, len(data), hashlib.sha256(data).hexdigest()))
        self.cache = os.path.join(self.tmp, "..", os.path.basename(self.tmp) + "-cache.json")
        self.addCleanup(lambda: os.path.exists(self.cache) and os.remove(self.cache))

    def test_ok_then_cached(self):
        r = verify_model.verify(self.tmp, self.entries, cache_path=self.cache)
        self.assertTrue(r["ok"])
        self.assertEqual(r["cached_files"], 0)
        r = verify_model.verify(self.tmp, self.entries, cache_path=self.cache)
        self.assertTrue(r["ok"])
        self.assertEqual(r["cached_files"], 3)
        self.assertEqual(r["hashed_bytes"], 0)
        r = verify_model.verify(self.tmp, self.entries, cache_path=self.cache, full=True)
        self.assertEqual(r["cached_files"], 0)

    def test_size_and_hash_mismatch_and_missing(self):
        with open(os.path.join(self.tmp, "a.bin"), "ab") as fh:
            fh.write(b"!")
        # same size, different bytes
        p = os.path.join(self.tmp, "c.safetensors")
        with open(p, "r+b") as fh:
            fh.write(b"y")
        os.remove(os.path.join(self.tmp, "sub", "b.json"))
        r = verify_model.verify(self.tmp, self.entries)
        self.assertFalse(r["ok"])
        by = {x["path"]: x["status"] for x in r["results"]}
        self.assertEqual(by, {"a.bin": "size", "c.safetensors": "sha256", "sub/b.json": "missing"})

    def test_cache_does_not_hide_a_modified_file(self):
        verify_model.verify(self.tmp, self.entries, cache_path=self.cache)
        p = os.path.join(self.tmp, "c.safetensors")
        with open(p, "r+b") as fh:
            fh.write(b"z")      # same size, new mtime
        os.utime(p, (1, 1))
        r = verify_model.verify(self.tmp, self.entries, cache_path=self.cache)
        self.assertFalse(r["ok"])

    def test_upload_manifest_reader(self):
        import json
        m = os.path.join(self.tmp, "UPLOAD_MANIFEST.json")
        json.dump({"files": [{"path": "a.bin", "size": 500, "sha256": "ab"}]}, open(m, "w"))
        self.assertEqual(verify_model.manifest_from_upload_manifest(m), [("a.bin", 500, "ab")])


if __name__ == "__main__":
    unittest.main()
