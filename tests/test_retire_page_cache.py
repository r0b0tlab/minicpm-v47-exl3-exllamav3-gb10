import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import helpers  # noqa: F401
import retire_page_cache as rpc

MIB = 1048576


class RetirePageCacheTest(unittest.TestCase):
    def test_only_shard_files_are_advised(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "m-1.safetensors").write_bytes(b"a" * 8192)
            (root / "config.json").write_bytes(b"{}")
            with mock.patch.object(rpc, "resident_bytes", side_effect=[4096, 0]), \
                 mock.patch.object(rpc.os, "posix_fadvise") as advise:
                res = rpc.retire(root)
            self.assertEqual((res["files"], res["before_bytes"], res["after_bytes"]), (1, 4096, 0))
            advise.assert_called_once()
            self.assertEqual((root / "m-1.safetensors").read_bytes(), b"a" * 8192)

    def test_rejects_symlinks_empty_dirs_and_wrong_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "no \\*.safetensors"):
                rpc.retire(root)
            (root / "one.safetensors").write_bytes(b"a")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                rpc.retire(root, expected=2)
            (root / "alias.safetensors").symlink_to(root / "one.safetensors")
            with mock.patch.object(rpc.os, "posix_fadvise") as advise:
                with self.assertRaisesRegex(ValueError, "unsafe shard"):
                    rpc.retire(root)
                advise.assert_not_called()

    def test_admission_is_fail_closed(self):
        self.assertEqual(rpc.admit_budget(100 * MIB, 0, 5000)["required_memfree_mib"], 4196)
        with self.assertRaisesRegex(ValueError, "cache residency"):
            rpc.admit_budget(100 * MIB, 65 * MIB, 5000)
        with self.assertRaisesRegex(ValueError, "MemFree"):
            rpc.admit_budget(100 * MIB, 0, 4095)

    def test_mincore_reports_bounded_residency(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.safetensors"
            p.write_bytes(b"a" * 8192)
            self.assertTrue(0 <= rpc.resident_bytes(p) <= 8192)


if __name__ == "__main__":
    unittest.main()
