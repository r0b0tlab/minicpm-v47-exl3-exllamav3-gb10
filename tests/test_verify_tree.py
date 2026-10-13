import json
import os
import shutil
import tempfile
import unittest

from tests import helpers
import verify_tree

ROOT = helpers.ROOT


def copy_tree():
    td = tempfile.mkdtemp()
    dst = os.path.join(td, "repo")
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", ".venv", ".state", ".cache", "models", ".kit",
                                                             "__pycache__"))
    return td, dst


class VerifyTreeTest(unittest.TestCase):
    def test_this_repository_conforms(self):
        rep = verify_tree.run(ROOT, skip_public=False)
        self.assertEqual(rep.findings, [])

    def setUp(self):
        self.td, self.root = copy_tree()
        self.addCleanup(shutil.rmtree, self.td, True)

    def classes(self, **kw):
        return verify_tree.run(self.root, **kw).classes()

    def test_missing_required_path_is_class_10(self):
        os.remove(os.path.join(self.root, "docs", "MEMORY.md"))
        self.assertIn(10, self.classes(skip_public=True))

    def test_manifest_mismatch_is_class_20(self):
        d = os.path.join(self.root, "evidence", "fidelity-20261013")
        with open(os.path.join(d, "kl.json"), "a") as fh:
            fh.write(" ")
        self.assertIn(20, self.classes(skip_public=True))

    def test_unlisted_file_in_bundle_is_class_20(self):
        open(os.path.join(self.root, "evidence", "fidelity-20261013", "extra.json"), "w").write("{}")
        self.assertIn(20, self.classes(skip_public=True))

    def test_pin_disagreement_is_class_30(self):
        p = os.path.join(self.root, "runtime.lock.json")
        lock = json.load(open(p))
        lock["models"]["target"]["revision"] = "0" * 40
        json.dump(lock, open(p, "w"))
        self.assertIn(30, self.classes(skip_public=True))

    def test_readme_quoting_a_wrong_commit_is_class_30(self):
        p = os.path.join(self.root, "README.md")
        with open(p, "a") as fh:
            fh.write("\nbuilt from commit `deadbeefcafe`\n")
        self.assertIn(30, self.classes(skip_public=True))

    def test_forbidden_string_is_class_40(self):
        with open(os.path.join(self.root, "docs", "TROUBLESHOOTING.md"), "a") as fh:
            fh.write("\n" + "re" + "xl3\n")
        self.assertIn(40, self.classes())

    def test_large_file_is_class_50(self):
        with open(os.path.join(self.root, "docs", "big.bin"), "wb") as fh:
            fh.write(b"0" * 1_100_000)
        self.assertIn(50, self.classes(skip_public=True))

    def test_generated_block_drift_is_class_60(self):
        p = os.path.join(self.root, "README.md")
        text = open(p).read()
        marker = "<!-- BEGIN GENERATED:results-summary -->\n"
        self.assertIn(marker, text)
        open(p, "w").write(text.replace(marker, marker + "| tampered | 1 |\n", 1))
        self.assertIn(60, self.classes(skip_public=True))

    def test_undefined_error_code_is_class_70(self):
        with open(os.path.join(self.root, "scripts", "extra.sh"), "w") as fh:
            fh.write("echo E_NOT_A_REAL_CODE\n")
        self.assertIn(70, self.classes(skip_public=True))

    def test_exit_status_is_lowest_class(self):
        os.remove(os.path.join(self.root, "docs", "MEMORY.md"))
        with open(os.path.join(self.root, "docs", "big.bin"), "wb") as fh:
            fh.write(b"0" * 1_100_000)
        self.assertEqual(verify_tree.main(["--root", self.root, "--skip-public"]), 10)


if __name__ == "__main__":
    unittest.main()
