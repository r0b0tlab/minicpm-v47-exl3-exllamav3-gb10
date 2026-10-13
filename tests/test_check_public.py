import os
import shutil
import subprocess
import tempfile
import unittest

from tests import helpers

ROOT = helpers.ROOT
SCRIPT = os.path.join(ROOT, "scripts", "check_public.sh")


def run(directory):
    p = subprocess.run(["bash", SCRIPT, directory], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    return p.returncode, p.stdout


# The forbidden words are assembled here at run time so this file itself stays clean.
BAD = {
    "engine": "re" + "xl3 notes",
    "host": "built on no" + "de2 yesterday",
    "path": "see /ho" + "me/someone/models",
    "user": "r0b0t" + "dgx",
    "ip": "19" + "2.168.1.20",
    "mail": "write to some" + "one@exam" + "ple.com",
    "crate": "gb" + "-model crate",
}


class CheckPublicTest(unittest.TestCase):
    def test_clean_tree_passes(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "a.md"), "w") as fh:
                fh.write("A perfectly ordinary document about EXL3 on a GB10. node.js is fine.\n")
            rc, out = run(td)
            self.assertEqual(rc, 0, out)

    def test_each_forbidden_class_fails(self):
        for name, text in BAD.items():
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "doc.txt"), "w") as fh:
                    fh.write("line one\n" + text + "\n")
                rc, out = run(td)
                self.assertNotEqual(rc, 0, name)
                self.assertIn("PUBLIC-CHECK HIT", out, name)

    def test_case_insensitive_and_binary_files_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "x.py"), "w") as fh:
                fh.write("# " + "RE" + "xL3\n")
            self.assertNotEqual(run(td)[0], 0)
            os.remove(os.path.join(td, "x.py"))
            with open(os.path.join(td, "blob.bin"), "wb") as fh:
                fh.write(b"\x00\x01" + ("re" + "xl3").encode() + b"\x00")
            self.assertEqual(run(td)[0], 0)

    def test_repository_is_clean(self):
        rc, out = run(ROOT)
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
