import os
import shutil
import subprocess
import tempfile
import unittest

from tests import helpers

ROOT = helpers.ROOT


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=noreply"] + list(args), cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class BootstrapTest(unittest.TestCase):
    def make_origin(self, tamper=False):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        origin = os.path.join(td, "origin")
        shutil.copytree(ROOT, origin, ignore=shutil.ignore_patterns(".git", ".venv", ".state", ".kit", "models",
                                                                    "__pycache__", ".state-test"))
        if tamper:
            with open(os.path.join(origin, "scripts", "click_run.sh"), "a") as fh:
                fh.write("\n# tampered\n")
        git("init", "-q", "-b", "main", cwd=origin)
        git("add", "-A", cwd=origin)
        git("commit", "-q", "-m", "x", cwd=origin)
        git("tag", "v2.0.0", cwd=origin)
        return td, origin

    def run_bootstrap(self, td, origin):
        dest = os.path.join(td, "dest")
        env = dict(os.environ, BOOTSTRAP_REPO_URL="file://" + origin, BOOTSTRAP_DIR=dest, BOOTSTRAP_NO_RUN="1")
        p = subprocess.run(["bash", os.path.join(ROOT, "scripts", "bootstrap.sh")], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        return p

    def test_matching_hashes_pass(self):
        td, origin = self.make_origin()
        p = self.run_bootstrap(td, origin)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("match the embedded hashes", p.stdout)

    def test_tampered_click_run_is_refused(self):
        td, origin = self.make_origin(tamper=True)
        p = self.run_bootstrap(td, origin)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing to execute", p.stdout)
        self.assertNotIn("match the embedded hashes", p.stdout)


if __name__ == "__main__":
    unittest.main()
