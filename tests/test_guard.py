import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest

from tests import helpers
import guard


class GuardTest(unittest.TestCase):
    def test_unsafe_logic_is_fail_closed(self):
        self.assertIsNone(guard.unsafe(8000, 6144))
        self.assertEqual(guard.unsafe(100, 6144), "low_memory")
        self.assertEqual(guard.unsafe(None, 6144), "probe_failed")

    def _victim(self):
        # a process whose cmdline contains serve_openai.py
        code = "import time; time.sleep(60)  # serve_openai.py"
        p = subprocess.Popen([sys.executable, "-c", code])
        self.addCleanup(lambda: (p.poll() is None) and p.kill())
        return p

    def test_refuses_foreign_pid(self):
        out = os.path.join(tempfile.mkdtemp(), "g.jsonl")
        rc = guard.main(["--pid", str(os.getpid()), "--out", out])
        self.assertEqual(rc, 5)

    def test_stops_only_after_consecutive_bad_samples(self):
        p = self._victim()
        out = os.path.join(tempfile.mkdtemp(), "g.jsonl")
        samples = iter([9000, 100, 9000, 100, 100, 100, 100])
        rc = guard.main(["--pid", str(p.pid), "--out", out, "--bad-samples", "3", "--interval", "0", "--floor-mib", "6144"],
                        probe=lambda: next(samples), sleep=lambda s: None)
        self.assertEqual(rc, 2)
        p.wait(timeout=30)
        events = [json.loads(l) for l in open(out)]
        self.assertEqual(events[-1]["event"], "stopped_server")
        self.assertEqual(sum(1 for e in events if e.get("reason") == "low_memory"), 5)

    def test_lease_expiry_leaves_server_running_by_default(self):
        p = self._victim()
        out = os.path.join(tempfile.mkdtemp(), "g.jsonl")
        ticks = iter(range(0, 1000))
        rc = guard.main(["--pid", str(p.pid), "--out", out, "--lease-seconds", "3", "--interval", "0"],
                        probe=lambda: 50000, sleep=lambda s: None, clock=lambda: float(next(ticks)))
        self.assertEqual(rc, 0)
        self.assertIsNone(p.poll())
        self.assertEqual(json.loads(open(out).read().splitlines()[-1])["event"], "lease_expired")


if __name__ == "__main__":
    unittest.main()
