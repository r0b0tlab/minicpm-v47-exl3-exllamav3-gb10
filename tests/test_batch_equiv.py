import threading
import unittest

from tests import helpers
import batch_equiv


def completion(ids):
    return {"choices": [{"index": 0, "text": "x", "finish_reason": "stop", "token_ids": ids}],
            "usage": {"prompt_tokens": 5, "completion_tokens": len(ids)}, "timings": {}}


class BatchEquivTest(unittest.TestCase):
    def prompts(self, n):
        return [{"name": "p%d" % i, "ids": [i, i + 1, i + 2]} for i in range(n)]

    def test_identical_and_divergent(self):
        lock = threading.Lock()
        calls = {"n": 0}
        N = 4

        def handler(path, body):
            with lock:
                calls["n"] += 1
                k = calls["n"]
            base = [body["prompt"][0] * 10 + j for j in range(8)]
            if k > N and body["prompt"][0] == 2:     # concurrent arm, prompt 2 diverges at position 5
                base[5] = -1
            return completion(base)
        mock = helpers.ScriptedCompletions(handler, models={"id": "m", "max_batch_size": 4})
        try:
            res = batch_equiv.run(mock.base, None, self.prompts(N), 4, 8, log=lambda *_: None)
        finally:
            mock.close()
        self.assertEqual((res["identical"], res["prompts"]), (3, 4))
        bad = [r for r in res["rows"] if not r["identical"]]
        self.assertEqual([(r["name"], r["first_divergence"], r["agree_prefix"]) for r in bad], [("p2", 5, 5)])
        self.assertEqual(batch_equiv.first_divergence([1, 2, 3], [1, 2, 3, 4]), 3)
        self.assertIsNone(batch_equiv.first_divergence([1, 2], [1, 2]))


if __name__ == "__main__":
    unittest.main()
