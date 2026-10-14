import hashlib
import json
import os
import tempfile
import unittest

from tests import helpers
import draft_parity


def sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class DraftParityTest(unittest.TestCase):
    def test_counts_identical_streams(self):
        rows = [{"name": "p%d" % i, "ids": [i + 1, i + 2], "max_new": 8} for i in range(4)]
        ref = {"p0": sha([1, 2]), "p1": sha([9, 9]), "p2": sha([3, 4]), "p3": sha([4, 5])}

        def handler(path, body):
            first = body["prompt"][0]
            ids = [first, first + 1] if first != 2 else [0, 0]           # prompt p1 diverges
            return {"choices": [{"text": "x", "finish_reason": "stop", "token_ids": ids}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2,
                              "completion_tokens_details": {"accepted_prediction_tokens": 3,
                                                            "rejected_prediction_tokens": 1}},
                    "timings": {"decode_tok_s": 120.0}}
        mock = helpers.ScriptedCompletions(handler)
        try:
            res = draft_parity.run(mock.base, rows, ref)
        finally:
            mock.close()
        self.assertEqual([r["identical"] for r in res], [True, False, True, True])
        self.assertEqual(sum(r["accepted"] for r in res), 12)

    def test_pick_spreads_over_the_kit_and_reference_loads(self):
        rows = [{"name": str(i)} for i in range(80)]
        self.assertEqual([r["name"] for r in draft_parity.pick(rows, 8)], ["0", "10", "20", "30", "40", "50", "60", "70"])
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"mode": "greedy", "name": "a", "tokens_sha256": "x"}) + "\n")
            fh.write(json.dumps({"mode": "sampled", "name": "a", "tokens_sha256": "y"}) + "\n")
        try:
            self.assertEqual(draft_parity.load_reference(fh.name), {"a": "x"})
        finally:
            os.remove(fh.name)

    def test_shipped_reference_covers_the_kit(self):
        ref = draft_parity.load_reference(draft_parity.default_reference())
        self.assertEqual(len(ref), 80)


if __name__ == "__main__":
    unittest.main()
