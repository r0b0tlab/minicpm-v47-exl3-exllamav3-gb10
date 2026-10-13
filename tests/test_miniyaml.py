import os
import unittest

from tests import helpers  # noqa: F401  (sys.path)
import miniyaml

ROOT = helpers.ROOT


class MiniYamlTest(unittest.TestCase):
    def test_scalars_lists_maps(self):
        doc = miniyaml.loads("""
# comment
a: 1
b: 2.5   # trailing
c: true
d: ~
e: "x: y # not a comment"
f: [1, two, "3"]
g: {k: v, n: 4}
h:
  - 10
  - name: z
    v: 1
i:
  nested:
    deep: yes-string
""")
        self.assertEqual(doc["a"], 1)
        self.assertEqual(doc["b"], 2.5)
        self.assertIs(doc["c"], True)
        self.assertIsNone(doc["d"])
        self.assertEqual(doc["e"], "x: y # not a comment")
        self.assertEqual(doc["f"], [1, "two", "3"])
        self.assertEqual(doc["g"], {"k": "v", "n": 4})
        self.assertEqual(doc["h"], [10, {"name": "z", "v": 1}])
        self.assertEqual(doc["i"]["nested"]["deep"], "yes-string")

    def test_rejects_unsupported(self):
        for bad in ("a: &x 1\n", "a: |\n  text\n", "a: 1\na: 2\n", "\ta: 1\n", "a: [1,\n  2]\n"):
            with self.assertRaises(miniyaml.YamlError, msg=bad):
                miniyaml.loads(bad)

    def test_shipped_configs_match_pyyaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        for rel in ("config/model.yaml", "config/profiles.yaml"):
            p = os.path.join(ROOT, rel)
            with open(p) as fh:
                self.assertEqual(miniyaml.load(p), yaml.safe_load(fh), rel)


if __name__ == "__main__":
    unittest.main()
