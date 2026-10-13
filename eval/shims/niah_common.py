"""Stand-in for the kit's `niah_common` module (the runner only uses strict_json_loads).

Strict: duplicate object keys are an error and the non-standard constants NaN / Infinity / -Infinity are rejected.
"""
import json


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key: %s" % k)
        out[k] = v
    return out


def _no_constants(name):
    raise ValueError("non-standard JSON constant: %s" % name)


def strict_json_loads(data):
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    return json.loads(data, object_pairs_hook=_no_dupes, parse_constant=_no_constants)
