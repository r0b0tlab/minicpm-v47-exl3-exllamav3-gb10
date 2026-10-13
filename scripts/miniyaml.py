"""Strict-subset YAML reader (standard library only).

The repository's configuration (config/model.yaml, config/profiles.yaml) has to
be readable by the pre-venv preflight, on a clean machine, with whatever
python3 the host ships. PyYAML is not guaranteed there, so this module reads
the small YAML subset the config files are written in:

  * block mappings and block sequences, indentation by spaces (any consistent
    width), comments (# ...) and blank lines
  * scalars: int, float, true/false, null/~, "double", 'single' and plain strings
  * single-line flow sequences  [a, b, 3]  and flow mappings  {a: 1, b: x}
  * "key:" with no value opens a nested block

Not supported (and rejected loudly rather than guessed): anchors, tags, multi-
document streams, multi-line scalars, "? " complex keys. tests/test_miniyaml.py
checks that every shipped config file parses identically with PyYAML when that
is installed.
"""
import json
import re
import sys


class YamlError(ValueError):
    pass


_INT = re.compile(r"^[-+]?(0|[1-9][0-9_]*)$")
_FLOAT = re.compile(r"^[-+]?([0-9][0-9_]*)?\.[0-9]+([eE][-+]?[0-9]+)?$|^[-+]?[0-9]+[eE][-+]?[0-9]+$")


def _strip_comment(line):
    out = []
    quote = None
    prev = " "
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'" and prev in " \t[{,:":
            quote = ch
            out.append(ch)
        elif ch == "#" and prev in " \t":
            break
        else:
            out.append(ch)
        prev = ch
    return "".join(out).rstrip()


def _split_flow(text, lineno):
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch in "[{":
            depth += 1
            cur.append(ch)
        elif ch in "]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail or parts:
        parts.append(tail)
    return [p for p in parts if p != ""]


def _scalar(text, lineno):
    t = text.strip()
    if t == "":
        return None
    if t[0] == '"':
        if len(t) < 2 or t[-1] != '"':
            raise YamlError(f"line {lineno}: unterminated double-quoted string")
        try:
            return json.loads(t)
        except ValueError as exc:
            raise YamlError(f"line {lineno}: bad double-quoted string: {exc}")
    if t[0] == "'":
        if len(t) < 2 or t[-1] != "'":
            raise YamlError(f"line {lineno}: unterminated single-quoted string")
        return t[1:-1].replace("''", "'")
    if t[0] == "[":
        if t[-1] != "]":
            raise YamlError(f"line {lineno}: flow sequence must be on one line")
        return [_scalar(p, lineno) for p in _split_flow(t[1:-1], lineno)]
    if t[0] == "{":
        if t[-1] != "}":
            raise YamlError(f"line {lineno}: flow mapping must be on one line")
        out = {}
        for p in _split_flow(t[1:-1], lineno):
            k, sep, v = p.partition(":")
            if not sep:
                raise YamlError(f"line {lineno}: flow mapping entry without ':'")
            out[_scalar(k, lineno) if k.strip()[:1] in "\"'" else k.strip()] = _scalar(v, lineno)
        return out
    if t[0] in "&*!|>%@`":
        raise YamlError(f"line {lineno}: unsupported YAML construct {t[0]!r}")
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    if _INT.match(t):
        return int(t.replace("_", ""))
    if _FLOAT.match(t):
        return float(t.replace("_", ""))
    return t


def _key(text, lineno):
    k = text.strip()
    if k[:1] in "\"'":
        return _scalar(k, lineno)
    return k


def _find_colon(text):
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == ":" and (i + 1 == len(text) or text[i + 1] in " \t"):
            return i
    return -1


def loads(text):
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YamlError(f"line {n}: tab indentation is not supported")
        s = _strip_comment(raw)
        if not s.strip():
            continue
        if s.strip() in ("---", "..."):
            if lines:
                raise YamlError(f"line {n}: multi-document streams are not supported")
            continue
        lines.append((n, len(s) - len(s.lstrip(" ")), s.strip()))
    if not lines:
        return {}
    value, pos = _block(lines, 0, lines[0][1])
    if pos != len(lines):
        raise YamlError(f"line {lines[pos][0]}: unexpected indentation")
    return value


def _block(lines, pos, indent):
    n, ind, content = lines[pos]
    if content.startswith("- ") or content == "-":
        return _seq(lines, pos, indent)
    return _map(lines, pos, indent)


def _map(lines, pos, indent):
    out = {}
    while pos < len(lines):
        n, ind, content = lines[pos]
        if ind < indent:
            break
        if ind > indent:
            raise YamlError(f"line {n}: unexpected indentation")
        if content.startswith("- "):
            raise YamlError(f"line {n}: sequence item inside a mapping")
        c = _find_colon(content)
        if c < 0:
            raise YamlError(f"line {n}: expected 'key: value', got {content!r}")
        key = _key(content[:c], n)
        rest = content[c + 1:].strip()
        pos += 1
        if key in out:
            raise YamlError(f"line {n}: duplicate key {key!r}")
        if rest == "":
            if pos < len(lines) and lines[pos][1] > indent:
                out[key], pos = _block(lines, pos, lines[pos][1])
            elif pos < len(lines) and lines[pos][1] == indent and lines[pos][2].startswith("- "):
                out[key], pos = _seq(lines, pos, indent)
            else:
                out[key] = None
        else:
            out[key] = _scalar(rest, n)
    return out, pos


def _seq(lines, pos, indent):
    out = []
    while pos < len(lines):
        n, ind, content = lines[pos]
        if ind < indent:
            break
        if ind > indent or not (content.startswith("- ") or content == "-"):
            raise YamlError(f"line {n}: bad sequence indentation")
        rest = content[1:].strip()
        pos += 1
        if rest == "":
            if pos < len(lines) and lines[pos][1] > indent:
                item, pos = _block(lines, pos, lines[pos][1])
            else:
                item = None
        elif _find_colon(rest) > 0 and rest[0] not in "\"'[{":
            # "- key: value" opens an inline mapping whose further keys are
            # indented to the column after "- "
            sub_indent = indent + 2
            fake = [(n, sub_indent, rest)]
            while pos < len(lines) and lines[pos][1] >= sub_indent:
                fake.append(lines[pos])
                pos += 1
            item, _ = _map(fake, 0, sub_indent)
        else:
            item = _scalar(rest, n)
        out.append(item)
    return out, pos


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: miniyaml.py FILE   (prints the parsed document as JSON)")
    print(json.dumps(load(sys.argv[1]), indent=2, sort_keys=True))
