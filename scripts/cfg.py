#!/usr/bin/env python3
"""Config access and memory arithmetic for the click-run driver (stdlib only).

Reads config/model.yaml and config/profiles.yaml with scripts/miniyaml.py.
Nothing about the model is hard-coded in the shell scripts: they call this
tool (or import it), so supporting another model is a data-only change.

  cfg.py get model.vocab_size
  cfg.py profiles                      # names, default first
  cfg.py env gb10-text-262k            # shell-safe KEY=VALUE lines (eval-able)
  cfg.py artifact target               # ART_REPO / ART_REV / ART_DIR / ART_REQUIRED ...
  cfg.py manifest target               # path <TAB> size <TAB> sha256
  cfg.py need gb10-text-262k [--json]  # memory budget table
"""
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import miniyaml  # noqa: E402

ROOT = os.path.dirname(HERE)
GIB = 1024 ** 3


class ConfigError(ValueError):
    pass


def load_model(root=ROOT):
    doc = miniyaml.load(os.path.join(root, "config", "model.yaml"))
    for key in ("model", "artifacts"):
        if key not in doc:
            raise ConfigError(f"config/model.yaml: missing '{key}'")
    return doc


def load_profiles(root=ROOT):
    doc = miniyaml.load(os.path.join(root, "config", "profiles.yaml"))
    if "profiles" not in doc or not doc["profiles"]:
        raise ConfigError("config/profiles.yaml: missing 'profiles'")
    return doc


_PROFILE_DEFAULTS = {
    "kv_bits": 16,
    "max_batch_size": 1,
    "chunk_size": 2048,
    "recurrent_cache_gib": 1.0,
    "workspace_gib": 4.0,
    "headroom_gib": 8.0,
    "default_max_tokens": 4096,
}


def get_profile(profiles_doc, name=None):
    name = name or profiles_doc.get("default")
    profiles = profiles_doc["profiles"]
    if name not in profiles:
        raise ConfigError(f"unknown profile {name!r}; available: {', '.join(profiles)}")
    prof = dict(_PROFILE_DEFAULTS)
    prof.update(profiles[name])
    if "window_tokens" not in prof:
        raise ConfigError(f"profile {name}: window_tokens is required")
    if prof["window_tokens"] % 256:
        raise ConfigError(f"profile {name}: window_tokens must be a multiple of 256 (cache page size)")
    kvb = prof["kv_bits"]
    if not (kvb == 16 or (isinstance(kvb, int) and 2 <= kvb <= 8)):
        raise ConfigError(f"profile {name}: kv_bits must be 16 or 2..8")
    prof["name"] = name
    return prof


def kv_bytes_per_token(model, kv_bits):
    base = model["kv_bytes_per_token"]
    if kv_bits >= 16:
        return base
    # quantised cache: kv_bits per element plus one fp16 scale per 32 elements
    return base * (4 * kv_bits + 2) / 64.0


def memory_budget(model, prof, weights_bytes=None):
    """Unified-memory need of one profile, in GiB (see docs/MEMORY.md)."""
    w = weights_bytes if weights_bytes is not None else model["weights_bytes"]
    kv = kv_bytes_per_token(model, prof["kv_bits"]) * prof["window_tokens"]
    state = model["recurrent_state_bytes"] * prof["max_batch_size"]
    parts = {
        "weights_gib": w / GIB,
        "kv_cache_gib": kv / GIB,
        "recurrent_state_gib": state / GIB,
        "recurrent_cache_gib": float(prof["recurrent_cache_gib"]),
        "workspace_gib": float(prof["workspace_gib"]),
    }
    resident = sum(parts.values())
    parts["resident_gib"] = resident
    parts["headroom_gib"] = float(prof["headroom_gib"])
    parts["need_memavailable_gib"] = resident + float(prof["headroom_gib"])
    return parts


def artifact(model_doc, name):
    arts = model_doc["artifacts"]
    if name not in arts:
        raise ConfigError(f"unknown artifact {name!r}; available: {', '.join(arts)}")
    return arts[name]


def manifest(model_doc, name):
    """[(path, size, sha256)] sorted by path."""
    art = artifact(model_doc, name)
    out = []
    for path, meta in sorted((art.get("files") or {}).items()):
        out.append((path, int(meta["size"]), str(meta["sha256"])))
    return out


def _env_line(key, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    return f"{key}={shlex.quote(str(value))}"


def _dotted(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"no such key: {path}")
        cur = cur[part]
    return cur


def _main(argv):
    root = os.environ.get("MC_ROOT", ROOT)
    if not argv:
        sys.exit(__doc__)
    cmd, args = argv[0], argv[1:]
    model_doc = load_model(root)
    if cmd == "get":
        val = _dotted({**model_doc, "profiles": load_profiles(root)["profiles"]}, args[0])
        print(json.dumps(val) if isinstance(val, (dict, list)) else val)
    elif cmd == "profiles":
        pdoc = load_profiles(root)
        default = pdoc.get("default")
        names = [n for n in pdoc["profiles"] if n == default] + [n for n in pdoc["profiles"] if n != default]
        print("\n".join(names))
    elif cmd == "env":
        prof = get_profile(load_profiles(root), args[0] if args else None)
        m = model_doc["model"]
        budget = memory_budget(m, prof)
        lines = [
            _env_line("P_NAME", prof["name"]),
            _env_line("P_WINDOW", prof["window_tokens"]),
            _env_line("P_KV_BITS", prof["kv_bits"]),
            _env_line("P_BATCH", prof["max_batch_size"]),
            _env_line("P_CHUNK", prof["chunk_size"]),
            _env_line("P_MAX_TOKENS", prof["default_max_tokens"]),
            _env_line("P_NEED_GIB", f"{budget['need_memavailable_gib']:.2f}"),
            _env_line("P_NONWEIGHT_MIB", int(-(-(budget["need_memavailable_gib"] - budget["weights_gib"]) * 1024 // 1))),
            _env_line("M_SERVED_NAME", m["served_name"]),
            _env_line("M_VOCAB", m["vocab_size"]),
        ]
        print("\n".join(lines))
    elif cmd == "artifact":
        a = artifact(model_doc, args[0])
        lines = [
            _env_line("ART_REPO", a["hf_repo"]),
            _env_line("ART_REV", a["revision"]),
            _env_line("ART_DIR", a["local_dir"]),
            _env_line("ART_REQUIRED", bool(a.get("required"))),
            _env_line("ART_FILES", len(a.get("files") or {})),
        ]
        print("\n".join(lines))
    elif cmd == "manifest":
        for path, size, sha in manifest(model_doc, args[0]):
            print(f"{path}\t{size}\t{sha}")
    elif cmd == "need":
        prof = get_profile(load_profiles(root), args[0] if args else None)
        b = memory_budget(model_doc["model"], prof)
        if "--json" in args:
            print(json.dumps({"profile": prof["name"], **b}, indent=2, sort_keys=True))
        else:
            print(f"profile {prof['name']}  (window {prof['window_tokens']} tokens, kv_bits {prof['kv_bits']}, "
                  f"batch {prof['max_batch_size']})")
            for k, label in (("weights_gib", "weights (pack shards)"), ("kv_cache_gib", "KV cache"),
                             ("recurrent_state_gib", "recurrent state (GDN)"),
                             ("recurrent_cache_gib", "host recurrent-cache store"),
                             ("workspace_gib", "workspace"), ("resident_gib", "= resident total"),
                             ("headroom_gib", "+ headroom"), ("need_memavailable_gib", "= MemAvailable needed")):
                print(f"  {label:30s} {b[k]:8.2f} GiB")
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    try:
        _main(sys.argv[1:])
    except ConfigError as exc:
        sys.exit(f"cfg.py: {exc}")
