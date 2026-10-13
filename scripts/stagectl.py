#!/usr/bin/env python3
"""Stage receipts, skip decisions and plans for scripts/click_run.sh (stdlib only).

click_run.sh is a fixed sequence of named stages:

    preflight  venv  torch  build  download  verify-model  retire-cache  serve  smoke

Each stage that runs writes .state/<stage>.json (inputs, inputs hash, versions,
duration, outcome). On a re-run a non-volatile stage is skipped when its last
receipt says "ok", its inputs hash is unchanged and its cheap validity check
(outputs still on disk) passes. preflight, retire-cache, serve and smoke are
"volatile": they look at live system state and always run.

  stagectl.py plan   [--from S | --only S] [--force] [--json]
  stagectl.py decide STAGE [--from S | --only S] [--force]     -> RUN|SKIP<TAB>reason
  stagectl.py record STAGE --status ok|failed --duration SEC [--detail T] [--extra-json J]
  stagectl.py show
  stagectl.py build-key

Context comes from environment variables exported by click_run.sh (ROOT, STATE,
VENV, MODEL_DIR, DRAFT_PATH, CACHE_DIR, PROFILE, HOST, PORT, ARTIFACTS (target|draft|target draft),
NO_DOWNLOAD, SKIP_BUILD, SERVE_ONLY, SKIP_VERIFY, HF_REPO, REVISION).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cfg  # noqa: E402

STAGES = ["preflight", "venv", "torch", "build", "download", "verify-model", "retire-cache", "serve", "smoke"]
VOLATILE = {"preflight", "retire-cache", "serve", "smoke"}
SCHEMA = 1


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def truthy(v):
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


class Ctx(object):
    def __init__(self, env):
        g = env.get
        self.env = env
        self.root = g("ROOT") or cfg.ROOT
        self.state = g("STATE") or os.path.join(self.root, ".state")
        self.venv = g("VENV") or os.path.join(self.root, ".venv")
        self.cache_dir = g("CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "minicpm-v47-exl3-gb10")
        self.profile = g("PROFILE") or ""
        self.host = g("HOST") or "127.0.0.1"
        self.port = int(g("PORT") or 8000)
        self.artifacts = (g("ARTIFACTS") or ("target draft" if truthy(g("WITH_DRAFT")) else "target")).split()
        self.no_download = truthy(g("NO_DOWNLOAD"))
        self.skip_build = truthy(g("SKIP_BUILD"))
        self.serve_only = truthy(g("SERVE_ONLY"))
        self.skip_verify = truthy(g("SKIP_VERIFY"))
        self.hf_repo = g("HF_REPO") or ""
        self.revision = g("REVISION") or ""
        self.lock = json.load(open(os.path.join(self.root, "runtime.lock.json")))
        self.model_doc = cfg.load_model(self.root)
        self.profiles_doc = cfg.load_profiles(self.root)
        self.target = cfg.artifact(self.model_doc, "target")
        self.model_dir = g("MODEL_DIR") or os.path.join(self.root, "models", self.target["local_dir"])
        draft = cfg.artifact(self.model_doc, "draft")
        self.draft_dir = g("DRAFT_PATH") or os.path.join(os.path.dirname(self.model_dir), draft["local_dir"])
        self.pyver = None  # filled lazily from the venv or the chosen interpreter

    # ---- derived
    def profile_name(self):
        return cfg.get_profile(self.profiles_doc, self.profile or None)["name"]

    def site_packages(self):
        pats = glob.glob(os.path.join(self.venv, "lib", "python3.*", "site-packages"))
        return pats[0] if pats else None

    def dist_versions(self):
        sp = self.site_packages()
        out = {}
        if not sp:
            return out
        for d in glob.glob(os.path.join(sp, "*.dist-info")):
            base = os.path.basename(d)[:-len(".dist-info")]
            name, _, ver = base.rpartition("-")
            if name:
                out[re.sub(r"[-_.]+", "-", name).lower()] = ver
        return out

    def venv_python_version(self):
        cfgf = os.path.join(self.venv, "pyvenv.cfg")
        try:
            for line in open(cfgf):
                k, _, v = line.partition("=")
                if k.strip() in ("version", "version_info"):
                    return v.strip()
        except OSError:
            pass
        # venv not created yet: the interpreter click_run.sh is about to use
        return self.env.get("MC_PYTHON_VERSION") or None

    def requirements(self):
        pins = {}
        for line in open(os.path.join(self.root, "requirements.txt")):
            line = line.split("#")[0].strip()
            m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;]+)", line)
            if m:
                pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(3)
        return pins

    def requirements_sha(self):
        with open(os.path.join(self.root, "requirements.txt"), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def exl_commit(self):
        return self.lock["exllamav3"]["commit"]

    def torch_pin(self):
        return self.lock["torch"]["version"]

    def arch(self):
        return self.lock["arch"]["torch_cuda_arch_list"]

    def build_key(self):
        py = (self.venv_python_version() or "unknown").rsplit(".", 1)[0]
        return "%s-torch%s-py%s-sm%s" % (self.exl_commit()[:12], self.torch_pin().replace("+", ""), py,
                                         self.arch().replace(".", ""))

    def wheel_dir(self):
        return os.path.join(self.cache_dir, "wheels", self.build_key())

    def artifact_for(self, name):
        art = dict(cfg.artifact(self.model_doc, name))
        if name == "target" and (self.hf_repo or self.revision):
            if self.hf_repo and self.hf_repo != art["hf_repo"]:
                art["files"] = {}
                art["unpinned"] = True
            art["hf_repo"] = self.hf_repo or art["hf_repo"]
            art["revision"] = self.revision or art["revision"]
            if self.revision and self.revision != cfg.artifact(self.model_doc, "target")["revision"]:
                art["files"] = {}
                art["unpinned"] = True
        return art

    def dir_for(self, name):
        return self.model_dir if name == "target" else self.draft_dir

    def artifacts_in_play(self):
        return list(self.artifacts)

    def files_fingerprint(self, name):
        art = self.artifact_for(name)
        d = self.dir_for(name)
        out = []
        for path in sorted(art.get("files") or {}):
            try:
                st = os.stat(os.path.join(d, path))
                out.append([path, st.st_size, st.st_mtime_ns])
            except OSError:
                out.append([path, None, None])
        return out


# ---------------------------------------------------------------- stage definitions

def _manifest_digest(ctx, name):
    art = ctx.artifact_for(name)
    return sha256_json({"repo": art["hf_repo"], "revision": art["revision"], "files": art.get("files") or {}})


def inputs_for(ctx, stage):
    if stage == "preflight":
        return {"profile": ctx.profile_name(), "port": ctx.port}
    if stage == "venv":
        return {"python": ctx.env.get("MC_PYTHON", ""), "min": ctx.lock["python"]["min"],
                "max": ctx.lock["python"]["max"], "venv": ctx.venv}
    if stage == "torch":
        return {"torch": ctx.torch_pin(), "index_url": ctx.lock["torch"]["index_url"],
                "requirements_sha256": ctx.requirements_sha(), "python": ctx.venv_python_version()}
    if stage == "build":
        return {"commit": ctx.exl_commit(), "torch": ctx.torch_pin(), "python": ctx.venv_python_version(),
                "arch": ctx.arch(), "key": ctx.build_key()}
    if stage == "download":
        return {"artifacts": {n: _manifest_digest(ctx, n) for n in ctx.artifacts_in_play()},
                "dirs": {n: ctx.dir_for(n) for n in ctx.artifacts_in_play()}}
    if stage == "verify-model":
        return {"artifacts": {n: _manifest_digest(ctx, n) for n in ctx.artifacts_in_play()},
                "fingerprint": {n: ctx.files_fingerprint(n) for n in ctx.artifacts_in_play()}}
    return {"profile": ctx.profile_name(), "port": ctx.port, "host": ctx.host}


def valid_for(ctx, stage):
    """Cheap check that a completed stage's outputs are still there. (ok, message)"""
    if stage in VOLATILE:
        return True, ""
    if stage == "venv":
        py = os.path.join(ctx.venv, "bin", "python")
        if not os.path.isfile(py):
            return False, "venv python missing"
        return True, ""
    d = ctx.dist_versions()
    if stage == "torch":
        if d.get("torch") != ctx.torch_pin():
            return False, "torch %s installed, want %s" % (d.get("torch"), ctx.torch_pin())
        for name, ver in ctx.requirements().items():
            if d.get(name) != ver:
                return False, "%s %s installed, want %s" % (name, d.get(name), ver)
        return True, ""
    if stage == "build":
        if "exllamav3" not in d:
            return False, "exllamav3 not installed in the venv"
        marker = os.path.join(ctx.state, "build-installed.json")
        try:
            m = json.load(open(marker))
        except (OSError, ValueError):
            return False, "no install marker"
        if m.get("key") != ctx.build_key():
            return False, "installed build key %s != %s" % (m.get("key"), ctx.build_key())
        sp = ctx.site_packages()
        if not glob.glob(os.path.join(sp, "exllamav3_ext*.so")):
            return False, "exllamav3_ext*.so not found in site-packages"
        return True, ""
    if stage == "download":
        for n in ctx.artifacts_in_play():
            art = ctx.artifact_for(n)
            files = art.get("files") or {}
            if not files:
                return os.path.isdir(ctx.dir_for(n)), "unpinned artifact: directory present"
            for path, meta in files.items():
                p = os.path.join(ctx.dir_for(n), path)
                try:
                    if os.stat(p).st_size != int(meta["size"]):
                        return False, "%s has the wrong size" % path
                except OSError:
                    return False, "%s missing" % path
        return True, ""
    if stage == "verify-model":
        return True, ""
    return True, ""


def receipt_path(ctx, stage):
    return os.path.join(ctx.state, stage + ".json")


def read_receipt(ctx, stage):
    try:
        with open(receipt_path(ctx, stage)) as fh:
            doc = json.load(fh)
        if doc.get("schema") != SCHEMA:
            return None
        return doc
    except (OSError, ValueError):
        return None


def _diff_keys(old, new):
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ["inputs"]
    keys = sorted(set(old) | set(new))
    return [k for k in keys if old.get(k) != new.get(k)]


def decide(ctx, stage, force=False, from_stage=None, only=None):
    """-> (action, reason). action in RUN / SKIP."""
    if stage not in STAGES:
        raise ValueError("unknown stage %r (stages: %s)" % (stage, ", ".join(STAGES)))
    idx = STAGES.index(stage)
    if only:
        if stage == only:
            return "RUN", "--only %s" % only
        return "SKIP", "--only %s" % only
    if from_stage and idx < STAGES.index(from_stage):
        return "SKIP", "before --from %s" % from_stage
    # flags that remove whole stages
    if stage in ("venv", "torch", "build", "download", "verify-model") and ctx.serve_only:
        return "SKIP", "SERVE_ONLY=1"
    if stage == "build" and ctx.skip_build:
        return "SKIP", "SKIP_BUILD=1 (using whatever the venv already has)"
    if stage == "download" and ctx.no_download:
        return "SKIP", "NO_DOWNLOAD=1 (using MODEL_DIR as is)"
    if stage == "verify-model" and ctx.skip_verify:
        return "SKIP", "SKIP_VERIFY=1 (not checking sha256)"
    if force:
        return "RUN", "--force"
    if from_stage and stage == from_stage:
        return "RUN", "--from %s" % from_stage
    if stage in VOLATILE:
        return "RUN", "always runs (reads live system state)"
    rec = read_receipt(ctx, stage)
    if rec is None:
        return "RUN", "no receipt (.state/%s.json)" % stage
    if rec.get("status") != "ok":
        return "RUN", "previous attempt failed"
    now = inputs_for(ctx, stage)
    if rec.get("inputs_hash") != sha256_json(now):
        return "RUN", "inputs changed: %s" % ", ".join(_diff_keys(rec.get("inputs"), now))
    ok, msg = valid_for(ctx, stage)
    if not ok:
        return "RUN", "outputs invalid: %s" % msg
    return "SKIP", "completed %s in %ss, inputs unchanged" % (rec.get("finished_utc", "?"), rec.get("duration_s", "?"))


def record(ctx, stage, status, duration, detail="", extra=None):
    os.makedirs(ctx.state, exist_ok=True)
    now = inputs_for(ctx, stage)
    d = ctx.dist_versions()
    doc = {
        "schema": SCHEMA, "stage": stage, "status": status,
        "inputs": now, "inputs_hash": sha256_json(now),
        "versions": {"python": ctx.venv_python_version(), "torch": d.get("torch"),
                     "exllamav3": d.get("exllamav3"), "exllamav3_commit": ctx.exl_commit()},
        "duration_s": round(float(duration), 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": detail,
    }
    if extra:
        doc["extra"] = extra
    tmp = receipt_path(ctx, stage) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, receipt_path(ctx, stage))
    return doc


def disk_needed_gib(ctx):
    """Free GiB required for what is still missing (model + venv + build + slack)."""
    gib = 1024 ** 3
    need = 0.0
    if not (ctx.no_download or ctx.serve_only):
        for n in ctx.artifacts_in_play():
            art = ctx.artifact_for(n)
            for path, meta in (art.get("files") or {}).items():
                try:
                    have = os.path.getsize(os.path.join(ctx.dir_for(n), path))
                except OSError:
                    have = 0
                if have != int(meta["size"]):
                    need += int(meta["size"])
    if not ctx.serve_only:
        if not os.path.isfile(os.path.join(ctx.venv, "bin", "python")):
            need += 14 * gib
        if not glob.glob(os.path.join(ctx.wheel_dir(), "exllamav3-*.whl")) and not ctx.skip_build:
            need += 6 * gib
    if need <= 0:
        return 0
    return int(-(-(need + 20 * gib) // gib))


def plan(ctx, force=False, from_stage=None, only=None):
    rows = []
    for st in STAGES:
        action, reason = decide(ctx, st, force=force, from_stage=from_stage, only=only)
        rec = read_receipt(ctx, st)
        rows.append({"stage": st, "action": action, "reason": reason,
                     "last": (rec or {}).get("finished_utc"), "last_status": (rec or {}).get("status"),
                     "last_duration_s": (rec or {}).get("duration_s")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "decide"):
        p = sub.add_parser(name)
        if name == "decide":
            p.add_argument("stage")
        p.add_argument("--force", action="store_true")
        p.add_argument("--from", dest="from_stage")
        p.add_argument("--only")
        p.add_argument("--json", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("stage")
    p.add_argument("--status", required=True, choices=("ok", "failed"))
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--detail", default="")
    p.add_argument("--extra-json", default="")
    sub.add_parser("show")
    sub.add_parser("disk-needed")
    sub.add_parser("build-key")
    sub.add_parser("wheel-dir")
    args = ap.parse_args(argv)
    ctx = Ctx(os.environ)
    if args.cmd == "plan":
        rows = plan(ctx, args.force, args.from_stage, args.only)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            print("  %-13s %-5s %s" % ("stage", "do", "why"))
            for r in rows:
                print("  %-13s %-5s %s" % (r["stage"], r["action"], r["reason"]))
    elif args.cmd == "decide":
        a, r = decide(ctx, args.stage, args.force, args.from_stage, args.only)
        print("%s\t%s" % (a, r))
    elif args.cmd == "record":
        extra = json.loads(args.extra_json) if args.extra_json else None
        record(ctx, args.stage, args.status, args.duration, args.detail, extra)
    elif args.cmd == "show":
        print("  %-13s %-7s %9s  %s" % ("stage", "status", "seconds", "finished (UTC)"))
        for st in STAGES:
            rec = read_receipt(ctx, st)
            if rec:
                print("  %-13s %-7s %9s  %s" % (st, rec["status"], rec["duration_s"], rec["finished_utc"]))
    elif args.cmd == "disk-needed":
        print(disk_needed_gib(ctx))
    elif args.cmd == "build-key":
        print(ctx.build_key())
    elif args.cmd == "wheel-dir":
        print(ctx.wheel_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
