#!/usr/bin/env python3
"""Preflight for the click-run driver (stdlib only, runs before any venv exists).

Every check has a stable error code (scripts/errors.tsv, docs/TROUBLESHOOTING.md)
and a remediation line. All host access goes through a Host object, so the
logic is unit-tested with mocked nvidia-smi / meminfo / filesystem output
(tests/test_preflight.py).

  preflight.py --profile gb10-text-262k [--need-build] [--need-download] ...
  preflight.py --find-python               # prints the interpreter to use
exit status: 0 = no failure, 2 = at least one failure, 3 = usage / config error
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cfg  # noqa: E402

GIB = 1024 ** 3


class Check(object):
    def __init__(self, code, name, status, detail, fix=""):
        self.code, self.name, self.status, self.detail, self.fix = code, name, status, detail, fix

    def as_dict(self):
        return {"code": self.code, "name": self.name, "status": self.status, "detail": self.detail,
                "fix": self.fix}


class Host(object):
    """Real host access. Tests substitute a fake with the same methods."""

    def machine(self):
        import platform
        return platform.machine()

    def run(self, cmd, timeout=15):
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                               universal_newlines=True)
            return p.returncode, p.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, str(exc)

    def which(self, name):
        return shutil.which(name)

    def isfile(self, path):
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def glob(self, pattern):
        import glob
        return sorted(glob.glob(pattern))

    def environ(self, key, default=None):
        return os.environ.get(key, default)

    def meminfo_text(self):
        with open("/proc/meminfo") as fh:
            return fh.read()

    def disk_free_bytes(self, path):
        p = path
        while p and not os.path.exists(p):
            p = os.path.dirname(p)
        st = os.statvfs(p or "/")
        return st.f_bavail * st.f_frsize

    def device_of(self, path):
        p = path
        while p and not os.path.exists(p):
            p = os.path.dirname(p)
        return os.stat(p or "/").st_dev

    def port_in_use(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True
        finally:
            s.close()


# ---------------------------------------------------------------- parsers

def parse_meminfo(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def parse_nvidia_smi_gpus(text):
    """'NVIDIA GB10, 12.1, 580.173.02' lines -> [(name, cc, driver)]"""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and re.match(r"^\d+\.\d+$", parts[1]):
            gpus.append((parts[0], parts[1], parts[2]))
    return gpus


def parse_compute_apps(text):
    pids = []
    for line in text.strip().splitlines():
        m = re.match(r"^\s*(\d+)\s*,?\s*(.*)$", line)
        if m:
            pids.append((int(m.group(1)), m.group(2).strip()))
    return pids


def parse_nvcc_version(text):
    m = re.search(r"release (\d+)\.(\d+), V(\d+\.\d+\.\d+)", text)
    if not m:
        return None
    return {"major": int(m.group(1)), "minor": int(m.group(2)), "full": m.group(3)}


def parse_python_version(text):
    m = re.search(r"Python (\d+)\.(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


# ---------------------------------------------------------------- finders

def find_nvcc(host):
    cands = []
    ch = host.environ("CUDA_HOME") or host.environ("CUDA_PATH")
    if ch:
        cands.append(os.path.join(ch, "bin", "nvcc"))
    w = host.which("nvcc")
    if w:
        cands.append(w)
    cands += ["/usr/local/cuda/bin/nvcc"] + sorted(host.glob("/usr/local/cuda-13*/bin/nvcc"), reverse=True)
    seen = set()
    best = None
    for c in cands:
        if c in seen or not host.isfile(c):
            continue
        seen.add(c)
        rc, out = host.run([c, "--version"])
        ver = parse_nvcc_version(out) if rc == 0 else None
        if ver and best is None:
            best = (c, ver)
        if ver and ver["major"] == 13:
            return c, ver
    return best if best else (None, None)


def find_python(host, lo=(3, 11), hi=(3, 13)):
    names = []
    override = host.environ("PYTHON")
    if override:
        names.append(override)
    names += ["python3.12", "python3.13", "python3.11", "python3"]
    for n in names:
        path = n if os.path.isabs(n) else host.which(n)
        if not path:
            continue
        rc, out = host.run([path, "--version"])
        ver = parse_python_version(out) if rc == 0 else None
        if ver and lo <= ver[:2] <= hi:
            return path, ver
    return None, None


# ---------------------------------------------------------------- checks

def run_checks(host, need_gib, opts):
    """opts: dict with keys need_build, need_download_bytes, need_venv, port, bind_host,
    allow_busy, paths (list of dirs for disk), disk_min_gib, driver_min, nvcc_major,
    profile_name, py_range"""
    out = []
    add = lambda *a, **k: out.append(Check(*a, **k))  # noqa: E731

    # architecture
    m = host.machine()
    if m in ("aarch64", "arm64"):
        add("E_ARCH", "architecture", "pass", m)
    else:
        add("E_ARCH", "architecture", "fail", "machine is %s, this runtime is aarch64 (GB10 / DGX Spark) only" % m,
            "run on a DGX Spark / GB10; on x86-64 use upstream ExLlamaV3 wheels instead")

    # nvidia-smi / driver / compute capability
    smi = host.which("nvidia-smi")
    gpus = []
    if not smi:
        add("E_NO_NVIDIA_SMI", "nvidia-smi", "fail", "nvidia-smi not found",
            "install the NVIDIA driver (>= %d): sudo apt-get install -y nvidia-driver-%d-open && sudo reboot"
            % (opts["driver_min"], opts["driver_min"]))
    else:
        rc, txt = host.run([smi, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
        gpus = parse_nvidia_smi_gpus(txt) if rc == 0 else []
        if not gpus:
            add("E_NO_NVIDIA_SMI", "nvidia-smi", "fail", "nvidia-smi returned no GPU: %s" % txt.strip()[:120],
                "check the driver: nvidia-smi; reboot after a driver install")
        else:
            drv = gpus[0][2]
            try:
                major = int(drv.split(".")[0])
            except ValueError:
                major = 0
            if major >= opts["driver_min"]:
                add("E_DRIVER", "driver", "pass", "driver %s (CUDA 13 capable, >= %d)" % (drv, opts["driver_min"]))
            else:
                add("E_DRIVER", "driver", "fail", "driver %s < %d" % (drv, opts["driver_min"]),
                    "sudo apt-get update && sudo apt-get install -y --only-upgrade nvidia-driver-%d-open && sudo reboot"
                    % opts["driver_min"])
            sm121 = [g for g in gpus if g[1] == "12.1"]
            if sm121:
                add("E_GPU_CC", "gpu", "pass", "%s, compute capability 12.1" % sm121[0][0])
            else:
                add("E_GPU_CC", "gpu", "fail",
                    "compute capability %s (%s); this build targets 12.1 (SM121, GB10)" % (gpus[0][1], gpus[0][0]),
                    "run on a GB10 / DGX Spark; other GPUs need a different TORCH_CUDA_ARCH_LIST and are untested")
            rc, txt = host.run([smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
            apps = parse_compute_apps(txt) if rc == 0 else []
            if apps and not opts.get("allow_busy"):
                add("E_GPU_BUSY", "gpu idle", "fail",
                    "%d process(es) already compute on the GPU (pid %s)" % (len(apps), ", ".join(str(a[0]) for a in apps[:5])),
                    "stop them (nvidia-smi --query-compute-apps=pid,process_name --format=csv) or re-run with "
                    "--allow-busy (unified memory is shared; the load may fail)")
            elif apps:
                add("E_GPU_BUSY", "gpu idle", "warn", "%d GPU process(es) present; --allow-busy given" % len(apps))
            else:
                add("E_GPU_BUSY", "gpu idle", "pass", "no other compute process")

    # toolchain
    missing = [t for t in ("git", "gcc", "g++", "curl") if not host.which(t)]
    if missing:
        add("E_TOOLCHAIN", "toolchain", "fail", "missing: %s" % ", ".join(missing),
            "sudo apt-get install -y build-essential git curl")
    else:
        add("E_TOOLCHAIN", "toolchain", "pass", "git, gcc, g++, curl present")
    if host.which("ninja"):
        add("E_TOOLCHAIN", "ninja", "pass", "ninja on PATH")
    else:
        add("E_TOOLCHAIN", "ninja", "pass", "ninja not on PATH; the pinned ninja wheel is installed into the venv")

    # python
    py, ver = find_python(host, *opts["py_range"])
    if py:
        add("E_PYTHON", "python", "pass", "%s (%d.%d.%d)" % (py, ver[0], ver[1], ver[2]))
    else:
        lo, hi = opts["py_range"]
        add("E_PYTHON", "python", "fail", "no Python %d.%d - %d.%d found" % (lo + hi),
            "sudo apt-get install -y python3.12 python3.12-venv   (or set PYTHON=/path/to/python3.12)")

    # nvcc (only when the extension must be compiled)
    nvcc, nv = find_nvcc(host)
    if opts.get("need_build"):
        if nv and nv["major"] == opts["nvcc_major"]:
            add("E_NVCC", "nvcc", "pass", "%s (CUDA %d.%d, V%s)" % (nvcc, nv["major"], nv["minor"], nv["full"]))
        elif nv:
            add("E_NVCC", "nvcc", "fail", "%s is CUDA %d.%d; need %d.x to match the cu130 torch wheel"
                % (nvcc, nv["major"], nv["minor"], opts["nvcc_major"]),
                "sudo apt-get install -y cuda-toolkit-13-0 && export CUDA_HOME=/usr/local/cuda-13.0")
        else:
            add("E_NVCC", "nvcc", "fail", "nvcc not found (needed once, to compile the ExLlamaV3 extension)",
                "sudo apt-get install -y cuda-toolkit-13-0 && export CUDA_HOME=/usr/local/cuda-13.0 "
                "(or SKIP_BUILD=1 with a cached wheel)")
    else:
        add("E_NVCC", "nvcc", "pass", "not needed: extension build is cached / skipped")

    # disk
    paths = opts.get("paths") or []
    if opts.get("disk_needed_gib", 0) > 0:
        byte_needed = opts["disk_needed_gib"] * GIB
        by_dev = {}
        for p in paths:
            by_dev.setdefault(host.device_of(p), []).append(p)
        worst = None
        for dev, ps in by_dev.items():
            free = host.disk_free_bytes(ps[0])
            if worst is None or free < worst[0]:
                worst = (free, ps[0])
        if worst is not None and worst[0] < byte_needed:
            add("E_DISK", "disk", "fail", "%.1f GiB free, need about %d GiB (model + venv + build)"
                % (worst[0] / GIB, opts["disk_needed_gib"]),
                "free space, or point MODEL_DIR / VENV / CACHE_DIR at a larger disk")
        elif worst is not None:
            add("E_DISK", "disk", "pass", "%.0f GiB free (need about %d GiB)" % (worst[0] / GIB, opts["disk_needed_gib"]))
    else:
        add("E_DISK", "disk", "pass", "nothing left to download or build")

    # memory
    mi = parse_meminfo(host.meminfo_text())
    avail = mi.get("MemAvailable", 0) / GIB
    if avail >= need_gib:
        add("E_MEM", "memory", "pass", "MemAvailable %.1f GiB >= %.1f GiB needed (%s)"
            % (avail, need_gib, opts["profile_name"]))
    else:
        add("E_MEM", "memory", "fail", "MemAvailable %.1f GiB < %.1f GiB needed for %s"
            % (avail, need_gib, opts["profile_name"]),
            "stop other memory users, or pick a smaller profile (--profile gb10-text-32k / gb10-text-16k-lowmem)")

    # port
    if opts.get("port"):
        if host.port_in_use(opts["bind_host"], opts["port"]):
            add("E_PORT", "port", "fail", "%s:%d is in use" % (opts["bind_host"], opts["port"]),
                "choose another port (PORT=8001 ./run.sh) or stop the owner: ss -ltnp | grep :%d" % opts["port"])
        else:
            add("E_PORT", "port", "pass", "%s:%d is free" % (opts["bind_host"], opts["port"]))
    return out


def render(checks):
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    lines = []
    for c in checks:
        lines.append("  [%s] %-14s %s" % (icon[c.status], c.name, c.detail))
        if c.status == "fail":
            lines.append("         %s  fix: %s" % (c.code, c.fix))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile")
    ap.add_argument("--root", default=cfg.ROOT)
    ap.add_argument("--need-build", action="store_true")
    ap.add_argument("--disk-needed-gib", type=float, default=0)
    ap.add_argument("--path", action="append", default=[], help="directories whose filesystems need space")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", help="also write the results as JSON to this file")
    ap.add_argument("--find-python", action="store_true")
    ap.add_argument("--find-nvcc", action="store_true", help="print the nvcc path (CUDA 13 preferred)")
    args = ap.parse_args(argv)
    host = Host()
    lock = json.load(open(os.path.join(args.root, "runtime.lock.json")))
    py_lo = tuple(int(x) for x in lock["python"]["min"].split("."))
    py_hi = tuple(int(x) for x in lock["python"]["max"].split("."))
    if args.find_nvcc:
        path, ver = find_nvcc(host)
        print(path or "")
        return 0 if path else 2
    if args.find_python:
        path, ver = find_python(host, py_lo, py_hi)
        if not path:
            print("", end="")
            return 2
        print(path)
        return 0
    try:
        model = cfg.load_model(args.root)
        prof = cfg.get_profile(cfg.load_profiles(args.root), args.profile)
    except cfg.ConfigError as exc:
        print("preflight: %s" % exc, file=sys.stderr)
        return 3
    budget = cfg.memory_budget(model["model"], prof)
    opts = {
        "need_build": args.need_build, "port": args.port, "bind_host": args.bind_host,
        "allow_busy": args.allow_busy, "paths": args.path or [args.root],
        "disk_needed_gib": args.disk_needed_gib, "driver_min": lock["driver"]["min_major"],
        "nvcc_major": lock["cuda_toolkit"]["nvcc_major"], "profile_name": prof["name"],
        "py_range": (py_lo, py_hi),
    }
    checks = run_checks(host, budget["need_memavailable_gib"], opts)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([c.as_dict() for c in checks], fh, indent=2)
    if args.json:
        print(json.dumps([c.as_dict() for c in checks], indent=2))
    else:
        print(render(checks))
    return 2 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
