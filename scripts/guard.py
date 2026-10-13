#!/usr/bin/env python3
"""Optional host-memory guard for the server (GUARD=1 ./run.sh), default off. Stdlib only.

On GB10 the CPU and the GPU share one pool, so a runaway allocation anywhere can stall the box instead of failing
one process. The guard samples MemAvailable every few seconds and, if it stays below a floor for several
consecutive samples (or cannot be read at all: fail closed), stops the ONE server process it was started for.

  - it only ever signals the exact PID it was given, after checking that /proc/<pid>/cmdline names serve_openai.py
  - the floor, the number of consecutive bad samples and the lease are bounded and explicit
  - the lease is bounded (default 24 h): at expiry the guard ends and logs it; --stop-on-expiry also stops the server
  - every sample and decision goes to a JSON-lines log

    scripts/guard.py --pid 12345 --out .state/logs/guard.jsonl [--floor-mib 6144] [--bad-samples 3]
                     [--interval 2] [--lease-seconds 86400] [--stop-on-expiry]
exit status: 0 lease ended or server exited, 2 server stopped because of memory, 5 could not arm
"""
import argparse
import json
import os
import signal
import sys
import time


def mem_available_mib(path="/proc/meminfo"):
    with open(path) as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing")


def unsafe(available_mib, floor_mib):
    """None when fine, else a reason string. available_mib None means the probe failed (fail closed)."""
    if available_mib is None:
        return "probe_failed"
    if available_mib < floor_mib:
        return "low_memory"
    return None


def owns(pid, needle="serve_openai.py", proc="/proc"):
    try:
        with open("%s/%d/cmdline" % (proc, pid), "rb") as fh:
            return needle in fh.read().decode("utf-8", "replace")
    except OSError:
        return False


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid, grace=15.0):
    os.kill(pid, signal.SIGTERM)
    end = time.time() + grace
    while time.time() < end and alive(pid):
        time.sleep(0.5)
    if alive(pid):
        os.kill(pid, signal.SIGKILL)


def main(argv=None, probe=mem_available_mib, sleep=time.sleep, clock=time.monotonic):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-mib", type=int, default=6144)
    ap.add_argument("--bad-samples", type=int, default=3)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--lease-seconds", type=int, default=86400)
    ap.add_argument("--stop-on-expiry", action="store_true")
    a = ap.parse_args(argv)
    if not 1 <= a.lease_seconds <= 7 * 86400:
        print("lease must be 1 s .. 7 days", file=sys.stderr)
        return 5
    if a.bad_samples < 1 or a.floor_mib < 256:
        print("bad-samples >= 1 and floor >= 256 MiB required", file=sys.stderr)
        return 5
    if not owns(a.pid):
        print("pid %d is not a serve_openai.py process" % a.pid, file=sys.stderr)
        return 5
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bad = 0
    with open(a.out, "a", buffering=1) as out:
        out.write(json.dumps({"event": "armed", "pid": a.pid, "floor_mib": a.floor_mib, "bad_samples": a.bad_samples,
                              "lease_s": a.lease_seconds, "t": time.time()}) + "\n")
        end = clock() + a.lease_seconds
        while clock() < end:
            if not alive(a.pid):
                out.write(json.dumps({"event": "server_exited", "t": time.time()}) + "\n")
                return 0
            try:
                avail = probe()
            except Exception:  # noqa: BLE001
                avail = None
            reason = unsafe(avail, a.floor_mib)
            out.write(json.dumps({"t": time.time(), "available_mib": avail, "reason": reason}) + "\n")
            bad = bad + 1 if reason else 0
            if bad >= a.bad_samples:
                if owns(a.pid):
                    stop(a.pid)
                out.write(json.dumps({"event": "stopped_server", "reason": reason, "available_mib": avail,
                                      "t": time.time()}) + "\n")
                return 2
            sleep(a.interval)
        if a.stop_on_expiry and owns(a.pid):
            stop(a.pid)
            out.write(json.dumps({"event": "lease_expired_server_stopped", "t": time.time()}) + "\n")
        else:
            out.write(json.dumps({"event": "lease_expired", "t": time.time()}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
