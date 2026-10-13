import unittest

from tests import helpers  # noqa: F401
import preflight

GIB = 1024 ** 3
SMI_OK = "NVIDIA GB10, 12.1, 580.173.02\n"
MEMINFO = "MemTotal: 127000000 kB\nMemFree: 90000000 kB\nMemAvailable: %d kB\n"


class FakeHost(object):
    def __init__(self, machine="aarch64", smi=SMI_OK, apps="", nvcc="Cuda compilation tools, release 13.0, V13.0.88",
                 avail_gib=100, disk_gib=500, port_busy=False, tools=("git", "gcc", "g++", "curl", "nvidia-smi", "python3.12", "nvcc"),
                 py="Python 3.12.3", env=None):
        self._m, self.smi, self.apps, self.nvcc = machine, smi, apps, nvcc
        self.avail, self.disk, self.port_busy, self.tools, self.py = avail_gib, disk_gib, port_busy, set(tools), py
        self.env = env or {}

    def machine(self):
        return self._m

    def which(self, name):
        return "/usr/bin/" + name if name in self.tools else None

    def isfile(self, path):
        return path.endswith("nvcc") and "nvcc" in self.tools

    def glob(self, pattern):
        return []

    def environ(self, key, default=None):
        return self.env.get(key, default)

    def run(self, cmd, timeout=15):
        exe = cmd[0]
        if exe.endswith("nvidia-smi"):
            if "--query-gpu=name,compute_cap,driver_version" in cmd[1]:
                return (0, self.smi) if self.smi is not None else (9, "NVIDIA-SMI has failed")
            return 0, self.apps
        if exe.endswith("nvcc"):
            return 0, self.nvcc
        if "python" in exe:
            return 0, self.py
        return 127, ""

    def meminfo_text(self):
        return MEMINFO % (self.avail * 1024 * 1024)

    def disk_free_bytes(self, path):
        return self.disk * GIB

    def device_of(self, path):
        return 1

    def port_in_use(self, host, port):
        return self.port_busy


def opts(**kw):
    o = {"need_build": True, "port": 8000, "bind_host": "127.0.0.1", "allow_busy": False, "paths": ["/x"],
         "disk_needed_gib": 70, "driver_min": 580, "nvcc_major": 13, "profile_name": "gb10-text-262k",
         "py_range": ((3, 11), (3, 13))}
    o.update(kw)
    return o


def status(checks):
    return {(c.code, c.name): c.status for c in checks}


class PreflightTest(unittest.TestCase):
    def test_all_pass_on_a_gb10(self):
        checks = preflight.run_checks(FakeHost(), 44.4, opts())
        self.assertFalse([c for c in checks if c.status == "fail"], [c.as_dict() for c in checks if c.status == "fail"])

    def test_wrong_architecture(self):
        c = status(preflight.run_checks(FakeHost(machine="x86_64"), 44.4, opts()))
        self.assertEqual(c[("E_ARCH", "architecture")], "fail")

    def test_old_driver(self):
        c = status(preflight.run_checks(FakeHost(smi="NVIDIA GB10, 12.1, 570.124.06\n"), 44.4, opts()))
        self.assertEqual(c[("E_DRIVER", "driver")], "fail")

    def test_wrong_compute_capability(self):
        c = status(preflight.run_checks(FakeHost(smi="NVIDIA H100, 9.0, 580.1\n"), 44.4, opts()))
        self.assertEqual(c[("E_GPU_CC", "gpu")], "fail")

    def test_no_nvidia_smi(self):
        c = status(preflight.run_checks(FakeHost(tools=("git", "gcc", "g++", "curl", "python3.12")), 44.4, opts()))
        self.assertEqual(c[("E_NO_NVIDIA_SMI", "nvidia-smi")], "fail")

    def test_busy_gpu_refused_unless_overridden(self):
        h = FakeHost(apps="4242, python\n")
        self.assertEqual(status(preflight.run_checks(h, 44.4, opts()))[("E_GPU_BUSY", "gpu idle")], "fail")
        self.assertEqual(status(preflight.run_checks(h, 44.4, opts(allow_busy=True)))[("E_GPU_BUSY", "gpu idle")], "warn")

    def test_nvcc_needed_only_when_building(self):
        h = FakeHost(tools=("git", "gcc", "g++", "curl", "nvidia-smi", "python3.12"))
        self.assertEqual(status(preflight.run_checks(h, 44.4, opts(need_build=True)))[("E_NVCC", "nvcc")], "fail")
        self.assertEqual(status(preflight.run_checks(h, 44.4, opts(need_build=False)))[("E_NVCC", "nvcc")], "pass")

    def test_old_nvcc_rejected(self):
        h = FakeHost(nvcc="Cuda compilation tools, release 12.8, V12.8.93")
        self.assertEqual(status(preflight.run_checks(h, 44.4, opts()))[("E_NVCC", "nvcc")], "fail")

    def test_memory_and_disk_and_port(self):
        c = status(preflight.run_checks(FakeHost(avail_gib=20, disk_gib=30, port_busy=True), 44.4, opts()))
        self.assertEqual(c[("E_MEM", "memory")], "fail")
        self.assertEqual(c[("E_DISK", "disk")], "fail")
        self.assertEqual(c[("E_PORT", "port")], "fail")

    def test_python_range(self):
        c = status(preflight.run_checks(FakeHost(py="Python 3.9.2"), 44.4, opts()))
        self.assertEqual(c[("E_PYTHON", "python")], "fail")

    def test_parsers(self):
        self.assertEqual(preflight.parse_nvidia_smi_gpus("NVIDIA GB10, 12.1, 580.173.02"), [("NVIDIA GB10", "12.1", "580.173.02")])
        self.assertEqual(preflight.parse_nvcc_version("Cuda compilation tools, release 13.0, V13.0.88")["full"], "13.0.88")
        self.assertEqual(preflight.parse_compute_apps("123, python\n456, other"), [(123, "python"), (456, "other")])
        self.assertEqual(preflight.parse_meminfo("MemAvailable: 2048 kB")["MemAvailable"], 2048 * 1024)


if __name__ == "__main__":
    unittest.main()
