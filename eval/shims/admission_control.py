"""Stand-in for the kit's `admission_control` module, which the published Q200v2 subset imports but does not ship.

The kit's coordinator gates each batch of requests on the serving host's resources. This one grants every request
immediately and records that it did so in the lease the runner stores on every row (policy "none"), so a result
produced with it says so. Disclose it when reporting.
"""
import contextlib
import json
import time
from pathlib import Path


class AdmissionError(Exception):
    pass


class AdmissionNotGranted(AdmissionError):
    pass


class AdmissionCoordinator:
    SHIM = "minicpm-v47-exl3-gb10 eval/shims/admission_control: grants every request, gates nothing"

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_path(cls, path):
        try:
            config = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise AdmissionError("cannot read admission config %s: %s" % (path, exc))
        if not isinstance(config, dict):
            raise AdmissionError("admission config must be a JSON object")
        return cls(config)

    @contextlib.contextmanager
    def request(self, row_id):
        yield {"granted": True, "row_id": row_id, "policy": self.SHIM, "t": time.time()}
