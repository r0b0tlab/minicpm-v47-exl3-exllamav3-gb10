"""Tiny stdlib HTTP client for OpenAI-compatible endpoints (used by smoke, bench and the eval lanes)."""
import json
import time
import urllib.error
import urllib.request


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__("HTTP %s: %s" % (status, body if isinstance(body, str) else json.dumps(body)[:400]))
        self.status, self.body = status, body


def _norm(base):
    return base.rstrip("/")


def get_json(base, path, timeout=30):
    req = urllib.request.Request(_norm(base) + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace"))


def post_json(base, path, payload, timeout=3600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_norm(base) + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
        raise ApiError(exc.code, raw)
    body["_client_seconds"] = time.perf_counter() - t0
    return body


def post_stream(base, path, payload, timeout=3600):
    """Yield (t_seconds_since_request, parsed_chunk_dict) for each SSE data event."""
    payload = dict(payload, stream=True)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_norm(base) + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace"))
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            yield time.perf_counter() - t0, json.loads(body)


def wait_ready(base, timeout=600, interval=2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if get_json(base, "/health", timeout=5).get("status") == "ok":
                return True
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(interval)
    raise TimeoutError("endpoint not ready after %ds (%s)" % (timeout, last))


def first_model(base):
    data = get_json(base, "/v1/models")["data"]
    return data[0]
