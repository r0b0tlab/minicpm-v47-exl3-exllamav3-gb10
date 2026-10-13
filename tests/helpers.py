"""Shared test helpers: path setup, a stand-in tensor, a fake engine and a scripted completion server."""
import json
import os
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


class FakeIds(object):
    """The little of a torch (1, n) tensor the server code touches."""

    def __init__(self, ids):
        self._ids = list(ids)
        self.shape = (1, len(self._ids))

    def flatten(self):
        return self

    def tolist(self):
        return list(self._ids)


def fake_encode(text):
    # one id per whitespace-separated word plus special markers: deterministic, reversible enough for tests
    words = text.replace("<", " <").replace(">", "> ").split()
    return FakeIds([(sum(map(ord, w)) % 5000) + 1 for w in words] or [1])


class FakeEngine(object):
    """Stands in for serve_openai.Engine: scripted generations, no GPU."""

    def __init__(self, script, window=2048, max_batch=1):
        import oai_common as oc
        self.ready = True
        self.window = window
        self.vocab = 248144
        self.draft_active = False
        self.prof = {"name": "test", "chunk_size": 2048, "kv_bits": 16, "max_batch_size": max_batch,
                     "default_max_tokens": 256, "window_tokens": window}
        self.template = oc.ChatTemplate(open(os.path.join(FIXTURES, "chat_template.jinja")).read())
        self.stop_ids = [248046, 248044]
        self.m = {"requests": 0, "errors": 0, "started": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.active = {}
        self.q = queue.Queue()
        self.torch = None
        self.script = script          # callable(request) -> (list_of_text_pieces, eos_reason) or raises
        self.seen = []

    def encode(self, text):
        return fake_encode(text)

    def tensor_from_ids(self, ids):
        return FakeIds(ids)

    def room(self, n):
        return self.window - 1 - n

    def submit(self, req):
        self.m["requests"] += 1
        self.seen.append(req)

        def run():
            try:
                pieces, reason = self.script(req)
            except Exception as exc:  # noqa: BLE001
                req.events.put(("error", exc))
                return
            n = 0
            for p in pieces:
                req.events.put(("delta", p))
                n += 1
            req.events.put(("done", {
                "eos_reason": reason, "completion_tokens": max(1, n), "prompt_tokens": req.input_ids.shape[-1],
                "cached_tokens": 0, "queue_s": 0.0, "prefill_s": 0.01, "decode_s": 0.1, "max_id": 5,
                "decode_tok_s": 10.0, "token_ids": list(range(n))}))
        threading.Thread(target=run, daemon=True).start()


def start_http(handler_cls, engine, served_name="test-model", facts=None):
    """Run serve_openai.Handler against a FakeEngine on an ephemeral port. Returns (server, base_url)."""
    import serve_openai as so
    so.ENGINE = engine
    so.SERVED_NAME = served_name
    so.FACTS = facts or {"sampling_defaults": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}
    srv = so.QuietServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


class ScriptedCompletions(object):
    """Tiny OpenAI-compatible endpoint for client tests: handler(path, body) -> dict response."""

    def __init__(self, handler, models=None):
        outer = self
        self.calls = []
        self.models = models or {"id": "mock", "object": "model", "max_model_len": 300000, "chunk_size": 2048}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, payload, status=200):
                b = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/v1/models"):
                    return self._send({"object": "list", "data": [outer.models]})
                if self.path == "/health":
                    return self._send({"status": "ok"})
                if self.path == "/metrics":
                    return self._send({"mem_available_gib": 80.0, "torch_allocated_gib": 30.0})
                self._send({"error": {"message": "nf"}}, 404)

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                outer.calls.append((self.path, body))
                if body.get("stream"):
                    chunks = handler(self.path, body)
                    if isinstance(chunks, dict):
                        chunks = [{"choices": [{"index": 0, "text": chunks["choices"][0]["text"]}]}]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for c in chunks:
                        self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._send(handler(self.path, body))

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def close(self):
        self.srv.shutdown()
