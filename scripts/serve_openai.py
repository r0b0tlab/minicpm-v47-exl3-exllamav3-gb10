#!/usr/bin/env python3
"""OpenAI-compatible server for the MiniCPM-V-4.7-35B-A3B EXL3 pack on ExLlamaV3 (text path).

Adapted from our minimal Qwen3.8 EXL3 server (MIT). One process, one GPU:

  GET  /health                         200 {"status":"ok"} when loaded, 503 while loading
  GET  /v1/models                      id, max_model_len (the profile's window)
  GET  /metrics                        counters, tokens/s, memory (JSON)
  POST /v1/chat/completions            stream and non-stream; tools; reasoning_content
  POST /v1/completions                 prompt: string | token ids | [token ids]
  POST /v1/chat/completions/render     rendered prompt + exact token ids (no generation)

Behaviour that matters for graders and clients:
  * thinking is ON by default (the prompt ends with "<think>\\n"); chat_template_kwargs
    {"enable_thinking": false} turns it off, the alias key "thinking" is honoured,
    unknown template kwargs are ignored. Text up to </think> is returned as
    message.reasoning_content, the rest as message.content.
  * generation defaults come from the model facts (temperature 1.0, top_k 20, top_p 0.95)
  * stop on <|im_end|> and <|endoftext|> (ids 248046 / 248044) plus request "stop" strings
  * usage.prompt_tokens / completion_tokens are the engine's own counts (the stop token
    is counted as a generated token, like vLLM)
  * max_tokens is capped by the window; a prompt that does not fit is a 400
  * image / video / audio content parts are a clear 400 (text path only)
  * requests are served FIFO through one worker thread; batch size is the profile's
    max_batch_size (1 unless a profile says otherwise and it was verified, docs/RESULTS.md)
  * a bad request returns an error and the server lives; ids >= vocab_size (head padding)
    are asserted never to be generated
  * --draft DIR enables DSpark drafting if (and only if) the ExLlamaV3 build supports it;
    otherwise the server refuses to start with a clear message.

Run through ./run.sh, or directly:
  python scripts/serve_openai.py --model-dir MODEL_DIR --profile gb10-text-262k --port 8000
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cfg  # noqa: E402
import oai_common as oc  # noqa: E402

MAX_BODY = 64 * 1024 * 1024
DRAFT_UNSUPPORTED_EXIT = 3


def log(msg):
    print("[serve] %s" % msg, flush=True)


def mem_available_gib():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1048576.0, 2)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------- engine

class GenRequest(object):
    def __init__(self, input_ids, sp, stop_ids, want_ids=False, min_new=0):
        self.id = uuid.uuid4().hex
        self.input_ids = input_ids          # torch (1, n) long, cpu
        self.sp = sp
        self.stop_ids = stop_ids
        self.min_new = min_new
        self.events = queue.Queue()
        self.token_ids = [] if want_ids else None
        self.cancelled = False
        self.t_submit = time.time()
        self.job = None


class Engine(object):
    def __init__(self, args, prof, facts):
        self.args, self.prof, self.facts = args, prof, facts
        self.window = int(prof["window_tokens"])
        self.vocab = int(facts["vocab_size"])
        self.ready = False
        self.draft_active = False
        self.q = queue.Queue()
        self.tok_lock = threading.Lock()
        self.m = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "pad_violations": 0,
                  "accounting_mismatches": 0, "cancelled": 0, "last_decode_tok_s": None, "started": time.time(),
                  "accepted_draft_tokens": 0, "rejected_draft_tokens": 0}
        self.active = {}

    # -- load
    def load(self):
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator
        from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
        self.torch = torch
        a, prof = self.args, self.prof
        self.template = oc.ChatTemplate.from_model_dir(a.model_dir)   # fail fast on a broken template
        t0 = time.time()
        log("loading %s (profile %s: window %d, kv_bits %s, batch %d, chunk %d)" % (
            os.path.basename(a.model_dir.rstrip("/")), prof["name"], self.window, prof["kv_bits"],
            prof["max_batch_size"], prof["chunk_size"]))
        free0, total0 = torch.cuda.mem_get_info()
        config = Config.from_directory(a.model_dir)
        model = Model.from_config(config)
        draft_model = draft_config = None
        if a.draft:
            draft_config = Config.from_directory(a.draft)
            draft_model = Model.from_config(draft_config)
        max_history = int(draft_model.caps.get("default_draft_size", 4)) if draft_model else 0

        def make_cache(m, **extra):
            if prof["kv_bits"] >= 16:
                return Cache(m, max_num_tokens=self.window, layer_type=CacheLayer_fp16, **extra)
            return Cache(m, max_num_tokens=self.window, layer_type=CacheLayer_quant,
                         k_bits=prof["kv_bits"], v_bits=prof["kv_bits"], **extra)

        cache = make_cache(model, max_batch_size=prof["max_batch_size"], max_history=max_history)
        model.load(progressbar=False, max_chunk_size=prof["chunk_size"], max_batch_size=prof["max_batch_size"])
        self.tokenizer = Tokenizer.from_config(config)
        draft_cache = None
        if draft_model is not None:
            draft_cache = make_cache(draft_model, max_batch_size=prof["max_batch_size"])
            draft_model.load(progressbar=False, max_chunk_size=prof["chunk_size"],
                             max_batch_size=prof["max_batch_size"])
        kwargs = {}
        if draft_model is not None:
            kwargs.update(draft_model=draft_model, draft_cache=draft_cache)
            if a.num_draft_tokens:
                kwargs["num_draft_tokens"] = a.num_draft_tokens
        self.generator = Generator(
            model, cache, self.tokenizer, max_batch_size=prof["max_batch_size"],
            max_chunk_size=prof["chunk_size"], recurrent_cache_size=int(float(prof["recurrent_cache_gib"]) * 1024 ** 3),
            **kwargs)
        self.model = model
        free1, total1 = torch.cuda.mem_get_info()
        self.load_seconds = time.time() - t0
        self.mem_after_load_gib = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
        log("loaded in %.0f s; torch allocated %.2f GiB, reserved %.2f GiB (free-memory reading: %.1f of %.1f GiB "
            "in use, includes page cache); MemAvailable %s GiB" % (
                self.load_seconds, torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3,
                (total1 - free1) / 1024 ** 3, total1 / 1024 ** 3, mem_available_gib()))
        if draft_model is not None:
            self._probe_draft()
        stop_ids = [int(i) for i in self.facts.get("stop_token_ids") or []]
        eos = getattr(config, "eos_token_id_list", None) or []
        for i in eos:
            if int(i) not in stop_ids:
                stop_ids.append(int(i))
        self.stop_ids = stop_ids
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.ready = True
        log("READY (%d stop ids %s)" % (len(stop_ids), stop_ids))

    def _probe_draft(self):
        """Feature-detect DSpark drafting: run a few drafted tokens; refuse cleanly when unsupported."""
        from exllamav3 import Job
        from exllamav3.generator.sampler import ArgmaxSampler
        ids = self.tokenizer.encode("The capital of France is", add_bos=False)
        try:
            job = Job(input_ids=ids, max_new_tokens=12, sampler=ArgmaxSampler(), identifier="probe")
            self.generator.enqueue(job)
            n = 0
            while self.generator.num_remaining_jobs():
                for r in self.generator.iterate():
                    n += 1
            self.draft_active = True
            log("draft probe ok: drafting active (%s)" % self.args.draft)
        except NotImplementedError as exc:
            log("DRAFT UNSUPPORTED: this ExLlamaV3 build cannot run the DSpark drafter (%s)" % exc)
            log("start without --draft / DRAFT_DIR, or use a build whose draft path is implemented")
            sys.stdout.flush()
            os._exit(DRAFT_UNSUPPORTED_EXIT)
        except Exception as exc:  # noqa: BLE001
            log("DRAFT PROBE FAILED: %s: %s" % (type(exc).__name__, exc))
            sys.stdout.flush()
            os._exit(DRAFT_UNSUPPORTED_EXIT)
        finally:
            try:
                self.generator.clear_queue()
            except Exception:  # noqa: BLE001
                pass

    # -- tokenisation helpers (called from handler threads)
    def encode(self, text):
        with self.tok_lock:
            return self.tokenizer.encode(text, add_bos=False, encode_special_tokens=True)

    def tensor_from_ids(self, ids):
        t = self.torch.tensor([ids], dtype=self.torch.long)
        return t

    def submit(self, req):
        self.m["requests"] += 1
        self.q.put(req)

    # -- worker
    def _sampler(self, sp):
        from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
        if sp["temperature"] == 0.0 or sp["top_k"] == 1:
            return ArgmaxSampler()
        return ComboSampler(rep_p=sp["rep_p"], pres_p=sp["pres_p"], freq_p=sp["freq_p"],
                            rep_sustain_range=4096, rep_decay_range=0,
                            temperature=sp["temperature"], min_p=sp["min_p"], top_k=sp["top_k"], top_p=sp["top_p"])

    def _start(self, req):
        from exllamav3 import Job
        sp = req.sp
        job = Job(input_ids=req.input_ids, max_new_tokens=sp["max_new"], min_new_tokens=req.min_new,
                  sampler=self._sampler(sp), seed=sp["seed"],
                  stop_conditions=list(req.stop_ids) + list(sp["stop_strings"]),
                  decode_special_tokens=False, identifier=req.id)
        req.job = job
        req.n_ids = 0
        req.max_id = -1
        req.pieces = []
        self.generator.enqueue(job)
        self.active[req.id] = req

    def _fail(self, req, message, status=500):
        self.m["errors"] += 1
        req.events.put(("error", oc.RequestError(message, status, err_type="server_error")))
        self.active.pop(req.id, None)

    def _worker(self):
        gen = self.generator
        while True:
            try:
                if not self.active and gen.num_remaining_jobs() == 0:
                    req = self.q.get()
                    self._admit(req)
                while len(self.active) < self.prof["max_batch_size"]:
                    try:
                        req = self.q.get_nowait()
                    except queue.Empty:
                        break
                    self._admit(req)
                for rid, req in list(self.active.items()):
                    if req.cancelled and req.job is not None:
                        gen.cancel(req.job)
                        self.active.pop(rid, None)
                        self.m["cancelled"] += 1
                if gen.num_remaining_jobs() == 0:
                    continue
                results = gen.iterate()
            except Exception as exc:  # noqa: BLE001
                log("worker error: %s\n%s" % (exc, traceback.format_exc()))
                for req in list(self.active.values()):
                    try:
                        if req.job is not None:
                            gen.cancel(req.job)
                    except Exception:  # noqa: BLE001
                        pass
                    self._fail(req, "%s: %s" % (type(exc).__name__, exc))
                try:
                    gen.clear_queue()
                except Exception:  # noqa: BLE001
                    pass
                continue
            for r in results:
                req = self.active.get(r.get("identifier"))
                if req is None:
                    continue
                try:
                    self._dispatch(req, r)
                except Exception as exc:  # noqa: BLE001
                    log("dispatch error: %s\n%s" % (exc, traceback.format_exc()))
                    self._fail(req, "%s: %s" % (type(exc).__name__, exc))

    def _admit(self, req):
        if req.cancelled:
            return
        try:
            self._start(req)
        except Exception as exc:  # noqa: BLE001
            log("job start failed: %s" % exc)
            self._fail(req, "%s: %s" % (type(exc).__name__, exc))

    def _dispatch(self, req, r):
        stage = r.get("stage")
        if stage != "streaming":
            return
        tid = r.get("token_ids")
        text = r.get("text") or ""
        if tid is not None:
            ids = tid.flatten().tolist() if hasattr(tid, "flatten") else list(tid)
            if ids:
                if req.token_ids is not None:
                    req.token_ids.extend(ids)
                req.n_ids += len(ids)
                mx = max(ids)
                req.max_id = max(req.max_id, mx)
                if mx >= self.vocab:
                    self.m["pad_violations"] += 1
                    raise AssertionError("generated id %d >= vocab_size %d (padding column)" % (mx, self.vocab))
        if text:
            req.pieces.append(text)
            req.events.put(("delta", text))
        if r.get("eos"):
            new_tokens = int(r.get("new_tokens") or 0)
            stop_tok = 1 if r.get("eos_reason") == "stop_token" else 0
            if req.n_ids + stop_tok != new_tokens:
                self.m["accounting_mismatches"] += 1
            info = {
                "eos_reason": r.get("eos_reason"),
                "completion_tokens": new_tokens,
                "prompt_tokens": int(r.get("prompt_tokens") or req.input_ids.shape[-1]),
                "cached_tokens": int(r.get("cached_tokens") or 0),
                "queue_s": float(r.get("time_enqueued") or 0.0),
                "prefill_s": float(r.get("time_prefill") or 0.0),
                "decode_s": float(r.get("time_generate") or 0.0),
                "max_id": req.max_id,
            }
            if req.token_ids is not None:
                info["token_ids"] = req.token_ids + ([self._stop_token_of(r)] if stop_tok else [])
            if "accepted_draft_tokens" in r:
                info["accepted_draft_tokens"] = int(r["accepted_draft_tokens"])
                info["rejected_draft_tokens"] = int(r.get("rejected_draft_tokens") or 0)
                self.m["accepted_draft_tokens"] += info["accepted_draft_tokens"]
                self.m["rejected_draft_tokens"] += info["rejected_draft_tokens"]
            if info["decode_s"] > 0 and new_tokens > 1:
                info["decode_tok_s"] = (new_tokens - 1) / info["decode_s"]
                self.m["last_decode_tok_s"] = round(info["decode_tok_s"], 2)
            self.m["prompt_tokens"] += info["prompt_tokens"]
            self.m["completion_tokens"] += new_tokens
            self.active.pop(req.id, None)
            req.events.put(("done", info))

    @staticmethod
    def _stop_token_of(r):
        v = r.get("eos_triggering_token_id")
        return int(v) if v is not None else -1

    def room(self, prompt_len):
        extra = 0
        if self.draft_active:
            extra = int(getattr(self.generator, "num_draft_tokens", 0))
        return self.window - 1 - extra - prompt_len


# ---------------------------------------------------------------------------- HTTP

ENGINE = None
SERVED_NAME = None
FACTS = None


def finish_reason_of(info):
    if info.get("eos_reason") == "max_new_tokens":
        return "length"
    return "stop"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "minicpm-v47-exl3-gb10"

    def log_message(self, fmt, *args):  # noqa: A003
        if os.environ.get("SERVE_HTTP_LOG"):
            log("%s %s" % (self.address_string(), fmt % args))

    # -- plumbing
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc):
        if isinstance(exc, oc.RequestError):
            ENGINE.m["errors"] += 1 if exc.status >= 500 else 0
            return self._json(exc.status, exc.body())
        ENGINE.m["errors"] += 1
        return self._json(500, {"error": {"message": "%s: %s" % (type(exc).__name__, exc), "type": "server_error",
                                           "param": None, "code": None}})

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise oc.RequestError("bad Content-Length")
        if length <= 0:
            raise oc.RequestError("empty request body")
        if length > MAX_BODY:
            raise oc.RequestError("request body too large", 413)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise oc.RequestError("invalid JSON: %s" % exc)
        if not isinstance(body, dict):
            raise oc.RequestError("request body must be a JSON object")
        return body

    # -- routes
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/health", "/v1/health", "/health/ready"):
            if ENGINE.ready:
                return self._json(200, {"status": "ok"})
            return self._json(503, {"status": "loading"})
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [self._model_card()]})
        if path.startswith("/v1/models/"):
            card = self._model_card()
            if path.rsplit("/", 1)[-1] == card["id"]:
                return self._json(200, card)
            return self._json(404, oc.RequestError("model not found", 404).body())
        if path == "/metrics":
            return self._json(200, self._metrics())
        return self._json(404, oc.RequestError("not found", 404).body())

    def _model_card(self):
        return {"id": SERVED_NAME, "object": "model", "created": int(ENGINE.m["started"]), "owned_by": "local",
                "max_model_len": ENGINE.window, "modalities": ["text"], "profile": ENGINE.prof["name"],
                "chunk_size": ENGINE.prof["chunk_size"], "kv_bits": ENGINE.prof["kv_bits"],
                "max_batch_size": ENGINE.prof["max_batch_size"], "drafting": ENGINE.draft_active}

    def _metrics(self):
        m = dict(ENGINE.m)
        m["uptime_s"] = round(time.time() - m.pop("started"), 1)
        m["active"] = len(ENGINE.active)
        m["queued"] = ENGINE.q.qsize()
        m["mem_available_gib"] = mem_available_gib()
        m["profile"] = ENGINE.prof["name"]
        m["window"] = ENGINE.window
        m["max_batch_size"] = ENGINE.prof["max_batch_size"]
        m["load_seconds"] = round(getattr(ENGINE, "load_seconds", 0.0), 1)
        m["mem_after_load_gib"] = getattr(ENGINE, "mem_after_load_gib", None)
        try:
            m["torch_allocated_gib"] = round(ENGINE.torch.cuda.memory_allocated() / 1024 ** 3, 2)
            m["torch_reserved_gib"] = round(ENGINE.torch.cuda.memory_reserved() / 1024 ** 3, 2)
            m["torch_max_reserved_gib"] = round(ENGINE.torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        except Exception:  # noqa: BLE001
            pass
        return m

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        try:
            if not ENGINE.ready:
                raise oc.RequestError("model is still loading", 503, err_type="server_error")
            body = self._body()
            if path == "/v1/chat/completions":
                return self._chat(body)
            if path == "/v1/completions":
                return self._completions(body)
            if path == "/v1/chat/completions/render":
                return self._render(body)
            raise oc.RequestError("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, oc.RequestError):
                log("request error: %s\n%s" % (exc, traceback.format_exc()))
            try:
                self._error(exc)
            except Exception:  # noqa: BLE001
                pass

    # -- chat
    def _prepare_chat(self, body):
        messages = oc.normalize_messages(body.get("messages"))
        tools = body.get("tools")
        tool_choice = body.get("tool_choice", "auto")
        want_tools = bool(tools) and tool_choice != "none"
        if tools is not None and not isinstance(tools, list):
            raise oc.RequestError("'tools' must be a list", param="tools")
        kwargs, thinking = oc.template_kwargs_from_body(body)
        text = ENGINE.template.render(messages, tools=tools if want_tools else None,
                                      add_generation_prompt=True, **kwargs)
        ids = ENGINE.encode(text)
        return messages, tools, want_tools, kwargs, thinking, text, ids

    def _render(self, body):
        messages, tools, want_tools, kwargs, thinking, text, ids = self._prepare_chat(body)
        flat = ids.flatten().tolist()
        return self._json(200, {"prompt": text, "token_ids": flat, "prompt_tokens": len(flat),
                                "thinking_open": oc.thinking_open_in(text)})

    def _make_request(self, ids, body, default_stop=True):
        n = int(ids.shape[-1])
        sp = oc.parse_sampling(body, FACTS["sampling_defaults"], ENGINE.room(0), n,
                               ENGINE.prof["default_max_tokens"])
        min_new = 0
        if body.get("ignore_eos") is True:
            min_new = sp["max_new"]
        elif isinstance(body.get("min_tokens"), int):
            min_new = max(0, min(int(body["min_tokens"]), sp["max_new"]))
        return GenRequest(ids, sp, ENGINE.stop_ids, want_ids=bool(body.get("return_token_ids")), min_new=min_new), n

    def _chat(self, body):
        messages, tools, want_tools, kwargs, thinking, text, ids = self._prepare_chat(body)
        req, n_prompt = self._make_request(ids, body)
        model_name = body.get("model") or SERVED_NAME
        stream = bool(body.get("stream"))
        thinking_open = oc.thinking_open_in(text)
        cid, created = oc.new_id("chatcmpl"), int(time.time())
        ENGINE.submit(req)
        if not stream:
            return self._chat_blocking(req, cid, created, model_name, thinking_open, tools, want_tools, n_prompt)
        return self._chat_stream(req, cid, created, model_name, thinking_open, tools, want_tools, body)

    def _drain(self, req):
        """Yield ('delta', text) ... then ('done', info); raises RequestError on ('error', exc)."""
        while True:
            try:
                ev = req.events.get(timeout=3600)
            except queue.Empty:
                req.cancelled = True
                raise oc.RequestError("generation timed out", 504, err_type="server_error")
            if ev[0] == "error":
                raise ev[1]
            yield ev
            if ev[0] == "done":
                return

    def _timings(self, info):
        t = {"queue_s": round(info["queue_s"], 4), "prefill_s": round(info["prefill_s"], 4),
             "decode_s": round(info["decode_s"], 4), "cached_tokens": info["cached_tokens"]}
        if "decode_tok_s" in info:
            t["decode_tok_s"] = round(info["decode_tok_s"], 2)
        return t

    def _usage(self, info, reasoning_tokens=None):
        u = oc.usage_obj(info["prompt_tokens"], info["completion_tokens"], info["cached_tokens"], reasoning_tokens)
        if "accepted_draft_tokens" in info:
            u.setdefault("completion_tokens_details", {}).update(
                accepted_prediction_tokens=info["accepted_draft_tokens"],
                rejected_prediction_tokens=info["rejected_draft_tokens"])
        return u

    def _chat_blocking(self, req, cid, created, model_name, thinking_open, tools, want_tools, n_prompt):
        parts, info = [], None
        try:
            for kind, val in self._drain(req):
                if kind == "delta":
                    parts.append(val)
                else:
                    info = val
        except oc.RequestError:
            raise
        text = "".join(parts)
        msg, has_calls = oc.final_message(text, thinking_open, tools, want_tools, finish_reason_of(info))
        reason = "tool_calls" if has_calls else finish_reason_of(info)
        return self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
            "usage": self._usage(info),
            "timings": self._timings(info),
        })

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _chat_stream(self, req, cid, created, model_name, thinking_open, tools, want_tools, body):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        splitter = oc.ReasoningSplitter(thinking_open)
        tfilter = oc.ToolStreamFilter() if want_tools else None
        self._sse_start()
        w = self.wfile
        try:
            w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"role": "assistant", "content": ""})))
            w.flush()
            info = None

            def emit_text(r, c):
                if r:
                    w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"reasoning_content": r})))
                if c:
                    if tfilter is not None:
                        c = tfilter.feed(c)
                    if c:
                        w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"content": c})))
                w.flush()

            for kind, val in self._drain(req):
                if kind == "delta":
                    emit_text(*splitter.feed(val))
                else:
                    info = val
            r, c = splitter.finish()
            emit_text(r, c)
            reason = finish_reason_of(info)
            if tfilter is not None:
                rest, tool_text = tfilter.finish()
                if rest:
                    w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"content": rest})))
                if tool_text:
                    content, calls = oc.parse_tool_calls(tool_text, tools)
                    if calls:
                        for i, call in enumerate(calls):
                            w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"tool_calls": [{
                                "index": i, "id": call["id"], "type": "function",
                                "function": {"name": call["function"]["name"],
                                             "arguments": call["function"]["arguments"]}}]})))
                        reason = "tool_calls"
                    else:
                        w.write(oc.sse(oc.chat_chunk(cid, model_name, created, {"content": tool_text})))
            final = oc.chat_chunk(cid, model_name, created, {}, finish_reason=reason)
            final["timings"] = self._timings(info)
            w.write(oc.sse(final))
            if include_usage:
                u = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                     "choices": [], "usage": self._usage(info)}
                w.write(oc.sse(u))
            w.write(oc.SSE_DONE)
            w.flush()
        except (BrokenPipeError, ConnectionResetError):
            req.cancelled = True
        except oc.RequestError as exc:
            try:
                w.write(oc.sse(exc.body()))
                w.write(oc.SSE_DONE)
                w.flush()
            except OSError:
                req.cancelled = True

    # -- raw completions
    def _prompt_ids(self, body):
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        if isinstance(prompt, str):
            if not prompt:
                raise oc.RequestError("'prompt' is empty", param="prompt")
            return ENGINE.encode(prompt)
        if isinstance(prompt, list) and prompt and all(isinstance(x, int) and not isinstance(x, bool) for x in prompt):
            if min(prompt) < 0 or max(prompt) >= ENGINE.vocab:
                raise oc.RequestError("token ids must be in [0, %d)" % ENGINE.vocab, param="prompt")
            return ENGINE.tensor_from_ids(prompt)
        raise oc.RequestError("'prompt' must be a string or a list of token ids", param="prompt")

    def _completions(self, body):
        ids = self._prompt_ids(body)
        req, n_prompt = self._make_request(ids, body)
        if body.get("max_tokens") is None:
            req.sp["max_new"] = min(req.sp["max_new"], 16)  # OpenAI's completions default
        model_name = body.get("model") or SERVED_NAME
        cid, created = oc.new_id("cmpl"), int(time.time())
        stream = bool(body.get("stream"))
        ENGINE.submit(req)
        if not stream:
            parts, info = [], None
            for kind, val in self._drain(req):
                if kind == "delta":
                    parts.append(val)
                else:
                    info = val
            choice = {"index": 0, "text": "".join(parts), "finish_reason": finish_reason_of(info), "logprobs": None}
            if "token_ids" in info:
                choice["token_ids"] = info["token_ids"]
            return self._json(200, {
                "id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [choice],
                "usage": self._usage(info), "timings": self._timings(info)})
        self._sse_start()
        w = self.wfile
        try:
            info = None
            for kind, val in self._drain(req):
                if kind == "delta":
                    w.write(oc.sse({"id": cid, "object": "text_completion", "created": created, "model": model_name,
                                    "choices": [{"index": 0, "text": val, "finish_reason": None}]}))
                    w.flush()
                else:
                    info = val
            w.write(oc.sse({"id": cid, "object": "text_completion", "created": created, "model": model_name,
                            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason_of(info)}],
                            "usage": self._usage(info), "timings": self._timings(info)}))
            w.write(oc.SSE_DONE)
            w.flush()
        except (BrokenPipeError, ConnectionResetError):
            req.cancelled = True
        except oc.RequestError as exc:
            try:
                w.write(oc.sse(exc.body()))
                w.write(oc.SSE_DONE)
                w.flush()
            except OSError:
                req.cancelled = True


class QuietServer(ThreadingHTTPServer):
    """Client disconnects are routine (probes with short timeouts): one log line, no traceback."""

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        log("http handler error from %s: %r" % (client_address[0], exc))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--draft", default=os.environ.get("DRAFT_DIR") or None,
                    help="DSpark draft directory (needs an ExLlamaV3 build with drafting; refuses otherwise)")
    ap.add_argument("--num-draft-tokens", type=int, default=0)
    ap.add_argument("--window", type=int, default=0, help="override the profile's window_tokens")
    ap.add_argument("--chunk-size", type=int, default=0, help="override the profile's prefill chunk size")
    ap.add_argument("--max-batch-size", type=int, default=0, help="override the profile's max_batch_size")
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--config-root", default=cfg.ROOT)
    return ap


def main(argv=None):
    global ENGINE, SERVED_NAME, FACTS
    args = build_parser().parse_args(argv)
    model_doc = cfg.load_model(args.config_root)
    FACTS = model_doc["model"]
    prof = cfg.get_profile(cfg.load_profiles(args.config_root), args.profile)
    if args.window:
        prof["window_tokens"] = args.window
    if args.chunk_size:
        prof["chunk_size"] = args.chunk_size
    if args.max_batch_size:
        prof["max_batch_size"] = args.max_batch_size
    SERVED_NAME = args.served_name or FACTS["served_name"]
    ENGINE = Engine(args, prof, FACTS)
    httpd = QuietServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.request_queue_size = 128
    # listen first so /health can say "loading"; then load the model
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    log("listening on http://%s:%d (loading ...)" % (args.host, args.port))
    try:
        ENGINE.load()
    except BaseException as exc:  # noqa: BLE001
        log("LOAD FAILED: %s: %s" % (type(exc).__name__, exc))
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(2)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
