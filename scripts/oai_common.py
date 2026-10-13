"""OpenAI-compatible request/response logic that needs neither torch nor a GPU.

serve_openai.py (the server) imports this module; tests/ exercises it offline:

  * chat-template rendering with the pack's own chat_template.jinja (same Jinja
    environment settings as Hugging Face's apply_chat_template)
  * request normalisation: content parts, tool-call arguments, template kwargs
    (enable_thinking, with the alias key "thinking"; unknown kwargs are ignored)
  * reasoning split at </think> (non-streaming and streaming, chunk-boundary safe)
  * Qwen-style XML tool calls -> OpenAI tool_calls
  * sampling parameter parsing and SSE framing

Standard library plus jinja2 (a torch dependency, always present in the venv).
"""
import json
import re
import time
import uuid

THINK_OPEN = "<think>"
THINK_END = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class RequestError(Exception):
    """A client error: becomes an HTTP 4xx with an OpenAI-style error body."""

    def __init__(self, message, status=400, err_type="invalid_request_error", code=None, param=None):
        super().__init__(message)
        self.message, self.status, self.err_type, self.code, self.param = message, status, err_type, code, param

    def body(self):
        return {"error": {"message": self.message, "type": self.err_type, "param": self.param, "code": self.code}}


VISION_MESSAGE = ("image/video input needs an engine that implements the vision tower; "
                  "ExLlamaV3 runs the text path only")

_VISION_TYPES = {"image_url", "image", "input_image", "video", "video_url", "input_video"}
_AUDIO_TYPES = {"input_audio", "audio", "audio_url"}


# ------------------------------------------------------------------ template

_ENV = None


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise_exception(message):
    import jinja2
    raise jinja2.exceptions.TemplateError(message)


def _strftime_now(fmt):
    return time.strftime(fmt)


def jinja_env():
    global _ENV
    if _ENV is None:
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        _ENV = env
    return _ENV


class ChatTemplate(object):
    def __init__(self, source):
        self.source = source
        self._tpl = jinja_env().from_string(source)

    @classmethod
    def from_model_dir(cls, directory):
        import os
        p = os.path.join(directory, "chat_template.jinja")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return cls(fh.read())
        with open(os.path.join(directory, "tokenizer_config.json"), encoding="utf-8") as fh:
            src = json.load(fh).get("chat_template")
        if not src:
            raise RequestError("no chat template found in the model directory", 500)
        return cls(src)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        import jinja2
        reserved = {"messages", "tools", "add_generation_prompt", "documents"}
        extra = {k: v for k, v in kwargs.items() if k not in reserved}
        try:
            return self._tpl.render(messages=messages, tools=tools or None, documents=None,
                                    add_generation_prompt=add_generation_prompt, **extra)
        except jinja2.exceptions.TemplateError as exc:
            raise RequestError("chat template rejected the conversation: %s" % exc, 400, code="bad_conversation")


def thinking_open_in(prompt_text):
    """True when the rendered prompt leaves the model inside an open <think> block."""
    tail = prompt_text.rstrip()
    return tail.endswith(THINK_OPEN)


# ------------------------------------------------------------------ request normalisation

def _text_of_content(content, role):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        raise RequestError("message content must be a string or a list of content parts", param="messages")
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            raise RequestError("content parts must be objects", param="messages")
        t = item.get("type")
        if t in _VISION_TYPES or "image_url" in item or "image" in item or "video" in item:
            raise RequestError(VISION_MESSAGE, code="vision_not_supported", param="messages")
        if t in _AUDIO_TYPES:
            raise RequestError("audio input is not supported by this model path", code="audio_not_supported",
                               param="messages")
        if t in ("text", "input_text", None) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        else:
            raise RequestError("unsupported content part type %r" % (t,), param="messages")
    return "\n".join(parts)


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise RequestError("'messages' must be a non-empty list", param="messages")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise RequestError("messages[%d] must be an object" % i, param="messages")
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role == "function":
            role = "tool"
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError("messages[%d].role %r is not supported" % (i, role), param="messages")
        msg = {"role": role, "content": _text_of_content(m.get("content"), role)}
        reasoning = m.get("reasoning_content", m.get("reasoning"))
        if isinstance(reasoning, str):
            msg["reasoning_content"] = reasoning
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = (tc or {}).get("function") if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    raise RequestError("messages[%d].tool_calls entries need function.name" % i, param="messages")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        raise RequestError("messages[%d] tool call arguments are not valid JSON" % i,
                                           param="messages")
                if not isinstance(args, dict):
                    raise RequestError("messages[%d] tool call arguments must be a JSON object" % i, param="messages")
                calls.append({"id": tc.get("id"), "type": "function", "function": {"name": fn["name"], "arguments": args}})
            msg["tool_calls"] = calls
        if role == "tool":
            for k in ("tool_call_id", "name"):
                if isinstance(m.get(k), str):
                    msg[k] = m[k]
        out.append(msg)
    return out


def template_kwargs_from_body(body):
    """chat_template_kwargs with the alias key and top-level enable_thinking folded in.

    Returns (kwargs_for_template, enable_thinking_effective). Unknown keys are
    passed through to the template (which ignores variables it does not use);
    reserved names are filtered by ChatTemplate.render.
    """
    ctk = body.get("chat_template_kwargs")
    if ctk is None:
        ctk = {}
    if not isinstance(ctk, dict):
        raise RequestError("'chat_template_kwargs' must be an object", param="chat_template_kwargs")
    kw = dict(ctk)
    for key in ("enable_thinking", "thinking"):
        if key in body and key not in kw:
            kw[key] = body[key]
    if "enable_thinking" not in kw and "thinking" in kw:
        kw["enable_thinking"] = _as_bool(kw["thinking"])
    if "enable_thinking" in kw:
        kw["enable_thinking"] = _as_bool(kw["enable_thinking"])
    # thinking is ON by default (the template renders "<think>\n" unless enable_thinking is false)
    return kw, kw.get("enable_thinking", True) is not False


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):  # {"type": "enabled" | "disabled"}
        return str(v.get("type", "enabled")).lower() not in ("disabled", "off", "false")
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "off", "disabled", "")
    if v is None:
        return True
    return bool(v)


def parse_sampling(body, defaults, window, prompt_len, default_max_tokens):
    """-> dict(temperature, top_k, top_p, min_p, rep_p, pres_p, freq_p, seed, max_new, stop_strings)."""
    def num(key, lo, hi, default, cast=float):
        v = body.get(key)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RequestError("'%s' must be a number" % key, param=key)
        v = cast(v)
        if not (lo <= v <= hi):
            raise RequestError("'%s' must be between %s and %s" % (key, lo, hi), param=key)
        return v

    temperature = num("temperature", 0.0, 5.0, float(defaults.get("temperature", 1.0)))
    top_p = num("top_p", 0.0, 1.0, float(defaults.get("top_p", 1.0)))
    top_k = num("top_k", -1, 100000, int(defaults.get("top_k", 0)), int)
    top_k = max(0, top_k)
    min_p = num("min_p", 0.0, 1.0, 0.0)
    pres_p = num("presence_penalty", -2.0, 2.0, 0.0)
    freq_p = num("frequency_penalty", -2.0, 2.0, 0.0)
    rep_p = num("repetition_penalty", 0.1, 5.0, 1.0)
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RequestError("'seed' must be an integer", param="seed")
    n = body.get("n", 1)
    if n != 1:
        raise RequestError("only n=1 is supported", param="n")
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if mt is None:
        mt = default_max_tokens
    if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
        raise RequestError("'max_tokens' must be a positive integer", param="max_tokens")
    room = window - prompt_len
    if room < 1:
        raise RequestError(
            "This model's maximum context length is %d tokens. However, your messages resulted in %d tokens."
            % (window, prompt_len), code="context_length_exceeded", param="messages")
    max_new = min(mt, room)
    stop = body.get("stop")
    if stop is None:
        stop_strings = []
    elif isinstance(stop, str):
        stop_strings = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop) and len(stop) <= 8:
        stop_strings = list(stop)
    else:
        raise RequestError("'stop' must be a string or a list of up to 8 strings", param="stop")
    return {"temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p, "rep_p": rep_p,
            "pres_p": pres_p, "freq_p": freq_p, "seed": seed, "max_new": max_new, "requested_max": mt,
            "capped_by_window": mt > room, "stop_strings": [s for s in stop_strings if s]}


# ------------------------------------------------------------------ reasoning split

class ReasoningSplitter(object):
    """Splits a generated stream into (reasoning, content) at the first </think>.

    thinking_open=True  : the prompt ended inside <think>, output starts as reasoning
    thinking_open=False : output is content from the start (</think> is not interpreted)

    Streaming-safe: a partial "</think>" at a chunk boundary is held back, leading
    newlines of each section are dropped and trailing whitespace of the reasoning
    is only released if more reasoning follows, so the concatenated deltas always
    equal the non-streaming split (reasoning.strip(), content.lstrip()).
    """

    def __init__(self, thinking_open=True):
        self.in_reasoning = bool(thinking_open)
        self.buf = ""
        self.pending_ws = ""
        self.started = {"r": False, "c": False}
        self.saw_end = False

    def feed(self, text):
        """-> (reasoning_delta, content_delta)"""
        if not text:
            return "", ""
        self.buf += text
        r_out, c_out = "", ""
        while True:
            if self.in_reasoning:
                idx = self.buf.find(THINK_END)
                if idx >= 0:
                    seg, self.buf = self.buf[:idx], self.buf[idx + len(THINK_END):]
                    r_out += self._reasoning_emit(seg, final=True)
                    self.in_reasoning = False
                    self.saw_end = True
                    continue
                hold = _partial_suffix(self.buf, THINK_END)
                seg, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
                r_out += self._reasoning_emit(seg, final=False)
                break
            else:
                seg, self.buf = self.buf, ""
                c_out += self._content_emit(seg)
                break
        return r_out, c_out

    def finish(self):
        """Flush held text at the end of generation -> (reasoning_delta, content_delta)."""
        rest, self.buf = self.buf, ""
        if self.in_reasoning:
            return self._reasoning_emit(rest, final=True), ""
        return "", self._content_emit(rest)

    def _reasoning_emit(self, seg, final):
        out = ""
        if seg:
            seg = self.pending_ws + seg
            self.pending_ws = ""
            if not self.started["r"]:
                seg = seg.lstrip()
            stripped = seg.rstrip()
            ws = seg[len(stripped):]
            if stripped:
                self.started["r"] = True
                out = stripped
                self.pending_ws = ws if not final else ""
            else:
                self.pending_ws = ws if self.started["r"] and not final else ""
        if final:
            self.pending_ws = ""
        return out

    def _content_emit(self, seg):
        if not self.started["c"]:
            seg = seg.lstrip()
        if seg:
            self.started["c"] = True
        return seg


def _partial_suffix(text, marker):
    """Length of the longest proper prefix of marker that ends text."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


def split_reasoning(text, thinking_open=True):
    s = ReasoningSplitter(thinking_open)
    r1, c1 = s.feed(text)
    r2, c2 = s.finish()
    return r1 + r2, c1 + c2


# ------------------------------------------------------------------ tool calls

_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _coerce(value, schema_type):
    v = value
    if schema_type is None:
        s = v.strip()
        if s[:1] in "{[" or s in ("true", "false", "null"):
            try:
                return json.loads(s)
            except ValueError:
                return v
        return v
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    for t in types:
        try:
            if t == "string":
                return v
            if t == "integer":
                return int(v.strip())
            if t == "number":
                f = float(v.strip())
                return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
            if t == "boolean":
                low = v.strip().lower()
                if low in ("true", "1", "yes"):
                    return True
                if low in ("false", "0", "no"):
                    return False
                raise ValueError(v)
            if t == "null":
                if v.strip().lower() in ("null", "none", ""):
                    return None
                raise ValueError(v)
            if t in ("object", "array"):
                return json.loads(v.strip())
        except ValueError:
            continue
    return v


def _param_types(tools, fn_name):
    for t in tools or []:
        fn = (t or {}).get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == fn_name:
            props = ((fn.get("parameters") or {}).get("properties")) or {}
            return {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    return {}


def parse_tool_calls(text, tools=None):
    """-> (content, tool_calls). On any malformed block returns (text, []) untouched."""
    if TOOL_OPEN not in text:
        return text, []
    calls = []
    for m in _CALL_RE.finditer(text):
        body = m.group(1)
        fm = _FUNC_RE.search(body)
        if not fm:
            return text, []
        name = fm.group(1).strip()
        if not name:
            return text, []
        types = _param_types(tools, name)
        args = {}
        for pm in _PARAM_RE.finditer(fm.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2)
            if val.startswith("\n"):
                val = val[1:]
            if val.endswith("\n"):
                val = val[:-1]
            args[key] = _coerce(val, types.get(key))
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return text, []
    # an opened but never closed block means the generation was cut: keep raw text
    if text.count(TOOL_OPEN) != len(calls):
        return text, []
    content = _CALL_RE.sub("", text).strip()
    return (content or None), calls


class ToolStreamFilter(object):
    """Streams content until <tool_call> appears, then buffers the rest."""

    def __init__(self):
        self.buf = ""
        self.tool_mode = False
        self.tail = ""

    def feed(self, text):
        if self.tool_mode:
            self.tail += text
            return ""
        self.buf += text
        idx = self.buf.find(TOOL_OPEN)
        if idx >= 0:
            out, self.tail = self.buf[:idx], self.buf[idx:]
            self.buf = ""
            self.tool_mode = True
            return out
        hold = _partial_suffix(self.buf, TOOL_OPEN)
        out, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
        return out

    def finish(self):
        """-> (content_delta, buffered_tool_text)"""
        if self.tool_mode:
            return "", self.tail
        rest, self.buf = self.buf, ""
        return rest, ""


# ------------------------------------------------------------------ SSE / responses

def sse(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")


SSE_DONE = b"data: [DONE]\n\n"


def new_id(prefix):
    return "%s-%s" % (prefix, uuid.uuid4().hex[:24])


def usage_obj(prompt_tokens, completion_tokens, cached_tokens=0, reasoning_tokens=None):
    u = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
         "total_tokens": prompt_tokens + completion_tokens,
         "prompt_tokens_details": {"cached_tokens": cached_tokens}}
    if reasoning_tokens is not None:
        u["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return u


def chat_chunk(cid, model, created, delta, finish_reason=None, usage=None):
    ch = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
          "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        ch["usage"] = usage
    return ch


def final_message(text, thinking_open, tools, want_tools, finish="stop"):
    """Non-streaming assistant message from the full generated text.

    A turn that ends normally (finish "stop") without ever closing the <think>
    block is treated like the pack's chat template treats it: the whole output is
    the answer (content), there is no reasoning_content. A truncated generation
    ("length") inside the block keeps everything as reasoning_content.
    Streaming responses cannot take this back (the deltas are already sent) and
    deliver such a turn as reasoning_content.

    -> (message_dict, tool_calls_present)"""
    reasoning, content = split_reasoning(text, thinking_open)
    if thinking_open and THINK_END not in text and finish == "stop":
        reasoning, content = "", text.strip()
    tool_calls = []
    if want_tools:
        parsed_content, tool_calls = parse_tool_calls(content, tools)
        if tool_calls:
            content = parsed_content
    msg = {"role": "assistant", "content": content if (content or not tool_calls) else None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    elif thinking_open:
        msg["reasoning_content"] = None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg, bool(tool_calls)
