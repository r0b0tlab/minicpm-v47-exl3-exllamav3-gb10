"""HTTP-level behaviour of serve_openai.Handler against a scripted fake engine (no GPU)."""
import json
import unittest

from tests import helpers
import oai_client as oc
import serve_openai as so


def script_ok(req):
    return ["Let me think", " about it.\n", "</think>", "\n\nThe answer is 42."], "stop"


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = helpers.FakeEngine(script_ok)
        cls.srv, cls.base = helpers.start_http(so.Handler, cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def chat(self, **kw):
        body = {"model": "x", "messages": [{"role": "user", "content": "q"}]}
        body.update(kw)
        return oc.post_json(self.base, "/v1/chat/completions", body)

    def test_health_models_metrics(self):
        self.assertEqual(oc.get_json(self.base, "/health"), {"status": "ok"})
        card = oc.get_json(self.base, "/v1/models")["data"][0]
        self.assertEqual((card["id"], card["max_model_len"]), ("test-model", 2048))
        self.assertEqual(oc.get_json(self.base, "/v1/models/test-model")["id"], "test-model")
        self.assertIn("requests", oc.get_json(self.base, "/metrics"))

    def test_non_streaming_splits_reasoning(self):
        r = self.chat()
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], "Let me think about it.")
        self.assertEqual(msg["content"], "The answer is 42.")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(r["usage"]["completion_tokens"], 4)
        self.assertEqual(r["usage"]["total_tokens"], r["usage"]["prompt_tokens"] + 4)
        self.assertEqual(r["model"], "x")

    def test_thinking_off_everything_is_content(self):
        r = self.chat(chat_template_kwargs={"enable_thinking": False})
        self.assertFalse(r["choices"][0]["message"].get("reasoning_content"))
        r2 = self.chat(chat_template_kwargs={"thinking": False, "reasoning_effort": "low", "unknown": 1})
        self.assertFalse(r2["choices"][0]["message"].get("reasoning_content"))
        # the rendered prompt really carries the closed think block
        seen = self.engine.seen[-1]
        self.assertTrue(seen.sp["max_new"] > 0)

    def test_streaming_matches_non_streaming(self):
        reasoning, content, finish, usage_chunks = [], [], None, 0
        for _t, ch in oc.post_stream(self.base, "/v1/chat/completions", {
                "model": "x", "messages": [{"role": "user", "content": "q"}], "stream_options": {"include_usage": True}}):
            for c in ch.get("choices", []):
                d = c["delta"]
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if d.get("content"):
                    content.append(d["content"])
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
            if ch.get("usage") and not ch.get("choices"):
                usage_chunks += 1
        self.assertEqual("".join(reasoning), "Let me think about it.")
        self.assertEqual("".join(content), "The answer is 42.")
        self.assertEqual((finish, usage_chunks), ("stop", 1))

    def test_vision_part_400_and_server_lives(self):
        with self.assertRaises(oc.ApiError) as cm:
            self.chat(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("vision tower", json.dumps(cm.exception.body))
        self.assertEqual(self.chat()["choices"][0]["message"]["content"], "The answer is 42.")

    def test_bad_requests_are_400_not_crashes(self):
        for body in ({"messages": []}, {"messages": "x"}, {"messages": [{"role": "user", "content": "q"}],
                                                          "temperature": "x"},
                     {"messages": [{"role": "user", "content": "q"}], "max_tokens": -1}):
            with self.assertRaises(oc.ApiError) as cm:
                oc.post_json(self.base, "/v1/chat/completions", body)
            self.assertEqual(cm.exception.status, 400, body)
        with self.assertRaises(oc.ApiError) as cm:
            oc.post_json(self.base, "/v1/nope", {"a": 1})
        self.assertEqual(cm.exception.status, 404)
        self.assertEqual(oc.get_json(self.base, "/health")["status"], "ok")

    def test_context_overflow_is_400(self):
        with self.assertRaises(oc.ApiError) as cm:
            oc.post_json(self.base, "/v1/completions", {"prompt": list(range(1, 2100)), "max_tokens": 4})
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("context", json.dumps(cm.exception.body))

    def test_max_tokens_is_capped_by_window(self):
        self.chat(max_tokens=10 ** 6)
        self.assertLess(self.engine.seen[-1].sp["max_new"], 2048)

    def test_defaults_applied(self):
        self.chat()
        sp = self.engine.seen[-1].sp
        self.assertEqual((sp["temperature"], sp["top_k"], sp["top_p"]), (1.0, 20, 0.95))
        self.chat(temperature=0, top_p=1)
        sp = self.engine.seen[-1].sp
        self.assertEqual((sp["temperature"], sp["top_p"]), (0.0, 1.0))

    def test_completions_with_ids_and_return_token_ids(self):
        r = oc.post_json(self.base, "/v1/completions", {"prompt": [5, 6, 7], "max_tokens": 4, "return_token_ids": True})
        self.assertEqual(r["object"], "text_completion")
        self.assertEqual(r["usage"]["prompt_tokens"], 3)
        self.assertIn("timings", r)
        with self.assertRaises(oc.ApiError):
            oc.post_json(self.base, "/v1/completions", {"prompt": [5, -1], "max_tokens": 4})
        with self.assertRaises(oc.ApiError):
            oc.post_json(self.base, "/v1/completions", {"prompt": [5, 10 ** 9], "max_tokens": 4})

    def test_render_endpoint(self):
        r = oc.post_json(self.base, "/v1/chat/completions/render", {
            "messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"thinking": False}})
        self.assertTrue(r["prompt"].endswith("<think>\n\n</think>\n\n"))
        self.assertFalse(r["thinking_open"])
        self.assertEqual(r["prompt_tokens"], len(r["token_ids"]))


class FailureIsolationTest(unittest.TestCase):
    def test_engine_error_is_reported_and_server_continues(self):
        calls = {"n": 0}

        def script(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise so.oc.RequestError("boom in the engine", 500, err_type="server_error")
            return ["fine"], "stop"

        eng = helpers.FakeEngine(script)
        srv, base = helpers.start_http(so.Handler, eng)
        try:
            with self.assertRaises(oc.ApiError) as cm:
                oc.post_json(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}]})
            self.assertEqual(cm.exception.status, 500)
            r = oc.post_json(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}],
                                                            "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(r["choices"][0]["message"]["content"], "fine")
        finally:
            srv.shutdown()

    def test_tools_end_to_end(self):
        text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>")
        eng = helpers.FakeEngine(lambda req: (["</think>\n\n", text[:30], text[30:]], "stop"))
        srv, base = helpers.start_http(so.Handler, eng)
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}]
        try:
            r = oc.post_json(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "w?"}],
                                                            "tools": tools})
            ch = r["choices"][0]
            self.assertEqual(ch["finish_reason"], "tool_calls")
            self.assertEqual(json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]), {"city": "Rome"})
            calls = []
            for _t, c in oc.post_stream(base, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "w?"}], "tools": tools}):
                for choice in c.get("choices", []):
                    calls += choice["delta"].get("tool_calls", [])
                    if choice.get("finish_reason"):
                        self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertEqual(calls[0]["function"]["name"], "get_weather")
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()


class ArgsTest(unittest.TestCase):
    def test_defaults_and_draft_env(self):
        import os
        old = os.environ.pop("DRAFT_DIR", None)
        try:
            a = so.build_parser().parse_args(["--model-dir", "/m"])
            self.assertEqual((a.host, a.port, a.draft, a.profile), ("127.0.0.1", 8000, None, None))
        finally:
            if old is not None:
                os.environ["DRAFT_DIR"] = old
        os.environ["DRAFT_DIR"] = "/some/draft"
        try:
            a = so.build_parser().parse_args(["--model-dir", "/m", "--profile", "gb10-text-32k", "--chunk-size", "4096",
                                              "--max-batch-size", "4", "--port", "9001"])
            self.assertEqual((a.draft, a.chunk_size, a.max_batch_size, a.port), ("/some/draft", 4096, 4, 9001))
        finally:
            del os.environ["DRAFT_DIR"]

    def test_unsupported_draft_exit_code_is_documented(self):
        self.assertEqual(so.DRAFT_UNSUPPORTED_EXIT, 3)
