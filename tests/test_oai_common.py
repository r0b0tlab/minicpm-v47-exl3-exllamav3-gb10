import json
import os
import random
import unittest

from tests import helpers
import oai_common as oc

FX = helpers.FIXTURES


def tpl():
    with open(os.path.join(FX, "chat_template.jinja")) as fh:
        return oc.ChatTemplate(fh.read())


class TemplateParityTest(unittest.TestCase):
    """The renderer must match Hugging Face apply_chat_template byte for byte (fixtures were produced with
    transformers 5.18.0 and the pack's chat_template.jinja)."""

    def test_fixtures(self):
        with open(os.path.join(FX, "chat_render.json")) as fh:
            fixtures = json.load(fh)
        t = tpl()
        self.assertGreaterEqual(len(fixtures), 10)
        for name, c in fixtures.items():
            msgs = oc.normalize_messages(c["messages"])
            out = t.render(msgs, tools=c["tools"], add_generation_prompt=True, **c["kwargs"])
            self.assertEqual(out, c["expected"], name)

    def test_thinking_is_open_by_default(self):
        t = tpl()
        text = t.render(oc.normalize_messages([{"role": "user", "content": "hi"}]))
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(oc.thinking_open_in(text))
        off = t.render(oc.normalize_messages([{"role": "user", "content": "hi"}]), enable_thinking=False)
        self.assertTrue(off.endswith("<think>\n\n</think>\n\n"))
        self.assertFalse(oc.thinking_open_in(off))

    def test_bad_conversation_is_a_400(self):
        t = tpl()
        with self.assertRaises(oc.RequestError) as cm:
            t.render(oc.normalize_messages([{"role": "assistant", "content": "x"}]))
        self.assertEqual(cm.exception.status, 400)


class KwargsTest(unittest.TestCase):
    def test_alias_and_unknown_kwargs(self):
        kw, think = oc.template_kwargs_from_body(
            {"chat_template_kwargs": {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"}})
        self.assertTrue(think)
        kw, think = oc.template_kwargs_from_body({"chat_template_kwargs": {"thinking": False, "weird": [1]}})
        self.assertFalse(think)
        self.assertIs(kw["enable_thinking"], False)
        self.assertEqual(kw["weird"], [1])
        kw, think = oc.template_kwargs_from_body({"enable_thinking": False})
        self.assertFalse(think)
        kw, think = oc.template_kwargs_from_body({})
        self.assertTrue(think)
        # enable_thinking wins over the alias
        kw, think = oc.template_kwargs_from_body({"chat_template_kwargs": {"enable_thinking": True, "thinking": False}})
        self.assertTrue(think)
        kw, think = oc.template_kwargs_from_body({"chat_template_kwargs": {"thinking": {"type": "disabled"}}})
        self.assertFalse(think)
        with self.assertRaises(oc.RequestError):
            oc.template_kwargs_from_body({"chat_template_kwargs": "nope"})

    def test_reserved_names_do_not_break_rendering(self):
        t = tpl()
        out = t.render(oc.normalize_messages([{"role": "user", "content": "x"}]), documents="boom",
                       enable_thinking=False)
        self.assertIn("<think>\n\n</think>", out)


class MessagesTest(unittest.TestCase):
    def test_vision_parts_are_rejected_clearly(self):
        for part in ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                     {"type": "image", "image": "x"}, {"type": "video_url", "video_url": {"url": "x"}}):
            with self.assertRaises(oc.RequestError) as cm:
                oc.normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"}, part]}])
            self.assertEqual(cm.exception.status, 400)
            self.assertIn("vision tower", cm.exception.message)
            self.assertIn("text path only", cm.exception.message)

    def test_tool_call_arguments_json_string_becomes_object(self):
        msgs = oc.normalize_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
            {"role": "tool", "content": "r", "tool_call_id": "1"}])
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(msgs[1]["content"], "")
        with self.assertRaises(oc.RequestError):
            oc.normalize_messages([{"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": "{not json"}}]}])

    def test_roles_and_empty(self):
        with self.assertRaises(oc.RequestError):
            oc.normalize_messages([])
        with self.assertRaises(oc.RequestError):
            oc.normalize_messages([{"role": "wizard", "content": "x"}])
        self.assertEqual(oc.normalize_messages([{"role": "developer", "content": "x"}])[0]["role"], "system")


class SamplingTest(unittest.TestCase):
    D = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}

    def test_defaults_from_generation_config(self):
        sp = oc.parse_sampling({}, self.D, 1000, 10, 256)
        self.assertEqual((sp["temperature"], sp["top_k"], sp["top_p"], sp["max_new"]), (1.0, 20, 0.95, 256))

    def test_max_tokens_capped_by_window_and_overflow_is_400(self):
        sp = oc.parse_sampling({"max_tokens": 5000}, self.D, 1000, 900, 256)
        self.assertEqual(sp["max_new"], 100)
        self.assertTrue(sp["capped_by_window"])
        with self.assertRaises(oc.RequestError) as cm:
            oc.parse_sampling({}, self.D, 1000, 1000, 256)
        self.assertEqual(cm.exception.code, "context_length_exceeded")

    def test_validation(self):
        for body in ({"temperature": "hot"}, {"top_p": 2}, {"n": 2}, {"max_tokens": 0}, {"stop": [1]},
                     {"seed": "x"}):
            with self.assertRaises(oc.RequestError, msg=str(body)):
                oc.parse_sampling(body, self.D, 1000, 10, 256)
        sp = oc.parse_sampling({"temperature": 0, "stop": "END", "max_completion_tokens": 7}, self.D, 1000, 10, 256)
        self.assertEqual((sp["temperature"], sp["stop_strings"], sp["max_new"]), (0.0, ["END"], 7))


class ReasoningSplitTest(unittest.TestCase):
    def test_basic(self):
        r, c = oc.split_reasoning("step one\nstep two\n</think>\n\nThe answer is 42.", True)
        self.assertEqual((r, c), ("step one\nstep two", "The answer is 42."))

    def test_thinking_off_is_all_content(self):
        r, c = oc.split_reasoning("\n42 </think> odd", False)
        self.assertEqual(r, "")
        self.assertIn("</think>", c)

    def test_chunking_never_changes_the_result(self):
        text = "  \nthink a\n\nthink b </th></think>\n\n  Answer: <x> done.\n"
        want = oc.split_reasoning(text, True)
        rng = random.Random(3)
        for _ in range(200):
            s = oc.ReasoningSplitter(True)
            r_all, c_all, i = "", "", 0
            while i < len(text):
                n = rng.randrange(1, 9)
                r, c = s.feed(text[i:i + n])
                r_all, c_all, i = r_all + r, c_all + c, i + n
            r, c = s.finish()
            self.assertEqual((r_all + r, c_all + c), want)

    def test_partial_marker_is_held_back(self):
        s = oc.ReasoningSplitter(True)
        r, c = s.feed("abc</thi")
        self.assertEqual((r, c), ("abc", ""))
        r, c = s.feed("nk>done")
        self.assertEqual((r, c), ("", "done"))

    def test_closing_without_end_tag_for_stop_is_content(self):
        msg, calls = oc.final_message("42", True, None, False, "stop")
        self.assertEqual(msg["content"], "42")
        self.assertIsNone(msg["reasoning_content"])
        msg, _ = oc.final_message("still thinking", True, None, False, "length")
        self.assertEqual((msg["content"], msg["reasoning_content"]), ("", "still thinking"))
        msg, _ = oc.final_message("r\n</think>\n\nanswer", True, None, False, "stop")
        self.assertEqual((msg["content"], msg["reasoning_content"]), ("answer", "r"))


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                      "metric": {"type": "boolean"}, "opts": {"type": "object"}}}}}]


class ToolCallTest(unittest.TestCase):
    def test_parse_single_call_with_types(self):
        text = ("Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n"
                "<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>")
        content, calls = oc.parse_tool_calls(text, TOOLS)
        self.assertEqual(content, "Let me check.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Paris", "days": 3, "metric": True, "opts": {"a": [1, 2]}})
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_two_calls_and_multiline_string(self):
        text = ("<tool_call>\n<function=a>\n<parameter=s>\nline1\nline2\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=b>\n</function>\n</tool_call>")
        content, calls = oc.parse_tool_calls(text, None)
        self.assertIsNone(content)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"s": "line1\nline2"})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {})

    def test_malformed_or_truncated_leaves_raw_text(self):
        for text in ("<tool_call>\n<function=a>\n<parameter=x>\n1", "<tool_call>no function</tool_call>",
                     "plain answer", "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>"):
            content, calls = oc.parse_tool_calls(text, None)
            if text.startswith("plain"):
                self.assertEqual((content, calls), (text, []))
            else:
                self.assertEqual((content, calls), (text, []), text)

    def test_final_message_with_tool_call(self):
        text = "think\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"
        msg, has = oc.final_message(text, True, TOOLS, True, "stop")
        self.assertTrue(has)
        self.assertIsNone(msg["content"])
        self.assertEqual(msg["reasoning_content"], "think")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")

    def test_stream_filter_holds_back_partial_marker(self):
        f = oc.ToolStreamFilter()
        out = f.feed("hello <tool_c")
        self.assertEqual(out, "hello ")
        out += f.feed("all>\n<function=a></function></tool_call>")
        self.assertEqual(out, "hello ")
        rest, tool_text = f.finish()
        self.assertEqual(rest, "")
        self.assertTrue(tool_text.startswith("<tool_call>"))
        f = oc.ToolStreamFilter()
        self.assertEqual(f.feed("plain <tool"), "plain ")
        rest, tool = f.finish()
        self.assertEqual((rest, tool), ("<tool", ""))


class FramingTest(unittest.TestCase):
    def test_sse(self):
        b = oc.sse({"a": "é"})
        self.assertTrue(b.startswith(b"data: {"))
        self.assertTrue(b.endswith(b"\n\n"))
        self.assertEqual(json.loads(b[6:].decode()), {"a": "é"})
        self.assertEqual(oc.SSE_DONE, b"data: [DONE]\n\n")

    def test_chat_chunk_and_usage(self):
        ch = oc.chat_chunk("id", "m", 1, {"content": "x"})
        self.assertEqual(ch["object"], "chat.completion.chunk")
        self.assertIsNone(ch["choices"][0]["finish_reason"])
        u = oc.usage_obj(10, 5, 3)
        self.assertEqual((u["total_tokens"], u["prompt_tokens_details"]["cached_tokens"]), (15, 3))


if __name__ == "__main__":
    unittest.main()
