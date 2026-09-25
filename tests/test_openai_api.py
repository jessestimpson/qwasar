#!/usr/bin/env python3
"""Integration tests: qwasar-server's OpenAI API against the OpenAI spec.

    make test-api                       # both APIs
    python3 tests/test_openai_api.py -v # this one

This drives a real server over HTTP and checks what comes back against the
shapes in OpenAI's published OpenAPI description (github.com/openai/openai-openapi)
-- the response schemas, the SSE framing of a streamed completion, the error
envelope, and the request parameters whose meaning the spec pins down (stop,
n, tool_choice, max_tokens, stream_options).

If the `openai` package happens to be installed, one extra class also runs the
official client against the server; it skips otherwise.  The server and the
environment variables that steer it are described in qwasar_api.py.
"""

import json
import socket
import time
import unittest
import urllib.parse

import qwasar_api as api
from qwasar_api import TIMEOUT, nullable, request

# Most tests only care about the envelope, so they turn thinking off (the
# server's documented extension) and keep the budget small.
FAST = {"enable_thinking": False, "temperature": 0, "max_tokens": 64}

# A document an agent might write: quotes and newlines, escaped twice on the
# way out (JSON arguments inside a JSON event), so a call carrying it is big.
LONG_DOC = "".join(f'{i:02d}. say "hello" to "{w}" and "goodbye" to "{w}s"\n'
                   for i, w in enumerate(["cat", "dog", "fox", "owl", "elk", "bee", "ant", "yak"] * 6))

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write text to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "Absolute path."},
                "content": {"type": "string", "description": "The whole file."},
            },
            "required": ["filePath", "content"],
        },
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"},
            },
            "required": ["location"],
        },
    },
}


# ---- schemas ----------------------------------------------------------------
#
# Transcribed from openai-openapi's components/schemas, keeping what the spec
# marks required and the enums it declares.  Extra properties are allowed
# everywhere, as they are in the spec -- reasoning_content is one.

FINISH_REASONS = ["stop", "length", "tool_calls", "content_filter", "function_call"]

COMPLETION_USAGE = {
    "type": "object",
    "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
    "properties": {
        "prompt_tokens": {"type": "integer"},
        "completion_tokens": {"type": "integer"},
        "total_tokens": {"type": "integer"},
    },
}

MESSAGE_TOOL_CALL = {
    "type": "object",
    "required": ["id", "type", "function"],
    "properties": {
        "id": {"type": "string"},
        "type": {"enum": ["function"]},
        "function": {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "string"},
            },
        },
    },
}

RESPONSE_MESSAGE = {
    "type": "object",
    "required": ["role", "content", "refusal"],
    "properties": {
        "role": {"enum": ["assistant"]},
        "content": nullable({"type": "string"}),
        "refusal": nullable({"type": "string"}),
        "tool_calls": {"type": "array", "items": MESSAGE_TOOL_CALL},
    },
}

CHAT_COMPLETION = {
    "type": "object",
    "required": ["id", "object", "created", "model", "choices"],
    "properties": {
        "id": {"type": "string"},
        "object": {"enum": ["chat.completion"]},
        "created": {"type": "integer"},
        "model": {"type": "string"},
        "system_fingerprint": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "message", "finish_reason", "logprobs"],
                "properties": {
                    "index": {"type": "integer"},
                    "message": RESPONSE_MESSAGE,
                    "finish_reason": {"enum": FINISH_REASONS},
                    "logprobs": nullable({"type": "object"}),
                },
            },
        },
        "usage": COMPLETION_USAGE,
    },
}

TOOL_CALL_CHUNK = {
    "type": "object",
    "required": ["index"],
    "properties": {
        "index": {"type": "integer"},
        "id": {"type": "string"},
        "type": {"enum": ["function"]},
        "function": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "string"},
            },
        },
    },
}

CHAT_COMPLETION_CHUNK = {
    "type": "object",
    "required": ["id", "object", "created", "model", "choices"],
    "properties": {
        "id": {"type": "string"},
        "object": {"enum": ["chat.completion.chunk"]},
        "created": {"type": "integer"},
        "model": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "delta", "finish_reason"],
                "properties": {
                    "index": {"type": "integer"},
                    "delta": {
                        "type": "object",
                        "properties": {
                            "role": {"enum": ["developer", "system", "user",
                                              "assistant", "tool"]},
                            "content": nullable({"type": "string"}),
                            "refusal": nullable({"type": "string"}),
                            "tool_calls": {"type": "array", "items": TOOL_CALL_CHUNK},
                        },
                    },
                    "finish_reason": nullable({"enum": FINISH_REASONS}),
                    "logprobs": nullable({"type": "object"}),
                },
            },
        },
        "usage": nullable(COMPLETION_USAGE),
    },
}

MODEL = {
    "type": "object",
    "required": ["id", "object", "created", "owned_by"],
    "properties": {
        "id": {"type": "string"},
        "object": {"enum": ["model"]},
        "created": {"type": "integer"},
        "owned_by": {"type": "string"},
    },
}

MODEL_LIST = {
    "type": "object",
    "required": ["object", "data"],
    "properties": {
        "object": {"enum": ["list"]},
        "data": {"type": "array", "items": MODEL},
    },
}

ERROR_RESPONSE = {
    "type": "object",
    "required": ["error"],
    "properties": {
        "error": {
            "type": "object",
            "required": ["message", "type", "param", "code"],
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string"},
                "param": nullable({"type": "string"}),
                "code": nullable({"type": "string"}),
            },
        },
    },
}


class Case(api.Case):
    SPEC = "the OpenAI spec"


def setUpModule():
    api.ensure_server()


def chat(**body):
    body.setdefault("model", "qwen3.8-27b")
    return request("POST", "/v1/chat/completions", body)


def chat_stream(**body):
    """A streamed completion.  Returns (status, headers, events), events being
    the `data:` payloads in order: dicts for JSON, "[DONE]" for the end."""
    body.setdefault("model", "qwen3.8-27b")
    body["stream"] = True
    status, headers, events = api.sse("/v1/chat/completions", body)
    if status == 200:
        names = [n for n, _ in events if n is not None]
        if names:
            headers["_other"].append(f"OpenAI streams carry no event names: {names[:3]}")
        events = [d for _, d in events]
    return status, headers, events


# ---- /v1/models -------------------------------------------------------------

class Models(Case):

    def test_list_shape(self):
        status, headers, body = request("GET", "/v1/models")
        self.assertOk(status, body)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        self.assertSchema(body, MODEL_LIST)
        self.assertGreaterEqual(len(body["data"]), 1)

    def test_retrieve_listed_model(self):
        _, _, listing = request("GET", "/v1/models")
        mid = listing["data"][0]["id"]
        status, _, body = request("GET", "/v1/models/" + urllib.parse.quote(mid, safe=""))
        self.assertOk(status, body)
        self.assertSchema(body, MODEL)
        self.assertEqual(body["id"], mid)

    def test_retrieve_unknown_model_is_404(self):
        # The spec's retrieve operation answers an id it does not serve with a
        # 404 error, not with some other model under the wrong name.
        status, headers, body = request("GET", "/v1/models/no-such-model-xyz")
        self.assertError(status, headers, body, 404)


# ---- /v1/chat/completions, not streamed -------------------------------------

class ChatCompletion(Case):

    @classmethod
    def setUpClass(cls):
        cls.status, cls.headers, cls.body = chat(
            messages=[{"role": "user", "content": "Say hello in one short sentence."}], **FAST)

    def test_status_and_content_type(self):
        self.assertOk(self.status, self.body)
        self.assertTrue(self.headers["content-type"].startswith("application/json"))

    def test_schema(self):
        self.assertSchema(self.body, CHAT_COMPLETION)

    def test_identity_fields(self):
        b = self.body
        self.assertTrue(b["id"].startswith("chatcmpl-"), b["id"])
        self.assertEqual(b["object"], "chat.completion")
        self.assertLess(abs(b["created"] - time.time()), 3600,
                        "created should be a Unix timestamp in seconds")
        _, _, listing = request("GET", "/v1/models")
        self.assertIn(b["model"], [m["id"] for m in listing["data"]])

    def test_single_choice(self):
        self.assertEqual(len(self.body["choices"]), 1)
        self.assertEqual(self.body["choices"][0]["index"], 0)

    def test_message(self):
        choice = self.body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIsInstance(choice["message"]["content"], str)
        self.assertTrue(choice["message"]["content"].strip())
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"],
                         "no tools were offered, so no tool_calls key")

    def test_usage_adds_up(self):
        u = self.body["usage"]
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertGreater(u["completion_tokens"], 0)
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def test_ids_are_unique(self):
        _, _, again = chat(messages=[{"role": "user", "content": "Say hi."}], **FAST)
        self.assertNotEqual(self.body["id"], again["id"])


class ChatParameters(Case):

    def test_max_tokens_truncates_with_length(self):
        status, _, body = chat(
            messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
            enable_thinking=False, temperature=0, max_tokens=5)
        self.assertOk(status, body)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertLessEqual(body["usage"]["completion_tokens"], 5)

    def test_max_completion_tokens(self):
        # The spec's current name for the budget; max_tokens is deprecated.
        status, _, body = chat(
            messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
            enable_thinking=False, temperature=0, max_completion_tokens=5)
        self.assertOk(status, body)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertLessEqual(body["usage"]["completion_tokens"], 5)

    def test_seed_is_reproducible(self):
        req = dict(messages=[{"role": "user", "content": "Name three fruits."}],
                   enable_thinking=False, temperature=0.9, seed=4242, max_tokens=24)
        _, _, a = chat(**req)
        _, _, b = chat(**req)
        self.assertEqual(a["choices"][0]["message"]["content"],
                         b["choices"][0]["message"]["content"])

    def test_system_message_is_honoured(self):
        status, _, body = chat(messages=[
            {"role": "system", "content": "Whatever the user asks, reply with exactly "
                                          "the single word BANANA and nothing else."},
            {"role": "user", "content": "What colour is the sky?"},
        ], **FAST)
        self.assertOk(status, body)
        self.assertIn("banana", body["choices"][0]["message"]["content"].lower())

    def test_content_parts_array(self):
        # User content may be an array of typed parts instead of a string.
        status, _, body = chat(messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is 2 + 3?"},
            {"type": "text", "text": "Answer with just the number."},
        ]}], **FAST)
        self.assertOk(status, body)
        self.assertIn("5", body["choices"][0]["message"]["content"])

    def test_multi_turn_history(self):
        status, _, body = chat(messages=[
            {"role": "user", "content": "My name is Zephyrine. Remember it."},
            {"role": "assistant", "content": "Got it, Zephyrine."},
            {"role": "user", "content": "What is my name? Answer with just the name."},
        ], **FAST)
        self.assertOk(status, body)
        self.assertIn("zephyrine", body["choices"][0]["message"]["content"].lower())

    def test_stop_sequence(self):
        # Output stops before the first stop sequence, which is not included.
        status, _, body = chat(
            messages=[{"role": "user", "content":
                       "Count from 1 to 10, digits separated by single spaces, nothing else."}],
            stop=[" 5"], enable_thinking=False, temperature=0, max_tokens=40)
        self.assertOk(status, body)
        content = body["choices"][0]["message"]["content"] or ""
        self.assertIn("4", content)
        self.assertNotIn("5", content, f"generation ran past the stop sequence: {content!r}")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_stop_as_string(self):
        status, _, body = chat(
            messages=[{"role": "user", "content":
                       "Count from 1 to 10, digits separated by single spaces, nothing else."}],
            stop=" 5", enable_thinking=False, temperature=0, max_tokens=40)
        self.assertOk(status, body)
        self.assertNotIn("5", body["choices"][0]["message"]["content"] or "")

    def test_n_is_honoured_or_rejected(self):
        # n asks for that many choices.  A server that cannot produce them may
        # refuse with a 400; silently returning one choice is the wrong answer.
        status, headers, body = chat(
            messages=[{"role": "user", "content": "Say hi."}], n=2, **FAST)
        if status == 200:
            self.assertEqual(len(body["choices"]), 2, "n=2 returned a different number of choices")
            self.assertEqual(sorted(c["index"] for c in body["choices"]), [0, 1])
        else:
            self.assertError(status, headers, body, 400)


class Reasoning(Case):
    """Thinking is on by default; the reasoning comes back beside the answer as
    reasoning_content, which the spec permits as an extra property."""

    def test_default_thinking_response_is_valid(self):
        status, _, body = chat(
            messages=[{"role": "user", "content": "What is 7 * 6? Answer briefly."}],
            temperature=0, reasoning_effort="low", max_tokens=1024)
        self.assertOk(status, body)
        msg = body["choices"][0]["message"]
        self.assertSchema(msg["content"], nullable({"type": "string"}))
        self.assertIsInstance(msg.get("reasoning_content"), str)
        if body["choices"][0]["finish_reason"] == "stop":
            self.assertIn("42", msg["content"] or "")


# ---- tools ------------------------------------------------------------------

TOOL_PROMPT = [{"role": "user", "content":
                "What's the weather in Paris right now? Use the get_weather tool."}]


class Tools(Case):

    @classmethod
    def setUpClass(cls):
        cls.status, _, cls.body = chat(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                       enable_thinking=False, temperature=0, max_tokens=256)

    def test_tool_calls_schema(self):
        # The envelope is ChatCompletion.test_schema's; this is the tool_calls
        # array a tool-using response adds to it.
        self.assertOk(self.status, self.body)
        self.assertSchema(self.body["choices"][0]["message"].get("tool_calls"),
                          {"type": "array", "items": MESSAGE_TOOL_CALL})

    def test_tool_call(self):
        choice = self.body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"].get("tool_calls")
        self.assertTrue(calls, "expected tool_calls")
        call = calls[0]
        self.assertTrue(call["id"])
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        # arguments is a JSON document serialised into a string.
        args = json.loads(call["function"]["arguments"])
        self.assertIsInstance(args, dict)
        self.assertIn("paris", str(args.get("location", "")).lower())

    def test_tool_call_ids_unique(self):
        calls = self.body["choices"][0]["message"].get("tool_calls") or []
        ids = [c["id"] for c in calls]
        self.assertEqual(len(ids), len(set(ids)))

    def test_tool_result_round_trip(self):
        call = self.body["choices"][0]["message"]["tool_calls"][0]
        status, _, body = chat(messages=TOOL_PROMPT + [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"],
             "content": json.dumps({"temperature_c": 17, "conditions": "light rain"})},
        ], tools=[WEATHER_TOOL], enable_thinking=False, temperature=0, max_tokens=128)
        self.assertOk(status, body)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        text = (choice["message"]["content"] or "").lower()
        self.assertTrue("17" in text or "rain" in text, f"answer ignores the tool result: {text!r}")

    def test_tool_choice_none(self):
        # tool_choice "none" means the model must answer in text.
        status, _, body = chat(messages=TOOL_PROMPT, tools=[WEATHER_TOOL], tool_choice="none",
                               enable_thinking=False, temperature=0, max_tokens=128)
        self.assertOk(status, body)
        choice = body["choices"][0]
        self.assertNotEqual(choice["finish_reason"], "tool_calls")
        self.assertFalse(choice["message"].get("tool_calls"))

    def test_tool_choice_required(self):
        status, _, body = chat(
            messages=[{"role": "user", "content": "Tell me a joke about Berlin."}],
            tools=[WEATHER_TOOL], tool_choice="required",
            enable_thinking=False, temperature=0, max_tokens=128)
        self.assertOk(status, body)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                         "get_weather")

    def test_tool_choice_required_with_thinking(self):
        # The forced call has to come after the reasoning block, not inside it.
        status, _, body = chat(
            messages=[{"role": "user", "content": "Is it raining in Oslo?"}],
            tools=[WEATHER_TOOL], tool_choice="required",
            temperature=0, reasoning_effort="low", max_tokens=1024)
        self.assertOk(status, body)
        msg = body["choices"][0]["message"]
        self.assertTrue(msg.get("tool_calls"), f"no tool call: {msg!r}"[:500])
        self.assertIsInstance(json.loads(msg["tool_calls"][0]["function"]["arguments"]), dict)

    def test_tool_choice_unknown_function_rejected(self):
        status, headers, body = chat(
            messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "launch_rockets"}}, **FAST)
        self.assertError(status, headers, body, 400)
        self.assertEqual(body["error"].get("param"), "tool_choice")

    def test_tool_choice_named_function(self):
        # Naming a function forces a call to it, even when the prompt does not
        # ask for one.
        status, _, body = chat(
            messages=[{"role": "user", "content": "Tell me a joke about Berlin."}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            enable_thinking=False, temperature=0, max_tokens=128)
        self.assertOk(status, body)
        calls = body["choices"][0]["message"].get("tool_calls") or []
        self.assertTrue(calls, "a named tool_choice must produce a call")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")


# ---- streaming --------------------------------------------------------------

class Streaming(Case):

    PROMPT = [{"role": "user", "content": "What is the capital of France? One word."}]

    @classmethod
    def setUpClass(cls):
        cls.status, cls.headers, cls.events = chat_stream(messages=cls.PROMPT, **FAST)

    def chunks(self):
        return [e for e in self.events if e != "[DONE]"]

    def test_status_and_content_type(self):
        self.assertOk(self.status, self.events)
        self.assertTrue(self.headers["content-type"].startswith("text/event-stream"),
                        self.headers["content-type"])

    def test_framing(self):
        self.assertEqual(self.headers["_other"], [], "non-data lines in the event stream")
        self.assertEqual(self.events[-1], "[DONE]", "stream must end with data: [DONE]")
        self.assertEqual(self.events.count("[DONE]"), 1)

    def test_every_chunk_matches_schema(self):
        for i, c in enumerate(self.chunks()):
            with self.subTest(chunk=i):
                self.assertSchema(c, CHAT_COMPLETION_CHUNK)

    def test_chunks_share_identity(self):
        chunks = self.chunks()
        self.assertTrue(chunks[0]["id"].startswith("chatcmpl-"))
        for key in ("id", "created", "model"):
            self.assertEqual(len({c[key] for c in chunks}), 1, f"{key} changes mid-stream")

    def test_first_delta_carries_role(self):
        first = self.chunks()[0]["choices"][0]
        self.assertEqual(first["delta"].get("role"), "assistant")

    def test_exactly_one_finish_reason_and_it_is_last(self):
        with_choices = [c for c in self.chunks() if c["choices"]]
        reasons = [c["choices"][0]["finish_reason"] for c in with_choices]
        finished = [r for r in reasons if r is not None]
        self.assertEqual(finished, ["stop"])
        self.assertIsNotNone(reasons[-1], "the finish_reason chunk must come last")

    def test_content_matches_non_streamed(self):
        text = "".join(c["choices"][0]["delta"].get("content") or ""
                       for c in self.chunks() if c["choices"])
        self.assertTrue(text.strip())
        _, _, body = chat(messages=self.PROMPT, **FAST)
        self.assertEqual(text, body["choices"][0]["message"]["content"])

    def test_include_usage(self):
        # With include_usage the spec adds one chunk just before [DONE] whose
        # choices array is empty and whose usage covers the whole request.
        status, _, events = chat_stream(messages=self.PROMPT,
                                        stream_options={"include_usage": True}, **FAST)
        self.assertOk(status, events)
        chunks = [e for e in events if e != "[DONE]"]
        last = chunks[-1]
        self.assertSchema(last, CHAT_COMPLETION_CHUNK)
        self.assertEqual(last["choices"], [], "the usage chunk has an empty choices array")
        u = last.get("usage")
        self.assertIsNotNone(u)
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def test_length_finish(self):
        status, _, events = chat_stream(
            messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
            enable_thinking=False, temperature=0, max_tokens=5)
        self.assertOk(status, events)
        reasons = [c["choices"][0]["finish_reason"] for c in events
                   if c != "[DONE]" and c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(reasons, ["length"])

    def test_streamed_tool_call(self):
        status, _, events = chat_stream(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                        enable_thinking=False, temperature=0, max_tokens=256)
        self.assertOk(status, events)
        calls = {}
        finish = None
        for c in events:
            if c == "[DONE]" or not c["choices"]:
                continue
            self.assertSchema(c, CHAT_COMPLETION_CHUNK)
            ch = c["choices"][0]
            finish = ch["finish_reason"] or finish
            for tc in ch["delta"].get("tool_calls") or []:
                if tc["index"] not in calls:
                    # The first fragment of each call carries its identity.
                    self.assertTrue(tc.get("id"), "first tool_call fragment has no id")
                    self.assertEqual(tc.get("type"), "function")
                    self.assertTrue(tc.get("function", {}).get("name"))
                    calls[tc["index"]] = {"name": tc["function"]["name"], "arguments": ""}
                calls[tc["index"]]["arguments"] += tc.get("function", {}).get("arguments") or ""
        self.assertEqual(finish, "tool_calls")
        content = "".join(c["choices"][0]["delta"].get("content") or ""
                          for c in events if c != "[DONE]" and c["choices"])
        self.assertNotIn("<function", content, "the tool call's XML leaked into content")
        self.assertNotIn("<parameter", content, "the tool call's XML leaked into content")
        self.assertEqual(sorted(calls), list(range(len(calls))), "tool_call indexes not 0..n-1")
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertIsInstance(json.loads(calls[0]["arguments"]), dict)

    def test_streamed_large_tool_call(self):
        # A tool call's arguments go out in one chunk.  An agent writing a file
        # makes that chunk tens of kilobytes -- and the server once formatted
        # every event through a 2 KB buffer, cutting it mid-string and running
        # the next event into it.  The content is quote- and newline-heavy, so
        # its JSON-in-JSON escaping crosses that size quickly.
        status, _, events = chat_stream(
            messages=[{"role": "user", "content":
                       "Use the write tool to save this exact text to /tmp/qw_notes.txt:\n\n"
                       + LONG_DOC}],
            tools=[WRITE_TOOL], tool_choice={"type": "function", "function": {"name": "write"}},
            enable_thinking=False, temperature=0, max_tokens=2048)
        self.assertOk(status, events)   # every event parsed as JSON on the way in
        args = ""
        for c in events:
            if c == "[DONE]" or not c["choices"]:
                continue
            for tc in c["choices"][0]["delta"].get("tool_calls") or []:
                args += tc.get("function", {}).get("arguments") or ""
        biggest = max(len(json.dumps(c)) for c in events if c != "[DONE]")
        if biggest < 2600:
            self.skipTest(f"the model wrote a short call ({biggest} bytes); nothing over 2 KB to check")
        self.assertIn("content", json.loads(args))

    def test_non_ascii_deltas_are_whole_characters(self):
        # A token can end partway through a multi-byte character.  Every delta
        # must still be valid UTF-8 (chat_stream decodes strictly), and the
        # pieces must add up to the non-streamed text.
        prompt = [{"role": "user", "content":
                   "Repeat exactly, nothing else: 你好世界 🌧️🐙🎉 naïve café"}]
        status, _, events = chat_stream(messages=prompt, **FAST)
        self.assertOk(status, events)
        text = "".join(c["choices"][0]["delta"].get("content") or ""
                       for c in events if c != "[DONE]" and c["choices"])
        _, _, body = chat(messages=prompt, **FAST)
        self.assertEqual(text, body["choices"][0]["message"]["content"])
        self.assertIn("🐙", text)

    def test_streamed_stop_sequence(self):
        status, _, events = chat_stream(
            messages=[{"role": "user", "content":
                       "Count from 1 to 10, digits separated by single spaces, nothing else."}],
            stop=[" 5"], enable_thinking=False, temperature=0, max_tokens=40)
        self.assertOk(status, events)
        chunks = [c for c in events if c != "[DONE]" and c["choices"]]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        self.assertIn("4", text)
        self.assertNotIn("5", text, f"streamed past the stop sequence: {text!r}")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_streamed_reasoning(self):
        status, _, events = chat_stream(
            messages=[{"role": "user", "content": "What is 7 * 6?"}],
            temperature=0, reasoning_effort="low", max_tokens=1024)
        self.assertOk(status, events)
        for c in events:
            if c != "[DONE]":
                self.assertSchema(c, CHAT_COMPLETION_CHUNK)
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content") or ""
                            for c in events if c != "[DONE]" and c["choices"])
        self.assertTrue(reasoning.strip(), "thinking was on but no reasoning_content streamed")


# ---- errors and HTTP --------------------------------------------------------

class Errors(Case):

    def test_malformed_json(self):
        status, headers, body = request("POST", "/v1/chat/completions", raw=b"{not json",
                                        headers={"Content-Type": "application/json"})
        self.assertError(status, headers, body, 400)

    def test_empty_messages(self):
        # messages has minItems 1 in the spec.
        status, headers, body = chat(messages=[], **FAST)
        self.assertError(status, headers, body, 400)

    def test_missing_messages(self):
        status, headers, body = request("POST", "/v1/chat/completions",
                                        {"model": "qwen3.8-27b"})
        self.assertError(status, headers, body, 400)

    def test_unknown_endpoint(self):
        status, headers, body = request("GET", "/v1/nope")
        self.assertError(status, headers, body, 404)

    def test_unimplemented_endpoints(self):
        for path in ("/v1/completions", "/v1/responses"):
            with self.subTest(path=path):
                status, headers, body = request("POST", path, {"model": "qwen3.8-27b"})
                self.assertIn(status, (404, 501))
                self.assertIsInstance(body.get("error", {}).get("message"), str)

    def test_envelope_schema(self):
        # Every error body is {"error": {message, type, param, code}}, with
        # param and code strings or null -- clients switch on code.
        cases = {
            "malformed JSON": lambda: request("POST", "/v1/chat/completions", raw=b"{",
                                              headers={"Content-Type": "application/json"}),
            "no messages": lambda: chat(messages=[], **FAST),
            "unknown endpoint": lambda: request("GET", "/v1/nope"),
            "unimplemented": lambda: request("POST", "/v1/completions", {"model": "x"}),
            "unknown model": lambda: request("GET", "/v1/models/no-such-model"),
            "unsupported parameter": lambda: chat(messages=[{"role": "user", "content": "hi"}],
                                                  n=3, **FAST),
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                self.assertSchema(call()[2], ERROR_RESPONSE)


class Http(Case):

    def test_keep_alive_serves_several_requests(self):
        conn = api.connect(30)
        try:
            for _ in range(3):
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                json.loads(resp.read())
        finally:
            conn.close()

    def test_idle_connection_does_not_block_others(self):
        # Pooled clients keep a connection open between requests.  That must
        # not lock anyone else out, whether it has sent nothing yet or has
        # finished a request and is sitting idle.
        silent = socket.create_connection((api.base().hostname, api.base().port), timeout=30)
        idle = api.connect(30)
        try:
            idle.request("GET", "/v1/models")
            idle.getresponse().read()
            status, _, body = request("GET", "/v1/models", timeout=15)
            self.assertOk(status, body)
            status, _, body = chat(messages=[{"role": "user", "content": "Say hi."}], **FAST)
            self.assertOk(status, body)
        finally:
            silent.close()
            idle.close()

    def raw_exchange(self, head, body_parts=(), wait_for_continue=False):
        """Sends a hand-built request and returns the raw response bytes."""
        sock = socket.create_connection((api.base().hostname, api.base().port), timeout=30)
        try:
            sock.sendall(head)
            if wait_for_continue:
                interim = sock.recv(4096)
                self.assertTrue(interim.startswith(b"HTTP/1.1 100 "),
                                f"no 100 Continue before the body: {interim[:80]!r}")
            for part in body_parts:
                sock.sendall(part)
            out = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                out += chunk
            return out
        finally:
            sock.close()

    def test_chunked_request_body(self):
        # Clients that stream their request body send it chunked, with no
        # Content-Length at all.
        body = json.dumps({"model": "qwen3.8-27b", "messages": [], **FAST}).encode()
        parts = [body[:10], body[10:]]
        chunked = b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in parts) + b"0\r\n\r\n"
        out = self.raw_exchange(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                                b"Content-Type: application/json\r\n"
                                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                                + chunked)
        # An empty messages list is refused -- by the handler, which means the
        # body arrived and parsed; an unread body fails as malformed JSON.
        self.assertTrue(out.startswith(b"HTTP/1.1 400 "), out[:200])
        self.assertIn(b"no messages", out)

    def test_expect_100_continue(self):
        body = json.dumps({"model": "qwen3.8-27b", "messages": []}).encode()
        out = self.raw_exchange(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                                b"Content-Type: application/json\r\n"
                                b"Content-Length: %d\r\nExpect: 100-continue\r\n"
                                b"Connection: close\r\n\r\n" % len(body),
                                [body], wait_for_continue=True)
        self.assertIn(b"no messages", out)

    def test_oversized_body_refused(self):
        out = self.raw_exchange(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                                b"Content-Type: application/json\r\n"
                                b"Content-Length: 99999999999\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 413 "), out[:200])

    def test_query_string_ignored(self):
        status, _, body = request("GET", "/v1/models?limit=5")
        self.assertOk(status, body)
        self.assertSchema(body, MODEL_LIST)

    def test_cors_preflight(self):
        if not api.spawned():
            self.skipTest("CORS depends on how the external server was started")
        status, headers, _ = request("OPTIONS", "/v1/chat/completions", headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization"})
        self.assertIn(status, (200, 204))
        self.assertEqual(headers.get("access-control-allow-origin"), "*")
        self.assertIn("POST", headers.get("access-control-allow-methods", ""))

    def test_bearer_auth_header_accepted(self):
        # Clients always send one; a server without auth must not choke on it.
        status, _, body = request("GET", "/v1/models",
                                  headers={"Authorization": "Bearer sk-test"})
        self.assertOk(status, body)


# ---- the official client ----------------------------------------------------

try:
    import openai  # noqa: E402
except ImportError:
    openai = None


@unittest.skipIf(openai is None, "the openai package is not installed")
class OfficialClient(Case):
    """One client for the class, closed at the end."""

    @classmethod
    def setUpClass(cls):
        cls._client = openai.OpenAI(base_url=f"http://{api.base().hostname}:{api.base().port}/v1",
                                    api_key="sk-unused", max_retries=0, timeout=TIMEOUT)

    @classmethod
    def tearDownClass(cls):
        cls._client.close()

    def client(self):
        return self._client

    def test_models_list(self):
        ids = [m.id for m in self.client().models.list()]
        self.assertTrue(ids)

    def test_create(self):
        r = self.client().chat.completions.create(
            model="qwen3.8-27b", messages=[{"role": "user", "content": "Say hi."}],
            temperature=0, max_tokens=32, extra_body={"enable_thinking": False})
        self.assertEqual(r.choices[0].message.role, "assistant")
        self.assertTrue(r.choices[0].message.content)

    def test_stream(self):
        text = ""
        for chunk in self.client().chat.completions.create(
                model="qwen3.8-27b", messages=[{"role": "user", "content": "Say hi."}],
                temperature=0, max_tokens=32, stream=True,
                extra_body={"enable_thinking": False}):
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        self.assertTrue(text)

    def test_tool_call(self):
        r = self.client().chat.completions.create(
            model="qwen3.8-27b", messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
            temperature=0, max_tokens=256, extra_body={"enable_thinking": False})
        call = r.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.name, "get_weather")
        json.loads(call.function.arguments)


if __name__ == "__main__":
    unittest.main()
