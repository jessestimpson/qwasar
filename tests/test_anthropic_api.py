#!/usr/bin/env python3
"""Integration tests: qwasar-server's Anthropic API against the Messages spec.

    make test-api                          # both APIs
    python3 tests/test_anthropic_api.py -v # this one

This drives a real server over HTTP and checks /v1/messages, its streaming
event protocol, /v1/messages/count_tokens and the Anthropic shape of
/v1/models against the Messages API as the official SDK types describe it
(anthropic-sdk-python, src/anthropic/types): the Message and content-block
schemas, the order and pairing of stream events, the error envelope, and the
request parameters whose meaning the API pins down -- stop_sequences,
tool_choice, thinking, max_tokens, system, and assistant prefill.

If the `anthropic` package happens to be installed, one extra class also runs
the official client, whose stream accumulator is strict about event order; it
skips otherwise.  The server and the environment variables that steer it are
described in qwasar_api.py.
"""

import datetime
import json
import unittest
import urllib.parse

import qwasar_api as api
from qwasar_api import TIMEOUT, nullable, request, tagged

# Every Anthropic client sends these; the server keys the shape of shared
# endpoints such as /v1/models on anthropic-version.
HEADERS = {"anthropic-version": "2023-06-01", "x-api-key": "sk-test"}

# Most tests only care about the envelope, so they turn thinking off and keep
# the budget small.
FAST = {"thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 64}

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name, e.g. Paris"}},
        "required": ["location"],
    },
}

TOOL_PROMPT = [{"role": "user", "content":
                "What's the weather in Paris right now? Use the get_weather tool."}]

# See the OpenAI suite: a quote- and newline-heavy document, for a big call.
LONG_DOC = "".join(f'{i:02d}. say "hello" to "{w}" and "goodbye" to "{w}s"\n'
                   for i, w in enumerate(["cat", "dog", "fox", "owl", "elk", "bee", "ant", "yak"] * 6))

WRITE_TOOL = {
    "name": "write",
    "description": "Write text to a file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Absolute path."},
            "content": {"type": "string", "description": "The whole file."},
        },
        "required": ["filePath", "content"],
    },
}

COUNT_PROMPT = [{"role": "user", "content":
                 "Count from 1 to 10, digits separated by single spaces, nothing else."}]


# ---- schemas ----------------------------------------------------------------
#
# From the SDK's pydantic models.  A field the SDK types as Optional with a
# default is left out of `required`; stop_reason and stop_sequence are
# required anyway, because the API reference shows them on every Message.

STOP_REASONS = ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn",
                "refusal", "model_context_window_exceeded"]

ERROR_TYPES = ["invalid_request_error", "authentication_error", "permission_error",
               "not_found_error", "request_too_large", "rate_limit_error", "timeout_error",
               "overloaded_error", "api_error", "billing_error"]

USAGE = {
    "type": "object",
    "required": ["input_tokens", "output_tokens"],
    "properties": {
        "input_tokens": {"type": "integer"},
        "output_tokens": {"type": "integer"},
        "cache_creation_input_tokens": nullable({"type": "integer"}),
        "cache_read_input_tokens": nullable({"type": "integer"}),
    },
}

TEXT_BLOCK = {
    "type": "object",
    "required": ["type", "text"],
    "properties": {"text": {"type": "string"}},
}

THINKING_BLOCK = {
    "type": "object",
    "required": ["type", "thinking", "signature"],
    "properties": {"thinking": {"type": "string"}, "signature": {"type": "string"}},
}

TOOL_USE_BLOCK = {
    "type": "object",
    "required": ["type", "id", "name", "input"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "input": {"type": "object"},
    },
}

CONTENT_BLOCK = tagged(text=TEXT_BLOCK, thinking=THINKING_BLOCK, tool_use=TOOL_USE_BLOCK)

MESSAGE = {
    "type": "object",
    "required": ["id", "type", "role", "content", "model", "stop_reason", "stop_sequence",
                 "usage"],
    "properties": {
        "id": {"type": "string"},
        "type": {"enum": ["message"]},
        "role": {"enum": ["assistant"]},
        "content": {"type": "array", "items": CONTENT_BLOCK},
        "model": {"type": "string"},
        "stop_reason": nullable({"enum": STOP_REASONS}),
        "stop_sequence": nullable({"type": "string"}),
        "usage": USAGE,
    },
}

ERROR_OBJECT = {
    "type": "object",
    "required": ["type", "message"],
    "properties": {"type": {"enum": ERROR_TYPES}, "message": {"type": "string"}},
}

ERROR_RESPONSE = {
    "type": "object",
    "required": ["type", "error"],
    "properties": {"type": {"enum": ["error"]}, "error": ERROR_OBJECT},
}

DELTA = tagged(
    text_delta={"type": "object", "required": ["type", "text"],
                "properties": {"text": {"type": "string"}}},
    input_json_delta={"type": "object", "required": ["type", "partial_json"],
                      "properties": {"partial_json": {"type": "string"}}},
    thinking_delta={"type": "object", "required": ["type", "thinking"],
                    "properties": {"thinking": {"type": "string"}}},
    signature_delta={"type": "object", "required": ["type", "signature"],
                     "properties": {"signature": {"type": "string"}}},
)

STREAM_EVENT = tagged(
    message_start={"type": "object", "required": ["type", "message"],
                   "properties": {"message": MESSAGE}},
    content_block_start={"type": "object", "required": ["type", "index", "content_block"],
                         "properties": {"index": {"type": "integer"},
                                        "content_block": CONTENT_BLOCK}},
    content_block_delta={"type": "object", "required": ["type", "index", "delta"],
                         "properties": {"index": {"type": "integer"}, "delta": DELTA}},
    content_block_stop={"type": "object", "required": ["type", "index"],
                        "properties": {"index": {"type": "integer"}}},
    message_delta={"type": "object", "required": ["type", "delta", "usage"],
                   "properties": {
                       "delta": {"type": "object",
                                 "required": ["stop_reason", "stop_sequence"],
                                 "properties": {
                                     "stop_reason": nullable({"enum": STOP_REASONS}),
                                     "stop_sequence": nullable({"type": "string"})}},
                       "usage": {"type": "object", "required": ["output_tokens"],
                                 "properties": {"output_tokens": {"type": "integer"}}}}},
    message_stop={"type": "object", "required": ["type"]},
    ping={"type": "object", "required": ["type"]},
    error={"type": "object", "required": ["type", "error"],
           "properties": {"error": ERROR_OBJECT}},
)

MODEL_INFO = {
    "type": "object",
    "required": ["type", "id", "display_name", "created_at"],
    "properties": {
        "type": {"enum": ["model"]},
        "id": {"type": "string"},
        "display_name": {"type": "string"},
        "created_at": {"type": "string"},
    },
}

MODEL_PAGE = {
    "type": "object",
    "required": ["data", "has_more", "first_id", "last_id"],
    "properties": {
        "data": {"type": "array", "items": MODEL_INFO},
        "has_more": {"type": "boolean"},
        "first_id": nullable({"type": "string"}),
        "last_id": nullable({"type": "string"}),
    },
}

# Which delta types each kind of block may receive.
DELTAS_FOR = {
    "text": {"text_delta"},
    "thinking": {"thinking_delta", "signature_delta"},
    "tool_use": {"input_json_delta"},
}


# ---- helpers ----------------------------------------------------------------

def setUpModule():
    api.ensure_server()


def messages(**body):
    body.setdefault("model", "qwen3.8-27b")
    body.setdefault("max_tokens", 64)
    return request("POST", "/v1/messages", body, headers=HEADERS)


def messages_stream(**body):
    body.setdefault("model", "qwen3.8-27b")
    body.setdefault("max_tokens", 64)
    body["stream"] = True
    return api.sse("/v1/messages", body, headers=HEADERS)


def text_of(message):
    return "".join(b["text"] for b in message["content"] if b["type"] == "text")


def blocks(message, kind):
    return [b for b in message["content"] if b["type"] == kind]


class Case(api.Case):
    SPEC = "the Anthropic Messages API"

    def assertStream(self, events):
        """Checks a Messages stream event by event and as a protocol, and
        returns the Message it adds up to -- content blocks assembled from
        their deltas, stop reason and usage from message_delta -- so tests can
        compare it with a non-streamed reply."""
        self.assertTrue(events, "empty stream")
        for i, (name, data) in enumerate(events):
            self.assertNotEqual(data, "[DONE]", "Anthropic streams have no [DONE] sentinel")
            self.assertSchema(data, STREAM_EVENT)
            self.assertEqual(name, data["type"], f"event {i}: `event:` name differs from its type")

        kinds = [d["type"] for _, d in events if d["type"] != "ping"]
        self.assertEqual(kinds[0], "message_start")
        self.assertEqual(kinds[-1], "message_stop")
        self.assertEqual(kinds[-2], "message_delta", "message_delta must come just before "
                                                     "message_stop")
        self.assertEqual(kinds.count("message_start"), 1)
        self.assertEqual(kinds.count("message_delta"), 1)

        message = dict(events[0][1]["message"])
        self.assertEqual(message["content"], [], "message_start carries no content yet")
        content, open_block, partial, last_delta = [], None, "", None
        for _, d in events[1:]:
            t = d["type"]
            if t == "content_block_start":
                self.assertIsNone(open_block, "a block started before the last one stopped")
                self.assertEqual(d["index"], len(content), "block indexes must run 0, 1, 2, ...")
                open_block = dict(d["content_block"])
                partial, last_delta = "", None
            elif t == "content_block_delta":
                self.assertIsNotNone(open_block, "a delta outside any block")
                self.assertEqual(d["index"], len(content), "delta for a block that is not open")
                dt = d["delta"]["type"]
                self.assertIn(dt, DELTAS_FOR[open_block["type"]],
                              f"{dt} sent to a {open_block['type']} block")
                if dt == "text_delta":
                    open_block["text"] += d["delta"]["text"]
                elif dt == "thinking_delta":
                    self.assertNotEqual(last_delta, "signature_delta",
                                        "thinking after the block's signature")
                    open_block["thinking"] += d["delta"]["thinking"]
                elif dt == "signature_delta":
                    open_block["signature"] = d["delta"]["signature"]
                else:
                    partial += d["delta"]["partial_json"]
                last_delta = dt
            elif t == "content_block_stop":
                self.assertIsNotNone(open_block, "a stop outside any block")
                self.assertEqual(d["index"], len(content))
                if open_block["type"] == "thinking":
                    # The SDK types: "Delivered in a signature_delta event just
                    # before the block's content_block_stop event."
                    self.assertEqual(last_delta, "signature_delta",
                                     "a thinking block must end with its signature_delta")
                    self.assertTrue(open_block.get("signature"))
                if open_block["type"] == "tool_use" and partial:
                    open_block["input"] = json.loads(partial)
                content.append(open_block)
                open_block = None
            elif t == "message_delta":
                self.assertIsNone(open_block, "message_delta while a block is still open")
                message["stop_reason"] = d["delta"]["stop_reason"]
                message["stop_sequence"] = d["delta"]["stop_sequence"]
                message["usage"] = dict(message["usage"], **d["usage"])
        message["content"] = content
        self.assertSchema(message, MESSAGE)
        return message


# ---- /v1/models -------------------------------------------------------------

class Models(Case):

    def test_list_shape(self):
        status, headers, body = request("GET", "/v1/models", headers=HEADERS)
        self.assertOk(status, body)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        self.assertSchema(body, MODEL_PAGE)
        self.assertGreaterEqual(len(body["data"]), 1)
        self.assertEqual(body["first_id"], body["data"][0]["id"])
        self.assertEqual(body["last_id"], body["data"][-1]["id"])

    def test_created_at_is_rfc3339(self):
        _, _, body = request("GET", "/v1/models", headers=HEADERS)
        when = datetime.datetime.fromisoformat(body["data"][0]["created_at"])
        self.assertIsNotNone(when.tzinfo, "created_at should carry a timezone")

    def test_retrieve(self):
        _, _, listing = request("GET", "/v1/models", headers=HEADERS)
        mid = listing["data"][0]["id"]
        status, _, body = request("GET", "/v1/models/" + urllib.parse.quote(mid, safe=""),
                                  headers=HEADERS)
        self.assertOk(status, body)
        self.assertSchema(body, MODEL_INFO)
        self.assertEqual(body["id"], mid)

    def test_retrieve_unknown_is_404(self):
        status, headers, body = request("GET", "/v1/models/no-such-model", headers=HEADERS)
        self.assertError(status, headers, body, 404)
        self.assertEqual(body["error"].get("type"), "not_found_error")


# ---- /v1/messages, not streamed ---------------------------------------------

class Message(Case):

    @classmethod
    def setUpClass(cls):
        cls.status, cls.headers, cls.body = messages(
            messages=[{"role": "user", "content": "Say hello in one short sentence."}], **FAST)

    def test_status_and_content_type(self):
        self.assertOk(self.status, self.body)
        self.assertTrue(self.headers["content-type"].startswith("application/json"))

    def test_schema(self):
        self.assertSchema(self.body, MESSAGE)

    def test_identity_fields(self):
        self.assertTrue(self.body["id"].startswith("msg_"), self.body["id"])
        self.assertEqual(self.body["type"], "message")
        self.assertEqual(self.body["role"], "assistant")

    def test_content(self):
        self.assertTrue(text_of(self.body).strip())
        self.assertEqual(blocks(self.body, "thinking"), [], "thinking was disabled")
        self.assertEqual(self.body["stop_reason"], "end_turn")
        self.assertIsNone(self.body["stop_sequence"])

    def test_usage(self):
        u = self.body["usage"]
        self.assertGreater(u["input_tokens"], 0)
        self.assertGreater(u["output_tokens"], 0)

    def test_ids_are_unique(self):
        _, _, again = messages(messages=[{"role": "user", "content": "Say hi."}], **FAST)
        self.assertNotEqual(self.body["id"], again["id"])


class Parameters(Case):

    def test_max_tokens(self):
        status, _, body = messages(
            messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
            thinking={"type": "disabled"}, temperature=0, max_tokens=5)
        self.assertOk(status, body)
        self.assertEqual(body["stop_reason"], "max_tokens")
        self.assertLessEqual(body["usage"]["output_tokens"], 5)

    def test_stop_sequences(self):
        # The reply ends before the sequence, which is named in stop_sequence.
        status, _, body = messages(messages=COUNT_PROMPT, stop_sequences=[" 5", "zzz"],
                                   thinking={"type": "disabled"}, temperature=0, max_tokens=40)
        self.assertOk(status, body)
        text = text_of(body)
        self.assertIn("4", text)
        self.assertNotIn("5", text, f"generation ran past the stop sequence: {text!r}")
        self.assertEqual(body["stop_reason"], "stop_sequence")
        self.assertEqual(body["stop_sequence"], " 5")

    def test_system_string(self):
        status, _, body = messages(
            system="Whatever the user asks, reply with exactly the single word BANANA "
                   "and nothing else.",
            messages=[{"role": "user", "content": "What colour is the sky?"}], **FAST)
        self.assertOk(status, body)
        self.assertIn("banana", text_of(body).lower())

    def test_system_blocks(self):
        # system may also be an array of text blocks.
        status, _, body = messages(
            system=[{"type": "text", "text": "Whatever the user asks, reply with exactly "
                                             "the single word BANANA and nothing else."}],
            messages=[{"role": "user", "content": "What colour is the sky?"}], **FAST)
        self.assertOk(status, body)
        self.assertIn("banana", text_of(body).lower())

    def test_content_blocks(self):
        status, _, body = messages(messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is 2 + 3?"},
            {"type": "text", "text": "Answer with just the number."},
        ]}], **FAST)
        self.assertOk(status, body)
        self.assertIn("5", text_of(body))

    def test_multi_turn(self):
        status, _, body = messages(messages=[
            {"role": "user", "content": "My name is Zephyrine. Remember it."},
            {"role": "assistant", "content": "Got it, Zephyrine."},
            {"role": "user", "content": "What is my name? Answer with just the name."},
        ], **FAST)
        self.assertOk(status, body)
        self.assertIn("zephyrine", text_of(body).lower())

    def test_prefill_continues_the_assistant_turn(self):
        # A conversation ending in an assistant turn is continued, not answered
        # afresh.  Counting tells the two apart: a new turn starts over from 1,
        # a continuation picks up at 6.
        status, _, body = messages(messages=[
            {"role": "user", "content": "Count from 1 to 10, separated by spaces."},
            {"role": "assistant", "content": "1 2 3 4 5"},
        ], **FAST)
        self.assertOk(status, body)
        text = text_of(body)
        self.assertTrue(text.lstrip().startswith("6"),
                        f"the reply did not continue the prefill: {text!r}")

    def test_prefill_streamed(self):
        status, _, events = messages_stream(messages=[
            {"role": "user", "content": "Count from 1 to 10, separated by spaces."},
            {"role": "assistant", "content": "1 2 3 4 5"},
        ], **FAST)
        self.assertOk(status, events)
        text = text_of(self.assertStream(events))
        self.assertTrue(text.lstrip().startswith("6"),
                        f"the reply did not continue the prefill: {text!r}")

    def test_count_tokens(self):
        # count_tokens renders exactly what a completion would -- thinking
        # included, since it changes the prompt.
        req = dict(model="qwen3.8-27b", system="Be brief.", tools=[WEATHER_TOOL],
                   thinking={"type": "disabled"},
                   messages=[{"role": "user", "content": "Hello there."}])
        status, _, counted = request("POST", "/v1/messages/count_tokens", req, headers=HEADERS)
        self.assertOk(status, counted)
        self.assertSchema(counted, {"type": "object", "required": ["input_tokens"],
                                    "properties": {"input_tokens": {"type": "integer"}}})
        _, _, body = messages(**req, temperature=0, max_tokens=8)
        self.assertEqual(counted["input_tokens"], body["usage"]["input_tokens"])


class Thinking(Case):

    @classmethod
    def setUpClass(cls):
        cls.prompt = [{"role": "user", "content": "What is 7 * 6? Answer briefly."}]
        cls.status, _, cls.body = messages(
            messages=cls.prompt, thinking={"type": "enabled", "budget_tokens": 1024},
            temperature=0, max_tokens=1024)

    def test_thinking_block(self):
        self.assertOk(self.status, self.body)
        self.assertSchema(self.body, MESSAGE)
        self.assertEqual(self.body["content"][0]["type"], "thinking",
                         "the thinking block comes first")
        th = self.body["content"][0]
        self.assertTrue(th["thinking"].strip())
        self.assertTrue(th["signature"], "a thinking block carries a signature")
        if self.body["stop_reason"] == "end_turn":
            self.assertIn("42", text_of(self.body))

    def test_thinking_round_trip(self):
        # Clients hand the whole assistant content back, thinking and
        # signature included, on the next turn.
        status, _, body = messages(messages=self.prompt + [
            {"role": "assistant", "content": self.body["content"]},
            {"role": "user", "content": "Now add 8 to that. Just the number."},
        ], thinking={"type": "enabled", "budget_tokens": 1024}, temperature=0, max_tokens=1024)
        self.assertOk(status, body)
        self.assertSchema(body, MESSAGE)
        if body["stop_reason"] == "end_turn":
            self.assertIn("50", text_of(body))


# ---- tools ------------------------------------------------------------------

class Tools(Case):

    @classmethod
    def setUpClass(cls):
        cls.status, _, cls.body = messages(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                           thinking={"type": "disabled"}, temperature=0,
                                           max_tokens=256)

    def test_schema(self):
        self.assertOk(self.status, self.body)
        self.assertSchema(self.body, MESSAGE)

    def test_tool_use(self):
        self.assertEqual(self.body["stop_reason"], "tool_use")
        uses = blocks(self.body, "tool_use")
        self.assertTrue(uses, "expected a tool_use block")
        self.assertTrue(uses[0]["id"].startswith("toolu_"), uses[0]["id"])
        self.assertEqual(uses[0]["name"], "get_weather")
        self.assertIn("paris", str(uses[0]["input"].get("location", "")).lower())
        self.assertNotIn("<function", text_of(self.body), "tool call XML leaked into text")

    def round_trip(self, result_content):
        use = blocks(self.body, "tool_use")[0]
        return messages(messages=TOOL_PROMPT + [
            {"role": "assistant", "content": self.body["content"]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": use["id"], "content": result_content}]},
        ], tools=[WEATHER_TOOL], thinking={"type": "disabled"}, temperature=0, max_tokens=128)

    def test_tool_result_string(self):
        status, _, body = self.round_trip(json.dumps({"temperature_c": 17,
                                                      "conditions": "light rain"}))
        self.assertOk(status, body)
        self.assertEqual(body["stop_reason"], "end_turn")
        text = text_of(body).lower()
        self.assertTrue("17" in text or "rain" in text, f"answer ignores the tool result: {text!r}")

    def test_tool_result_blocks(self):
        status, _, body = self.round_trip([{"type": "text",
                                            "text": "17 degrees C and light rain"}])
        self.assertOk(status, body)
        text = text_of(body).lower()
        self.assertTrue("17" in text or "rain" in text, f"answer ignores the tool result: {text!r}")

    def test_tool_choice_none(self):
        status, _, body = messages(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                   tool_choice={"type": "none"}, thinking={"type": "disabled"},
                                   temperature=0, max_tokens=128)
        self.assertOk(status, body)
        self.assertNotEqual(body["stop_reason"], "tool_use")
        self.assertEqual(blocks(body, "tool_use"), [])

    def test_tool_choice_any(self):
        status, _, body = messages(
            messages=[{"role": "user", "content": "Tell me a joke about Berlin."}],
            tools=[WEATHER_TOOL], tool_choice={"type": "any"}, thinking={"type": "disabled"},
            temperature=0, max_tokens=128)
        self.assertOk(status, body)
        self.assertEqual(body["stop_reason"], "tool_use")
        self.assertEqual(blocks(body, "tool_use")[0]["name"], "get_weather")

    def test_tool_choice_tool(self):
        status, _, body = messages(
            messages=[{"role": "user", "content": "Tell me a joke about Berlin."}],
            tools=[WEATHER_TOOL], tool_choice={"type": "tool", "name": "get_weather"},
            thinking={"type": "disabled"}, temperature=0, max_tokens=128)
        self.assertOk(status, body)
        uses = blocks(body, "tool_use")
        self.assertTrue(uses, "a named tool_choice must produce a tool_use")
        self.assertEqual(uses[0]["name"], "get_weather")

    def test_tool_choice_unknown_tool_rejected(self):
        status, headers, body = messages(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                         tool_choice={"type": "tool", "name": "launch_rockets"},
                                         **FAST)
        self.assertError(status, headers, body, 400)
        self.assertEqual(body["error"].get("type"), "invalid_request_error")


# ---- streaming --------------------------------------------------------------

class Streaming(Case):

    PROMPT = [{"role": "user", "content": "What is the capital of France? One word."}]

    @classmethod
    def setUpClass(cls):
        cls.status, cls.headers, cls.events = messages_stream(messages=cls.PROMPT, **FAST)

    def test_status_and_content_type(self):
        self.assertOk(self.status, self.events)
        self.assertTrue(self.headers["content-type"].startswith("text/event-stream"),
                        self.headers["content-type"])

    def test_framing(self):
        self.assertEqual(self.headers["_other"], [], "stray lines in the event stream")
        for name, _ in self.events:
            self.assertIsNotNone(name, "every Anthropic event is named")

    def test_protocol(self):
        message = self.assertStream(self.events)
        self.assertEqual(message["stop_reason"], "end_turn")
        self.assertTrue(message["id"].startswith("msg_"))
        self.assertGreater(message["usage"]["input_tokens"], 0)
        self.assertGreater(message["usage"]["output_tokens"], 0)

    def test_matches_non_streamed(self):
        streamed = self.assertStream(self.events)
        _, _, body = messages(messages=self.PROMPT, **FAST)
        self.assertEqual(text_of(streamed), text_of(body))
        self.assertEqual(streamed["usage"]["input_tokens"], body["usage"]["input_tokens"])

    def test_max_tokens(self):
        status, _, events = messages_stream(
            messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
            thinking={"type": "disabled"}, temperature=0, max_tokens=5)
        self.assertOk(status, events)
        self.assertEqual(self.assertStream(events)["stop_reason"], "max_tokens")

    def test_stop_sequences(self):
        status, _, events = messages_stream(messages=COUNT_PROMPT, stop_sequences=[" 5"],
                                            thinking={"type": "disabled"}, temperature=0,
                                            max_tokens=40)
        self.assertOk(status, events)
        message = self.assertStream(events)
        self.assertNotIn("5", text_of(message))
        self.assertEqual(message["stop_reason"], "stop_sequence")
        self.assertEqual(message["stop_sequence"], " 5")

    def test_tool_use(self):
        status, _, events = messages_stream(messages=TOOL_PROMPT, tools=[WEATHER_TOOL],
                                            thinking={"type": "disabled"}, temperature=0,
                                            max_tokens=256)
        self.assertOk(status, events)
        message = self.assertStream(events)
        self.assertEqual(message["stop_reason"], "tool_use")
        uses = blocks(message, "tool_use")
        self.assertEqual(uses[0]["name"], "get_weather")
        self.assertIn("paris", str(uses[0]["input"].get("location", "")).lower())
        self.assertNotIn("<function", text_of(message), "tool call XML leaked into text")
        starts = [d for _, d in events if d["type"] == "content_block_start"
                  and d["content_block"]["type"] == "tool_use"]
        self.assertEqual(starts[0]["content_block"]["input"], {},
                         "a streamed tool_use starts with empty input")

    def test_large_tool_use(self):
        # As the OpenAI test of the same name: a tool call's input goes out in
        # one input_json_delta, which a 2 KB formatting buffer once truncated.
        status, _, events = messages_stream(
            messages=[{"role": "user", "content":
                       "Use the write tool to save this exact text to /tmp/qw_notes.txt:\n\n"
                       + LONG_DOC}],
            tools=[WRITE_TOOL], tool_choice={"type": "tool", "name": "write"},
            thinking={"type": "disabled"}, temperature=0, max_tokens=2048)
        self.assertOk(status, events)   # every event parsed as JSON on the way in
        biggest = max(len(json.dumps(d)) for _, d in events)
        if biggest < 2600:
            self.skipTest(f"the model wrote a short call ({biggest} bytes); nothing over 2 KB to check")
        message = self.assertStream(events)
        self.assertIn("content", blocks(message, "tool_use")[0]["input"])

    def test_thinking(self):
        status, _, events = messages_stream(
            messages=[{"role": "user", "content": "What is 7 * 6?"}],
            thinking={"type": "enabled", "budget_tokens": 1024}, temperature=0,
            max_tokens=1024)
        self.assertOk(status, events)
        message = self.assertStream(events)
        self.assertEqual(message["content"][0]["type"], "thinking")
        self.assertTrue(message["content"][0]["thinking"].strip())

    def test_non_ascii_deltas_are_whole_characters(self):
        prompt = [{"role": "user", "content":
                   "Repeat exactly, nothing else: 你好世界 🌧️🐙🎉 naïve café"}]
        status, _, events = messages_stream(messages=prompt, **FAST)
        self.assertOk(status, events)
        streamed = text_of(self.assertStream(events))
        _, _, body = messages(messages=prompt, **FAST)
        self.assertEqual(streamed, text_of(body))
        self.assertIn("🐙", streamed)


# ---- errors -----------------------------------------------------------------

class Errors(Case):

    def test_malformed_json(self):
        status, headers, body = request("POST", "/v1/messages", raw=b"{not json",
                                        headers=dict(HEADERS, **{"Content-Type":
                                                                 "application/json"}))
        self.assertError(status, headers, body, 400)

    def test_empty_messages(self):
        status, headers, body = messages(messages=[], **FAST)
        self.assertError(status, headers, body, 400)

    def test_envelope_schema(self):
        # Every error is {"type": "error", "error": {"type", "message"}}, with
        # the error type following the status.
        cases = {
            "malformed JSON": (400, lambda: request(
                "POST", "/v1/messages", raw=b"{",
                headers=dict(HEADERS, **{"Content-Type": "application/json"}))),
            "no messages": (400, lambda: messages(messages=[], **FAST)),
            "bad tool_choice": (400, lambda: messages(
                messages=TOOL_PROMPT, tools=[WEATHER_TOOL], tool_choice={"type": "sometimes"},
                **FAST)),
            "bad count_tokens": (400, lambda: request(
                "POST", "/v1/messages/count_tokens", {"model": "x", "messages": []},
                headers=HEADERS)),
            "unknown endpoint": (404, lambda: request("GET", "/v1/nope", headers=HEADERS)),
            "unknown model": (404, lambda: request("GET", "/v1/models/nope", headers=HEADERS)),
        }
        for name, (want, call) in cases.items():
            with self.subTest(case=name):
                status, _, body = call()
                self.assertEqual(status, want)
                self.assertSchema(body, ERROR_RESPONSE)
                self.assertEqual(body["error"]["type"],
                                 "not_found_error" if want == 404 else "invalid_request_error")


# ---- the official client ----------------------------------------------------

try:
    import anthropic  # noqa: E402
except ImportError:
    anthropic = None


@unittest.skipIf(anthropic is None, "the anthropic package is not installed")
class OfficialClient(Case):
    """The SDK's stream accumulator applies each event to the message it is
    building, so an event out of order or of the wrong shape fails here.
    Recent SDKs no longer take temperature as an argument; it goes through
    extra_body, which the server still honours."""

    @classmethod
    def setUpClass(cls):
        cls.client = anthropic.Anthropic(
            base_url=f"http://{api.base().hostname}:{api.base().port}",
            api_key="sk-unused", max_retries=0, timeout=TIMEOUT)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_models_list(self):
        self.assertTrue([m.id for m in self.client.models.list()])

    def test_create(self):
        r = self.client.messages.create(
            model="qwen3.8-27b", max_tokens=32, extra_body={"temperature": 0},
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Say hi."}])
        self.assertEqual(r.role, "assistant")
        self.assertEqual(r.content[0].type, "text")

    def test_stream_final_message(self):
        with self.client.messages.stream(
                model="qwen3.8-27b", max_tokens=1024, extra_body={"temperature": 0},
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": "What is 2 + 2?"}]) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        self.assertEqual(final.content[0].type, "thinking")
        self.assertTrue(final.content[0].signature)
        self.assertEqual(text, "".join(b.text for b in final.content if b.type == "text"))

    def test_tool_use(self):
        r = self.client.messages.create(
            model="qwen3.8-27b", max_tokens=256, extra_body={"temperature": 0},
            thinking={"type": "disabled"}, tools=[WEATHER_TOOL], messages=TOOL_PROMPT)
        use = next(b for b in r.content if b.type == "tool_use")
        self.assertEqual(use.name, "get_weather")
        self.assertIsInstance(use.input, dict)

    def test_count_tokens(self):
        n = self.client.messages.count_tokens(
            model="qwen3.8-27b", messages=[{"role": "user", "content": "Hello."}])
        self.assertGreater(n.input_tokens, 0)


if __name__ == "__main__":
    unittest.main()
