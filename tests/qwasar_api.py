"""Shared harness for the HTTP API integration tests.

test_openai_api.py and test_anthropic_api.py check qwasar-server against the
OpenAI and Anthropic specs respectively.  This module holds what they share:
the server under test, HTTP and server-sent-event helpers, and a small schema
validator.  Standard library only.

One server serves every test module in a run: the first module to need it
starts it, and it is stopped when the process exits.  Environment:

  QWASAR_SERVER_URL     test an already-running server instead (e.g.
                        http://127.0.0.1:8080); tests that depend on how the
                        server was started skip
  QWASAR_TEST_MODEL     model directory for the spawned server (default: the
                        server's own resolution -- QWASAR_MODEL, ./qwasar-model)
  QWASAR_SERVER_BIN     server binary (default ./qwasar-server)
  QWASAR_STARTUP_SECS   how long to wait for the model to load (default 600)

Every helper opens a connection, uses it, and closes it, so the tests never
depend on connection reuse except where that is what they test.
"""

import atexit
import http.client
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generation is slow next to an HTTP round trip; a thinking turn on a cold
# session can take minutes.
TIMEOUT = 900


# ---- schemas ----------------------------------------------------------------

def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def tagged(**by_type):
    """A union discriminated by the object's "type" field, as both APIs shape
    content blocks and stream events.  Reports against the matching variant
    rather than a bare "matches none of the allowed shapes"."""
    return {"tagged": by_type}


def validate(value, schema, path="$"):
    """Returns every violation of `schema` in `value`, as readable strings.

    The subset of JSON Schema the test schemas use -- type, enum, anyOf,
    required, properties, items -- plus `tagged` above.  Collecting rather
    than raising means one failed test names every problem with a response,
    not just the first."""
    errs = []
    if "anyOf" in schema:
        if all(validate(value, s, path) for s in schema["anyOf"]):
            errs.append(f"{path}: {json.dumps(value)[:80]} matches none of the allowed shapes")
        return errs
    if "tagged" in schema:
        variants = schema["tagged"]
        tag = value.get("type") if isinstance(value, dict) else None
        if tag not in variants:
            return [f"{path}: type {tag!r} is not one of {sorted(variants)}"]
        return validate(value, variants[tag], path)
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {json.dumps(value)[:80]} is not one of {schema['enum']}")
    t = schema.get("type")
    if t is not None:
        ok = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }[t]
        if not ok:
            errs.append(f"{path}: expected {t}, got {type(value).__name__} "
                        f"{json.dumps(value)[:80]}")
            return errs
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing required property '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errs.extend(validate(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errs.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errs


# ---- the server under test --------------------------------------------------

_BASE = None         # urllib.parse result for the server
_SPAWNED = None      # the Popen, when this run started the server
_LOG = None


def base():
    return _BASE


def spawned():
    """True when this run started the server, so its flags are known."""
    return _SPAWNED is not None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log_tail(n=40):
    if not _LOG:
        return ""
    _LOG.flush()
    with open(_LOG.name, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def ensure_server():
    """Starts the server on first use, or checks the one QWASAR_SERVER_URL
    names.  Call from each test module's setUpModule."""
    global _BASE, _SPAWNED, _LOG
    if _BASE is not None:
        return
    url = os.environ.get("QWASAR_SERVER_URL")
    if url:
        _BASE = urllib.parse.urlparse(url)
        status, _, _ = request("GET", "/health", timeout=10)
        if status != 200:
            raise RuntimeError(f"{url}/health answered {status}")
        return

    binary = os.environ.get("QWASAR_SERVER_BIN", os.path.join(ROOT, "qwasar-server"))
    if not os.access(binary, os.X_OK):
        raise unittest.SkipTest(f"{binary} not built; run make")
    port = _free_port()
    cmd = [binary, "--port", str(port), "--no-cache", "--cors"]
    model = os.environ.get("QWASAR_TEST_MODEL")
    if model:
        cmd += ["-m", model]

    _LOG = tempfile.NamedTemporaryFile("w+", prefix="qwasar-server-", suffix=".log", delete=False)
    _SPAWNED = subprocess.Popen(cmd, cwd=ROOT, stdout=_LOG, stderr=subprocess.STDOUT)
    _BASE = urllib.parse.urlparse(f"http://127.0.0.1:{port}")
    atexit.register(_stop_server)

    deadline = time.time() + float(os.environ.get("QWASAR_STARTUP_SECS", "600"))
    while time.time() < deadline:
        if _SPAWNED.poll() is not None:
            tail = _log_tail()
            _BASE = None
            raise RuntimeError(f"qwasar-server exited with {_SPAWNED.returncode}:\n{tail}")
        try:
            if request("GET", "/health", timeout=2)[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    tail = _log_tail()
    _stop_server()
    raise RuntimeError(f"qwasar-server did not come up in time:\n{tail}")


def _stop_server():
    global _SPAWNED, _BASE, _LOG
    if _SPAWNED and _SPAWNED.poll() is None:
        _SPAWNED.terminate()
        try:
            _SPAWNED.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _SPAWNED.kill()
    _SPAWNED = None
    _BASE = None
    if _LOG:
        _LOG.close()
        os.unlink(_LOG.name)
        _LOG = None


# ---- http -------------------------------------------------------------------

def connect(timeout=TIMEOUT):
    return http.client.HTTPConnection(_BASE.hostname, _BASE.port or 80, timeout=timeout)


def request(method, path, body=None, headers=None, timeout=TIMEOUT, raw=None):
    """One request on its own connection.  Returns (status, headers, body),
    with the body parsed as JSON when it is JSON."""
    conn = connect(timeout)
    try:
        hdrs = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        rh = {k.lower(): v for k, v in resp.getheaders()}
        if rh.get("content-type", "").startswith("application/json") and payload:
            payload = json.loads(payload)
        return resp.status, rh, payload
    finally:
        conn.close()


def sse(path, body, headers=None):
    """POSTs `body` and reads the reply as a server-sent event stream.

    Returns (status, headers, events).  Each event is (name, data): name is the
    `event:` field or None, data the parsed JSON of its `data:` field, or the
    literal string "[DONE]".  Lines that are neither field nor comment land in
    headers["_other"], for framing checks.  A non-200 reply comes back as
    (status, headers, parsed body) instead.

    Lines are decoded as strict UTF-8, so a delta carrying half a character
    fails here rather than being quietly repaired."""
    conn = connect(TIMEOUT)
    try:
        hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        hdrs.update(headers or {})
        conn.request("POST", path, body=json.dumps(body).encode(), headers=hdrs)
        resp = conn.getresponse()
        rh = {k.lower(): v for k, v in resp.getheaders()}
        if resp.status != 200:
            payload = resp.read()
            if rh.get("content-type", "").startswith("application/json") and payload:
                payload = json.loads(payload)
            return resp.status, rh, payload
        events, other, name, data = [], [], None, []
        # http.client undoes the chunked transfer encoding; what is left is the
        # event stream itself, one field per line, events split by blank lines.
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8").rstrip("\r\n")
            if line == "":
                if data:
                    payload = "\n".join(data)
                    events.append((name, payload if payload == "[DONE]" else json.loads(payload)))
                elif name:
                    other.append(f"event {name!r} with no data")
                name, data = None, []
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif not line.startswith(":"):
                other.append(line)
        if data or name:
            other.append(f"unterminated event: {name!r} {data!r}")
        rh["_other"] = other
        return resp.status, rh, events
    finally:
        conn.close()


class Case(unittest.TestCase):
    maxDiff = None
    SPEC = "the spec"

    def assertSchema(self, value, schema):
        errs = validate(value, schema)
        if errs:
            self.fail(f"does not match {self.SPEC}:\n  " + "\n  ".join(errs)
                      + "\n\nresponse: " + json.dumps(value)[:2000])

    def assertOk(self, status, body):
        if status != 200:
            self.fail(f"HTTP {status}: {body!r}"[:2000])

    def assertError(self, status, headers, body, expect_status):
        """Status, content type, and an error object with a message.  Each
        suite checks its full error schema once, so one deviation there does
        not fail every test that provokes an error."""
        self.assertEqual(status, expect_status, f"body: {body!r}")
        self.assertTrue(headers.get("content-type", "").startswith("application/json"),
                        f"error Content-Type is {headers.get('content-type')!r}")
        self.assertIsInstance(body, dict)
        self.assertIsInstance(body.get("error", {}).get("message"), str, f"body: {body!r}")
