#!/usr/bin/env python3
"""Integration tests: qwasar-server's Session API (API.md).

    make test-api                                   # with the compat suites
    python3 tests/test_session_api.py -v            # this one

Drives a real server over HTTP and checks the contract API.md states: a
prefix fixed at open, deltas only, warmth reported and guaranteed, prefill
progress that spans the whole of a resume, refusals as HTTP statuses, a
stream that can be reattached, sessions that survive a restart.

The model's words are never checked -- on a toy fixture they are noise --
only the shape and the accounting, so the suite runs on
tests/fixtures/flashnext-tiny-q4 (with the tokenizer tools/toy_tokenizer.py
writes) in a few seconds, and on the real model unchanged.  Set
QWASAR_TEST_MODEL to choose; QWASAR_SERVER_BIN to the binary.

The tool-call path (`tool_call` events and `continue`) needs a model that
actually writes a call, so those tests skip on a toy and run on the real
one.  Standard library only.

This module runs servers of its own rather than the shared harness's: it
needs checkpoints on, a state directory it controls, and a restart.  HOME
is pointed at a temp directory so the toy's checkpoints never enter the
user's own ~/.cache/qwasar/kv.
"""

import http.client
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("QWASAR_SERVER_BIN", os.path.join(ROOT, "qwasar-server"))
# Absolute, because the agent runs in a working directory of its own and
# passes the path on to a server it starts.
MODEL = os.path.abspath(os.environ.get("QWASAR_TEST_MODEL")
                        or os.path.join(ROOT, "tests/fixtures/flashnext-tiny-q4"))
TOY = "fixtures" in MODEL
TIMEOUT = 900

# Long enough that the prefix passes the checkpoint store's 256-token floor
# on the toy tokenizer (one token a character), which is what makes the
# shared-prefix and park tests meaningful.
SYSTEM = "You are a careful assistant who reads before answering. " * 6
READ_TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read a file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
FAST = {"max_tokens": 12, "sampling": {"temperature": 0}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One qwasar-server on a port and state directory of this run's."""

    def __init__(self, state_dir, home, ctx=4096, extra=()):
        self.port = free_port()
        self.state_dir = state_dir
        self.home = home
        self.log = open(os.path.join(home, f"server-{self.port}.log"), "w")
        cmd = [BIN, "-m", MODEL, "--port", str(self.port), "--ctx", str(ctx),
               "--state-dir", state_dir, "-v", *extra]
        env = dict(os.environ, HOME=home)
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + float(os.environ.get("QWASAR_STARTUP_SECS", "600"))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode}:\n{self.tail()}")
            try:
                if self.request("GET", "/health", timeout=2)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("server did not come up:\n" + self.tail())

    def tail(self, n=30):
        self.log.flush()
        with open(self.log.name, errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def connect(self, timeout=TIMEOUT):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def request(self, method, path, body=None, headers=None, timeout=TIMEOUT):
        conn = self.connect(timeout)
        try:
            hdrs = dict(headers or {})
            data = None
            if body is not None:
                data = json.dumps(body).encode()
                hdrs["Content-Type"] = "application/json"
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            payload = resp.read()
            ctype = resp.getheader("Content-Type", "")
            if ctype.startswith("application/json") and payload:
                payload = json.loads(payload)
            return resp.status, resp, payload
        finally:
            conn.close()

    def stream(self, method, path, body=None, headers=None, stop_after=None):
        """Sends a request and reads its event stream.  Returns
        (status, events) where events are (id, name, data) triples;
        `stop_after(events)` true closes the socket early, which is how the
        reattach test loses its connection."""
        conn = self.connect()
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        if not resp.getheader("Content-Type", "").startswith("text/event-stream"):
            payload = resp.read()
            try:
                payload = json.loads(payload)
            except ValueError:
                pass
            conn.close()
            return resp.status, payload
        events, cur = [], {}
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line == "":
                    if "data" in cur:
                        events.append((cur.get("id"), cur.get("event"), json.loads(cur["data"])))
                        if stop_after and stop_after(events):
                            break
                    cur = {}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    cur[k] = v.lstrip(" ")
        finally:
            conn.close()
        return resp.status, events

    def open(self, **kw):
        body = {"system": SYSTEM, "tools": [READ_TOOL], "effort": "medium"}
        body.update(kw)
        status, _, r = self.request("POST", "/v1/sessions", body)
        assert status == 201, (status, r)
        return r["id"]


def names(events):
    return [e[1] for e in events]


def find(events, name):
    return [e[2] for e in events if e[1] == name]


HOME = None
STATE = None
SRV = None


def setUpModule():
    global HOME, STATE, SRV
    if not os.access(BIN, os.X_OK):
        raise unittest.SkipTest(f"{BIN} not built; run make")
    if not os.path.exists(os.path.join(MODEL, "tokenizer.json")):
        raise unittest.SkipTest(f"{MODEL} has no tokenizer.json (tools/toy_tokenizer.py writes the toy's)")
    HOME = tempfile.mkdtemp(prefix="qwasar-api-")
    STATE = os.path.join(HOME, "state")
    SRV = Server(STATE, HOME)


def tearDownModule():
    if SRV:
        SRV.stop()
    if HOME and os.environ.get("QWASAR_KEEP_TEMP") is None:
        shutil.rmtree(HOME, ignore_errors=True)


class ServerInfo(unittest.TestCase):
    def test_server(self):
        status, _, r = SRV.request("GET", "/v1/server")
        self.assertEqual(status, 200)
        self.assertEqual(r["context"], 4096)
        self.assertIn(r["model"]["family"], ("qwen3_5", "qwen4_exp"))
        self.assertEqual(r["model"]["path"], MODEL)
        for k in ("physical_bytes", "working_set_bytes", "weights_bytes", "kv_bytes_per_token",
                  "session_fixed_bytes", "reserve", "max_context"):
            self.assertIn(k, r["profile"])
        self.assertGreater(r["profile"]["weights_bytes"], 0)
        self.assertEqual(set(r["capabilities"]), {"reasoning", "images", "video", "speculation", "rewind", "fork"})
        self.assertEqual(r["state_dir"], STATE)

    def test_health_and_compat_still_there(self):
        self.assertEqual(SRV.request("GET", "/health")[0], 200)
        self.assertEqual(SRV.request("GET", "/v1/models")[0], 200)


class Opening(unittest.TestCase):
    def test_open_and_describe(self):
        status, _, r = SRV.request("POST", "/v1/sessions",
                                   {"system": SYSTEM, "tools": [READ_TOOL], "effort": "medium",
                                    "metadata": {"client": "test", "title": "opening"}})
        self.assertEqual(status, 201, r)
        self.assertTrue(r["id"].startswith("s_"))
        self.assertGreater(r["prefix_tokens"], 0)
        self.assertEqual(r["warmth"], {"state": "cold", "covered": 0})
        status, _, d = SRV.request("GET", f"/v1/sessions/{r['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(d["tokens"], 0)
        self.assertEqual(d["state"], "idle")
        self.assertEqual(d["metadata"]["title"], "opening")
        self.assertEqual(d["prefix_tokens"], r["prefix_tokens"])
        self.assertNotIn("last_step", d)
        status, _, lst = SRV.request("GET", "/v1/sessions")
        self.assertIn(r["id"], [s["id"] for s in lst["sessions"]])
        # On disk, as the record and the (empty) token log.
        self.assertTrue(os.path.exists(os.path.join(STATE, "sessions", r["id"], "record.json")))

    def test_refusals(self):
        status, _, r = SRV.request("POST", "/v1/sessions", {"system": "x", "effort": "extreme"})
        self.assertEqual(status, 400)
        self.assertEqual(set(r["error"]), {"code", "message"})
        status, _, r = SRV.request("POST", "/v1/sessions", {"system": "x", "tools": "not a list"})
        self.assertEqual(status, 400)
        status, _, r = SRV.request("GET", "/v1/sessions/s_doesnotexist")
        self.assertEqual(status, 404)
        status, _, r = SRV.request("POST", "/v1/sessions/s_doesnotexist/turn", {"text": "hi"})
        self.assertEqual(status, 404)


class Steps(unittest.TestCase):
    def test_first_turn_cold_then_live(self):
        sid = SRV.open()
        _, d0 = None, SRV.request("GET", f"/v1/sessions/{sid}")[2]
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="List the files."))
        self.assertEqual(status, 200)
        self.assertEqual(names(ev)[0], "resume", ev[:3])
        self.assertEqual(names(ev)[-1], "done", ev[-3:])
        resume = ev[0][2]
        # First ever step of this prefix in this store: nothing to read.
        self.assertIn(resume["from"], ("cold", "checkpoint"))
        # Prefill progress spans the whole of what was evaluated, not a chunk.
        pre = find(ev, "prefill")
        if pre:
            self.assertTrue(all(p["total"] == resume["prefill"] for p in pre), pre)
            self.assertEqual(pre[-1]["done"], resume["prefill"])
        done = ev[-1][2]
        self.assertIn(done["stop"], ("end_turn", "length"))
        self.assertEqual(done["usage"]["prompt"], resume["prefill"])
        self.assertLessEqual(done["usage"]["generated"], FAST["max_tokens"])
        self.assertEqual(done["warmth"]["state"], "live")
        self.assertEqual(done["context"]["limit"], 4096)
        self.assertGreater(done["context"]["used"], d0["prefix_tokens"])
        # Every event carries an id of the form step.seq, in order.
        ids = [e[0] for e in ev]
        self.assertTrue(all(i and i.startswith("1.") for i in ids), ids)
        self.assertEqual([int(i.split(".")[1]) for i in ids], list(range(1, len(ids) + 1)))

        # The second turn continues the live session: no re-evaluation.
        d1 = SRV.request("GET", f"/v1/sessions/{sid}")[2]
        self.assertEqual(d1["warmth"]["state"], "live")
        self.assertEqual(d1["tokens"], done["context"]["used"])
        status, ev2 = SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="Again."))
        r2 = ev2[0][2]
        self.assertEqual(r2["from"], "live")
        self.assertEqual(r2["restored"], d1["tokens"])
        self.assertLess(r2["prefill"], 40)
        self.assertEqual(ev2[-1][2]["usage"]["prompt"], r2["prefill"])
        self.assertTrue(ev2[0][0].startswith("2."))
        # Text and reasoning deltas, when any, are objects with text.
        for t in find(ev2, "text") + find(ev2, "reasoning"):
            self.assertIsInstance(t["text"], str)
        for r in find(ev2, "reasoning"):
            self.assertGreaterEqual(r["tokens"], 1)

    def test_shared_prefix_is_read_by_the_next_session(self):
        a = SRV.open()
        SRV.stream("POST", f"/v1/sessions/{a}/turn", dict(FAST, text="one"))
        b = SRV.open()
        status, ev = SRV.stream("POST", f"/v1/sessions/{b}/turn", dict(FAST, text="two"))
        resume = ev[0][2]
        self.assertEqual(resume["from"], "checkpoint", resume)
        self.assertTrue(resume["prefix_cached"], resume)
        pfx = SRV.request("GET", f"/v1/sessions/{b}")[2]["prefix_tokens"]
        self.assertEqual(resume["restored"], pfx)
        self.assertEqual(ev[-1][2]["usage"]["prompt"], resume["prefill"])

    def test_park_then_resume_from_checkpoint(self):
        sid = SRV.open()
        SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="hello"))
        n = SRV.request("GET", f"/v1/sessions/{sid}")[2]["tokens"]
        status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/park")
        self.assertEqual(status, 200, r)
        self.assertEqual(r["warmth"]["state"], "warm", r)
        self.assertEqual(r["warmth"]["covered"], n)
        self.assertIn("estimate_seconds", r["warmth"])
        d = SRV.request("GET", f"/v1/sessions/{sid}")[2]
        self.assertEqual(d["warmth"]["state"], "warm")
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="back"))
        resume = ev[0][2]
        self.assertEqual(resume["from"], "checkpoint", resume)
        self.assertEqual(resume["restored"], n)
        self.assertEqual(ev[-1][2]["warmth"]["state"], "live")
        # Parking again is idempotent on an idle session; a second park of a
        # parked one says so without a write.
        self.assertEqual(SRV.request("POST", f"/v1/sessions/{sid}/park")[0], 200)
        self.assertEqual(SRV.request("POST", f"/v1/sessions/{sid}/park")[0], 200)

    def test_continue_without_pending_calls_is_409(self):
        sid = SRV.open()
        SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="hi"))
        status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/continue", {"results": []})
        self.assertEqual(status, 409, r)
        self.assertEqual(r["error"]["code"], "conflict")

    def test_context_full_ends_the_session(self):
        sid = SRV.open()
        # No budget: the step runs to the end of the window.
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn", {"text": "go", "max_tokens": 0})
        done = ev[-1][2]
        self.assertIn(done["stop"], ("context_full", "end_turn", "length"))
        if done["stop"] != "context_full":
            self.skipTest("the model stopped before the window filled; nothing to check")
        d = SRV.request("GET", f"/v1/sessions/{sid}")[2]
        self.assertEqual(d["state"], "full")
        status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/turn", {"text": "more"})
        self.assertEqual(status, 409, r)

    def test_context_full_before_evaluating(self):
        """A turn that cannot fit is refused before a token is evaluated,
        whatever the model would have done -- deterministic where the test
        above depends on the model running to the end of the window."""
        sid = SRV.open()
        # More tokens than the window whatever the tokenizer makes of "x ":
        # one token each on the real one, two on the toy's.
        ctx = SRV.request("GET", "/v1/server")[2]["context"]
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn", {"text": "x " * (ctx + 64), "max_tokens": 8})
        self.assertEqual(status, 200)
        self.assertEqual([e[1] for e in ev][-1], "done")
        self.assertEqual(ev[-1][2]["stop"], "context_full", ev[-1])
        self.assertEqual(ev[-1][2]["usage"]["generated"], 0)
        self.assertNotIn("resume", names(ev))
        d = SRV.request("GET", f"/v1/sessions/{sid}")[2]
        self.assertEqual(d["state"], "full")
        self.assertEqual(d["tokens"], 0)
        status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/turn", {"text": "more"})
        self.assertEqual(status, 409, r)
        self.assertEqual(SRV.request("DELETE", f"/v1/sessions/{sid}")[0], 204)

    def test_cancel(self):
        sid = SRV.open()
        got = []

        def run():
            got.append(SRV.stream("POST", f"/v1/sessions/{sid}/turn", {"text": "go", "max_tokens": 0}))

        t = threading.Thread(target=run)
        t.start()
        # Once the step is running -- describe says so -- cancel it.
        for _ in range(200):
            d = SRV.request("GET", f"/v1/sessions/{sid}")[2]
            if d["state"] == "running":
                break
            time.sleep(0.02)
        status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/cancel")
        t.join(60)
        self.assertEqual(status, 200)
        status, ev = got[0]
        done = ev[-1][2]
        if r["cancelled"]:
            self.assertEqual(done["stop"], "cancelled", done)
        self.assertEqual(SRV.request("GET", f"/v1/sessions/{sid}")[2]["state"], "idle")
        # The tokens it generated are the session's; the next turn continues.
        status, ev2 = SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="and"))
        self.assertEqual(ev2[0][2]["from"], "live")

    def test_reattach_after_losing_the_stream(self):
        sid = SRV.open()
        # Drop the connection after the resume event; the step goes on.
        status, first = SRV.stream("POST", f"/v1/sessions/{sid}/turn", {"text": "go", "max_tokens": 64},
                                   stop_after=lambda ev: len(ev) >= 2)
        self.assertGreaterEqual(len(first), 2)
        last_id = first[-1][0]
        status, rest = SRV.stream("GET", f"/v1/sessions/{sid}/events", headers={"Last-Event-ID": last_id})
        self.assertEqual(status, 200)
        self.assertTrue(rest, "nothing replayed")
        self.assertEqual(rest[-1][1], "done", rest[-2:])
        # Contiguous: the replay starts right after what was seen.
        seen = [int(e[0].split(".")[1]) for e in first + rest]
        self.assertEqual(seen, list(range(1, len(seen) + 1)), seen)
        # Without Last-Event-ID the whole step is replayed.
        status, whole = SRV.stream("GET", f"/v1/sessions/{sid}/events")
        self.assertEqual(whole[0][1], "resume")
        self.assertEqual(whole[-1][1], "done")
        self.assertEqual(len(whole), len(seen))

    def test_delete(self):
        sid = SRV.open()
        SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="x"))
        status, _, _ = SRV.request("DELETE", f"/v1/sessions/{sid}")
        self.assertEqual(status, 204)
        self.assertEqual(SRV.request("GET", f"/v1/sessions/{sid}")[0], 404)
        self.assertFalse(os.path.exists(os.path.join(STATE, "sessions", sid)))

    def test_turn_while_running_is_409(self):
        # On a toy a whole step takes about a second, so the step can end
        # between seeing it run and sending the second turn -- and then 200
        # is the right answer (the server refuses a turn only while a step is
        # queued or running).  So: a few fresh attempts, and at least one must
        # land while the first step runs and be refused.
        statuses = []
        for _ in range(5):
            sid = SRV.open()
            got = []
            t = threading.Thread(target=lambda: got.append(
                SRV.stream("POST", f"/v1/sessions/{sid}/turn", {"text": "go", "max_tokens": 0})))
            t.start()
            for _ in range(200):
                if SRV.request("GET", f"/v1/sessions/{sid}")[2]["state"] in ("running", "queued"):
                    break
                time.sleep(0.01)
            status, _, r = SRV.request("POST", f"/v1/sessions/{sid}/turn", {"text": "again", "max_tokens": 4})
            SRV.request("POST", f"/v1/sessions/{sid}/cancel")
            t.join(60)
            statuses.append(status)
            self.assertIn(status, (409, 200), r)
            if status == 409:
                self.assertEqual(r["error"]["code"], "conflict")
                return
        self.fail(f"never caught a step running: {statuses}")

class Compat(unittest.TestCase):
    def test_completions_share_the_engine(self):
        sid = SRV.open()
        SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="one"))
        status, _, r = SRV.request("POST", "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": "hi"}],
                                    "max_tokens": 8, "enable_thinking": False})
        self.assertEqual(status, 200, r)
        self.assertEqual(r["object"], "chat.completion")
        # And the API session is still usable after it.
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="two"))
        self.assertEqual(ev[-1][1], "done")


@unittest.skipIf(TOY, "needs a model that writes tool calls")
class ToolCalls(unittest.TestCase):
    def test_call_and_continue(self):
        sid = SRV.open()
        status, ev = SRV.stream("POST", f"/v1/sessions/{sid}/turn",
                                {"text": "Read the file README.md with the read tool; do not answer otherwise.",
                                 "max_tokens": 2048, "sampling": {"temperature": 0}})
        calls = find(ev, "tool_call")
        self.assertTrue(calls, names(ev))
        self.assertEqual(ev[-1][2]["stop"], "tool_calls")
        self.assertEqual(calls[0]["name"], "read")
        self.assertIsInstance(calls[0]["arguments"], dict)
        d = SRV.request("GET", f"/v1/sessions/{sid}")[2]
        self.assertEqual(d["state"], "awaiting_tools")
        self.assertEqual([c["id"] for c in d["last_step"]["pending_calls"]], [c["id"] for c in calls])
        status, ev2 = SRV.stream("POST", f"/v1/sessions/{sid}/continue",
                                 {"results": [{"id": c["id"], "content": "hello world\n"} for c in calls],
                                  "max_tokens": 256})
        self.assertEqual(status, 200, ev2)
        self.assertEqual(ev2[0][2]["from"], "live")
        self.assertIn(ev2[-1][2]["stop"], ("end_turn", "tool_calls", "length"))


AGENT = os.environ.get("QWASAR_AGENT_BIN", os.path.join(ROOT, "qwasar-agent"))


@unittest.skipUnless(os.access(AGENT, os.X_OK), "qwasar-agent not built")
class Agent(unittest.TestCase):
    """qwasar-agent as a client: one-shot tasks against the module's server,
    a resumed conversation, and a server the agent starts for itself."""

    def run_agent(self, args, port=None, timeout=180):
        # realpath: the agent records getcwd(), which resolves /var to /private/var.
        work = os.path.realpath(tempfile.mkdtemp(prefix="agent-work-", dir=HOME))
        env = dict(os.environ, HOME=HOME)
        url = f"http://127.0.0.1:{port or SRV.port}"
        cmd = [AGENT, "--server", url, "-y", "-n", "12", "--temperature", "0", *args]
        p = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode(errors="replace"), work

    def agent_sessions(self, work, server=None):
        _, _, r = (server or SRV).request("GET", "/v1/sessions")
        return [s for s in r["sessions"]
                if s["metadata"].get("client") == "qwasar-agent" and s["metadata"].get("cwd") == work]

    def test_one_shot_and_resume(self):
        rc, out, work = self.run_agent(["say hi"])
        self.assertEqual(rc, 0, out)
        self.assertIn("tools", out)
        sess = self.agent_sessions(work)
        self.assertEqual(len(sess), 1, out)
        self.assertEqual(sess[0]["metadata"]["title"], "say hi")
        self.assertGreater(sess[0]["tokens"], sess[0]["prefix_tokens"])
        # Parked on the way out: warm if it passed the store's floor.
        self.assertIn(sess[0]["warmth"]["state"], ("warm", "cold"))
        n = sess[0]["tokens"]
        # The same directory resumes its last conversation, which grows.
        env = dict(os.environ, HOME=HOME)
        p = subprocess.run([AGENT, "--server", f"http://127.0.0.1:{SRV.port}", "-y", "-n", "8",
                            "--resume", "last", "and again"], cwd=work, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        out2 = p.stdout.decode(errors="replace")
        self.assertEqual(p.returncode, 0, out2)
        self.assertIn("resuming", out2)
        sess2 = self.agent_sessions(work)
        self.assertEqual(len(sess2), 1)
        self.assertEqual(sess2[0]["id"], sess[0]["id"])
        self.assertGreater(sess2[0]["tokens"], n)

    def test_refuses_without_a_server_or_model(self):
        env = dict(os.environ, HOME=HOME)
        env.pop("QWASAR_MODEL", None)
        # A copy of the binary in a directory of its own: the agent also
        # looks for a qwasar-model link beside itself, and a checkout that
        # has run download_model.sh has one -- the test would then start a
        # real server on the real model instead of testing the refusal.
        lone = os.path.join(tempfile.mkdtemp(prefix="lone-agent-", dir=HOME), "qwasar-agent")
        shutil.copy2(AGENT, lone)
        p = subprocess.run([lone, "--server", f"http://127.0.0.1:{free_port()}", "hi"], cwd=HOME, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(p.returncode, 1)
        self.assertIn("nothing is listening", p.stdout.decode(errors="replace"))

    def test_starts_its_own_server(self):
        port = free_port()
        rc, out, work = self.run_agent(["-m", MODEL, "hello there"], port=port)
        self.assertEqual(rc, 0, out)
        self.assertIn("starting qwasar-server", out)
        self.assertIn("server ready", out)
        # The lifeline: the server goes when the agent does.
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    time.sleep(0.1)
                    continue
            except OSError:
                break
        else:
            self.fail("the agent's server outlived it")


class Restart(unittest.TestCase):
    """A second server on the same state directory finds the sessions."""

    def test_sessions_survive(self):
        global SRV
        sid = SRV.open(metadata={"title": "survivor"})
        SRV.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="remember this"))
        n = SRV.request("GET", f"/v1/sessions/{sid}")[2]["tokens"]
        SRV.stop()                      # SIGTERM: the live session is checkpointed on the way out
        second = Server(STATE, HOME)
        try:
            status, _, d = second.request("GET", f"/v1/sessions/{sid}")
            self.assertEqual(status, 200)
            self.assertEqual(d["tokens"], n)
            self.assertEqual(d["metadata"]["title"], "survivor")
            self.assertIn(d["warmth"]["state"], ("warm", "cold"))
            status, ev = second.stream("POST", f"/v1/sessions/{sid}/turn", dict(FAST, text="still here?"))
            resume = ev[0][2]
            self.assertIn(resume["from"], ("checkpoint", "cold"))
            # What was read plus what was prefilled covers the old timeline
            # and the new turn; nothing the timeline held was skipped.
            self.assertLessEqual(resume["restored"], n)
            self.assertGreater(resume["restored"] + resume["prefill"], n)
            self.assertEqual(ev[-1][1], "done")
            self.assertEqual(ev[-1][2]["usage"]["prompt"], resume["prefill"])
            self.assertEqual(ev[-1][2]["context"]["used"], n + resume["prefill"] - (n - resume["restored"]) + ev[-1][2]["usage"]["generated"])
        finally:
            second.stop()
            # The module's server, back for any test that runs after this one.
            SRV = Server(STATE, HOME)


if __name__ == "__main__":
    unittest.main()
