"""The cache plugin's server-side interface (S1v3 spec, section 4): the Unix-socket client, and the engine hooks
around the GEN line (acquire / DUMP / release) that must never fail a request.  No GPU, no real engine: the plugin
is a fake Unix-socket server on a thread, the engine a lazy StrataEngine with a fake stdin and a scripted line queue."""

import json
import queue
import socket
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import serve.server as srv
from serve.server import PluginClient, StrataEngine

REPLY_MISS = {"ok": True, "result": {"tokens": 0}}      # S1: acquire is always miss semantics


class FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


class FakePlugin:
    """One NDJSON reply per request, from a script.  A reply entry: a dict (sent with the request's id),
    "silent" (record, hold the connection, never reply), "raw:<text>" (send it verbatim), "close" (record, close)."""

    def __init__(self, path, replies=()):
        self.replies = list(replies)
        self.requests = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.path = self.sock.getsockname()
        self.sock.listen(8)
        self.stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        self.sock.settimeout(0.1)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            conn.settimeout(2)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            req = json.loads(buf.decode())
            self.requests.append(req)
            reply = self.replies.pop(0) if self.replies else REPLY_MISS
            if reply == "silent":
                self.stop.wait(1.0)
                return
            if reply == "close":
                return
            if isinstance(reply, str) and reply.startswith("raw:"):
                conn.sendall(reply[4:].encode() + b"\n")
                return
            conn.sendall((json.dumps({"id": req["id"], **reply}) + "\n").encode())

    def close(self):
        self.stop.set()
        self.sock.close()
        Path(self.path).unlink(missing_ok=True)


class PluginClientTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "plugin.sock")

    def test_acquire_miss_returns_the_reply(self):
        plugin = FakePlugin(self.path, [REPLY_MISS])
        self.addCleanup(plugin.close)
        client = PluginClient(self.path)
        reply = client.call({"op": "acquire", "epoch": 0, "tokens": 3, "identity": {}, "ids": [1, 2, 3]})
        self.assertEqual(reply, {"id": "1", "ok": True, "result": {"tokens": 0}})
        self.assertEqual(plugin.requests[0]["op"], "acquire")
        self.assertEqual(plugin.requests[0]["ids"], [1, 2, 3])

    def test_silent_plugin_times_out_and_retries(self):
        plugin = FakePlugin(self.path, ["silent", "silent", "silent"])
        self.addCleanup(plugin.close)
        with unittest.mock.patch.object(srv, "PLUGIN_CONNECT_TIMEOUT_S", 0.5), \
                unittest.mock.patch.object(srv, "PLUGIN_REPLY_TIMEOUT_S", 0.15), \
                unittest.mock.patch.object(srv, "PLUGIN_RETRIES", 2):
            started = time.monotonic()
            self.assertIsNone(PluginClient(self.path).call({"op": "acquire", "epoch": 0}))
            self.assertGreaterEqual(time.monotonic() - started, 0.4)   # three attempts, each to its reply timeout
        self.assertEqual(len(plugin.requests), 3)

    def test_malformed_reply_is_a_failure(self):
        for bad in ("raw:not json", "raw:[]", {"id": "999", "ok": True, "result": {}}):
            with self.subTest(bad=bad):
                plugin = FakePlugin(self.path, [bad, bad, bad])
                self.addCleanup(plugin.close)
                with unittest.mock.patch.object(srv, "PLUGIN_REPLY_TIMEOUT_S", 0.3), \
                        unittest.mock.patch.object(srv, "PLUGIN_RETRIES", 0):
                    self.assertIsNone(PluginClient(self.path).call({"op": "acquire", "epoch": 0}))
                plugin.close()

    def test_closed_connection_is_a_failure(self):
        plugin = FakePlugin(self.path, ["close", "close", "close"])
        self.addCleanup(plugin.close)
        with unittest.mock.patch.object(srv, "PLUGIN_REPLY_TIMEOUT_S", 0.3), \
                unittest.mock.patch.object(srv, "PLUGIN_RETRIES", 0):
            self.assertIsNone(PluginClient(self.path).call({"op": "release", "epoch": 0}))
        self.assertEqual(len(plugin.requests), 1)


class EngineHookTests(unittest.TestCase):
    """The engine hooks with a lazy StrataEngine (no process) and a fake stdin / line queue."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "plugin.sock")

    def engine(self, client):
        e = StrataEngine("strata", ["--serve"], lazy=True)
        e.proc = SimpleNamespace(stdin=FakeStdin(), poll=lambda: None)
        e.ended = False
        e.plugin_client = client
        return e

    def test_acquire_miss_sends_no_dump(self):
        plugin = FakePlugin(self.path, [REPLY_MISS])
        self.addCleanup(plugin.close)
        e = self.engine(PluginClient(self.path))
        StrataEngine._plugin_acquire(e, [1, 2, 3])
        self.assertEqual([r["op"] for r in plugin.requests], ["acquire"])
        self.assertEqual(e.proc.stdin.written, [])

    def test_need_dump_writes_dump_then_acquires_again(self):
        plugin = FakePlugin(self.path, [{"ok": True, "result": {"need_dump": True}}, REPLY_MISS])
        self.addCleanup(plugin.close)
        e = self.engine(PluginClient(self.path))
        e.lines = queue.Queue()
        e.lines.put("INFO something the engine said")
        e.lines.put("DUMPED")
        StrataEngine._plugin_acquire(e, [1, 2, 3])
        self.assertEqual(e.proc.stdin.written, ["DUMP\n"])
        self.assertEqual([r["op"] for r in plugin.requests], ["acquire", "acquire"])
        self.assertEqual(plugin.requests[1]["tokens"], 3)

    def test_dump_wait_is_bounded_and_still_acquires_again(self):
        plugin = FakePlugin(self.path, [{"ok": True, "result": {"need_dump": True}}, REPLY_MISS])
        self.addCleanup(plugin.close)
        e = self.engine(PluginClient(self.path))
        e.lines = queue.Queue()                        # no DUMPED ever comes
        with unittest.mock.patch.object(srv, "PLUGIN_DUMP_WAIT_S", 0.2):
            started = time.monotonic()
            StrataEngine._plugin_acquire(e, [1, 2, 3])
            self.assertGreaterEqual(time.monotonic() - started, 0.15)
        self.assertEqual([r["op"] for r in plugin.requests], ["acquire", "acquire"])

    def test_release_reaches_the_plugin(self):
        plugin = FakePlugin(self.path, [{"ok": True, "result": {"ack": True}}])
        self.addCleanup(plugin.close)
        StrataEngine._plugin_release(self.engine(PluginClient(self.path)))
        self.assertEqual(plugin.requests[0]["op"], "release")
        self.assertEqual(plugin.requests[0]["epoch"], 0)


class GenerateWithPluginTests(unittest.TestCase):
    """The plugin must never stop the GEN line: a dead plugin changes nothing but the log."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "plugin.sock")
        self.dead_path = str(Path(self.dir.name) / "absent.sock")

    def engine(self, client):
        e = StrataEngine("strata", ["--serve"], lazy=True)
        e.proc = SimpleNamespace(stdin=FakeStdin(), poll=lambda: None)
        e.ended = False
        e.plugin_client = client
        e.lines = queue.Queue()
        e.lines.put("DONE 2 3 100.0 50.0 stop")
        return e

    def run_generate(self, e):
        return list(e.generate([1, 2, 3], 5, {}, threading.Event()))

    def test_local_default_stays_the_same(self):
        self.assertIsNone(StrataEngine.plugin_client)
        e = self.engine(None)
        self.assertEqual(self.run_generate(e), [])
        self.assertEqual(e.proc.stdin.written, ["GEN 5 1,2,3\n"])

    def test_dead_plugin_does_not_prevent_gen(self):
        e = self.engine(PluginClient(self.dead_path))  # nothing listens there
        self.assertEqual(self.run_generate(e), [])
        self.assertEqual(e.proc.stdin.written, ["GEN 5 1,2,3\n"])
        self.assertEqual(e.last["generated"], 2)       # the DONE was parsed all the same

    def test_plugin_miss_and_release_around_gen(self):
        plugin = FakePlugin(self.path, [REPLY_MISS, REPLY_MISS])
        self.addCleanup(plugin.close)
        e = self.engine(PluginClient(self.path))
        self.assertEqual(self.run_generate(e), [])
        self.assertEqual(e.proc.stdin.written, ["GEN 5 1,2,3\n"])
        self.assertEqual([r["op"] for r in plugin.requests], ["acquire", "release"])


if __name__ == "__main__":
    unittest.main()
