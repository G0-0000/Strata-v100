"""serve/test_session_composite.py - POST /v1/chat/completions' opt-in strata_session: restore -> generate -> save
inside one engine turn, and splicing new turns onto the saved base's recorded token ids
(mock engine: no GPU, no pack).

    python -m unittest serve.test_session_composite -v
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, IM_END, MockEngine, Service, SessionRefused,  # noqa: E402
                          serve, strata_session_of)

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine(MockEngine):
    """MockEngine + the SAVE/RESTORE command a strata_session composite drives, recording every command in order.

    It counts tokens the way the engine does: when the first submitted id is not BOS it prepends one to its live
    session (this server's ids carry none), so its RESTORE and SAVE answers can be one more than the ids the server
    holds.  Hand-set counts hid that off-by-one; a real machine answered every fresh save -> splice roundtrip with
    409 splice_stale."""

    def __init__(self, tokenizer, **kw):
        super().__init__(tokenizer, kw.pop("script", "Understood."), **kw)
        self.bos = tokenizer.encode("<|endoftext|>", parse_special=True)[0]   # the byte tokenizer's BOS stand-in
        self.commands = []                    # GEN / RESTORE / SAVE, in the order the engine saw them
        self.lock = threading.Lock()
        self.restore_tokens = None            # hand-set count of session_file("restore") (a file no save wrote here)
        self.save_tokens = None               # hand-set count of session_file("save") (a count no convention explains)
        self.restore_error = None             # (kind, reason): session_file("restore") raises SessionRefused
        self.save_refusal = None              # (kind, reason): session_file("save") raises SessionRefused
        self.reused = 0                       # the RESUME figure Service.run reads after the generation
        self.next_reused = None               # set self.reused on the next generate()
        self.gen_gate = None                  # threading.Event: when set, generate() waits on it before GEN
        self.gen_entered = None               # threading.Event: generate() sets it once it waits on the gate
        self.live = 0                         # the engine's live session length, BOS included
        self.files = {}                       # path -> the engine-side count a SAVE recorded there
        self.prompt_tokens = 0                # the engine-side length of the last generate()'s prompt
        self.generated = 0                    # the tokens the last generate() yielded

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if self.gen_gate is not None:         # hold this turn inside the fifo until the test opens the gate
            self.gen_entered.set()            #   the engine is at the gate: commands still holds RESTORE only
            self.gen_gate.wait()
        with self.lock:
            self.commands.append("GEN")
            engine_ids = list(ids)
            if not engine_ids or engine_ids[0] != self.bos:
                engine_ids.insert(0, self.bos)             # the engine's leading BOS: this server's ids carry none
            self.prompt_tokens = len(engine_ids)
            self.reused = min(self.live, self.prompt_tokens)    # RESUME: what the live session shares
            if self.next_reused is not None:
                self.reused, self.next_reused = self.next_reused, None
        self.generated = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings=embeddings):
                self.generated += 1
                yield t
        finally:
            self.live = self.prompt_tokens + self.generated      # the session this request leaves live

    def session_file(self, action, path):
        with self.lock:
            self.commands.append(action.upper())
        if action == "restore":
            if self.restore_error:
                raise SessionRefused(*self.restore_error)
            # the engine reports the count its own SAVE recorded in that file
            tokens = self.restore_tokens if self.restore_tokens is not None else self.files.get(path, 0)
            self.live = tokens
            return {"tokens": tokens, "bytes": tokens * 4, "ms": 12.5}
        if self.save_refusal:
            raise SessionRefused(*self.save_refusal)
        tokens = self.save_tokens if self.save_tokens is not None else self.live
        self.files[path] = tokens
        return {"tokens": tokens, "bytes": tokens * 4, "ms": 34.5}


class Composite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.tok = ByteTokenizer()
        cls.engine = FakeEngine(cls.tok)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.httpd = serve(cls.svc, port=0)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def setUp(self):
        end = self.tok.encode(IM_END, parse_special=True)
        self.engine.scripts = [self.tok.encode("Understood.", parse_special=True) + end]
        self.engine.turns = 0
        self.engine.script = self.engine.scripts[0]
        self.engine.commands = []
        self.engine.restore_tokens = None
        self.engine.save_tokens = None
        self.engine.restore_error = None
        self.engine.save_refusal = None
        self.engine.reused = 0
        self.engine.next_reused = None
        self.engine.gen_gate = None
        self.engine.gen_entered = None
        self.engine.live = 0
        self.engine.files = {}
        self.engine.prompt_tokens = 0
        self.engine.generated = 0

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def sidecar(self, name):
        return Path(self.dir.name, name + ".tokens.json")

    def write_sidecar(self, name, ids, **kw):
        kw.setdefault("engine_tokens", self.engine_count(ids))       # what the engine's SAVE reported for them
        data = {"version": 2, "tokenizer": self.svc.session_tok_hash, "template": self.svc.session_tpl_hash,
                "created": 0, "ids": ids}
        data.update(kw)
        self.sidecar(name).write_text(json.dumps(data), encoding="utf-8")

    def engine_count(self, ids):
        """The tokens the engine keeps for these server-side ids: it prepends one BOS when the first is not BOS."""
        return len(ids) + (0 if ids and ids[0] == self.engine.bos else 1)

    def base_ids(self, user_text, assistant_text, **kw):
        """What a save after one turn records: the submitted prompt + the generated ids (stop token included).
        With thinking on, the generation prompt ends on an open "<think>\n" and the reply closes it before the
        answer - that "\n</think>\n\n" is generated text, recorded in the base like any other token."""
        kw.setdefault("enable_thinking", True)
        submitted = self.svc.encode_prompt([{"role": "user", "content": user_text}], None, dict(kw), store=False)
        reply = self.tok.encode(assistant_text, parse_special=True) + self.tok.encode(IM_END, parse_special=True)
        if kw["enable_thinking"] is not False:
            reply = self.tok.encode("\n</think>\n\n", parse_special=True) + reply
        return submitted + reply

    # T1: the splice invariant - base + separator + delta is exactly the full render of the whole conversation
    def test_t1_spliced_prompt_equals_the_full_render(self):
        pairs = [("First question, with <|im_start|> quoted, ünïcode and\nline breaks.", "First answer: done — 完成。"),
                 ("Second turn asks about <tool_call> verbatim.", "Second answer: 42, with a tab\there.")]
        new = {"role": "user", "content": "A third turn: what is 6×7? And <|im_end|> as text, please."}
        for thinking in (True, False):
            with self.subTest(enable_thinking=thinking):
                kw = {"enable_thinking": thinking}
                whole = [{"role": "user", "content": pairs[0][0]},
                         {"role": "assistant", "content": pairs[0][1]},
                         {"role": "user", "content": pairs[1][0]},
                         {"role": "assistant", "content": pairs[1][1]}, dict(new)]
                # the base is what the second turn's save recorded: the whole submitted prompt of that turn + its reply
                submitted = self.svc.encode_prompt([dict(m) for m in whole[:3]], None, dict(kw), store=False)
                reply = self.tok.encode(pairs[1][1], parse_special=True) + self.tok.encode(IM_END, parse_special=True)
                if thinking:
                    reply = self.tok.encode("\n</think>\n\n", parse_special=True) + reply
                base = submitted + reply
                self.write_sidecar("t1.sess", base)
                spec = strata_session_of({"strata_session": {"restore": "t1.sess", "splice": True}}, [dict(new)],
                                         self.dir.name, False)
                ids, _, _ = self.svc.prepare([dict(new)], None, dict(kw), 0, session=spec)
                want = self.svc.encode_prompt([dict(m) for m in whole], None, dict(kw), store=False)
                self.assertEqual(ids, want)
                self.assertEqual(spec["splice_base_len"], len(base))

    # T2: one fifo turn - a plain chat's GEN never lands between another request's RESTORE, GEN and SAVE
    def test_t2_restore_generate_save_is_atomic_with_a_plain_chat(self):
        self.engine.restore_tokens = 11                                # the file it loads: no save here wrote it
        self.engine.gen_gate = threading.Event()                       # hold the composite's generate in the fifo
        self.engine.gen_entered = threading.Event()
        composite = {"messages": [{"role": "user", "content": "composite"}],
                     "strata_session": {"restore": "t2.sess", "save": "t2.sess"}}
        out = {}
        t = threading.Thread(target=lambda: out.update(r=self.post("/v1/chat/completions", composite)))
        t.start()
        self.assertTrue(self.engine.gen_entered.wait(30))              # the composite is inside the latched generate
        self.assertEqual(self.engine.commands, ["RESTORE"])          # GEN is recorded only after the gate opens
        plain = {}
        t2 = threading.Thread(target=lambda: plain.update(r=self.post(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "plain"}]})))
        t2.start()
        # settle on a fact, not a clock: the server counts the plain request as queued before it waits on the fifo
        # (Service.run), so once the counter moves the request exists - and while the gate is shut the fifo still
        # holds it.  The final order below remains the verdict.
        deadline = time.monotonic() + 30
        while self.svc.status["queued"] < 1:
            self.assertLess(time.monotonic(), deadline, "the plain request was never queued")
            time.sleep(0.01)
        self.assertEqual(self.engine.commands, ["RESTORE"])          # its GEN has not slipped into the turn
        self.engine.gen_gate.set()                                     # release: GEN, SAVE, then the plain chat's GEN
        t2.join(30)
        t.join(30)
        self.assertFalse(t.is_alive() or t2.is_alive())
        self.assertEqual(out["r"][0], 200, out)
        self.assertEqual(plain["r"][0], 200, plain)
        self.assertEqual(self.engine.commands, ["RESTORE", "GEN", "SAVE", "GEN"])
        session = out["r"][1]["session"]
        self.assertEqual(session["restored"], {"tokens": 11, "bytes": 44, "ms": 12.5})
        sidecar = json.loads(self.sidecar("t2.sess").read_text())
        self.assertEqual(session["saved"]["tokens"], sidecar["engine_tokens"])   # the engine's own count
        self.assertEqual(session["saved"]["tokens"], self.engine_count(sidecar["ids"]))
        self.assertNotIn("splice", session)
        self.assertNotIn("save_error", session)
        self.assertNotIn("session", plain["r"][1])                   # the plain request is untouched

    # T3: a restore refusal fails the whole request; a save refusal rides a 200 and writes no sidecar
    def test_t3_restore_refusal_fails_the_whole_request(self):
        self.engine.restore_error = ("invalid", "session file: saved with another model (model fingerprint differs)")
        body = {"messages": [{"role": "user", "content": "hi"}],
                "strata_session": {"restore": "t3.sess", "save": "t3.sess"}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 400, b)
        self.assertIn("fingerprint", b["error"]["message"])
        self.assertEqual(self.engine.commands, ["RESTORE"])          # no GEN, no SAVE

    def test_t3_save_refusal_rides_a_200_and_leaves_the_sidecar_untouched(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t3b.sess", base)
        before = hashlib.sha256(self.sidecar("t3b.sess").read_bytes()).hexdigest()
        self.engine.save_refusal = ("storage", "No space left on device")
        body = {"messages": [{"role": "user", "content": "hi"}],
                "strata_session": {"save": "t3b.sess"}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.commands, ["GEN", "SAVE"])
        session = b["session"]
        self.assertNotIn("saved", session)
        self.assertEqual(session["save_error"], {"kind": "storage", "reason": "No space left on device"})
        self.assertEqual(hashlib.sha256(self.sidecar("t3b.sess").read_bytes()).hexdigest(), before)

    # T4: every bad shape is a 400 before anything reaches the engine
    def test_t4_bad_shapes_are_400_before_anything_is_queued(self):
        img = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        hi = [{"role": "user", "content": "hi"}]
        cases = [
            ({"messages": hi, "strata_session": {"restore": "../evil.sess"}}, "plain file name"),
            ({"messages": hi, "strata_session": {"restore": 5}}, "non-empty string"),
            ({"messages": hi, "strata_session": {"splice": True}}, "needs"),
            ({"messages": hi, "strata_session": {"restore": "a.sess", "splice": "yes"}}, "true or false"),
            ({"messages": [{"role": "user", "content": img}],
              "strata_session": {"restore": "a.sess", "splice": True}}, "images"),
            ({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
              "strata_session": {"restore": "a.sess", "splice": True}}, "system"),
            ({"messages": [{"role": "assistant", "content": "hi"}],
              "strata_session": {"restore": "a.sess", "splice": True}}, 'role "user"'),
            ({"messages": hi, "stream": True, "strata_session": {"restore": "a.sess"}}, "stream"),
            ({"messages": hi, "strata_session": {"restore": "a.sess", "nope": 1}}, "unknown key"),
        ]
        for body, word in cases:
            with self.subTest(word=word):
                s, b = self.post("/v1/chat/completions", body)
                self.assertEqual(s, 400, (word, b))
                self.assertIn(word, b["error"]["message"])
        self.assertEqual(self.engine.commands, [])

    def test_t4_an_empty_strata_session_asks_for_nothing(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "strata_session": {}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 200, b)
        self.assertNotIn("session", b)
        self.assertEqual(self.engine.commands, ["GEN"])

    # T5: the sidecar names its writer - another one, or a restore count that disagrees with it, is a 409
    def test_t5_a_sidecar_from_another_tokenizer_is_409(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t5.sess", base, tokenizer="deadbeefdeadbeef")
        saved = self.svc.session_tok_hash
        self.svc.session_tok_hash = "cafecafecafecafe"
        try:
            body = {"messages": [{"role": "user", "content": "next"}],
                    "strata_session": {"restore": "t5.sess", "splice": True}}
            s, b = self.post("/v1/chat/completions", body)
        finally:
            self.svc.session_tok_hash = saved
        self.assertEqual(s, 409, b)
        self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(self.engine.commands, [])                   # refused before the engine

    def test_t5_a_restore_count_below_the_sidecar_is_409_with_expected_and_got(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t5b.sess", base)
        engine_base = self.engine_count(base)                       # the engine's count: the restore check's number
        self.engine.restore_tokens = engine_base - 3                 # the file changed underneath: it restores fewer
        body = {"messages": [{"role": "user", "content": "next"}],
                "strata_session": {"restore": "t5b.sess", "splice": True}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 409, b)
        self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(b["error"]["expected"], engine_base)
        self.assertEqual(b["error"]["got"], engine_base - 3)
        self.assertIn(f"the engine restored {engine_base - 3} tokens, the sidecar's engine count is {engine_base}",
                      b["error"]["message"])                        # both conventions named
        self.assertEqual(self.engine.commands, ["RESTORE"])

    # T6: the engine reused fewer tokens than the base holds - a warning and resume_ok=false, the answer still stands
    def test_t6_a_short_resume_degrades_to_resume_ok_false(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t6.sess", base)
        engine_base = self.engine_count(base)
        self.engine.restore_tokens = engine_base
        self.engine.next_reused = engine_base - 10                   # the engine's RESUME, short of the base
        body = {"messages": [{"role": "user", "content": "next"}],
                "strata_session": {"restore": "t6.sess", "save": "t6.sess", "splice": True}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 200, b)
        session = b["session"]
        self.assertEqual(session["restored"]["tokens"], engine_base)
        splice = session["splice"]
        self.assertEqual(splice["base_tokens"], len(base))           # the server-side ids the sidecar holds
        self.assertEqual(splice["resume"], engine_base - 10)
        self.assertFalse(splice["resume_ok"])
        self.assertIn("saved", session)
        got = json.loads(self.sidecar("t6.sess").read_text())["ids"]
        self.assertEqual(got[:len(base)], base)                      # sidecar: the submitted prompt + the generated ids
        self.assertGreater(len(got), len(base))

    # T7: a base cut mid-turn cannot be spliced; a reasoning recovery saves the session but skips the sidecar
    def test_t7_a_base_not_ending_on_a_stop_token_is_409(self):
        base = self.base_ids("one", "one done")[:-1]                 # the stop token cut off: a mid-turn base
        self.write_sidecar("t7.sess", base)
        body = {"messages": [{"role": "user", "content": "next"}],
                "strata_session": {"restore": "t7.sess", "splice": True}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 409, b)
        self.assertEqual(b["error"]["type"], "splice_base_seam")
        self.assertEqual(self.engine.commands, [])

    def test_t7_reasoning_recovery_skips_the_sidecar(self):
        self.svc.reasoning_loop_recovery = "recover"
        try:
            base = self.base_ids("one", "one done")
            self.write_sidecar("t7b.sess", base)
            self.engine.restore_tokens = self.engine_count(base)
            loop = "<think>" + "ab" * 1500 + "</think>\n\nThe answer is 42."
            end = self.tok.encode(IM_END, parse_special=True)
            self.engine.scripts = [self.tok.encode(loop, parse_special=True) + end]
            self.engine.turns = 0
            self.engine.script = self.engine.scripts[0]
            body = {"messages": [{"role": "user", "content": "next"}],
                    "strata_session": {"restore": "t7b.sess", "save": "t7c.sess", "splice": True}}
            s, b = self.post("/v1/chat/completions", body)
            self.assertEqual(s, 200, b)
            session = b["session"]
            self.assertIn("saved", session)                          # the .sess save itself succeeded
            self.assertEqual(session["sidecar"], "skipped_recovery")
            self.assertFalse(self.sidecar("t7c.sess").exists())      # no sidecar was written
        finally:
            self.svc.reasoning_loop_recovery = False

    # T8: branches the happy path never reached - validation, identity, and a failed sidecar write
    def test_t8_needs_slot_save_path_rejects_parallel_and_the_mcp_mutex(self):
        hi = [{"role": "user", "content": "hi"}]
        with self.subTest(case="no --slot-save-path"):
            with self.assertRaisesRegex(ValueError, "--slot-save-path"):
                strata_session_of({"strata_session": {"restore": "a.sess"}}, hi, None, False)
        with self.subTest(case="parallel requests"):
            with self.assertRaisesRegex(ValueError, "parallel"):
                strata_session_of({"strata_session": {"restore": "a.sess"}}, hi, self.dir.name, True)
        with self.subTest(case="strata_mcp mutex"):
            class _McpStub:                       # just enough for the handler's MCP-tool branch
                def wait(self, timeout):
                    pass

                def template_tools(self, exclude=()):
                    return [{"name": "stub_tool", "description": "t", "parameters": {"type": "object"}}]

            saved = self.svc.mcp
            self.svc.mcp = _McpStub()
            try:
                body = {"messages": hi, "strata_mcp": True,
                        "strata_session": {"save": "t8mcp.sess"}}
                s, b = self.post("/v1/chat/completions", body)
            finally:
                self.svc.mcp = saved
            self.assertEqual(s, 400, b)
            self.assertIn("strata_session with MCP tools", b["error"]["message"])
        self.assertEqual(self.engine.commands, [])                     # none of these reached the engine

    def test_t8_a_sidecar_from_another_template_is_409(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t8tpl.sess", base, template="deadbeefdeadbeef")
        body = {"messages": [{"role": "user", "content": "next"}],
                "strata_session": {"restore": "t8tpl.sess", "splice": True}}
        s, b = self.post("/v1/chat/completions", body)
        self.assertEqual(s, 409, b)
        self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(self.engine.commands, [])

    def test_t8_a_bad_sidecar_version_or_garbage_is_409(self):
        base = self.base_ids("one", "one done")
        for make, word in ((lambda: json.dumps({"version": 1, "ids": base}), "version 1 (no engine count)"),
                           (lambda: json.dumps({"version": 2, "ids": base}), "version 2 without engine_tokens"),
                           (lambda: "{not json", "garbage")):
            with self.subTest(word=word):
                self.sidecar("t8ver.sess").write_text(make(), encoding="utf-8")
                body = {"messages": [{"role": "user", "content": "next"}],
                        "strata_session": {"restore": "t8ver.sess", "splice": True}}
                s, b = self.post("/v1/chat/completions", body)
                self.assertEqual(s, 409, b)
                self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(self.engine.commands, [])

    def test_t8_splice_render_mismatch_is_the_400_shape(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("t8rend.sess", base)
        spec = strata_session_of({"strata_session": {"restore": "t8rend.sess", "splice": True}},
                                 [{"role": "user", "content": "hi"}], self.dir.name, False)
        bare = self.tok.encode("no turn opening here", parse_special=True)
        with self.assertRaisesRegex(ValueError, "splice_render_mismatch"):
            self.svc._spliced_prompt(bare, [{"role": "user", "content": "hi"}], spec)

    def test_t8_a_sidecar_write_failure_still_reports_saved(self):
        blocker = Path(self.dir.name, "t8io.sess.tokens.json.tmp")
        blocker.mkdir()                            # the sidecar's temp path is a directory: open() fails with OSError
        try:
            body = {"messages": [{"role": "user", "content": "hi"}],
                    "strata_session": {"save": "t8io.sess"}}
            s, b = self.post("/v1/chat/completions", body)
        finally:
            blocker.rmdir()
        self.assertEqual(s, 200, b)
        session = b["session"]
        self.assertIn("saved", session)                            # the .sess save itself succeeded
        self.assertEqual(session["save_error"]["kind"], "io")

    # T9: the sidecar is first read in prepare(), outside the fifo - a same-name save that swaps it between that read
    # and the turn is a 409 (the in-turn re-read catches what the restore-count check cannot)
    def test_t9_a_same_name_save_between_prepare_and_the_turn_is_409(self):
        base = self.base_ids("one", "one done")
        self.write_sidecar("race.sess", base)
        self.engine.restore_tokens = self.engine_count(base)        # both restores answer the base's count
        self.engine.gen_gate = threading.Event()                   # B holds the fifo inside its generate
        self.engine.gen_entered = threading.Event()
        outb, outa = {}, {}
        tb = threading.Thread(target=lambda: outb.update(r=self.post("/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "the replacement turn"}],
                      "strata_session": {"restore": "race.sess", "save": "race.sess"}})))
        tb.start()
        self.assertTrue(self.engine.gen_entered.wait(30))          # B is in the fifo: RESTORE done, SAVE not yet
        ta = threading.Thread(target=lambda: outa.update(r=self.post("/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "next"}],
                      "strata_session": {"restore": "race.sess", "splice": True}})))
        ta.start()
        deadline = time.monotonic() + 30
        while self.svc.status["queued"] < 1:                       # A finished prepare (read the old sidecar) and
            self.assertLess(time.monotonic(), deadline, "the spliced request was never queued")
            time.sleep(0.01)                                       # waits on the fifo behind B
        self.assertEqual(json.loads(self.sidecar("race.sess").read_text())["ids"], base)  # B has not saved yet
        self.engine.gen_gate.set()                                 # B: GEN, SAVE (sidecar swapped), release the fifo
        tb.join(30)
        ta.join(30)
        self.assertFalse(ta.is_alive() or tb.is_alive())
        self.assertEqual(outb["r"][0], 200, outb)
        self.assertIn("saved", outb["r"][1]["session"])            # B's save succeeded: the sidecar now holds B's ids
        s, b = outa["r"]
        self.assertEqual(s, 409, b)                                # A: the base it rendered with is gone
        self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(b["error"]["expected"], len(base))
        self.assertEqual(b["error"]["got"],
                         len(json.loads(self.sidecar("race.sess").read_text())["ids"]))

    # T10: the counts the engine reports are its own - it prepends one BOS this server's ids do not hold, so a fresh
    # save -> splice roundtrip must compare the restore against the sidecar's engine count, not against its ids
    def test_t10_a_fresh_save_then_splice_roundtrip_succeeds(self):
        s, b = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "one"}],
                                                  "strata_session": {"save": "t10.sess"}})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.commands, ["GEN", "SAVE"])
        first = json.loads(self.sidecar("t10.sess").read_text())
        self.assertEqual(first["version"], 2)
        self.assertEqual(b["session"]["saved"]["tokens"], first["engine_tokens"])   # the SAVED count, as reported
        self.assertEqual(first["engine_tokens"], self.engine_count(first["ids"]))
        self.assertEqual(first["engine_tokens"], len(first["ids"]) + 1)             # the engine's one leading BOS
        s, b = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "two"}],
                                                  "strata_session": {"restore": "t10.sess", "splice": True}})
        self.assertEqual(s, 200, b)                                   # it answered 409 splice_stale before the fix
        session = b["session"]
        self.assertEqual(session["restored"]["tokens"], first["engine_tokens"])
        splice = session["splice"]
        self.assertEqual(splice["base_tokens"], len(first["ids"]))     # the ids the new turns were spliced onto
        self.assertEqual(splice["resume"], first["engine_tokens"])
        self.assertTrue(splice["resume_ok"], splice)
        self.assertEqual(self.engine.commands, ["GEN", "SAVE", "RESTORE", "GEN"])

    def test_t10_a_version_1_sidecar_heals_with_one_full_resend(self):
        base = self.base_ids("one", "one done")
        old = {"version": 1, "tokenizer": self.svc.session_tok_hash, "template": self.svc.session_tpl_hash,
               "created": 0, "ids": base}
        self.sidecar("t10v1.sess").write_text(json.dumps(old), encoding="utf-8")
        s, b = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "next"}],
                                                  "strata_session": {"restore": "t10v1.sess", "splice": True}})
        self.assertEqual(s, 409, b)                                    # a v1 sidecar is stale: resend once
        self.assertEqual(b["error"]["type"], "splice_stale")
        self.assertEqual(self.engine.commands, [])
        whole = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "one done"},
                 {"role": "user", "content": "next"}]
        s, b = self.post("/v1/chat/completions", {"messages": whole,
                                                  "strata_session": {"save": "t10v1.sess"}})
        self.assertEqual(s, 200, b)                                    # the full resend rebuilds the base
        self.assertEqual(json.loads(self.sidecar("t10v1.sess").read_text())["version"], 2)
        s, b = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "third"}],
                                                  "strata_session": {"restore": "t10v1.sess", "splice": True}})
        self.assertEqual(s, 200, b)                                    # and the splice now works
        self.assertTrue(b["session"]["splice"]["resume_ok"], b["session"])

    # T11: an engine count that is no BOS explanation is logged, not fixed: the sidecar still records it as reported
    def test_t11_an_unusual_save_count_is_logged_and_still_written(self):
        self.engine.save_tokens = 5                                    # neither the ids' count nor that plus one
        s, b = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "one"}],
                                                  "strata_session": {"save": "t11.sess"}})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["session"]["saved"]["tokens"], 5)
        data = json.loads(self.sidecar("t11.sess").read_text())
        self.assertEqual(data["engine_tokens"], 5)                   # the engine's count: authoritative for restore
        self.assertGreater(len(data["ids"]), 6)


if __name__ == "__main__":
    unittest.main()