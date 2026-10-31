"""serve/test_cache_gate_start.py - the disk cache's recompute gate runs BEFORE the engine takes the GPU
(docs/CACHE_REUSE_RULES.md R1/R2/R4: the program verifies, nothing is judged or deleted before it does, and a cache
with nothing to check costs the start nothing).

The verifier itself is tools/cache_verify.py, loaded by path; here it is replaced with a stand-in that records what it
was asked, so these tests never start an engine and need no GPU.

    python -m unittest serve.test_cache_gate_start -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402


class Pending:
    def __init__(self, verify: bool, detail: str = "detail", refused: bool = False):
        self.verify = verify
        self.detail = detail
        self.target = None
        self.sources = ()
        self.refused = refused


class FakeVerifier:
    """Stands in for tools/cache_verify.py: records the calls and claims whatever it is told to."""

    Refused = type("Refused", (Exception,), {})
    UsageError = type("UsageError", (Exception,), {})

    def __init__(self, pending: Pending, status: int = 0):
        self.pending, self.status = pending, status
        self.calls: list[tuple] = []

    def probe_engine_build(self, exe, env=None):
        self.calls.append(("build", exe, env))
        return "0.1.52.20"

    def pending_verification(self, cache_dir, slot_dirs, build):
        self.calls.append(("pending", str(cache_dir), [str(d) for d in slot_dirs], build))
        return self.pending

    def build_parser(self):
        class Parser:
            def parse_args(inner, argv):
                self.calls.append(("argv", list(argv)))
                return argv
        return Parser()

    def run(self, options, passthrough=(), *, env=None, cwd=None):
        self.calls.append(("run", env, cwd))
        return self.status


class RealParserVerifier(FakeVerifier):
    """tools/cache_verify.py's own parser: the command line the server builds must be one it accepts, and the parsed
    options must still carry the engine's arguments whole - a flag-shaped one included."""

    def __init__(self, pending: Pending, module, status: int = 0):
        super().__init__(pending, status)
        self.module = module

    def build_parser(self):
        real = self.module.build_parser()
        recorder = self

        class Parser:
            def parse_args(inner, argv):
                recorder.calls.append(("argv", list(argv)))
                return real.parse_args(argv)
        return Parser()

    def run(self, options, passthrough=(), *, env=None, cwd=None):
        self.calls.append(("options", options, env, cwd))
        return self.status


class CacheGateStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spill = self.root / "spill"
        self.spill.mkdir()
        self.slots = self.root / "slots"
        self.slots.mkdir()
        self.engine = str(self.root / "engine")

    def args(self, gate=True, spill=True, sep=" "):
        args = ["--conversation-cache-disk-only", "--max-context", "131072"]
        if spill:
            if sep == "=":
                args.append(f"--conversation-cache-spill-dir={self.spill}")
            else:
                args += ["--conversation-cache-spill-dir", str(self.spill)]
        if gate:
            args.append("--conversation-cache-disk-recompute-gate")
        return args

    def real_module(self):
        module = server._cache_verify_module()
        if module is None:
            self.skipTest("tools/cache_verify.py is not in this install")
        return module

    def call(self, verifier, args=None, slots=None, env=None, cwd=None):
        out = io.StringIO()
        with mock.patch.object(server, "_cache_verify_module", return_value=verifier), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            server.cache_gate_verify(self.engine, self.args() if args is None else args,
                                     [str(self.slots)] if slots is None else slots, env=env, cwd=cwd)
        return out.getvalue()

    def test_the_gate_off_touches_nothing(self):
        verifier = FakeVerifier(Pending(True))
        self.assertEqual(self.call(verifier, args=self.args(gate=False)), "")
        self.assertEqual(verifier.calls, [])

    def test_no_spill_directory_means_the_engine_turns_the_gate_off(self):
        verifier = FakeVerifier(Pending(True))
        self.assertEqual(self.call(verifier, args=self.args(spill=False)), "")
        self.assertEqual(verifier.calls, [])

    def test_a_spill_directory_that_does_not_exist_yet_verifies_nothing(self):
        verifier = FakeVerifier(Pending(True))
        args = ["--conversation-cache-spill-dir", str(self.root / "missing"),
                "--conversation-cache-disk-recompute-gate"]
        self.assertEqual(self.call(verifier, args=args), "")
        self.assertEqual(verifier.calls, [])

    def test_nothing_pending_prints_nothing_and_starts_no_engine(self):
        verifier = FakeVerifier(Pending(False))
        self.assertEqual(self.call(verifier), "")
        self.assertEqual([c[0] for c in verifier.calls], ["build", "pending"])
        self.assertEqual(Path(verifier.calls[1][1]), self.spill)
        self.assertEqual(verifier.calls[1][2], [str(self.slots)])

    def test_the_gate_and_slot_directory_reach_the_verifier(self):
        verifier = FakeVerifier(Pending(False))
        self.call(verifier, args=self.args(sep="="))
        kind, cache_dir, slot_dirs, build = verifier.calls[1]
        self.assertEqual(kind, "pending")
        self.assertEqual(Path(cache_dir), self.spill)
        self.assertEqual(slot_dirs, [str(self.slots)])
        self.assertEqual(build, "0.1.52.20")

    def test_a_pending_source_runs_the_verifier_once_with_the_engine_arguments(self):
        verifier = FakeVerifier(Pending(True, "2 file(s) of another configuration await the recompute"))
        text = self.call(verifier)
        self.assertEqual([c[0] for c in verifier.calls], ["build", "pending", "argv", "run"])
        argv = verifier.calls[2][1]
        self.assertEqual(argv[:6], ["--cache-dir", str(self.spill), "--engine", self.engine, "--engine-build",
                                    "0.1.52.20"])
        self.assertIn("--slot-dir", argv)
        self.assertIn(str(self.slots), argv)
        # every engine argument reaches the verification run, and the gate switch itself does not (it is --serve's);
        # each one is a single --engine-arg=ARG token (see the test below)
        self.assertIn("--engine-arg=--max-context", argv)
        self.assertIn("--engine-arg=131072", argv)
        self.assertNotIn("--engine-arg=--conversation-cache-disk-recompute-gate", argv)
        self.assertEqual(len([a for a in argv if a.startswith("--engine-arg=")]), 5)   # args() minus the gate switch
        self.assertIn("await the recompute", text)
        self.assertIn("verified for this engine build", text)

    def test_the_real_parser_takes_the_servers_command_line(self):
        # P0 (review): the engine's own flags reached argparse as bare tokens, and "--engine-arg --kv" is what
        # argparse calls "expected one argument" - SystemExit(2), which is no Exception, so the whole start died.
        # One token per argument (--engine-arg=ARG) keeps a flag-shaped value intact.
        verifier = RealParserVerifier(Pending(True), self.real_module())
        try:
            text = self.call(verifier)
        except SystemExit:                         # in the server this one leaves cache_gate_verify and ends the start
            self.fail("the real parser must accept the command line cache_gate_verify builds")
        argv = verifier.calls[2][1]
        engine_args = ["--conversation-cache-disk-only", "--max-context", "131072",
                       "--conversation-cache-spill-dir", str(self.spill)]
        self.assertEqual([a for a in argv if a.startswith("--engine-arg=")],
                         [f"--engine-arg={a}" for a in engine_args])
        options = verifier.calls[-1][1]
        self.assertEqual(options.engine_arg, engine_args)
        self.assertEqual(options.cache_dir, self.spill)
        self.assertEqual(options.engine, self.engine)
        self.assertEqual(options.engine_build, "0.1.52.20")
        self.assertEqual(options.slot_dir, [self.slots])
        self.assertIn("verified for this engine build", text)

    def test_a_refused_command_line_is_reported_and_the_start_goes_on(self):
        # SystemExit is not an Exception: uncaught, the parser's refusal ended the whole start before the engine was
        # started.  R1/R2 (docs/CACHE_REUSE_RULES.md): a verification that cannot run approves nothing, deletes
        # nothing and judges nothing - the engine refuses these files exactly as it does without the gate.
        verifier = FakeVerifier(Pending(True))

        def refuse(argv):
            raise SystemExit(2)
        verifier.build_parser = lambda: mock.Mock(parse_args=refuse)
        text = self.call(verifier)
        self.assertIn("could not run", text)
        self.assertIn("nothing is approved or deleted", text)
        self.assertIn("starting the engine anyway", text)
        self.assertEqual([c[0] for c in verifier.calls], ["build", "pending"])

    def test_a_failed_verification_is_reported_and_the_start_goes_on(self):
        verifier = FakeVerifier(Pending(True), status=1)
        text = self.call(verifier)
        self.assertIn("did not pass", text)
        self.assertIn("as before", text)
        self.assertEqual([c[0] for c in verifier.calls], ["build", "pending", "argv", "run"])

    def test_the_verification_engine_gets_the_serving_engines_environment_and_cwd(self):
        # R1 (docs/CACHE_REUSE_RULES.md): the K/V the verification engine recomputes are compared with the cached
        # file's, so it runs as the engine it verifies for - the same GPU, kernel switches and library path
        # (child_env) in the same working directory
        verifier = FakeVerifier(Pending(True))
        child_env = {"CUDA_VISIBLE_DEVICES": "0", "STRATA_PF_WMMA": "0"}
        self.call(verifier, env=child_env, cwd="/srv/strata")
        self.assertEqual(verifier.calls[0], ("build", self.engine, child_env))
        self.assertEqual(verifier.calls[-1], ("run", child_env, "/srv/strata"))

    def test_a_refused_scan_is_said_once_and_starts_no_engine(self):
        # the tool refused to scan (an unreadable slot directory): silence would read as "checked"
        verifier = FakeVerifier(Pending(False, "cannot read the slot directory: permission denied", refused=True))
        text = self.call(verifier)
        self.assertIn("could not be scanned", text)
        self.assertIn("permission denied", text)
        self.assertIn("no file is deleted", text)
        self.assertNotIn("run", [c[0] for c in verifier.calls])

    def test_a_verifier_that_raises_never_stops_the_start(self):
        verifier = FakeVerifier(Pending(True))
        def explode(*_a, **_k):
            raise RuntimeError("boom")
        verifier.pending_verification = explode
        text = self.call(verifier)
        self.assertIn("starting the engine anyway", text)

    def test_an_install_without_the_tools_directory_says_so(self):
        out = io.StringIO()
        with mock.patch.object(server, "_cache_verify_module", return_value=None), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            server.cache_gate_verify(self.engine, self.args(), [])
        self.assertIn("tools/cache_verify.py is not in this install", out.getvalue())
        self.assertIn("no file is deleted", out.getvalue())


class LoadsTheRealVerifier(unittest.TestCase):
    """The path the server resolves and the module it loads: one verifier, not a copy of it."""

    def test_the_module_loads_from_the_repository(self):
        module = server._cache_verify_module()
        if not (Path(__file__).resolve().parents[1] / "tools" / "cache_verify.py").is_file():
            self.assertIsNone(module)
            return
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, "pending_verification"))
        self.assertTrue(hasattr(module, "run"))
        self.assertIs(module, module)

    def test_a_verifier_that_cannot_be_loaded_does_not_stop_the_start(self):
        # a half-copied tools/cache_verify.py raises a SyntaxError, not an ImportError: a verifier that cannot be
        # loaded must leave the gate fail-closed and the server running, never end the start
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "serve").mkdir()
            (root / "tools").mkdir()
            (root / "serve" / "server.py").write_text("")
            (root / "tools" / "cache_verify.py").write_text("def broken(:\n")
            with mock.patch.object(server, "__file__", str(root / "serve" / "server.py")):
                self.assertIsNone(server._cache_verify_module())

    def test_the_server_does_not_carry_its_own_verifier(self):
        # the tool's ledger parser and hashes exist once: in tools/cache_verify.py
        source = Path(server.__file__).read_text(encoding="utf-8")
        self.assertNotIn("strata-cache-gate 1", source)
        self.assertNotIn("def format_ledger", source)


if __name__ == "__main__":
    unittest.main()
