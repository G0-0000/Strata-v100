"""tools/cache_verify.py: the offline recompute-gate verifier (sidecar v1/v2, the ledger hash and format, sample
choice, digest comparison, atomic ledger update, and the CLI against mocked engine/digest subprocesses)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import cache_verify as CV  # noqa: E402

MODEL = 0x92015B27A6E3EFB8
OLD_CONFIG = 0x3074D88348B69B80
NEW_CONFIG = 0xD88978F614302157


def sidecar_bytes(version=2, model=MODEL, config=OLD_CONFIG, ids=(1, 2, 3), images=(), checkpoints=(2,),
                  last_used=1793037175538, origin=0, corrupt=False):
    body = b"SCSM" + struct.pack("<QQQQQ", version, model, config, 1, len(ids))
    for token in ids:
        body += struct.pack("<q", token)
    body += struct.pack("<Q", len(images))
    for start, key in images:
        body += struct.pack("<qQ", start, key)
    body += struct.pack("<Q", len(checkpoints))
    for length in checkpoints:
        body += struct.pack("<Q", length)
    if version >= 2:
        body += struct.pack("<Q", last_used)
    if version >= 3:
        body += struct.pack("<Q", origin)
    data = body + struct.pack("<Q", CV.session_hash64(body, 0))
    if corrupt:
        data = data[:-1] + bytes([data[-1] ^ 0xFF])
    return data


def digest(tokens=1, kv=2, state=3, exact=True, ref=False, rht=False):
    return CV.Digest(3, tokens, kv, state, exact, ref, rht)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cache_file(self, stem, size, version=2, config=OLD_CONFIG, ids=(1, 2, 3), images=(), checkpoints=(2,),
                   write_meta=True, dlog=False):
        (self.root / f"{stem}.sess").write_bytes(b"x" * size)
        if write_meta:
            (self.root / f"{stem}.meta").write_bytes(
                sidecar_bytes(version=version, config=config, ids=ids, images=images, checkpoints=checkpoints))
        if dlog:
            (self.root / f"{stem}.dlog").write_bytes(b"delta")
        return self.root / f"{stem}.sess"


class Hash(TempDir):
    def test_matches_the_cpp_implementation(self):
        # produced by the C++ SessionHasher (src/core/conversation_file.cpp) in a throwaway harness
        vectors = [(b"", 0, "3fdf455f9dcf1e62"), (b"", 1, "1157986175266e71"), (b"a", 0, "722f437eb72ecc23"),
                   (b"abc", 0, "c6286258a8ea2c6a"), (b"x" * 31, 0, "282b32bacd563b74"),
                   (b"x" * 32, 0, "e2df261fc2ec30eb"), (b"x" * 33, 0, "b3fa465f554208a6"),
                   (b"x" * 64, 0, "a97161d5402f516e"),
                   (b"strata-cache-gate 1\nbatch b1 0x0000000000000001 0x0000000000000002 0x0000000000000001 "
                    b"0x0000000000000003 0.1.52.20 s1 pass 123 -", 0, "c6e1a559c896059f")]
        for data, seed, want in vectors:
            self.assertEqual(f"{CV.session_hash64(data, seed):016x}", want, data)

    def test_streaming_equals_one_shot(self):
        data = b"q" * 1000
        hasher = CV.SessionHasher(0)
        for at in range(0, len(data), 7):
            hasher.update(data[at:at + 7])
        self.assertEqual(hasher.digest(), CV.session_hash64(data, 0))


class Sidecar(TempDir):
    def test_v2_roundtrip(self):
        parsed = CV.parse_sidecar(sidecar_bytes())
        self.assertEqual((parsed.version, parsed.model, parsed.config, parsed.cvec), (2, MODEL, OLD_CONFIG, True))
        self.assertEqual(parsed.ids, (1, 2, 3))
        self.assertEqual(parsed.checkpoints, (2,))
        self.assertEqual(parsed.last_used, 1793037175538)

    def test_v1_has_no_last_use_stamp(self):
        parsed = CV.parse_sidecar(sidecar_bytes(version=1))
        self.assertEqual(parsed.version, 1)
        self.assertEqual(parsed.last_used, 0)
        self.assertEqual(parsed.ids, (1, 2, 3))

    def test_images_are_parsed(self):
        parsed = CV.parse_sidecar(sidecar_bytes(images=((1, 99),)))
        self.assertEqual(parsed.images, ((1, 99),))

    def test_bad_checksum_is_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(corrupt=True))

    def test_truncation_and_bad_magic_are_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes()[:-8])
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(b"XCSM" + sidecar_bytes()[4:])

    def test_v3_carries_the_conversation_origin(self):
        # the engine writes v3 when --conversation-cache-origin-split is on and the conversation is tagged
        # (conversation_spill.cpp kMetaVersionOrigin): same parser, one more u64 after the last-use stamp
        parsed = CV.parse_sidecar(sidecar_bytes(version=3, origin=1))
        self.assertEqual((parsed.version, parsed.origin, parsed.last_used), (3, 1, 1793037175538))
        self.assertEqual((parsed.model, parsed.config, parsed.ids), (MODEL, OLD_CONFIG, (1, 2, 3)))
        self.assertEqual(CV.parse_sidecar(sidecar_bytes(version=3, origin=2)).origin, 2)   # disk_restore
        self.assertEqual(CV.parse_sidecar(sidecar_bytes(version=3)).origin, 0)             # ordinary

    def test_a_v3_origin_past_the_engines_range_is_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(version=3, origin=3))     # ConversationOrigin::disk_restore is the last

    def test_unsupported_version_is_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(version=4))

    def test_invalid_chains_are_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(checkpoints=(9,)))            # past the live ids
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(images=((5, 1),)))           # image start past the ids
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(ids=()))                     # no tokens: nothing to replay
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(ids=(-1,)))                  # outside the int32 token range

    def test_reads_from_disk(self):
        path = self.root / "a.meta"
        path.write_bytes(sidecar_bytes())
        self.assertEqual(CV.read_sidecar(path).ids, (1, 2, 3))


class Scan(TempDir):
    def test_groups_and_skips(self):
        self.cache_file("strata-conv-a", 100)
        smallest = self.cache_file("strata-conv-b", 50)
        self.cache_file("strata-conv-same", 60, config=NEW_CONFIG)
        self.cache_file("strata-conv-img", 70, images=((1, 5),))
        self.cache_file("strata-conv-chain", 80, dlog=True)
        self.cache_file("strata-conv-nometa", 90, write_meta=False)
        self.cache_file("strata conv space", 40)
        self.cache_file("strata-conv-v1", 55, version=1, ids=(4, 5, 6, 7))
        files = CV.scan_cache_dir(self.root)
        eligible = [f for f in files if not f.skip]
        self.assertEqual([f.stem for f in eligible], ["strata-conv-a", "strata-conv-b", "strata-conv-same",
                                                      "strata-conv-v1"])
        skipped = {f.stem: f.skip for f in files if f.skip}
        self.assertIn("images", skipped["strata-conv-img"])
        self.assertIn("delta chain", skipped["strata-conv-chain"])
        self.assertIn("sidecar is missing", skipped["strata-conv-nometa"])
        self.assertIn("whitespace", skipped["strata conv space"])
        legacy = CV.group_by_source(eligible)[CV.Identity(MODEL, OLD_CONFIG)]
        self.assertEqual(CV.choose_sample(legacy).stem, "strata-conv-b")
        self.assertEqual({f.stem for f in legacy}, {"strata-conv-a", "strata-conv-b", "strata-conv-v1"})
        self.assertEqual(len(CV.group_by_source(eligible)), 2)

    def test_a_shared_prefix_root_is_never_the_sample(self):
        # 0.1.52.22: the smallest file of the group was a root prefix (strata-conv-prefix-2), so the sample was a
        # prompt's shared root and the whole source failed kv_mismatch - its two complete conversations
        # (strata-conv-1, strata-conv-3) were never tried
        self.cache_file("strata-conv-prefix-2", 40, ids=(1, 2))
        self.cache_file("strata-conv-1", 100, ids=(1, 2, 3, 4, 5))
        self.cache_file("strata-conv-3", 120, ids=(1, 2, 3))
        group = CV.group_by_source(CV.scan_cache_dir(self.root))[CV.Identity(MODEL, OLD_CONFIG)]
        self.assertEqual(sorted(f.stem for f in group),
                         ["strata-conv-1", "strata-conv-3", "strata-conv-prefix-2"])   # membership is unchanged
        self.assertEqual(CV.choose_sample(group).stem, "strata-conv-1")
        self.assertTrue(CV.is_prefix_root(next(f for f in group if f.stem == "strata-conv-prefix-2")))
        self.assertFalse(CV.is_prefix_root(next(f for f in group if f.stem == "strata-conv-1")))

    def test_a_group_of_only_prefix_roots_has_no_sample(self):
        self.cache_file("strata-conv-prefix-7", 40, ids=(1, 2))
        group = CV.group_by_source(CV.scan_cache_dir(self.root))[CV.Identity(MODEL, OLD_CONFIG)]
        self.assertIsNone(CV.choose_sample(group))

    def test_a_v3_sidecar_is_scanned_like_the_others(self):
        # --conversation-cache-origin-split writes a v3 sidecar for a tagged conversation (kMetaVersionOrigin), and
        # the engine still holds such a file of another configuration "not indexed until the recompute gate
        # approves" it: a scan that skipped v3 would leave it unverifiable for ever (R1, docs/CACHE_REUSE_RULES.md)
        self.cache_file("strata-conv-v3", 42, version=3)
        files = CV.scan_cache_dir(self.root)
        self.assertEqual([f.skip for f in files], [""])
        self.assertEqual(files[0].sidecar.version, 3)
        self.assertEqual(CV.group_by_source(files)[CV.Identity(MODEL, OLD_CONFIG)][0].stem, "strata-conv-v3")

    def test_a_stem_that_is_not_utf8_is_skipped(self):
        # os.listdir surfaces undecodable name bytes as lone surrogates; the ledger is UTF-8, so such a file can
        # never be named in one and must be skipped rather than crash the write in format_ledger.
        raw = os.fsencode(self.root) + b"/strata-conv-\xff"
        with open(raw + b".sess", "wb") as out:
            out.write(b"x" * 10)
        with open(raw + b".meta", "wb") as out:
            out.write(sidecar_bytes())
        files = CV.scan_cache_dir(self.root)
        self.assertEqual([f.skip for f in files],
                         ["stem is not valid UTF-8 (cannot be written to the ledger)"])
        self.assertEqual(CV.group_by_source([f for f in files if not f.skip]), {})


class Comparison(unittest.TestCase):
    # compare_digests takes whether the OLD file carries a checkpoint (a spill sidecar's, or - for a .bin, which
    # this tool does not decode that far - always True): only a live-only old file compares tokens and state.
    def test_equal_digests_pass(self):
        self.assertEqual(CV.compare_digests(digest(), digest(), False), "")

    @unittest.skip("OWNER DECISION 2026-11-01: KV test bypassed for the cross-version reuse trial")
    def test_kv_always_counts(self):
        self.assertEqual(CV.compare_digests(digest(), digest(kv=9), True), "kv_mismatch")

    @unittest.skip("OWNER DECISION 2026-11-01: KV test bypassed for the cross-version reuse trial")
    def test_live_only_token_mismatch_fails(self):
        self.assertEqual(CV.compare_digests(digest(), digest(tokens=9), False), "tokens_mismatch")

    @unittest.skip("OWNER DECISION 2026-11-01: KV test bypassed for the cross-version reuse trial")
    def test_live_only_state_mismatch_fails(self):
        self.assertEqual(CV.compare_digests(digest(), digest(state=9), False), "state_mismatch")

    def test_rht_state_is_not_compared(self):
        self.assertEqual(CV.compare_digests(digest(), digest(state=9, exact=False), False), "")

    def test_a_checkpoint_bearing_file_compares_kv_only(self):
        # a fresh --session-out export holds no checkpoint, so its token/state digests are not comparable
        self.assertEqual(CV.compare_digests(digest(), digest(tokens=9, state=9), True), "")


class DigestParsing(unittest.TestCase):
    TEXT = ("version=2\ntokens=0000000000000001\nkv=0000000000000002\nstate=0000000000000003\n"
            "state_exact=1\nstate_ref=0\nstate_rht=0\n")

    def test_parses_the_documented_keys(self):
        parsed = CV.parse_digest(self.TEXT)
        self.assertEqual((parsed.version, parsed.tokens, parsed.kv, parsed.state), (2, 1, 2, 3))
        self.assertTrue(parsed.state_exact)
        self.assertFalse(parsed.state_rht)

    def test_missing_and_unknown_keys_are_refused(self):
        with self.assertRaises(CV.VerifyError) as caught:
            CV.parse_digest(self.TEXT.replace("state_rht=0\n", ""))
        self.assertEqual(caught.exception.reason, "digest_failed")
        with self.assertRaises(CV.VerifyError):
            CV.parse_digest(self.TEXT + "extra=1\n")


class FreshLog(unittest.TestCase):
    def test_identity_and_token_count(self):
        text = ("strata generate: --session-out: wrote 7175 tokens and 60 K/V layers (model 92015b27a6e3efb8, "
                "config d88978f614302157) to /tmp/x.sess, 237000000 bytes\n")
        identities, counts = CV.parse_fresh_log(text)
        self.assertEqual(identities, [CV.Identity(MODEL, NEW_CONFIG)])
        self.assertEqual(counts, [7175])

    def test_ambiguity_is_visible(self):
        text = ("(model 92015b27a6e3efb8, config d88978f614302157)\n"
                "(model 92015b27a6e3efb8, config 3074d88348b69b80)\n")
        identities, _ = CV.parse_fresh_log(text)
        self.assertEqual(len(set(identities)), 2)

    def test_two_identities_in_one_run_are_an_ambiguity_not_a_type_error(self):
        # the distinct identities are ordered by (model, config): an Identity has no order of its own
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fake_run(command, **_kwargs):
                Path(command[command.index("--session-out") + 1]).write_bytes(b"x" * 10)
                return mock.Mock(returncode=0, stderr="", stdout=(
                    f"(model {MODEL:016x}, config {NEW_CONFIG:016x})\n"
                    f"(model {MODEL:016x}, config {OLD_CONFIG:016x})\n--session-out: wrote 3 tokens\n"))
            with mock.patch.object(CV.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(CV.VerifyError) as caught:
                    CV.run_generate("/opt/engine/strata", [], [], (1, 2, 3), root / "x.ids", root / "x.fresh.sess",
                                    None, 60.0)
        self.assertEqual(caught.exception.reason, "identity_ambiguous")


class Ledger(TempDir):
    def batch(self, source=CV.Identity(MODEL, OLD_CONFIG), target=CV.Identity(MODEL, NEW_CONFIG), passed=True):
        return CV.GateBatch("b" + "a" * 16, source, target, "0.1.52.20", "strata-conv-1", passed,
                            1793037175538, "" if passed else "kv_mismatch")

    def test_roundtrip_write_and_read(self):
        entry = CV.GateEntry("strata-conv-1", "b" + "a" * 16, 249000000, 8170, 1793037175538)
        CV.write_ledger(self.root, [self.batch()], [entry])
        batches, entries = CV.read_ledger(self.root)
        self.assertEqual(batches, [self.batch()])
        self.assertEqual(entries, [entry])
        self.assertFalse((self.root / CV.GATE_TEMP).exists())
        self.assertEqual((self.root / CV.GATE_NAME).stat().st_mode & 0o777, 0o600)

    def test_empty_reason_is_written_as_dash_and_sanitised(self):
        failed = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "s", False, 5,
                              "route counts differ")
        text = CV.format_ledger([failed], []).decode()
        self.assertIn(" fail 5 route_counts_differ\n", text)
        passed = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "s", True, 5, "")
        self.assertIn(" pass 5 -\n", CV.format_ledger([passed], []).decode())

    def test_a_token_with_u0085_and_u2028_roundtrips(self):
        # std::getline splits on LF only: these characters are ordinary bytes inside a token, not line breaks
        batch = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "sample\x85name", True, 5,
                             "reason\u2028here")
        CV.write_ledger(self.root, [batch], [])
        self.assertEqual((self.root / CV.GATE_NAME).read_bytes().count(b"\n"), 3)
        self.assertEqual(CV.read_ledger(self.root), ([batch], []))

    def test_the_closing_newline_ends_the_line_and_is_not_one(self):
        body = "strata-cache-gate 1\n"
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        self.assertEqual(CV.parse_ledger(data), ([], []))

    def test_an_empty_line_before_the_checksum_is_refused(self):
        # std::getline yields that empty line, and the engine refuses it as an unknown line
        body = "strata-cache-gate 1\n\n"
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        with self.assertRaises(CV.LedgerError):
            CV.parse_ledger(data)

    def test_truncated_edited_and_unknown_ledgers_are_refused(self):
        CV.write_ledger(self.root, [self.batch()], [])
        path = self.root / CV.GATE_NAME
        original = path.read_bytes()
        for broken in (original[:-1], original.replace(b"pass", b"fail"), original.replace(b"strata-cache-gate 1",
                                                                                         b"strata-cache-gate 2"),
                       original + b"junk\n"):
            path.write_bytes(broken)
            with self.assertRaises(CV.LedgerError):
                CV.read_ledger(self.root)

    def test_a_malformed_field_is_refused(self):
        body = ("strata-cache-gate 1\n"
                "entry strata-conv-1 b1 12 NOTANUMBER 0\n")
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        with self.assertRaises(CV.LedgerError):
            CV.parse_ledger(data)

    def test_merge_replaces_only_the_processed_source(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        other = CV.GateBatch("bother", CV.Identity(7, 8), CV.Identity(7, 9), "build", "strata-conv-9", True, 2)
        entries = [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1),
                   CV.GateEntry("strata-conv-9", "bother", 20, 4, 2)]
        new_batch = self.batch()
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "pass", new_batch,
                                [CV.GateEntry("strata-conv-1", new_batch.id, 11, 5, 3)])
        batches, kept = CV.merge_ledger([old_batch, other], entries, [result])
        self.assertEqual(batches, [other, new_batch])
        self.assertEqual(kept, [CV.GateEntry("strata-conv-9", "bother", 20, 4, 2),
                                CV.GateEntry("strata-conv-1", new_batch.id, 11, 5, 3)])

    def test_a_failed_batch_leaves_no_entry_rows(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        failed = CV.GateBatch("bnew", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                              "0.1.52.20", "strata-conv-1", False, 2, "kv_mismatch")
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "fail", failed, [])
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [failed])
        self.assertEqual(entries, [])

    def test_a_failed_run_without_target_drops_old_rows(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "fail", None, [], "engine_failed")
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [])
        self.assertEqual(entries, [])

    def test_a_skip_leaves_everything_untouched(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "skip", None, [], "same identity")
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [old_batch])
        self.assertEqual(entries, [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)])


class CommandLine(TempDir):
    def setUp(self):
        super().setUp()
        self.engine = self.root / "engine"
        self.engine.write_text("#!/bin/sh\n")
        self.engine.chmod(0o755)
        self.digest_tool = self.root / "digest"
        self.digest_tool.write_text("#!/bin/sh\n")
        self.digest_tool.chmod(0o755)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.work = self.root / "work"
        patcher = mock.patch.object(CV, "resolve_executable", side_effect=lambda value, what: value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, argv, generate, digest_run):
        out = io.StringIO()
        with mock.patch.object(CV, "run_generate", side_effect=generate) as gen, \
                mock.patch.object(CV, "run_digest", side_effect=digest_run), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = CV.main(argv)
        return code, gen, out.getvalue()

    def base_args(self):
        return ["--cache-dir", str(self.cache), "--engine", str(self.engine),
                "--digest-tool", str(self.digest_tool), "--engine-build", "0.1.52.20",
                "--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG), "--work-dir", str(self.work)]

    def test_a_passing_sample_writes_a_batch_and_every_eligible_row(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 100)
        (self.cache / "strata-conv-a.meta").write_bytes(
            sidecar_bytes(ids=(1, 2, 3, 4, 5), checkpoints=(), last_used=11))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 50)
        (self.cache / "strata-conv-b.meta").write_bytes(
            sidecar_bytes(ids=(1, 2, 3), checkpoints=(), last_used=22))
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout, env=None):
            self.assertEqual(list(ids), [1, 2, 3])           # the smallest file is the sample
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), len(ids))
        def digest_run(tool, path, identity, timeout):
            return digest()
        code, gen, text = self.run_main(self.base_args(), generate, digest_run)
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 1)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].passed)
        self.assertEqual(batches[0].sample, "strata-conv-b")
        self.assertEqual(batches[0].source, CV.Identity(MODEL, OLD_CONFIG))
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))
        self.assertEqual({(e.stem, e.file_bytes, e.tokens) for e in entries},
                         {("strata-conv-a", 100, 5), ("strata-conv-b", 50, 3)})
        self.assertTrue(all(e.batch_id == batches[0].id for e in entries))

    def test_the_sample_is_a_complete_conversation_and_covers_the_prefix_entry_too(self):
        # the group of the 0.1.52.22 startup, replayed: prefix-2 is the smallest, so it was the sample and the
        # group failed; it must be the smallest COMPLETE base instead, and its pass still approves the prefix row
        for stem, size, ids in (("strata-conv-prefix-2", 40, (1, 2)),
                                ("strata-conv-1", 100, (1, 2, 3, 4, 5)),
                                ("strata-conv-3", 120, (1, 2, 3))):
            (self.cache / f"{stem}.sess").write_bytes(b"x" * size)
            (self.cache / f"{stem}.meta").write_bytes(sidecar_bytes(ids=ids, checkpoints=()))
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout, env=None):
            self.assertEqual(list(ids), [1, 2, 3, 4, 5])          # strata-conv-1, not the smaller prefix-2
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), len(ids))
        code, gen, text = self.run_main(self.base_args(), generate, lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 1)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual([(b.passed, b.sample) for b in batches], [(True, "strata-conv-1")])
        self.assertEqual({(e.stem, e.file_bytes, e.tokens) for e in entries},
                         {("strata-conv-prefix-2", 40, 2), ("strata-conv-1", 100, 5), ("strata-conv-3", 120, 3)})

    def test_a_group_of_only_prefix_roots_is_skipped_not_failed(self):
        # no complete conversation to recompute: the source is skipped with its reason, the engine is not started
        # and no failed batch is left behind (0.1.52.22 failed such a group on its prefix root)
        for stem, size, ids in (("strata-conv-prefix-2", 40, (1, 2)), ("strata-conv-prefix-5", 60, (1, 2, 3))):
            (self.cache / f"{stem}.sess").write_bytes(b"x" * size)
            (self.cache / f"{stem}.meta").write_bytes(sidecar_bytes(ids=ids, checkpoints=()))
        code, gen, text = self.run_main(self.base_args(), mock.Mock(), mock.Mock())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("skip", text)
        self.assertIn("only shared prefix roots", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())    # nothing failed, nothing approved

    @unittest.skip("OWNER DECISION 2026-11-01: KV test bypassed for the cross-version reuse trial")
    def test_a_kv_mismatch_writes_a_failed_batch_and_no_rows(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3)
        def digest_run(tool, path, identity, timeout):
            return digest() if path == self.cache / "strata-conv-a.sess" else digest(kv=99)
        code, _, text = self.run_main(self.base_args(), generate, digest_run)
        self.assertEqual(code, 1, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0].passed)
        self.assertEqual(batches[0].reason, "kv_mismatch")
        self.assertEqual(entries, [])

    def test_a_changed_target_identity_refuses_the_whole_run(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            return CV.FreshRun(CV.Identity(MODEL, OLD_CONFIG), 3)
        code, _, _ = self.run_main(self.base_args(), generate, lambda *a: digest())
        self.assertEqual(code, 3)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_a_foreign_model_is_skipped_without_running(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(model=MODEL ^ 1))
        code, gen, text = self.run_main(self.base_args(), mock.Mock(), lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("another model", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_build_is_probed_from_version_when_not_given(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--engine-build", "0.1.52.20")]
        with mock.patch.object(CV, "probe_engine_build", return_value="0.1.52.20") as probe:
            code, _, _ = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                       lambda *a: digest())
        self.assertEqual(code, 0)
        probe.assert_called_once()

    def test_the_engine_runs_in_the_serving_engines_environment_and_cwd(self):
        # R1 (docs/CACHE_REUSE_RULES.md): the server hands run() the environment and the working directory its own
        # engine starts with (serve/server.py: child_env, cfg["cwd"]).  An engine in another environment - another
        # card, other kernel switches, another library path - recomputes other K/V and would refuse a good file.
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        cwd = self.root / "engine-cwd"
        cwd.mkdir()
        child_env = dict(os.environ, STRATA_PF_WMMA="0", LD_LIBRARY_PATH="/opt/engine/lib")
        seen = {}
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, the_cwd, timeout, engine_env=None):
            seen["cwd"], seen["env"] = the_cwd, engine_env
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3)
        options = CV.build_parser().parse_args(self.base_args())
        with mock.patch.object(CV, "run_generate", side_effect=generate), \
                mock.patch.object(CV, "run_digest", side_effect=lambda *a: digest()):
            self.assertEqual(CV.run(options, env=child_env, cwd=cwd), 0)
        self.assertEqual(seen["cwd"], cwd)
        self.assertEqual(seen["env"]["STRATA_PF_WMMA"], "0")
        self.assertEqual(seen["env"]["LD_LIBRARY_PATH"], "/opt/engine/lib")

    def test_the_build_probe_runs_in_the_engines_environment(self):
        # the binary the server starts finds its libraries the way that engine does, not the way this process does
        child_env = {"LD_LIBRARY_PATH": "/opt/engine/lib"}
        with mock.patch.object(CV.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="strata 0.1.52.20\n", stderr="")
            self.assertEqual(CV.probe_engine_build("/opt/engine/strata", env=child_env), "0.1.52.20")
        self.assertEqual(run.call_args[1]["env"], child_env)

    def test_flag_shaped_engine_arguments_reach_the_engine_command_line(self):
        # serve/server.py passes the serving engine's own flags: "--engine-arg --kv" is two tokens and argparse
        # refuses a flag-shaped value ("expected one argument", SystemExit 2, which is no Exception).  The equals
        # form carries them, and they reach the engine's own command line unchanged and in order.
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = self.base_args() + ["--engine-arg=--conversation-cache-disk-only", "--engine-arg=--kv",
                                   "--engine-arg=--max-context", "--engine-arg=131072"]
        commands = []

        def fake_run(command, **_kwargs):
            Path(command[command.index("--session-out") + 1]).write_bytes(b"x" * 10)
            commands.append(list(command))
            return mock.Mock(returncode=0, stderr="",
                             stdout=f"(model {MODEL:016x}, config {NEW_CONFIG:016x})\n--session-out: wrote 3 tokens\n")
        with mock.patch.object(CV.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(CV, "run_digest", side_effect=lambda *a: digest()), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = CV.main(argv)
        self.assertEqual(code, 0)
        work = Path(self.work).resolve()
        self.assertEqual(commands, [[str(self.engine), "--conversation-cache-disk-only", "--kv", "--max-context",
                                     "131072", "--tokens-file", str(work / "strata-conv-a.ids"), "--max-new", "1",
                                     "--greedy", "--session-out", str(work / "strata-conv-a.fresh.sess")]])

    def test_serve_in_the_passthrough_is_a_usage_error(self):
        code = CV.main(self.base_args() + ["--", "--serve"])
        self.assertEqual(code, 2)

    def test_the_verifier_owns_its_engine_arguments(self):
        for owned in ("--session-out", "--tokens-file", "--tokens", "--max-new", "--serve"):
            with self.assertRaises(CV.UsageError):
                CV.check_engine_args(["--pack", "p", owned])

    def test_the_serve_refusal_does_not_claim_the_verifier_sets_it(self):
        for given in ("--serve", "--serve=8080"):
            with self.assertRaises(CV.UsageError) as caught:
                CV.check_engine_args(["--pack", "p", given])
            self.assertIn("--serve is not allowed", str(caught.exception))
            self.assertNotIn("sets --serve", str(caught.exception))

    def test_a_malformed_ledger_starts_empty_and_unrelated_records_survive(self):
        (self.cache / CV.GATE_NAME).write_bytes(b"strata-cache-gate 1\ngarbage\n")
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        code, _, text = self.run_main(self.base_args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                      lambda *a: digest())
        self.assertEqual(code, 0, text)
        batches, _ = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].passed)

    def test_an_existing_unrelated_batch_is_preserved(self):
        other = CV.GateBatch("bother", CV.Identity(7, 8), CV.Identity(7, 9), "old", "strata-conv-z", True, 1)
        stale = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG), "old",
                             "strata-conv-a", False, 2, "kv_mismatch")
        CV.write_ledger(self.cache, [other, stale], [CV.GateEntry("strata-conv-z", "bother", 5, 1, 1)])
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        code, _, _ = self.run_main(self.base_args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                   lambda *a: digest())
        self.assertEqual(code, 0)
        batches, entries = CV.read_ledger(self.cache)
        self.assertIn(other, batches)
        self.assertNotIn(stale, batches)
        self.assertIn(CV.GateEntry("strata-conv-z", "bother", 5, 1, 1), entries)
        self.assertEqual(len(entries), 2)

    def test_dry_run_starts_nothing_and_writes_nothing(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--digest-tool", str(self.digest_tool))]
        code, gen, text = self.run_main(argv + ["--dry-run"], mock.Mock(), mock.Mock())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("plan", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_target_is_learned_from_the_fresh_export(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        code, _, text = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                      lambda *a: digest())
        self.assertEqual(code, 0, text)
        batches, _ = CV.read_ledger(self.cache)
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))

    def test_a_learned_target_of_another_model_fails_each_group(self):
        # a foreign model must not be pinned as the target: the later groups would then be skipped as
        # "another model" and the run would end 0 with a ledger that approves nothing
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG, checkpoints=()))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 60)
        (self.cache / "strata-conv-b.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG ^ 1, checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        code, gen, text = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL ^ 1, NEW_CONFIG), 3),
                                        lambda *a: digest())
        self.assertEqual(code, 1, text)
        self.assertEqual(gen.call_count, 2)              # the second group was verified, not skipped
        self.assertIn("the fresh export is model", text)
        self.assertEqual(CV.read_ledger(self.cache), ([], []))

    def test_two_different_identities_refuse_the_run(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG, checkpoints=()))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 60)
        (self.cache / "strata-conv-b.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG ^ 1, checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        answers = [CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                   CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG ^ 1), 3)]
        code, _, _ = self.run_main(argv, lambda *a: answers.pop(0), lambda *a: digest())
        self.assertEqual(code, 3)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_an_engine_failure_is_a_failed_batch_with_no_rows(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            raise CV.VerifyError("engine_failed", "exit 1: no model")
        code, _, text = self.run_main(self.base_args(), generate, lambda *a: digest())
        self.assertEqual(code, 1, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0].passed)
        self.assertEqual(batches[0].reason, "engine_failed")
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))
        self.assertEqual(entries, [])

    def test_a_work_dir_inside_the_cache_is_refused(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--work-dir", str(self.work))]
        code = CV.main(argv + ["--work-dir", str(self.cache / "work")])
        self.assertEqual(code, 2)

    def test_target_model_without_config_is_a_usage_error(self):
        self.assertEqual(CV.main(["--cache-dir", str(self.cache), "--engine", str(self.engine),
                                  "--target-model", "1"]), 2)

    def test_a_config_json_supplies_the_args_and_never_prints_them(self):
        config = self.root / "cfg.json"
        config.write_text(json.dumps({"args": ["--pack", "/p", "--api-key", "SECRET"], "cwd": str(self.root)}))
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        seen = {}
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout, env=None):
            seen["args"] = config_args
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3)
        code, _, text = self.run_main(self.base_args() + ["--engine-config", str(config)], generate,
                                      lambda *a: digest())
        self.assertEqual(code, 0)
        self.assertEqual(seen["args"], ["--pack", "/p", "--api-key", "SECRET"])
        self.assertNotIn("SECRET", text)


def session_bytes(version=2, model=MODEL, config=OLD_CONFIG, ids=(11, 12, 13), layers=0,
                  part_sizes=(4, 0, 0, 0, 0)):
    """A session file a slot directory would hold (conversation_file.cpp): the 64-byte header with its checksum, the
    geometry prefix, for v3 the K/V directory, then the live checkpoint's token ids.  The payload past the ids is
    never read by this tool, so it is left empty and the header is sized to match."""
    geo = struct.pack("<18q", *[100 + i for i in range(18)]) + struct.pack("<qq", 0, 48) + struct.pack("<Q", 0)
    kv = b""
    if version == 3:
        kv += struct.pack("<Q", layers)
        for _ in range(layers):
            kv += struct.pack("<7q", 2, 1024, 2, 256, 64, 9, 128)
            for size in part_sizes:
                kv += struct.pack("<Q", size) + bytes(size)
    body = struct.pack("<Q", len(ids)) + struct.pack(f"<{len(ids)}i", *ids)
    payload = geo + kv + body
    header = bytearray(64)
    header[0:8] = CV.SESSION_MAGIC
    struct.pack_into("<II", header, 8, version, CV.SESSION_HEADER)
    struct.pack_into("<QQQ", header, 16, model, config, len(payload))
    struct.pack_into("<QQ", header, 40, 0, 0)
    struct.pack_into("<Q", header, 56, CV.session_hash64(bytes(header[:56]), 0))
    return bytes(header) + payload + b"\x00" * CV.SESSION_TRAILER


class AssumePass(TempDir):
    """--assume-pass: the owner's decision to reuse the cache now, without the KV test (docs/CACHE_REUSE_RULES.md).

    Every same-model source is approved - one batch whose reason says assume_pass, one entry row per eligible file -
    with no engine started and no digest compared; a foreign model is still never approved.
    """

    BUILD = "0.1.52.23"

    def setUp(self):
        super().setUp()
        self.engine = self.root / "engine"
        self.engine.write_text("#!/bin/sh\n")
        self.engine.chmod(0o755)
        self.cache = self.root / "cache"
        self.cache.mkdir()

    def base_args(self, target=True):
        argv = ["--cache-dir", str(self.cache), "--engine", str(self.engine), "--engine-build", self.BUILD,
                "--assume-pass"]
        if target:
            argv += ["--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG)]
        return argv

    def run_main(self, argv):
        out = io.StringIO()
        with mock.patch.object(CV, "run_generate", side_effect=AssertionError("no engine may be started")) as gen, \
                mock.patch.object(CV, "run_digest", side_effect=AssertionError("no digest may be compared")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = CV.main(argv)
        return code, gen, out.getvalue()

    def write_group(self, config=OLD_CONFIG, model=MODEL):
        for stem, size, ids in (("strata-conv-prefix-2", 40, (1, 2)), ("strata-conv-1", 100, (1, 2, 3, 4, 5)),
                                ("strata-conv-3", 120, (1, 2, 3))):
            (self.cache / f"{stem}.sess").write_bytes(b"x" * size)
            (self.cache / f"{stem}.meta").write_bytes(
                sidecar_bytes(model=model, config=config, ids=ids, checkpoints=()))

    def test_every_eligible_file_of_a_source_is_approved_without_an_engine(self):
        self.write_group()
        code, gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertTrue(batch.passed)
        self.assertEqual(batch.reason, "assume_pass")
        self.assertEqual(batch.sample, "-")                 # nothing was recomputed, so no file was the sample
        self.assertEqual(batch.engine_build, self.BUILD)
        self.assertEqual((batch.source, batch.target),
                         (CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG)))
        self.assertEqual({(e.stem, e.file_bytes, e.tokens) for e in entries},
                         {("strata-conv-prefix-2", 40, 2), ("strata-conv-1", 100, 5), ("strata-conv-3", 120, 3)})
        self.assertTrue(all(e.batch_id == batch.id for e in entries))
        self.assertIn("ASSUMPTION", text)                   # the run says what it did, in words

    def test_the_batch_row_itself_says_assume_pass(self):
        # the ledger is the record: a reader of the file, not only of this tool, must see the row was not measured
        self.write_group()
        code, _gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        lines = (self.cache / CV.GATE_NAME).read_text(encoding="utf-8").splitlines()
        fields = next(line for line in lines if line.startswith("batch ")).split(" ")
        self.assertEqual((fields[0], fields[7], fields[8], fields[10]), ("batch", "-", "pass", "assume_pass"))
        self.assertEqual(CV.parse_ledger((self.cache / CV.GATE_NAME).read_bytes())[0][0].reason, "assume_pass")

    def test_the_next_start_has_nothing_pending(self):
        self.write_group()
        code, _gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        pending = CV.pending_verification(self.cache, [], self.BUILD)
        self.assertFalse(pending.verify, pending.detail)

    def test_a_foreign_model_is_never_approved(self):
        self.write_group()                                  # this model
        (self.cache / "strata-conv-theirs.sess").write_bytes(b"y" * 90)
        (self.cache / "strata-conv-theirs.meta").write_bytes(
            sidecar_bytes(model=MODEL ^ 1, config=OLD_CONFIG, checkpoints=()))
        code, _gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual([b.source.model for b in batches], [MODEL])
        self.assertNotIn("strata-conv-theirs", {e.stem for e in entries})
        self.assertIn("a foreign model is never approved", text)

    def test_a_foreign_model_alone_is_not_approved_and_writes_nothing(self):
        (self.cache / "strata-conv-theirs.sess").write_bytes(b"y" * 90)
        (self.cache / "strata-conv-theirs.meta").write_bytes(sidecar_bytes(model=MODEL ^ 1, checkpoints=()))
        code, _gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_a_slot_directory_is_approved_on_the_same_terms(self):
        slots = self.root / "slots"
        slots.mkdir()
        data = session_bytes(config=OLD_CONFIG, ids=(11, 12, 13))
        (slots / "chat-a.bin").write_bytes(data)
        code, _gen, text = self.run_main(self.base_args() + ["--slot-dir", str(slots)])
        self.assertEqual(code, 0, text)
        _batches, entries = CV.read_ledger(self.cache)
        self.assertEqual({(e.stem, e.file_bytes, e.tokens) for e in entries}, {("chat-a.bin", len(data), 3)})

    def test_an_unrelated_batch_and_its_entries_are_kept(self):
        kept = CV.GateBatch("b-kept", CV.Identity(MODEL, NEW_CONFIG), CV.Identity(MODEL, NEW_CONFIG), "0.1.52.19",
                            "old", True, 5)
        CV.write_ledger(self.cache, [kept], [CV.GateEntry("old-stem", "b-kept", 10, 2, 0)])
        self.write_group()
        code, _gen, text = self.run_main(self.base_args())
        self.assertEqual(code, 0, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual({b.id for b in batches}, {kept.id, CV.batch_id(CV.Identity(MODEL, OLD_CONFIG))})
        self.assertIn("old-stem", {e.stem for e in entries})

    def test_a_dry_run_only_reports(self):
        self.write_group()
        code, gen, text = self.run_main(self.base_args() + ["--dry-run"])
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("assume pass", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_identity_the_ledger_recorded_for_this_build_is_used(self):
        # no --target-model/--target-config: the identity can only be the one the ledger already names for this build
        source = CV.Identity(MODEL, OLD_CONFIG)
        sign = CV.GateBatch(CV.batch_id(source), source, CV.Identity(MODEL, NEW_CONFIG), self.BUILD, "x", False, 1,
                            "kv_mismatch")
        CV.write_ledger(self.cache, [sign], [])
        self.write_group()
        code, _gen, text = self.run_main(self.base_args(target=False))
        self.assertEqual(code, 0, text)
        batches, _entries = CV.read_ledger(self.cache)
        self.assertEqual({b.target for b in batches}, {CV.Identity(MODEL, NEW_CONFIG)})
        self.assertTrue(all(b.reason == "assume_pass" for b in batches if b.source != b.target))

    def test_no_identity_anywhere_is_a_usage_error_and_writes_nothing(self):
        self.write_group()
        code, gen, _text = self.run_main(self.base_args(target=False))
        self.assertEqual(code, 2)
        self.assertEqual(gen.call_count, 0)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_build_must_come_from_the_command_line(self):
        # no engine is started, so none can report its build
        self.write_group()
        argv = [a for a in self.base_args() if a not in ("--engine-build", self.BUILD)]
        code, gen, _text = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertEqual(gen.call_count, 0)

    def test_assume_pass_is_off_by_default(self):
        self.assertFalse(CV.build_parser().parse_args(["--cache-dir", "c", "--engine", "e"]).assume_pass)


class SlotScan(TempDir):
    """A --slot-dir: a .bin has no sidecar, so its identity and token count come from its own header."""

    def write(self, name, **kwargs):
        path = self.root / name
        path.write_bytes(session_bytes(**kwargs))
        return path

    def test_identity_and_ids_come_from_the_header(self):
        path = self.write("chat-1.bin", ids=(11, 12, 13))
        files = CV.scan_slot_dir(self.root)
        self.assertEqual([f.stem for f in files], ["chat-1.bin"])
        self.assertEqual(files[0].skip, "")
        self.assertEqual(files[0].source, CV.Identity(MODEL, OLD_CONFIG))
        self.assertEqual(files[0].ids, (11, 12, 13))
        self.assertEqual(files[0].bytes, path.stat().st_size)
        self.assertEqual(files[0].path, path)
        self.assertEqual(files[0].last_used, 0)
        # a .bin usually holds the deepest checkpoint, and the count is not in the header: K/V only
        self.assertTrue(files[0].has_checkpoints)

    def test_a_v3_file_reads_its_ids_after_the_kv_directory(self):
        self.write("chat-v3.bin", version=3, ids=(21, 22), layers=2, part_sizes=(4, 5, 6, 0, 0))
        files = CV.scan_slot_dir(self.root)
        self.assertEqual(files[0].skip, "", files[0].skip)
        self.assertEqual(files[0].source, CV.Identity(MODEL, OLD_CONFIG))
        self.assertEqual(files[0].ids, (21, 22))

    def test_a_file_that_is_not_a_session_file_is_skipped(self):
        (self.root / "notes.bin").write_bytes(b"not a session file, just a name that ends in .bin" * 4)
        (self.root / "readme.txt").write_text("not scanned at all")
        files = CV.scan_slot_dir(self.root)
        self.assertEqual([f.stem for f in files], ["notes.bin"])
        self.assertIn("magic", files[0].skip)

    def test_a_bad_header_or_size_is_skipped(self):
        good = bytearray(session_bytes())
        corrupt = bytearray(good)
        corrupt[20] ^= 0xFF                       # the model fingerprint, under the header's checksum
        (self.root / "corrupt.bin").write_bytes(bytes(corrupt))
        (self.root / "short.bin").write_bytes(bytes(good[:40]))
        grown = session_bytes()
        (self.root / "grown.bin").write_bytes(grown + b"x")   # the header no longer sizes the file
        skipped = {f.stem: f.skip for f in CV.scan_slot_dir(self.root)}
        self.assertIn("checksum", skipped["corrupt.bin"])
        self.assertIn("smaller than its header", skipped["short.bin"])
        self.assertIn("does not match its header", skipped["grown.bin"])

    def test_a_name_that_cannot_go_in_the_ledger_is_skipped(self):
        (self.root / "chat 1.bin").write_bytes(session_bytes())
        files = CV.scan_slot_dir(self.root)
        self.assertIn("whitespace", files[0].skip)

    def test_slot_and_spill_files_group_together(self):
        self.write("chat-1.bin", ids=(11, 12))
        (self.root / "strata-conv-1.sess").write_bytes(b"x" * 20)
        (self.root / "strata-conv-1.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG, ids=(1, 2), checkpoints=()))
        everything = CV.scan_cache_dir(self.root) + CV.scan_slot_dir(self.root)
        groups = CV.group_by_source([f for f in everything if not f.skip])
        self.assertEqual({f.stem for f in groups[CV.Identity(MODEL, OLD_CONFIG)]},
                         {"chat-1.bin", "strata-conv-1"})


class Pending(TempDir):
    """pending_verification: does a start have to run the verification, and for which sources?"""

    BUILD = "0.1.52.20"

    def slots(self):
        directory = self.root / "slots"
        directory.mkdir(exist_ok=True)
        return directory

    def write_slot(self, name, config=OLD_CONFIG, model=MODEL):
        (self.slots() / name).write_bytes(session_bytes(model=model, config=config))

    def identity_batch(self, target=CV.Identity(MODEL, NEW_CONFIG), build=None):
        return CV.GateBatch(CV.batch_id(target), target, target, build or self.BUILD, "-", True,
                            int(time.time() * 1000))

    def test_nothing_at_all_runs_nothing(self):
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertFalse(pending.verify)
        self.assertEqual(pending.sources, ())

    def test_a_v3_file_of_another_config_is_pending_too(self):
        (self.root / "strata-conv-1.sess").write_bytes(b"x" * 20)
        (self.root / "strata-conv-1.meta").write_bytes(sidecar_bytes(version=3, origin=1, checkpoints=()))
        pending = CV.pending_verification(self.root, [], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertEqual(pending.sources, ((CV.Identity(MODEL, OLD_CONFIG), 1),))

    def test_a_cache_directory_that_cannot_be_read_is_a_refusal_not_a_pass(self):
        pending = CV.pending_verification(self.root / "gone", [], self.BUILD)
        self.assertFalse(pending.verify)
        self.assertTrue(pending.refused)                   # the start says so; silence would read as "checked"
        self.assertIn("cannot read the cache directory", pending.detail)

    def test_a_file_of_another_config_is_pending_until_the_ledger_names_this_build(self):
        self.write_slot("chat-1.bin")
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertIsNone(pending.target)
        self.assertEqual(pending.sources, ((CV.Identity(MODEL, OLD_CONFIG), 1),))

    def test_a_ledger_of_another_build_proves_nothing(self):
        self.write_slot("chat-1.bin")
        CV.write_ledger(self.root, [self.identity_batch(build="0.1.51")], [])
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertIsNone(pending.target)

    def test_the_identity_batch_alone_leaves_the_other_config_pending(self):
        self.write_slot("chat-1.bin")
        CV.write_ledger(self.root, [self.identity_batch()], [])
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertEqual(pending.target, CV.Identity(MODEL, NEW_CONFIG))

    def test_a_passed_batch_settles_its_source(self):
        self.write_slot("chat-1.bin")
        batch = CV.GateBatch("b1", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG), self.BUILD,
                             "chat-1.bin", True, int(time.time() * 1000) + 1000)
        CV.write_ledger(self.root, [batch], [CV.GateEntry("chat-1.bin", "b1", 1, 3, 0)])
        self.assertFalse(CV.pending_verification(self.root, [self.slots()], self.BUILD).verify)

    def test_a_failed_batch_settles_it_too_and_a_changed_file_does_not(self):
        path = self.slots() / "chat-1.bin"
        self.write_slot("chat-1.bin")
        failed = CV.GateBatch("b1", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG), self.BUILD,
                              "chat-1.bin", False, int(time.time() * 1000) + 1000, "kv_mismatch")
        CV.write_ledger(self.root, [failed], [])
        self.assertFalse(CV.pending_verification(self.root, [self.slots()], self.BUILD).verify)
        # a file rewritten after the run is not the file that was checked: the group is verified again
        os.utime(path, (time.time() + 10, time.time() + 10))
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertEqual(pending.sources, ((CV.Identity(MODEL, OLD_CONFIG), 1),))

    def test_a_foreign_model_is_never_pending(self):
        self.write_slot("chat-1.bin", model=MODEL ^ 1)
        CV.write_ledger(self.root, [self.identity_batch()], [])
        self.assertFalse(CV.pending_verification(self.root, [self.slots()], self.BUILD).verify)

    def test_this_identitys_own_files_need_no_gate(self):
        self.write_slot("chat-1.bin", config=NEW_CONFIG)
        CV.write_ledger(self.root, [self.identity_batch()], [])
        self.assertFalse(CV.pending_verification(self.root, [self.slots()], self.BUILD).verify)

    def test_two_source_identities_are_reported_in_model_config_order(self):
        # two sources in one group map: sorting the pairs themselves raised TypeError (0.1.52.22, the gate never ran)
        self.write_slot("chat-a.bin", config=NEW_CONFIG)      # scanned first, but the larger config
        self.write_slot("chat-z.bin", config=OLD_CONFIG)
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertIsNone(pending.target)
        self.assertEqual(pending.sources, ((CV.Identity(MODEL, OLD_CONFIG), 1), (CV.Identity(MODEL, NEW_CONFIG), 1)))

    def test_the_sources_of_a_known_target_are_in_model_config_order(self):
        target = CV.Identity(MODEL, NEW_CONFIG ^ 1)
        CV.write_ledger(self.root, [self.identity_batch(target=target)], [])
        self.write_slot("chat-a.bin", config=NEW_CONFIG)
        self.write_slot("chat-z.bin", config=OLD_CONFIG)
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)
        self.assertEqual(pending.target, target)
        self.assertEqual(pending.sources, ((CV.Identity(MODEL, OLD_CONFIG), 1), (CV.Identity(MODEL, NEW_CONFIG), 1)))

    def test_a_malformed_ledger_is_not_an_error(self):
        self.write_slot("chat-1.bin")
        (self.root / CV.GATE_NAME).write_bytes(b"strata-cache-gate 1\ngarbage\n")
        pending = CV.pending_verification(self.root, [self.slots()], self.BUILD)
        self.assertTrue(pending.verify)          # approves nothing, so the run starts a fresh ledger


class IdentityBatch(unittest.TestCase):
    def test_it_replaces_older_ones_and_leaves_the_others(self):
        target = CV.Identity(MODEL, NEW_CONFIG)
        old = CV.GateBatch("bold", CV.Identity(7, 8), CV.Identity(7, 8), "0.1.51", "-", True, 1)
        foreign = CV.GateBatch("bf", CV.Identity(MODEL, OLD_CONFIG), target, "0.1.52.20", "s", True, 2)
        batches = CV.with_identity_batch([old, foreign], target, "0.1.52.20", "chat-1.bin")
        self.assertEqual({b.id for b in batches}, {"bf", CV.batch_id(target)})
        identity = [b for b in batches if b.source == b.target]
        self.assertEqual(len(identity), 1)
        self.assertEqual((identity[0].target, identity[0].engine_build, identity[0].sample),
                         (target, "0.1.52.20", "chat-1.bin"))
        self.assertTrue(identity[0].passed)
        self.assertEqual(CV.current_identity(batches, "0.1.52.20"), target)
        self.assertIsNone(CV.current_identity([old], "0.1.52.20"))


class SlotCommandLine(TempDir):
    """The tool against a slot directory: the same ledger, rows keyed by the .bin's own name."""

    def setUp(self):
        super().setUp()
        self.engine = self.root / "engine"
        self.engine.write_text("#!/bin/sh\n")
        self.engine.chmod(0o755)
        self.digest_tool = self.root / "digest"
        self.digest_tool.write_text("#!/bin/sh\n")
        self.digest_tool.chmod(0o755)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.slots = self.root / "slots"
        self.slots.mkdir()
        self.work = self.root / "work"
        patcher = mock.patch.object(CV, "resolve_executable", side_effect=lambda value, what: value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *extra):
        return ["--cache-dir", str(self.cache), "--slot-dir", str(self.slots), "--engine", str(self.engine),
                "--digest-tool", str(self.digest_tool), "--engine-build", "0.1.52.20", "--work-dir", str(self.work),
                *extra]

    def run_main(self, argv, generate, digest_run):
        out = io.StringIO()
        with mock.patch.object(CV, "run_generate", side_effect=generate) as gen, \
                mock.patch.object(CV, "run_digest", side_effect=digest_run), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = CV.main(argv)
        return code, gen, out.getvalue()

    def test_a_slot_file_is_sampled_and_written_under_its_own_name(self):
        path = self.slots / "chat-1.bin"
        path.write_bytes(session_bytes(ids=(11, 12, 13)))
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout, env=None):
            self.assertEqual(list(ids), [11, 12, 13])
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), len(ids))
        code, gen, text = self.run_main(self.args(), generate, lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 1)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)                     # a learned identity that verified needs no marker
        self.assertEqual(batches[0].source, CV.Identity(MODEL, OLD_CONFIG))
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))
        self.assertEqual(batches[0].sample, "chat-1.bin")
        self.assertEqual([(e.stem, e.file_bytes, e.tokens) for e in entries],
                         [("chat-1.bin", path.stat().st_size, 3)])

    def test_the_stem_is_the_whole_file_name_extension_included(self):
        path = self.slots / "strata-conv-1.bin"               # the same base name a spill entry would have
        path.write_bytes(session_bytes(ids=(11, 12)))
        code, _, text = self.run_main(self.args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 2),
                                      lambda *a: digest())
        self.assertEqual(code, 0, text)
        _, entries = CV.read_ledger(self.cache)
        self.assertEqual([e.stem for e in entries], ["strata-conv-1.bin"])

    def test_a_run_that_learned_the_identity_and_verified_nothing_records_it(self):
        # every file is this engine's own: no batch is written, so the ledger has to record the identity instead,
        # or the next start would load the engine again to learn it
        (self.slots / "chat-1.bin").write_bytes(session_bytes(ids=(11, 12), config=NEW_CONFIG))
        code, gen, text = self.run_main(self.args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 2),
                                        lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 1)                    # the identity is learned from one export
        batches, _ = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0].source, batches[0].target), (CV.Identity(MODEL, NEW_CONFIG),
                                                                 CV.Identity(MODEL, NEW_CONFIG)))
        self.assertEqual(CV.current_identity(batches, "0.1.52.20"), CV.Identity(MODEL, NEW_CONFIG))
        # and with that record a start sees there is nothing to verify, so it starts no engine at all
        self.assertFalse(CV.pending_verification(self.cache, [self.slots], "0.1.52.20").verify)

    def test_a_slot_dir_that_is_the_cache_dir_is_a_usage_error(self):
        argv = ["--cache-dir", str(self.cache), "--slot-dir", str(self.cache), "--engine", str(self.engine)]
        self.assertEqual(CV.main(argv), 2)

    def test_work_dir_inside_a_slot_dir_is_a_usage_error(self):
        argv = self.args()
        argv[argv.index("--work-dir") + 1] = str(self.slots / "work")
        self.assertEqual(CV.main(argv), 2)

    def test_dry_run_starts_nothing_and_writes_nothing(self):
        (self.slots / "chat-1.bin").write_bytes(session_bytes(ids=(11, 12)))
        code, gen, text = self.run_main(self.args("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG),
                                                  "--dry-run"), mock.Mock(), lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("sample=chat-1.bin", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())


if __name__ == "__main__":
    unittest.main()

