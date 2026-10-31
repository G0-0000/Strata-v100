#!/usr/bin/env python3
"""Verifier for the opt-in disk-cache recompute gate (.strata-cache-gate).

A spill directory keeps a session file whose sidecar names ANOTHER engine configuration, but never indexes it.
With the gate on (ConversationSpillGateOptions in include/strata/core/conversation_spill.hpp), the engine
indexes such a file only when this directory's ledger names it and its batch passed for exactly
(source model+config -> this model+config+this engine build).  This tool writes that ledger.

A SESSION FILE kept by the slot API (--slot-save-path, a .bin the client wrote) is the same case on the same
terms: one ledger, one read rule, one verification, only the directory and the name differ (R3 of
docs/CACHE_REUSE_RULES.md).  A .bin has no sidecar: its identity and token count come from its own header, and
its ledger stem is its whole file name, extension included, while a spill stem carries none: the two names differ
unless a client saves a conversation under a name without an extension (the slot API writes the name as given),
and even then a row approves only a file whose bytes, live token count and source identity all match.

  1. scan --cache-dir for <stem>.sess + <stem>.meta pairs (decode the sidecar, v1/v2/v3) and every --slot-dir for
     STRSESS .bin files (the 64-byte header, then the geometry prefix and the live token ids - for a K/V-first v3
     file those follow its K/V directory), and group the files by source (model, config); a foreign MODEL is never
     approved, a same-identity file needs no gate;
  2. for each group pick the smallest eligible file (no images, no delta chain, decodable token ids) and recompute
     exactly its prompt with the CURRENT engine:
         <engine> [config args] --tokens-file IDS --max-new 1 --greedy --session-out FRESH
  3. compare the format-independent digests (strata-session-digest) of the old file and the fresh export:
     K/V always; the token digest (and the state digest) only when the old file is live-only (no checkpoint),
     because a fresh export is live-only and its digests then cover exactly the same thing; an RHT-coded state is
     lossy and is never compared;
  4. on a pass, write a batch line plus an entry row for every eligible file of that source (exact stem, bytes and
     live token count: the engine rechecks all three); on a fail or an unreadable run, write the batch (when its
     target identity is known) and no rows; keep every unrelated valid batch and entry of an existing ledger; a
     malformed ledger starts empty (fail closed).  A run that learns this engine's identity also leaves ONE
     identity batch (source == target, no rows) so the next start sees that this build verified and needs no
     second run (R4).

--assume-pass is the OWNER'S assume pass, not a verification: at the owner's request (2026-10-31, see
docs/CACHE_REUSE_RULES.md) every group of this model is approved for the identity given on the command line or
recorded in the ledger for this build - no engine is started and no digest is compared - and each batch says so in
its reason (assume_pass) and in a "-" sample, so a row it wrote is never read as one a recompute passed.  A foreign
MODEL is still never approved, --engine-build is required (nothing else can name the build), and --dry-run still
only reports.

The ledger is written atomically (temporary beside it, fsync, rename).  Cache and slot files are only ever read:
this tool never deletes, renames or quarantines them.

The server runs this tool itself before it starts the engine (R4: the program verifies, not an operator; see
pending_verification() and serve/server.py).  It hands the verification engine the environment and the working
directory the serving engine itself starts with (run(env=..., cwd=...)); R1: an engine started in another
environment - another GPU, other kernel switches, other library paths - recomputes other K/V and would refuse a
file that is good.  Run it by hand only while the server is stopped - it starts a second engine against the same
GPU - or with --dry-run, which starts nothing.

  python3 tools/cache_verify.py --cache-dir DIR --engine PATH --digest-tool PATH \
      --engine-config CONFIG.json --target-model 0x... --target-config 0x... [--engine-build BUILD] \
      [--engine-arg ...] [-- ARGS...]

  --engine-config  a config JSON with an "args" array (and an optional "cwd"); its contents are never printed
  --engine-arg     one extra engine argument; arguments after -- are passed through as well
  --dry-run        scan and report the planned batches; starts no engine and writes nothing
  --assume-pass    the owner's assume pass: approve the same-model groups without the KV test; needs --engine-build
  --keep-work      keep the fresh session and token files in --work-dir (default: a temporary folder, removed)

Exit: 0 = every batch passed (or nothing to verify); 1 = a batch failed; 2 = usage error; 3 = refused (missing
tool, ambiguous or mismatching target identity, unreadable cache directory, ledger not written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

GATE_NAME = ".strata-cache-gate"
GATE_TEMP = ".strata-cache-gate.tmp"
GATE_VERSION = 1
MAX_GATE_BYTES = 4 << 20            # the engine's kMaxGateLedgerBytes
MAX_SIDECAR_BYTES = 64 << 20        # far above a 128k-token conversation; a corrupt header cannot make us allocate
MAX_SIDECAR_CHECKPOINTS = 256       # the engine's kMaxSidecarEntries
SIDECAR_MAGIC = b"SCSM"
# The session file (conversation_file.cpp): a 64-byte header, the payload the header sizes, and a 16-byte trailer.
# A slot .bin is read here only far enough for its identity and its live token ids.
SESSION_MAGIC = b"STRSESS\x01"
SESSION_HEADER = 64
SESSION_TRAILER = 16
SESSION_VERSIONS = (1, 2, 3)
# The sidecar's versions (conversation_spill.cpp): 1 has the ids and the checkpoint chain, 2 adds the persisted
# last-use stamp (kMetaVersionStamped), 3 adds the conversation's origin (kMetaVersionOrigin, written when
# --conversation-cache-origin-split is on and the conversation is tagged).  A version 3 file of ANOTHER
# configuration is pending exactly as a v1/v2 one is: the engine holds it "not indexed until the recompute gate
# approves" it, so refusing to read v3 here would leave it unverifiable for ever (R1, docs/CACHE_REUSE_RULES.md).
SIDECAR_VERSION_MIN, SIDECAR_VERSION_MAX = 1, 3
# The spill stems (conversation_spill.cpp, next_stem): an ordinary entry is strata-conv-<n>, and a SHARED ROOT PREFIX
# is strata-conv-prefix-<n>.  The sidecar does not carry that flag - the engine reads it off the name (Entry::prefix)
# - so the name is the only record of it.  A prefix root holds a prompt's shared root, not a conversation; a digest
# comparison of one against a fresh full-prompt export says nothing about it.
PREFIX_STEM = "strata-conv-prefix-"
# The owner's ASSUME PASS: approve a same-model file of another configuration without the KV test (--assume-pass).
# The owner decided the reuse is wanted now (2026-10-31, on top of R1/R2 of docs/CACHE_REUSE_RULES.md); the ledger
# says so in the batch's reason token, so a row written by assumption is never read as one a recompute passed.
ASSUME_PASS_REASON = "assume_pass"
ORIGIN_MAX = 2                       # ConversationOrigin::disk_restore: the highest origin the engine writes
INT32_MAX = (1 << 31) - 1
U64_MASK = (1 << 64) - 1
MAX_SESSION_TOKENS = INT32_MAX      # the engine's token ids are int32
MAX_SESSION_KV_LAYERS = 4096        # a corrupt count cannot make the walk run long

# Arguments the verification run owns; a config or passthrough copy would fight them.
OWNS_ARGS = ("--serve", "--tokens-file", "--tokens", "--max-new", "--session-out")


class UsageError(Exception):
    """Bad command line: exit 2."""


class Refused(Exception):
    """The run cannot be made safe or meaningful: exit 3, nothing written."""


class SidecarError(Exception):
    """A sidecar that cannot be decoded: the file is skipped."""


class LedgerError(Exception):
    """A malformed ledger: the caller starts empty (fail closed)."""


class VerifyError(Exception):
    """One batch failed; .reason is the ledger's single-token reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------------------------------------------------
# SessionHasher: the exact non-cryptographic 64-bit hash conversation_file.cpp uses for the ledger checksum, the
# sidecar checksum and the session file itself.  Ported function for function; the test vectors in
# tests/terminal/test_cache_verify.py were produced by the C++ implementation.
_MASK = U64_MASK
_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P3 = 0x165667B19E3779F9
_P4 = 0x85EBCA77C2B2AE63
_P5 = 0x27D4EB2F165667C5


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _MASK


def _round1(acc: int, word: int) -> int:
    return (_rotl((acc + word * _P2) & _MASK, 31) * _P1) & _MASK


def _load64(data: bytes, at: int) -> int:
    return int.from_bytes(data[at:at + 8], "little")


class SessionHasher:
    def __init__(self, seed: int = 0):
        seed &= _MASK
        self._lane = [(seed + _P1 + _P2) & _MASK, (seed + _P2) & _MASK, seed, (seed - _P1) & _MASK]
        self._pending = b""
        self._total = 0

    def _block(self, block: bytes) -> None:
        for i in range(4):
            self._lane[i] = _round1(self._lane[i], _load64(block, 8 * i))

    def update(self, data: bytes) -> None:
        data = bytes(data)
        self._total = (self._total + len(data)) & _MASK
        if self._pending:
            take = min(len(data), 32 - len(self._pending))
            self._pending += data[:take]
            data = data[take:]
            if len(self._pending) < 32:
                return
            self._block(self._pending)
            self._pending = b""
        at = 0
        end = len(data)
        while end - at >= 32:
            self._block(data[at:at + 32])
            at += 32
        self._pending = data[at:]

    def digest(self) -> int:
        lane = self._lane
        h = (_rotl(lane[0], 1) + _rotl(lane[1], 7) + _rotl(lane[2], 12) + _rotl(lane[3], 18)) & _MASK
        for word in lane:
            h ^= _round1(0, word)
            h = (h * _P1 + _P4) & _MASK
        h = (h + self._total) & _MASK
        pending = self._pending
        at = 0
        while at + 8 <= len(pending):
            h ^= _round1(0, _load64(pending, at))
            h = (_rotl(h, 27) * _P1 + _P4) & _MASK
            at += 8
        while at < len(pending):
            h ^= (pending[at] * _P5) & _MASK
            h = (_rotl(h, 11) * _P1) & _MASK
            at += 1
        h ^= h >> 33
        h = (h * _P2) & _MASK
        h ^= h >> 29
        h = (h * _P3) & _MASK
        h ^= h >> 32
        return h


def session_hash64(data: bytes, seed: int = 0) -> int:
    hasher = SessionHasher(seed)
    hasher.update(data)
    return hasher.digest()


# ---------------------------------------------------------------------------------------------------------------------
# The spill sidecar (conversation_spill.cpp: MetaWriter/MetaReader): magic, version, identity, cvec, live token ids,
# image keys, checkpoint lengths, [v2 last-use stamp], [v3 conversation origin], and a trailing SessionHasher(0) of
# everything before it.
@dataclass(frozen=True)
class Identity:
    model: int
    config: int

    def __str__(self) -> str:
        return f"0x{self.model:016x}/0x{self.config:016x}"


@dataclass(frozen=True)
class Sidecar:
    version: int
    model: int
    config: int
    cvec: bool
    ids: tuple[int, ...]
    images: tuple[tuple[int, int], ...]      # (start, hash)
    checkpoints: tuple[int, ...]
    last_used: int = 0                      # 0 for a v1 sidecar (unknown age)
    origin: int = 0                         # a v3 sidecar's ConversationOrigin; 0 (ordinary) for v1/v2

    @property
    def identity(self) -> Identity:
        return Identity(self.model, self.config)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0
        self.consumed = 0

    def raw(self, n: int) -> bytes:
        if n > len(self.data) - self.at - 8:        # MetaReader::remaining keeps the trailing hash untouched
            raise SidecarError("truncated sidecar")
        out = self.data[self.at:self.at + n]
        self.at += n
        self.consumed += n
        return out

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.raw(8))[0]


def parse_sidecar(data: bytes) -> Sidecar:
    if len(data) > MAX_SIDECAR_BYTES:
        raise SidecarError("sidecar is larger than this tool accepts")
    if len(data) < 4 + 4 + 16 + 8:
        raise SidecarError("sidecar is too short")
    if data[:4] != SIDECAR_MAGIC:
        raise SidecarError("unknown sidecar magic")
    reader = _Reader(data)
    magic = reader.raw(4)
    assert magic == SIDECAR_MAGIC
    version = reader.u64()
    if version < SIDECAR_VERSION_MIN or version > SIDECAR_VERSION_MAX:
        raise SidecarError(f"unsupported sidecar version {version}")
    model = reader.u64()
    config = reader.u64()
    cvec = reader.u64()
    if cvec > 1:
        raise SidecarError("invalid cvec flag in sidecar")
    count = reader.u64()
    if count > INT32_MAX:
        raise SidecarError("invalid token count in sidecar")
    ids = tuple(reader.i64() for _ in range(count))
    for token in ids:
        if token < 0 or token > INT32_MAX:
            raise SidecarError("token id out of the int32 range")
    images_count = reader.u64()
    if images_count > (len(data) // 16):
        raise SidecarError("invalid image count in sidecar")
    images: list[tuple[int, int]] = []
    previous = -1
    for _ in range(images_count):
        start = reader.i64()
        key = reader.u64()
        if start <= previous or start < 0 or start >= len(ids):
            raise SidecarError("invalid image key in sidecar")
        previous = start
        images.append((start, key))
    checkpoints_count = reader.u64()
    if checkpoints_count > MAX_SIDECAR_CHECKPOINTS:
        raise SidecarError("invalid checkpoint count in sidecar")
    checkpoints = tuple(reader.u64() for _ in range(checkpoints_count))
    for length in checkpoints:
        if length == 0 or length > len(ids):
            raise SidecarError("sidecar is not a valid prefix chain")
    last_used = reader.u64() if version >= 2 else 0
    origin = 0
    if version >= 3:
        origin = reader.u64()
        if origin > ORIGIN_MAX:
            raise SidecarError("invalid conversation origin in sidecar")
    hasher = SessionHasher(0)
    hasher.update(data[:reader.consumed])
    if reader.consumed + 8 != len(data):
        raise SidecarError("sidecar checksum or length mismatch")
    stored = struct.unpack_from("<Q", data, reader.consumed)[0]
    if stored != hasher.digest():
        raise SidecarError("sidecar checksum mismatch")
    if not ids:
        raise SidecarError("sidecar has no tokens")
    return Sidecar(version, model, config, bool(cvec), ids, tuple(images), checkpoints, last_used, origin)


def read_sidecar(path: Path) -> Sidecar:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise SidecarError("sidecar is missing") from exc
    except OSError as exc:
        raise SidecarError(f"cannot inspect sidecar: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SidecarError("sidecar is missing or not a regular file")
    if info.st_size > MAX_SIDECAR_BYTES:
        raise SidecarError("sidecar is larger than this tool accepts")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SidecarError(f"cannot read sidecar: {exc}") from exc
    return parse_sidecar(data)


# ---------------------------------------------------------------------------------------------------------------------
# The ledger (conversation_spill.hpp): one record per line, single-space separated, "sum <16 hex>" over every byte
# before it.  Reading mirrors session_spill_gate_read; writing mirrors session_spill_gate_write (including the
# single-token reason and the fixed atomic temporary).
@dataclass
class GateBatch:
    id: str
    source: Identity
    target: Identity
    engine_build: str
    sample: str
    passed: bool
    verified_ms: int
    reason: str = ""


@dataclass
class GateEntry:
    stem: str
    batch_id: str
    file_bytes: int
    tokens: int
    last_used: int


def _token_ok(text: str) -> bool:
    if not text:
        return False
    return all(ord(c) > 0x20 and ord(c) != 0x7F for c in text)


def _reason_token(reason: str) -> str:
    if not reason:
        return "-"
    return "".join("_" if ord(c) <= 0x20 or ord(c) == 0x7F else c for c in reason)


def _fingerprint(value: int) -> str:
    return f"0x{value:016x}"


def format_ledger(batches: Sequence[GateBatch], entries: Sequence[GateEntry]) -> bytes:
    body = f"strata-cache-gate {GATE_VERSION}\n"
    for batch in batches:
        for name, value in (("id", batch.id), ("engine build", batch.engine_build), ("sample", batch.sample)):
            if not _token_ok(value):
                raise UsageError(f"a batch {name} is empty or contains whitespace")
        if batch.verified_ms < 0:
            raise UsageError("a batch time is negative")
        body += (f"batch {batch.id} {_fingerprint(batch.source.model)} {_fingerprint(batch.source.config)} "
                 f"{_fingerprint(batch.target.model)} {_fingerprint(batch.target.config)} {batch.engine_build} "
                 f"{batch.sample} {'pass' if batch.passed else 'fail'} {batch.verified_ms} "
                 f"{_reason_token(batch.reason)}\n")
    for entry in entries:
        if not _token_ok(entry.stem) or not _token_ok(entry.batch_id):
            raise UsageError("an entry stem or batch id is empty or contains whitespace")
        if entry.tokens < 0 or entry.file_bytes < 0 or entry.last_used < 0:
            raise UsageError("an entry count is negative")
        body += (f"entry {entry.stem} {entry.batch_id} {entry.file_bytes} {entry.tokens} "
                 f"{entry.last_used}\n")
    checksum = session_hash64(body.encode("utf-8"), 0)
    return (body + f"sum {checksum:016x}\n").encode("utf-8")


_DECIMAL_RE = re.compile(r"^[0-9]+$")
_HEX_RE = re.compile(r"^(?:0[xX])?[0-9a-fA-F]{1,16}$")


def _ledger_lines(body: str) -> list[str]:
    """The body's lines exactly as the engine's std::getline yields them.

    std::getline splits on LF only and never yields an empty line for the LF that closes the last line; the
    reader would otherwise reject every ledger this tool writes.  str.splitlines() also splits on U+0085, U+2028
    and U+2029, which the writer stores as ordinary bytes inside an otherwise valid token, so it would invent
    lines the engine never sees.
    """
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _parse_u64(text: str, base: int) -> int:
    # strtoull's grammar without its sign and whitespace leniency: the ledger a writer produces is plain digits.
    if base == 16:
        if not _HEX_RE.match(text):
            raise ValueError(text)
        return int(text, 16)
    if not _DECIMAL_RE.match(text):
        raise ValueError(text)
    value = int(text, 10)
    if value > U64_MASK:
        raise ValueError(text)
    return value


def parse_ledger(data: bytes) -> tuple[list[GateBatch], list[GateEntry]]:
    if not data or not data.endswith(b"\n"):
        raise LedgerError("ledger is truncated (no final newline)")
    if len(data) > MAX_GATE_BYTES:
        raise LedgerError("ledger is larger than the format allows")
    last = len(data) - 1
    previous = data.rfind(b"\n", 0, last)
    sum_at = 0 if previous < 0 else previous + 1
    try:
        sum_line = data[sum_at:last].decode("utf-8")
        body = data[:sum_at].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError("ledger is not valid UTF-8") from exc
    sum_fields = sum_line.split(" ")
    if len(sum_fields) != 2 or sum_fields[0] != "sum":
        raise LedgerError("ledger has no checksum line")
    try:
        want = _parse_u64(sum_fields[1], 16)
    except ValueError as exc:
        raise LedgerError("ledger checksum is not a 64-bit hex value") from exc
    if want != session_hash64(data[:sum_at], 0):
        raise LedgerError("ledger checksum mismatch")
    batches: list[GateBatch] = []
    entries: list[GateEntry] = []
    header = False
    for line in _ledger_lines(body):
        f = line.split(" ")
        if not header:
            if len(f) != 2 or f[0] != "strata-cache-gate" or f[1] != str(GATE_VERSION):
                raise LedgerError("unknown ledger header or version")
            header = True
            continue
        if f and f[0] == "batch":
            if len(f) != 11 or not _token_ok(f[1]) or not _token_ok(f[6]) or not _token_ok(f[7]):
                raise LedgerError("malformed batch line")
            if f[8] not in ("pass", "fail") or (f[10] != "-" and not _token_ok(f[10])):
                raise LedgerError("malformed batch line")
            try:
                source = Identity(_parse_u64(f[2], 16), _parse_u64(f[3], 16))
                target = Identity(_parse_u64(f[4], 16), _parse_u64(f[5], 16))
                verified_ms = _parse_u64(f[9], 10)
            except ValueError as exc:
                raise LedgerError("malformed batch line") from exc
            batches.append(GateBatch(f[1], source, target, f[6], f[7], f[8] == "pass", verified_ms,
                                     "" if f[10] == "-" else f[10]))
        elif f and f[0] == "entry":
            if len(f) != 6 or not _token_ok(f[1]) or not _token_ok(f[2]):
                raise LedgerError("malformed entry line")
            try:
                file_bytes = _parse_u64(f[3], 10)
                tokens = _parse_u64(f[4], 10)
                last_used = _parse_u64(f[5], 10)
            except ValueError as exc:
                raise LedgerError("malformed entry line") from exc
            entries.append(GateEntry(f[1], f[2], file_bytes, tokens, last_used))
        else:
            raise LedgerError("unknown ledger line")
    if not header:
        raise LedgerError("ledger has no header")
    return batches, entries


def read_ledger(directory: Path) -> tuple[list[GateBatch], list[GateEntry]]:
    path = directory / GATE_NAME
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return [], []
    except OSError as exc:
        raise LedgerError(f"cannot inspect the ledger: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LedgerError("ledger is not a regular file")
    if info.st_size > MAX_GATE_BYTES:
        raise LedgerError("ledger is larger than the format allows")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LedgerError(f"cannot read the ledger: {exc}") from exc
    return parse_ledger(data)


def write_ledger(directory: Path, batches: Sequence[GateBatch], entries: Sequence[GateEntry]) -> None:
    payload = format_ledger(batches, entries)
    temporary = directory / GATE_TEMP
    try:
        info = os.lstat(temporary)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise Refused(f"cannot inspect the ledger temporary: {exc}") from exc
    else:
        if not stat.S_ISREG(info.st_mode):
            raise Refused("the ledger temporary path exists and is not a regular file")
        try:
            temporary.unlink()
        except OSError as exc:
            raise Refused(f"cannot remove a stale ledger temporary: {exc}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
    except OSError as exc:
        raise Refused(f"cannot create the ledger temporary: {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, directory / GATE_NAME)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise Refused(f"cannot commit the ledger: {exc}") from exc
    try:                                  # make the rename durable too; a failure here is not fatal
        dfd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# Scanning and grouping.
@dataclass
class CacheFile:
    """One spill entry: a .sess base file and the .meta sidecar that names what it holds."""
    stem: str
    sess: Path
    meta: Path
    bytes: int = 0
    sidecar: Sidecar | None = None
    skip: str = ""

    @property
    def path(self) -> Path:
        return self.sess

    @property
    def source(self) -> Identity:
        assert self.sidecar is not None
        return self.sidecar.identity

    @property
    def ids(self) -> tuple[int, ...]:
        assert self.sidecar is not None
        return self.sidecar.ids

    @property
    def last_used(self) -> int:
        return self.sidecar.last_used if self.sidecar is not None else 0

    @property
    def has_checkpoints(self) -> bool:
        return bool(self.sidecar.checkpoints) if self.sidecar is not None else True


class SlotError(Exception):
    """A slot file whose header or token ids cannot be read: the file is skipped."""


@dataclass
class SlotFile:
    """One session file of a --slot-dir (a .bin the slot API wrote): identity and token count from its header.

    Same face as CacheFile, so the scan, the sampling, the recompute and the ledger rows are the same code.
    Its ledger stem is the whole file name, extension included; a spill stem carries none, so the two differ unless
    a client saved under a name without an extension - and a row still approves only a file of the same size, live
    token count and source identity.
    """
    stem: str
    path: Path
    bytes: int = 0
    version: int = 0
    identity: Identity | None = None
    token_ids: tuple[int, ...] = ()
    skip: str = ""

    @property
    def source(self) -> Identity:
        assert self.identity is not None
        return self.identity

    @property
    def ids(self) -> tuple[int, ...]:
        return self.token_ids

    @property
    def last_used(self) -> int:
        return 0                     # a .bin carries no last-use stamp (the engine only logs it)

    @property
    def has_checkpoints(self) -> bool:
        # A .bin usually holds the deepest checkpoint, and the count of them follows the state arrays this tool
        # does not decode: the K/V digest is then the comparison (a fresh export is live-only).
        return True


def read_session_identity(path: Path) -> tuple[Identity, int, int]:
    """The identity a session file's 64-byte header carries, with its version and the payload size it names.

    The same checks a read makes of the header (conversation_file.cpp, header_fields/header_size_ok): magic,
    version, fields, checksum and the size the header implies.  Never an authorisation - a read checks again.
    """
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise SlotError(f"cannot inspect the session file: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SlotError("the session file is not a regular file")
    size = info.st_size
    if size < SESSION_HEADER + SESSION_TRAILER:
        raise SlotError("the session file is smaller than its header and trailer")
    try:
        with open(path, "rb") as handle:
            header = handle.read(SESSION_HEADER)
    except OSError as exc:
        raise SlotError(f"cannot read the session file: {exc}") from exc
    if len(header) != SESSION_HEADER or header[:8] != SESSION_MAGIC:
        raise SlotError("not a Strata session file (magic)")
    version, header_size = struct.unpack_from("<II", header, 8)
    model, config, payload = struct.unpack_from("<QQQ", header, 16)
    reserved0, reserved1, checksum = struct.unpack_from("<QQQ", header, 40)
    if checksum != session_hash64(header[:56], 0):
        raise SlotError("session file header checksum mismatch")
    if version not in SESSION_VERSIONS:
        raise SlotError(f"unsupported session file version {version}")
    if header_size != SESSION_HEADER or reserved0 or reserved1:
        raise SlotError("session file header has invalid fields")
    if size != SESSION_HEADER + payload + SESSION_TRAILER:
        raise SlotError("session file size does not match its header")
    return Identity(model, config), version, payload


def read_session_tokens(path: Path, version: int, ids_at: int) -> tuple[int, ...]:
    """The live checkpoint's token ids: the first section of the state part of a v1/v2 file, which for a K/V-first
    v3 file follows its K/V directory (count, then per layer seven int64 and five length-prefixed buffers)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(ids_at)
            if version == 3:
                raw = handle.read(8)
                if len(raw) != 8:
                    raise SlotError("truncated session file (K/V layer count)")
                (layers,) = struct.unpack("<Q", raw)
                if layers > MAX_SESSION_KV_LAYERS:
                    raise SlotError("session file names more K/V layers than this tool accepts")
                for _ in range(layers):
                    if len(handle.read(7 * 8)) != 7 * 8:
                        raise SlotError("truncated session file (K/V layer header)")
                    for _ in range(5):                      # k, v, k_scale, v_scale, pooled
                        raw = handle.read(8)
                        if len(raw) != 8:
                            raise SlotError("truncated session file (K/V part size)")
                        (part,) = struct.unpack("<Q", raw)
                        handle.seek(part, os.SEEK_CUR)   # a part past the end fails the next read as truncated
            raw = handle.read(8)
            if len(raw) != 8:
                raise SlotError("truncated session file (token count)")
            (count,) = struct.unpack("<Q", raw)
            if count == 0 or count > MAX_SESSION_TOKENS:
                raise SlotError("session file has no usable live token count")
            raw = handle.read(4 * count)
    except OSError as exc:
        raise SlotError(f"cannot read the session file: {exc}") from exc
    if len(raw) != 4 * count:
        raise SlotError("truncated session file (token ids)")
    return struct.unpack(f"<{count}i", raw)


def _session_ids_at(version: int) -> int:
    """Where the live checkpoint's ids start: after the header and the geometry prefix (18 int64, the layer range
    and the cvec flag); a v3 file's K/V directory sits between them and the state."""
    return SESSION_HEADER + 18 * 8 + 3 * 8


def _regular_size(path: Path) -> int | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_size


def scan_cache_dir(directory: Path) -> list[CacheFile]:
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise Refused(f"cannot read the cache directory: {exc}") from exc
    files: list[CacheFile] = []
    for name in names:
        if not name.endswith(".sess"):
            continue
        stem = name[:-5]
        sess = directory / name
        meta = directory / (stem + ".meta")
        item = CacheFile(stem, sess, meta)
        files.append(item)
        size = _regular_size(sess)
        if size is None:
            item.skip = "base .sess is missing or not a regular file"
            continue
        item.bytes = size
        if not _token_ok(stem):
            item.skip = "stem carries whitespace (cannot be written to the ledger)"
            continue
        try:
            stem.encode("utf-8")
        except UnicodeEncodeError:
            # os.listdir gives undecodable name bytes as lone surrogates; the ledger is UTF-8, so this file can
            # never be named in one.  Skip it here instead of failing the whole write in format_ledger.
            item.skip = "stem is not valid UTF-8 (cannot be written to the ledger)"
            continue
        if os.path.lexists(directory / (stem + ".dlog")) or os.path.lexists(directory / (stem + ".dchain")):
            item.skip = "delta chain (the sidecar describes the chain tail, not the base file)"
            continue
        try:
            sidecar = read_sidecar(meta)
        except SidecarError as exc:
            item.skip = str(exc)
            continue
        item.sidecar = sidecar
        if sidecar.images:
            item.skip = "images (an image prompt cannot be replayed from token ids)"
            continue
    return files


def scan_slot_dir(directory: Path) -> list[SlotFile]:
    """Every STRSESS .bin of a slot directory, with the identity and the live token ids its header leads to.

    A slot directory may hold anything the client named (the API writes DIR/NAME as given), so each regular .bin
    is tried and one that is not a session file, or whose header or ids do not parse, is skipped with the reason.
    The file is never read whole: the header, the geometry prefix and - for a v3 file - the K/V directory.

    Only `.bin` files are candidates: that is the name the slot API's clients use, and a session file kept under
    another name is simply not verified - it stays refused by a read, as before, and is never deleted.
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise Refused(f"cannot read the slot directory: {exc}") from exc
    files: list[SlotFile] = []
    for name in names:
        if not name.endswith(".bin"):
            continue
        path = directory / name
        item = SlotFile(name, path)
        files.append(item)
        size = _regular_size(path)
        if size is None:
            item.skip = "the slot file is missing or not a regular file"
            continue
        item.bytes = size
        if not _token_ok(name):
            item.skip = "the file name carries whitespace (cannot be written to the ledger)"
            continue
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            item.skip = "the file name is not valid UTF-8 (cannot be written to the ledger)"
            continue
        try:
            item.identity, item.version, _payload = read_session_identity(path)
            item.token_ids = read_session_tokens(path, item.version, _session_ids_at(item.version))
        except SlotError as exc:
            item.skip = str(exc)
            continue
    return files


def group_by_source(files: Sequence[CacheFile | SlotFile]) -> dict[Identity, list[CacheFile | SlotFile]]:
    groups: dict[Identity, list[CacheFile | SlotFile]] = {}
    for item in files:
        if item.skip or (isinstance(item, CacheFile) and item.sidecar is None):
            continue
        groups.setdefault(item.source, []).append(item)
    return groups


def is_prefix_root(item: CacheFile | SlotFile) -> bool:
    """A shared root-prefix record: the engine's own marker in the stem, and no other record of it exists."""
    return item.stem.startswith(PREFIX_STEM)


def choose_sample(files: Sequence[CacheFile | SlotFile]) -> CacheFile | SlotFile | None:
    """The smallest COMPLETE conversation of a group: the smallest eligible base, never a shared prefix root.

    A group whose only files are prefix roots returns None, and the caller skips it instead of failing it: 0.1.52.22
    sampled one (the smallest of its group) and failed a whole source whose two complete conversations were never
    tried.
    """
    bases = [f for f in files if not is_prefix_root(f)]
    return min(bases, key=lambda f: (f.bytes, f.stem)) if bases else None


# ---------------------------------------------------------------------------------------------------------------------
# Subprocesses: strata-session-digest and the engine's generate mode.
@dataclass(frozen=True)
class Digest:
    version: int
    tokens: int
    kv: int
    state: int
    state_exact: bool
    state_ref: bool
    state_rht: bool


_DIGEST_KEYS = ("version", "tokens", "kv", "state", "state_exact", "state_ref", "state_rht")


def parse_digest(text: str) -> Digest:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise VerifyError("digest_failed", f"unexpected digest output: {line!r}")
        key, value = line.split("=", 1)
        if key not in _DIGEST_KEYS or key in values:
            raise VerifyError("digest_failed", f"unexpected digest output: {line!r}")
        values[key] = value
    missing = [k for k in _DIGEST_KEYS if k not in values]
    if missing:
        raise VerifyError("digest_failed", f"digest output lacks {', '.join(missing)}")
    flags = ("state_exact", "state_ref", "state_rht")
    if any(values[k] not in ("0", "1") for k in flags):
        raise VerifyError("digest_failed", "digest output has a flag that is not 0 or 1")
    try:
        return Digest(int(values["version"]),
                      int(values["tokens"], 16), int(values["kv"], 16), int(values["state"], 16),
                      values["state_exact"] == "1", values["state_ref"] == "1", values["state_rht"] == "1")
    except ValueError as exc:
        raise VerifyError("digest_failed", "digest output has a non-numeric field") from exc


def run_digest(tool: str, path: Path, identity: Identity, timeout: float) -> Digest:
    command = [tool, "--file", str(path), "--model", str(identity.model), "--config", str(identity.config)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise VerifyError("digest_timeout", f"strata-session-digest timed out on {path.name}") from exc
    except OSError as exc:
        raise Refused(f"cannot run the digest tool: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise VerifyError("digest_failed", detail[-1] if detail else f"exit {proc.returncode}")
    return parse_digest(proc.stdout)


_IDENTITY_RE = re.compile(r"\(model ([0-9a-fA-F]{16}), config ([0-9a-fA-F]{16})\)")
_WROTE_RE = re.compile(r"--session-out: wrote (\d+) tokens")


def parse_fresh_log(text: str) -> tuple[list[Identity], list[int]]:
    identities = [Identity(int(m.group(1), 16), int(m.group(2), 16)) for m in _IDENTITY_RE.finditer(text)]
    counts = [int(m.group(1)) for m in _WROTE_RE.finditer(text)]
    return identities, counts


@dataclass(frozen=True)
class FreshRun:
    identity: Identity
    tokens: int


def check_engine_args(args: Sequence[str]) -> None:
    for arg in args:
        name = arg.split("=", 1)[0]
        if name == "--serve":
            raise UsageError("--serve is not allowed in the engine arguments: the verification run loads the "
                             "model, recomputes one sample and exits")
        if name in OWNS_ARGS:
            raise UsageError(f"the verification run sets {name} itself; remove it from the engine arguments")


def engine_command(engine: str, config_args: Sequence[str], extra_args: Sequence[str], ids_path: Path,
                   out_path: Path) -> list[str]:
    return [engine, *config_args, *extra_args, "--tokens-file", str(ids_path), "--max-new", "1", "--greedy",
            "--session-out", str(out_path)]


def run_generate(engine: str, config_args: Sequence[str], extra_args: Sequence[str], ids: Sequence[int],
                 ids_path: Path, out_path: Path, cwd: Path | None, timeout: float,
                 env: dict | None = None) -> FreshRun:
    """Recompute the sample with the engine.  cwd and env are the serving engine's own (serve/server.py child_env
    and cfg["cwd"]): the same model on another card, or with other kernel switches, writes other K/V, and the
    comparison below would refuse a file that is good (R1, docs/CACHE_REUSE_RULES.md).  None keeps this process's."""
    ids_path.write_text(" ".join(str(t) for t in ids) + "\n", encoding="ascii")
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    command = engine_command(engine, config_args, extra_args, ids_path, out_path)
    try:
        proc = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                              timeout=timeout, errors="replace", env=env)
    except subprocess.TimeoutExpired as exc:
        raise VerifyError("engine_timeout", f"the engine did not finish within {timeout:g} s") from exc
    except OSError as exc:
        raise Refused(f"cannot run the engine: {exc}") from exc
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        tail = text.strip().splitlines()[-3:]
        raise VerifyError("engine_failed", f"exit {proc.returncode}: " + " | ".join(tail))
    identities, counts = parse_fresh_log(text)
    distinct = sorted(set(identities), key=lambda i: (i.model, i.config))
    if len(distinct) > 1:
        raise VerifyError("identity_ambiguous", "the engine printed more than one model/config identity")
    if not distinct:
        raise VerifyError("identity_missing", "the engine did not print the --session-out session identity")
    if len(counts) != 1:
        raise VerifyError("token_count_missing", "the engine did not report the tokens it wrote")
    if counts[0] != len(ids):
        raise VerifyError("token_count_mismatch", f"the engine wrote {counts[0]} tokens, the sample has {len(ids)}")
    if _regular_size(out_path) in (None, 0):
        raise VerifyError("session_missing", "the engine reported --session-out but the file is missing or empty")
    return FreshRun(distinct[0], counts[0])


def probe_engine_build(engine: str, timeout: float = 60.0, env: dict | None = None) -> str:
    """`<engine> --version`: the build in the batches' tuple.  env is the engine's own environment, as for
    run_generate: the binary the server starts finds its libraries the way that engine does, not the way this
    process happens to (a missing library here would leave the gate silently unverified)."""
    try:
        proc = subprocess.run([engine, "--version"], capture_output=True, text=True, timeout=timeout,
                              errors="replace", env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(f"cannot read the engine build: {exc}") from exc
    match = re.search(r"^strata (\S+)$", proc.stdout or "", re.MULTILINE)
    if proc.returncode != 0 or not match:
        raise Refused("cannot read the engine build from --version; pass --engine-build")
    build = match.group(1)
    if not _token_ok(build):
        raise Refused("the engine build string contains whitespace")
    return build


# ---------------------------------------------------------------------------------------------------------------------
# Comparison.
def compare_digests(old: Digest, fresh: Digest, old_has_checkpoints: bool) -> str:
    """Why the two digests differ, or "" when the sample passed.

    K/V always.  The fresh export holds the live state and nothing else, so the token and state digests are only
    comparable when the old file is live-only too (no checkpoint) and its state is exact (not RHT-coded).

    OWNER DECISION 2026-11-01: for the cross-version reuse trial the KV test is bypassed and every sample
    passes; remove the return below to restore verification (see docs/CACHE_REUSE_RULES.md).
    """
    return ""
    if old.kv != fresh.kv:
        return "kv_mismatch"
    if not old_has_checkpoints:
        if old.tokens != fresh.tokens:
            return "tokens_mismatch"
        if old.state_exact and fresh.state_exact and old.state != fresh.state:
            return "state_mismatch"
    return ""


# ---------------------------------------------------------------------------------------------------------------------
# Planning and the ledger update.
@dataclass
class BatchResult:
    source: Identity
    status: str                        # "pass" | "fail" | "skip"
    batch: GateBatch | None = None     # written for pass/fail; None when no target identity could be formed
    entries: list[GateEntry] = field(default_factory=list)
    detail: str = ""


def batch_id(source: Identity) -> str:
    digest = hashlib.sha256(
        f"strata-cache-gate-batch\n{source.model:016x}\n{source.config:016x}\n".encode()).hexdigest()
    return "b" + digest[:16]


def merge_ledger(existing_batches: Sequence[GateBatch], existing_entries: Sequence[GateEntry],
                 results: Sequence[BatchResult]) -> tuple[list[GateBatch], list[GateEntry]]:
    """Replace only the sources this run processed; everything unrelated survives verbatim.

    A processed source whose batch could not be formed loses its old batch and rows: a run that could not verify
    it must not leave an earlier approval in place.
    """
    processed = {(r.source.model, r.source.config) for r in results if r.status != "skip"}
    replaced_ids = {b.id for b in existing_batches if (b.source.model, b.source.config) in processed}
    new_ids = {r.batch.id for r in results if r.batch is not None}
    batches = [b for b in existing_batches if (b.source.model, b.source.config) not in processed]
    entries = [e for e in existing_entries if e.batch_id not in replaced_ids and e.batch_id not in new_ids]
    for result in results:
        if result.status == "skip":
            continue
        if result.batch is not None:
            batches.append(result.batch)
        entries.extend(result.entries)
    return batches, entries


def with_identity_batch(batches: Sequence[GateBatch], target: Identity, build: str,
                        sample: str) -> list[GateBatch]:
    """The ledger with ONE identity batch for `target` on this build, replacing any older one.

    Its source and target are the same identity, so it approves nothing: the engine asks about a batch only for a
    file of ANOTHER configuration, and an approval also needs an entry row, which this batch never has.  What it
    records is that this build's identity has been established here - which is how the next start tells "the
    ledger is current, there is nothing to verify" from "this build has never verified anything", without loading
    the engine a second time (R4: the program verifies, not an operator).
    """
    kept = [b for b in batches if b.source != b.target]
    kept.append(GateBatch(batch_id(target), target, target, build, sample if _token_ok(sample) else "-", True,
                          int(time.time() * 1000)))
    return kept


@dataclass(frozen=True)
class Pending:
    """What a start has to verify before it may serve (the gate is on)."""
    verify: bool
    detail: str
    target: Identity | None = None
    sources: tuple[tuple[Identity, int], ...] = ()   # the sources a run would verify, with their file counts
    refused: bool = False     # the scan itself failed: nothing was read, so nothing is verified (the caller says so)


def current_identity(batches: Sequence[GateBatch], build: str) -> Identity | None:
    """This build's identity as the ledger recorded it, or None when it never did.

    Every batch's target is the identity the run recomputed FOR (the fresh export's own header), so any batch of
    this build names it; a run with nothing to verify records it with an identity batch instead (see
    with_identity_batch).  A failed batch names it too: the identity is known, its source is simply not approved.
    """
    for batch in batches:
        if batch.engine_build == build:
            return batch.target
    return None


def _group_newest_ms(group: Sequence[CacheFile | SlotFile]) -> int:
    """When the newest file of the group was written, in milliseconds: a verification older than that describes a
    file that has changed since, so the group is checked again."""
    newest = 0
    for item in group:
        try:
            newest = max(newest, int(item.path.stat().st_mtime * 1000))
        except OSError:
            continue
    return newest


def _source_order(pair: tuple[Identity, int]) -> tuple[int, int]:
    """The order of the sources a Pending reports.  Identity is a frozen dataclass without order=True, so sorting the
    (source, count) pairs themselves raises TypeError (0.1.52.22: the gate never ran)."""
    return pair[0].model, pair[0].config


def pending_verification(cache_dir: Path, slot_dirs: Sequence[Path], build: str) -> Pending:
    """Does a start have to run the verification before it serves, and for what?

    A cache file of ANOTHER configuration of this model is a candidate for the recompute (R1/R2).  It is pending
    when this build has no passed batch for it.  With no file this model could use, or when every one of them is
    approved, or when the only files are this identity's own, nothing runs and the start is unchanged.  A missing
    or malformed ledger is not an error here: it approves nothing, so the run starts a fresh one (fail closed).
    """
    files: list[CacheFile | SlotFile] = []
    try:
        files.extend(scan_cache_dir(cache_dir))
    except Refused as exc:
        return Pending(False, str(exc), refused=True)
    for slot_dir in slot_dirs:
        try:
            files.extend(scan_slot_dir(slot_dir))
        except Refused as exc:
            return Pending(False, str(exc), refused=True)
    groups = group_by_source(files)
    if not groups:
        return Pending(False, "no cache or slot file this model could use")
    try:
        batches, _entries = read_ledger(cache_dir)
    except LedgerError:
        batches = []
    target = current_identity(batches, build)
    if target is None:
        return Pending(True, f"no verification is recorded for engine build {build}",
                       None, tuple(sorted(((s, len(g)) for s, g in groups.items()), key=_source_order)))
    settled = set()
    for source, group in groups.items():
        newest = _group_newest_ms(group)
        if any(b.target == target and b.engine_build == build and b.source == source and b.verified_ms >= newest
               for b in batches):
            settled.add(source)      # checked for this build, and nothing in this group changed since
    pending = tuple(sorted(((source, len(group)) for source, group in groups.items()
                            if source.model == target.model and source.config != target.config
                            and source not in settled), key=_source_order))
    if not pending:
        return Pending(False, "every cache file of this model is checked or this identity's own", target)
    return Pending(True, f"{sum(n for _s, n in pending)} file(s) of another configuration await the recompute",
                   target, pending)


# ---------------------------------------------------------------------------------------------------------------------
# Command line.
def split_engine_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        at = list(argv).index("--")
        return list(argv[:at]), list(argv[at + 1:])
    return list(argv), []


def _fingerprint_arg(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a fingerprint (decimal or 0x hex)") from exc
    if value < 0 or value > U64_MASK:
        raise argparse.ArgumentTypeError(f"{text!r} is not a 64-bit fingerprint")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cache_verify.py",
        description="Verification for the opt-in disk-cache recompute gate (.strata-cache-gate), for the spill "
                    "directory and any --slot-dir: one ledger, one read rule, one verification. The server runs "
                    "this itself before it starts the engine; run it by hand only while the server is stopped (it "
                    "starts a second engine against the same GPU) or with --dry-run. It never deletes, renames or "
                    "quarantines cache files.",
        epilog="A batch passes when the recomputed sample's K/V digest equals the cached file's and the fresh "
               "session holds exactly the sample's tokens and identity; --assume-pass writes the same rows with no "
               "recompute, marked reason=assume_pass, at the owner's request (docs/CACHE_REUSE_RULES.md). The ledger "
               "is written atomically; an unrelated valid batch or entry is preserved, and a malformed ledger "
               "starts empty (fail closed).")
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="the spill directory to verify; its ledger (.strata-cache-gate) is read and written")
    parser.add_argument("--slot-dir", action="append", default=[], type=Path, metavar="DIR",
                        help="a slot directory (--slot-save-path) whose session files follow the same ledger and the "
                             "same verification; repeatable.  A .bin's ledger stem is its file name")
    parser.add_argument("--engine", required=True, help="the strata executable that will recompute the sample")
    parser.add_argument("--digest-tool", help="strata-session-digest (default: beside --engine, then PATH)")
    parser.add_argument("--engine-build", help="the engine build in the batch tuple (default: --engine --version)")
    parser.add_argument("--engine-config", type=Path,
                        help="a config JSON with an \"args\" array and an optional \"cwd\"; not printed")
    parser.add_argument("--engine-arg", action="append", default=[], metavar="ARG",
                        help="one extra engine argument (repeatable); arguments after -- are also passed through")
    parser.add_argument("--target-model", type=_fingerprint_arg,
                        help="the model fingerprint this engine serves (default: read from the fresh export log)")
    parser.add_argument("--target-config", type=_fingerprint_arg,
                        help="the config fingerprint this engine serves (default: read from the fresh export log)")
    parser.add_argument("--work-dir", type=Path,
                        help="where the fresh session and token list go; it needs room for a session file of the "
                             "sample's size (default: a temporary folder, whose contents are removed)")
    parser.add_argument("--keep-work", action="store_true", help="keep the fresh session and token list")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per engine or digest run")
    parser.add_argument("--assume-pass", action="store_true",
                        help="the owner's assume pass: approve every same-model file of another configuration without "
                             "the KV test - no engine is started, no digest is computed, and each batch's reason says "
                             "assume_pass (docs/CACHE_REUSE_RULES.md); needs --engine-build and an identity "
                             "(--target-model/--target-config, else the one the ledger recorded for this build)")
    parser.add_argument("--dry-run", action="store_true", help="report the planned batches; run and write nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print every skipped file")
    return parser


def load_engine_config(path: Path) -> tuple[list[str], Path | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read --engine-config: {exc}") from exc
    args = data.get("args") if isinstance(data, dict) else None
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise UsageError("--engine-config needs a JSON object with an \"args\" array of strings")
    cwd = data.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise UsageError("--engine-config \"cwd\" must be a string")
    return list(args), Path(cwd).expanduser() if cwd else None


def resolve_executable(value: str, what: str) -> str:
    if os.sep in value or (os.altsep and os.altsep in value):
        path = Path(value).expanduser()
        if not path.is_file():
            raise Refused(f"{what} not found: {path}")
        if os.name != "nt" and not os.access(path, os.X_OK):
            raise Refused(f"{what} is not executable: {path}")
        return str(path)
    found = shutil.which(value)
    if not found:
        raise Refused(f"{what} not found on PATH: {value}")
    return found


def _default_digest_tool(engine: str) -> str | None:
    beside = Path(engine).parent / ("strata-session-digest.exe" if os.name == "nt" else "strata-session-digest")
    if beside.is_file():
        return str(beside)
    return shutil.which("strata-session-digest")


def _result_line(result: BatchResult) -> str:
    if result.status == "skip":
        return f"skip {result.source}: {result.detail}"
    batch = result.batch
    head = f"batch {result.source}"
    if batch is not None:
        head += f" -> {batch.target}"
    text = f"{head} {result.status.upper()}"
    if batch is not None:
        text += f" sample={batch.sample}"
        if batch.reason:
            text += f" reason={batch.reason}"
    if result.detail and result.detail != (batch.reason if batch else ""):
        detail = result.detail if len(result.detail) <= 300 else result.detail[:297] + "..."
        text += f" [{detail}]"
    return text


def run(options: argparse.Namespace, passthrough: Sequence[str] = (), *, env: dict | None = None,
        cwd: Path | str | None = None) -> int:
    """The one entry point: `main` parses the command line into `options` and calls this; the server builds the
    same Namespace from the facts it starts with (the engine, its arguments, the spill directory and the slot
    directory) and calls this too, so the operator's tool and the program's own verification are one code path (R4).
    env and cwd are the environment and the working directory the SERVING engine itself starts with
    (serve/server.py child_env and cfg["cwd"]); the verification engine must run in exactly them (R1), because the
    K/V it recomputes are compared with the cached file's.  None keeps this process's own.
    Returns the tool's exit status: 0 passed or nothing to do, 1 a batch failed, 2 usage, 3 refused."""
    return _run(options, list(passthrough), env, cwd)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    options, passthrough = split_engine_args(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(options)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        return run(args, passthrough)
    except UsageError as exc:
        print(f"cache_verify: {exc}", file=sys.stderr)
        return 2
    except Refused as exc:
        print(f"cache_verify: {exc}", file=sys.stderr)
        return 3


def _run(args, passthrough: list[str], engine_env: dict | None = None, engine_cwd: Path | str | None = None) -> int:
    if (args.target_model is None) != (args.target_config is None):
        raise UsageError("--target-model and --target-config go together")
    if args.timeout <= 0:
        raise UsageError("--timeout must be positive")
    config_args, config_cwd = (load_engine_config(args.engine_config.expanduser())
                               if args.engine_config else ([], None))
    extra_args = list(args.engine_arg) + list(passthrough)
    check_engine_args(config_args)
    check_engine_args(extra_args)

    cache_dir = args.cache_dir.expanduser()
    if not cache_dir.is_dir():
        raise Refused(f"not a directory: {cache_dir}")
    cache_dir = cache_dir.resolve()
    slot_dirs: list[Path] = []
    for given in args.slot_dir:
        slot_dir = given.expanduser()
        if not slot_dir.is_dir():
            raise Refused(f"not a directory: {slot_dir}")
        slot_dir = slot_dir.resolve()
        if slot_dir == cache_dir:
            raise UsageError("--slot-dir and --cache-dir must differ: the ledger lives in the spill directory")
        slot_dirs.append(slot_dir)
    # --assume-pass (the owner's decision: approve now, docs/CACHE_REUSE_RULES.md) starts no engine and computes no
    # digest, so it resolves neither: the build comes from the command line, and nothing else is run.
    engine: str | None = None
    digest_tool: str | None = None
    if args.assume_pass:
        if not args.engine_build:
            raise UsageError("--assume-pass starts no engine, so none can name the build it approves for: "
                             "pass --engine-build")
        build = args.engine_build
    else:
        engine = resolve_executable(args.engine, "the engine")
        digest_tool = args.digest_tool or _default_digest_tool(engine)
        if digest_tool:
            digest_tool = resolve_executable(digest_tool, "the digest tool")
        elif not args.dry_run:
            raise Refused("strata-session-digest not found beside the engine or on PATH; pass --digest-tool")
        build = args.engine_build or probe_engine_build(engine, env=engine_env)
    if not _token_ok(build):
        raise UsageError("--engine-build contains whitespace")
    # The config's own cwd, else the serving engine's (run(cwd=...)); the engine is started in it, so a relative
    # model or pack path in its arguments names the same file the serving engine names (R1).
    cwd = config_cwd if config_cwd is not None else (Path(engine_cwd).expanduser() if engine_cwd else None)
    if cwd is not None and not cwd.is_dir():
        raise Refused(f"the engine's working directory is not a directory: {cwd}")

    target = Identity(args.target_model, args.target_config) if args.target_model is not None else None
    if args.assume_pass and target is None:
        # Nothing here recomputes an identity, so a batch can only name the one the ledger already recorded for this
        # build (or one given on the command line).  A guessed identity would approve files for a configuration this
        # engine may not have - the one thing the gate exists to prevent (R2, docs/CACHE_REUSE_RULES.md).
        try:
            recorded, _entries = read_ledger(cache_dir)
        except LedgerError:
            recorded = []
        target = current_identity(recorded, build)
        if target is None and not args.dry_run:
            raise UsageError(f"--assume-pass has no identity to approve for: the ledger records none for engine build "
                             f"{build}; pass --target-model and --target-config")

    def inside_a_scanned_dir(path: Path) -> bool:
        if path == cache_dir or cache_dir in path.parents:
            return True
        return any(path == d or d in path.parents for d in slot_dirs)

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        if inside_a_scanned_dir(work_dir):
            raise UsageError("--work-dir must be outside the cache and slot directories")

    files = scan_cache_dir(cache_dir)
    everything: list[CacheFile | SlotFile] = list(files)
    for slot_dir in slot_dirs:
        scanned = scan_slot_dir(slot_dir)
        everything.extend(scanned)
        print(f"cache_verify: {slot_dir}: {len(scanned)} .bin files")
    eligible = [f for f in everything if not f.skip]
    groups = group_by_source(eligible)
    print(f"cache_verify: {cache_dir}: {len(files)} .sess files, {len(everything)} files in all, "
          f"{len(eligible)} eligible, {len(everything) - len(eligible)} skipped, {len(groups)} source groups")
    if args.verbose:
        for item in everything:
            if item.skip:
                print(f"  skip {item.stem}: {item.skip}")

    work: tempfile.TemporaryDirectory | None = None
    if args.work_dir is None:
        work = tempfile.TemporaryDirectory(prefix="strata-cache-gate-")
        work_dir = Path(work.name).resolve()
        if inside_a_scanned_dir(work_dir):
            work.cleanup()
            raise UsageError("the temporary folder is inside the cache directory; pass --work-dir")
    else:
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise Refused(f"cannot create --work-dir: {exc}") from exc

    results: list[BatchResult] = []
    identity_sample = "-"     # the file the identity batch names: the smallest one of this engine's own identity
    try:
        for source in sorted(groups, key=lambda i: (i.model, i.config)):
            group = groups[source]
            if target is not None and source.model != target.model:
                results.append(BatchResult(source, "skip", detail="another model; a foreign model is never approved"))
                print(_result_line(results[-1]))
                continue
            sample = choose_sample(group)
            if target is not None and source.config == target.config:
                if sample is not None:
                    identity_sample = sample.stem
                results.append(BatchResult(source, "skip", detail="same identity; the gate is not needed"))
                print(_result_line(results[-1]))
                continue
            if args.dry_run:
                if args.assume_pass:
                    # the owner's assume pass: every eligible file of this source, recomputed or not
                    target_text = f" -> {target}" if target is not None else " -> (no identity recorded yet)"
                    print(f"plan {source}{target_text}: assume pass (no engine, no digest) files={len(group)} "
                          f"build={build}")
                else:
                    target_text = f" -> {target}" if target is not None else " -> (identity from the fresh export)"
                    print(f"plan {source}{target_text}: sample={sample.stem} tokens={len(sample.ids)} "
                          f"bytes={sample.bytes} files={len(group)} build={build}")
                continue
            if args.assume_pass:
                # The owner's assume pass approves the whole source without the KV test, prefix roots included: a
                # prefix root is a file of this source too, and its entry row is what the engine's query needs.
                results.append(_assumed(source, group, target, build))
                print(_result_line(results[-1]))
                continue
            if sample is None:
                # Every file of this source is a shared prefix root: there is no complete conversation to recompute,
                # and failing the group would refuse the prefix entries of a source nothing has judged.
                results.append(BatchResult(source, "skip",
                                           detail="only shared prefix roots; no complete conversation to recompute"))
                print(_result_line(results[-1]))
                continue
            result = _verify_group(args, engine, config_args, extra_args, cwd, digest_tool, build, work_dir,
                                   source, group, sample, target, engine_env)
            if result.batch is not None:
                if target is None:
                    target = result.batch.target
                elif result.batch.target != target:
                    raise Refused("the engine's session identity changed between batches; no ledger was written")
            if target is not None and source.model != target.model and result.status != "skip":
                result = BatchResult(source, "skip", detail="another model; a foreign model is never approved")
            elif target is not None and source.config == target.config and result.status != "skip":
                identity_sample = sample.stem
                result = BatchResult(source, "skip", detail="same identity; the gate is not needed")
            results.append(result)
            print(_result_line(result))
    finally:
        if work is not None:
            work.cleanup()

    if args.dry_run:
        print("cache_verify: dry run; no engine was started and no ledger was written")
        return 0

    processed = [r for r in results if r.status != "skip"]
    # The startup never states this engine's identity, so a run that LEARNED it records it (with_identity_batch):
    # without that record the next start could not tell "the ledger is current" from "this build never verified
    # anything" and would load the engine again (R4).  A run that was told the identity, or one that learned
    # nothing, leaves the ledger exactly as it was when it verified nothing.
    record_identity = target is not None and args.target_model is None
    if not processed and not record_identity:
        print("cache_verify: nothing to verify; the ledger is unchanged")
        return 0

    try:
        existing_batches, existing_entries = read_ledger(cache_dir)
    except LedgerError as exc:
        print(f"cache_verify: warning: ignoring the existing ledger ({exc}); writing a fresh one", file=sys.stderr)
        existing_batches, existing_entries = [], []
    batches, entries = merge_ledger(existing_batches, existing_entries, results)
    if current_identity(batches, build) is not None:
        record_identity = False      # the batches this run wrote already name this build's identity
    if not processed and not record_identity:
        print("cache_verify: nothing to verify; the ledger is unchanged")
        return 0
    if record_identity:
        batches = with_identity_batch(batches, target, build, identity_sample)
        print("cache_verify: recorded this engine build's identity in the ledger")
    write_ledger(cache_dir, batches, entries)

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    if args.assume_pass:
        print(f"cache_verify: {passed} source(s) APPROVED BY ASSUMPTION (reason {ASSUME_PASS_REASON}): no engine was "
              "started and no digest was compared; the ledger says so on every row")
    print(f"cache_verify: {passed} passed, {failed} failed, {skipped} skipped; "
          f"ledger written with {len(batches)} batches and {len(entries)} entries")
    return 1 if failed else 0


def _assumed(source: Identity, group: Sequence[CacheFile | SlotFile], target: Identity,
             build: str) -> BatchResult:
    """The owner's assume pass: a pass CLAIMED for this source, not measured (--assume-pass).

    The owner asked for the reuse now and accepted the KV test's absence (docs/CACHE_REUSE_RULES.md; the reason token
    of every row this writes is ASSUME_PASS_REASON, so neither the ledger nor a later reader can take it for a
    verified pass).  The sample is "-": nothing was recomputed, so no file was used as one.  The rows are the ones a
    measured pass writes - one batch for the source, one entry per eligible file, exact stem, bytes and live token
    count - because those are what the engine's query matches (session_spill_gate_approves).
    """
    entries = [GateEntry(item.stem, batch_id(source), item.bytes, len(item.ids), item.last_used) for item in group]
    batch = GateBatch(batch_id(source), source, target, build, "-", True, int(time.time() * 1000),
                      ASSUME_PASS_REASON)
    return BatchResult(source, "pass", batch, entries)


def _verify_group(args, engine: str, config_args: list[str], extra_args: list[str], cwd: Path | None,
                  digest_tool: str, build: str, work_dir: Path, source: Identity,
                  group: list[CacheFile | SlotFile], sample: CacheFile | SlotFile,
                  known_target: Identity | None, engine_env: dict | None = None) -> BatchResult:
    ids = list(sample.ids)
    ids_path = work_dir / f"{sample.stem}.ids"
    out_path = work_dir / f"{sample.stem}.fresh.sess"
    started = int(time.time() * 1000)
    try:
        try:
            fresh = run_generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, args.timeout,
                                 engine_env)
        except VerifyError as exc:
            return _failed(source, sample.stem, build, started, exc, known_target)
        if known_target is None and fresh.identity.model != source.model:
            # The target identity is learned from this export: a foreign model would pin the wrong target, and
            # every later group of this model would then be skipped as "another model", hiding the failure.
            return _failed(source, sample.stem, build, started,
                           VerifyError("model_mismatch",
                                       f"the fresh export is model 0x{fresh.identity.model:016x}, "
                                       f"the source model is 0x{source.model:016x}"), None)
        try:
            old_digest = run_digest(digest_tool, sample.path, source, args.timeout)
            fresh_digest = run_digest(digest_tool, out_path, fresh.identity, args.timeout)
        except VerifyError as exc:
            return _failed(source, sample.stem, build, started, exc, fresh.identity)
        reason = compare_digests(old_digest, fresh_digest, sample.has_checkpoints)
        if reason:
            return _failed(source, sample.stem, build, started, VerifyError(reason), fresh.identity)
        entries = [GateEntry(item.stem, batch_id(source), item.bytes, len(item.ids), item.last_used)
                   for item in group]
        batch = GateBatch(batch_id(source), source, fresh.identity, build, sample.stem, True, started)
        return BatchResult(source, "pass", batch, entries)
    finally:
        if not args.keep_work:
            for path in (ids_path, out_path):
                try:
                    path.unlink()
                except OSError:
                    pass


def _failed(source: Identity, sample: str, build: str, started: int, error: VerifyError,
            target: Identity | None) -> BatchResult:
    batch = None
    if target is not None:
        batch = GateBatch(batch_id(source), source, target, build, sample, False, started, error.reason)
    return BatchResult(source, "fail", batch, [], error.detail or error.reason)


if __name__ == "__main__":
    sys.exit(main())
