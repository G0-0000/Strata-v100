// The engine's side of the plugin's shared segment as an allocator: one process-wide arena bound once at
// startup (generate.cpp, right after the share's memfd is mapped and checked), from which the current
// session's memory is carved - vma_alloc's session_active class, and the KV host pools that follow - so the
// session LIVES in the segment and a release publishes it where it already lies (no copy: the rows name
// offsets into pages the plugin maps too).
//
// A bump arena: no free, no reuse.  A session rebuild carves fresh bytes (the documented cost of one rebuild
// per process); a carve that does not fit returns kNoOffset and the caller keeps its own mapping - a session
// never fails mid-request because the share ran out, it only stops being published.  Single-threaded by
// construction: the carving happens while the session is built, before requests run.
//
// The rows the allocators record (shared_arena_record) live here until a release publishes them: the control
// block's table is written only by shared_arena_publish, which copies the rows in, counts them, counts the
// handover in the epoch and flips the state - after which the plugin reads the table.
//
// Host code only (no CUDA).  Header-only, like vma_alloc.hpp.
#pragma once

#include "strata/core/shared_control.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One page: every published slice starts page aligned (SharedControlSegment's contract), so this is the
/// arena's minimum alignment and the rounding the carve sites apply to their sizes.
inline constexpr uint64_t kSharedArenaPageBytes = 4096;

/// The first offset the arena hands out: behind the control block and the segment table (28672, the extent
/// shared_control.hpp's static_assert fixes).
inline constexpr uint64_t kSharedArenaFirstOffset =
    SharedControl::kTableOffset + (uint64_t) SharedControl::kTableCapacity * sizeof(SharedControlSegment);

/// shared_arena_alloc()'s answer when the arena is not bound or the bytes do not fit: the caller keeps its own
/// mapping and no row is recorded.  A valid offset is always < the share's bytes, so no real offset collides.
inline constexpr uint64_t kNoOffset = UINT64_MAX;

/// The process-wide arena.  One instance (shared_arena_instance), bound once by shared_arena_bind; every
/// accessor below reads that instance.
struct SharedArena {
    uint8_t* base = nullptr;           ///< the mapped share's first byte (the control block lives here)
    uint64_t bytes = 0;                ///< the share's size, as the plugin announced it
    SharedControl* control = nullptr;  ///< the control block at base: publish writes the table through it
    uint64_t bump = kSharedArenaFirstOffset;   ///< the next free offset (a bump arena: no free, no reuse)
    std::vector<SharedControlSegment> rows;    ///< the rows the allocators recorded, published on release
    bool bound = false;
};

inline SharedArena& shared_arena_instance() {
    static SharedArena arena;
    return arena;
}

/// Set once when any piece of the session could not be carved out of the share (a KV pool fell back to its own
/// mapping).  A session carrying such a piece must not be published.
inline bool& shared_arena_incomplete_flag() {
    static bool incomplete = false;
    return incomplete;
}
inline void shared_arena_mark_incomplete() { shared_arena_incomplete_flag() = true; }
inline void shared_arena_clear_incomplete() { shared_arena_incomplete_flag() = false; }
inline bool shared_arena_incomplete() { return shared_arena_incomplete_flag(); }

/// The bound arena, or null when none is bound: what a VmaAllocator's session_active cell holds.  A null cell
/// keeps one own mapping per allocation - the behaviour before the arena existed, byte for byte.
inline SharedArena* shared_arena_handle() {
    SharedArena& a = shared_arena_instance();
    return a.bound ? &a : nullptr;
}

/// Bind the process-wide arena to the mapped share: once, at startup, before anything allocates.  False (with
/// `error` saying why) on a second bind, a null argument, or a share too small to hold the control block and
/// the segment table - the same fail-fast as every other handshake failure.
inline bool shared_arena_bind(void* base, uint64_t bytes, SharedControl* control, std::string& error) {
    SharedArena& a = shared_arena_instance();
    if (a.bound) { error = "the share arena is already bound"; return false; }
    if (base == nullptr || control == nullptr) { error = "null base or control block"; return false; }
    if (bytes < kSharedArenaFirstOffset) {
        error = "the share is too small for the control block and the segment table (need at least " +
                std::to_string(kSharedArenaFirstOffset) + " bytes)";
        return false;
    }
    a.base = static_cast<uint8_t*>(base);
    a.bytes = bytes;
    a.control = control;
    a.bump = kSharedArenaFirstOffset;
    a.rows.clear();
    a.bound = true;
    shared_arena_incomplete_flag() = false;   // a fresh bind starts a fresh session
    return true;
}

/// Carve `bytes` at alignment `align` out of the bump: the offset to hand to shared_arena_ptr(), or kNoOffset
/// when the arena is not bound or the bytes do not fit.  Page alignment is enforced (a smaller `align` is
/// raised to it, a larger one is honored), so a slice is always a publishable row.
inline uint64_t shared_arena_alloc(uint64_t bytes, uint64_t align) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound) return kNoOffset;
    if (bytes == 0) bytes = 1;
    const uint64_t al = align < kSharedArenaPageBytes ? kSharedArenaPageBytes : align;
    if (al > a.bytes) return kNoOffset;   // keeps the align_up below overflow-free
    const uint64_t off = (a.bump + al - 1) / al * al;
    if (off > a.bytes || bytes > a.bytes - off) return kNoOffset;
    a.bump = off + bytes;
    return off;
}

/// The address of an offset the arena handed out: null when unbound or out of range.
inline void* shared_arena_ptr(uint64_t offset) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound || offset >= a.bytes) return nullptr;
    return a.base + offset;
}

/// The inverse of shared_arena_ptr: the offset of an address inside the bound arena, or kNoOffset.  A pointer
/// whose allocation fell back to a private mapping (the carve did not fit) lies outside, and so does null.
inline uint64_t shared_arena_offset(const void* p) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound || p == nullptr) return kNoOffset;
    const uint8_t* b = static_cast<const uint8_t*>(p);
    if (b < a.base || b >= a.base + a.bytes) return kNoOffset;
    return (uint64_t) (b - a.base);
}

/// The bound control block (publish writes the table through it), or null when no arena is bound.
inline SharedControl* shared_arena_control() {
    SharedArena& a = shared_arena_instance();
    return a.bound ? a.control : nullptr;
}

/// The bound share's size in bytes, or 0 when no arena is bound.
inline uint64_t shared_arena_bytes() {
    SharedArena& a = shared_arena_instance();
    return a.bound ? a.bytes : 0;
}

/// Record one row for the release to publish: what a slice IS - its offset and bytes, the kind (the plugin's
/// fixed numbering in shared_control.hpp) and its index, the layer for the K/V kinds.  False when the arena is
/// not bound or the rows already fill the table they will publish into: the row stays unrecorded and the
/// recorded ones are untouched.
inline bool shared_arena_record(uint64_t offset, uint64_t bytes, uint32_t kind, uint32_t index) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound || a.rows.size() >= SharedControl::kTableCapacity) return false;
    a.rows.push_back(SharedControlSegment{offset, bytes, kind, index});
    return true;
}

/// The same row, with the arguments in the order a caller states them - what the span is, then where and how
/// big.  One wrapper over shared_arena_record; its failure contract is the same.
inline bool shared_arena_record_bytes(uint32_t kind, uint32_t index, uint64_t offset, uint64_t bytes) {
    return shared_arena_record(offset, bytes, kind, index);
}

/// The share could not hold a piece of this session (a KV pool fell back to a private mapping): the session
/// must not be published, or a reader would restore a partial image and answer from stale K/V.
inline void shared_arena_mark_incomplete();
inline bool shared_arena_incomplete();

/// Drop the rows a DUMP records (the running state: ids, gdn, ple, sidecar, and the chain's kinds 1-3 rows at
/// index >= 1).  A request that computed without a dump leaves rows that name the previous session's bytes;
/// publishing them would hand the reader a stale image.  The live ids' row (kind 1, index 0) stays: the caller
/// re-points it at the session's own ids with shared_arena_set_row.
inline void shared_arena_erase_running_rows() {
    SharedArena& a = shared_arena_instance();
    if (!a.bound) return;
    a.rows.erase(std::remove_if(a.rows.begin(), a.rows.end(),
                                [](const SharedControlSegment& r) {
                                    return r.kind == kSegmentGdn || r.kind == kSegmentPle ||
                                           r.kind == kSegmentSidecar ||
                                           (r.kind == kSegmentIds && r.index != 0);
                                }),
                 a.rows.end());
}

/// Record or replace the one row for (kind, index): a caller that republishes the same region (the live ids move
/// with every request) must not grow the table by one row per publish.  False when the arena is not bound.
inline bool shared_arena_set_row(uint32_t kind, uint32_t index, uint64_t offset, uint64_t bytes) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound) return false;
    for (auto& row : a.rows) {
        if (row.kind == kind && row.index == index) {
            row.offset = offset;
            row.bytes = bytes;
            return true;
        }
    }
    return shared_arena_record(offset, bytes, kind, index);
}

/// Publish the recorded rows and hand the share over: the rows are copied into the control block's table (a
/// pure copy - no page of the session is touched, so a release costs microseconds), the plugin's geometry
/// numbers are re-asserted from the bound arena, the epoch counts the handover and the state says the engine
/// owns the share now.  False (with `error`) when no arena is bound.
inline bool shared_arena_publish(uint64_t tokens, std::string& error) {
    SharedArena& a = shared_arena_instance();
    if (!a.bound) { error = "the share arena is not bound"; return false; }
    if (shared_arena_incomplete()) {
        // Fail closed: a partial session is not offered.  An empty table with tokens 0 reads as "nothing to
        // restore", so the next request recomputes instead of resuming from stale K/V.
        a.control->bytes = a.bytes;
        a.control->table_offset = SharedControl::kTableOffset;
        a.control->table_capacity = SharedControl::kTableCapacity;
        a.control->segment_count = 0;
        a.control->tokens = 0;
        a.control->epoch += 1;
        std::atomic_thread_fence(std::memory_order_release);
        a.control->state = SharedControl::kStateEngine;
        return true;
    }
    if (a.rows.size() > a.control->table_capacity) {
        error = "more recorded rows than the segment table holds";
        return false;
    }
    a.control->bytes = a.bytes;
    a.control->table_offset = SharedControl::kTableOffset;
    a.control->table_capacity = SharedControl::kTableCapacity;
    auto* table = reinterpret_cast<SharedControlSegment*>(a.base + SharedControl::kTableOffset);
    for (size_t i = 0; i < a.rows.size(); ++i) table[i] = a.rows[i];
    a.control->segment_count = static_cast<uint32_t>(a.rows.size());
    a.control->tokens = tokens;   // how many tokens the published session holds
    a.control->epoch += 1;
    std::atomic_thread_fence(std::memory_order_release);   // the rows and the count are visible before the handover
    a.control->state = SharedControl::kStateEngine;
    return true;
}

/// Unbind and forget every recorded row and carved offset.  Tests only: production binds once, at startup,
/// and never resets (the mapping outlives the process's work).
inline void shared_arena_reset() {
    SharedArena& a = shared_arena_instance();
    a.base = nullptr;
    a.bytes = 0;
    a.control = nullptr;
    a.bump = kSharedArenaFirstOffset;
    a.rows.clear();
    a.bound = false;
    shared_arena_incomplete_flag() = false;   // a reset arena starts a new session
}

}   // namespace strata::core
