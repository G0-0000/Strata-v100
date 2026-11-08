// The conversation-cache plugin's sidecar: the published session's directory and small state, in one
// self-describing byte string inside the share (ported from the disk-cache line's conversation_share.hpp,
// commit 941e1faf - the format the plugin's parser is built against, segment_sidecar.hpp).
//
// The share's segment table (shared_control.hpp, kinds 1-9) names only {offset, bytes, kind, index} spans, so
// everything a reader needs besides the spans themselves travels here: the geometry, the layer carve, the cvec
// flag, the live image keys, the three small running-state arrays (tails, dead, block_pos), one shape per
// published K/V layer, and the checkpoint chain's directory - one entry per checkpoint a save would write
// (session_checkpoints_to_save: the deepest one).  The chain's own bytes are not published here: the running
// state that is already in the share (session_active-class ids/gdn/ple) is published as kinds 1-3 rows whose
// `index` is 1 + the checkpoint's position in the directory (0 stays the live state's).
//
// The kind numbering and the sidecar's layout are the agreement with the plugin, which is built against this
// header: a field added here is a new version, not a compatible one.
//
// Host code only (no CUDA).  Header-only, like shared_arena.hpp.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/shared_arena.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

/// The sidecar's own magic, then the layout version: a block whose first word is not this is not ours to read.
inline constexpr uint64_t kSharedSidecarMagic = 0x5354524153494445ull;   ///< "STRASIDE"
inline constexpr uint32_t kSharedSidecarVersion = 1;

/// One K/V layer's shape, so a reader can cut the layer's spans into the five parts without a second source.
struct SharedKvShape {
    int64_t format = 0;       ///< ConversationKv::format
    int64_t cells = 0;        ///< ConversationKv::cells
    int64_t heads = 0;        ///< ConversationKv::heads
    int64_t head_dim = 0;     ///< ConversationKv::head_dim
    int64_t page_size = 0;    ///< ConversationKv::page_size
    int64_t pooled_rows = 0;  ///< ConversationKv::pooled_rows
    int64_t idx_dim = 0;      ///< ConversationKv::idx_dim
    uint64_t parts[5] = {};   ///< k, v, k_scale, v_scale, pooled: the bytes each part holds
};

/// One checkpoint of the chain, as the image holds it: the directory entry (what the checkpoint is), not its
/// bytes - the running state it holds is published as kinds 1-3 rows (see conversation_share_chain_publish).
struct SharedChainEntry {
    uint64_t tokens = 0;   ///< the checkpoint's token count (ids.size())
    uint64_t images = 0;   ///< its image keys (imgs.size())
    uint64_t used = 0;     ///< its LRU stamp
    uint32_t pinned = 0;   ///< holds a shared prefix (ConversationCheckpoint::pinned)
    uint32_t reserved = 0;
};

/// What one sidecar member holds.
enum SharedSidecarKind : uint32_t {
    kSidecarImages = 1,   ///< the live checkpoint's image keys, ConversationImageKey[bytes / 16]
    kSidecarTails = 2,    ///< the indexer tails, one per owned layer, back to back
    kSidecarDead = 3,     ///< the indexer dead rows, the same order
    kSidecarBlockPos = 4, ///< the indexer block positions, the same order
    kSidecarKvShape = 5,  ///< one SharedKvShape per K/V layer: index = the layer
    kSidecarChain = 6,    ///< the checkpoint chain's directory, SharedChainEntry[bytes / 32], index 0
};

/// One member: where it is inside the sidecar and what it holds.  `offset` is from the sidecar's own start, so
/// the reader needs nothing but the span the control block points at.
struct SharedSidecarMember {
    uint32_t kind = 0;    ///< a SharedSidecarKind
    uint32_t index = 0;   ///< which entry of that kind
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

/// The sidecar's first bytes, then `members` table entries, then the payload - each member 8-byte aligned.
struct SharedSidecarHeader {
    uint64_t magic = kSharedSidecarMagic;
    uint32_t version = kSharedSidecarVersion;
    uint32_t members = 0;      ///< filled entries of the table that follows this header
    uint64_t bytes = 0;        ///< the sidecar's whole length: header, table and payload
    uint64_t tokens = 0;       ///< the live image's token count (ids.size())
    uint64_t kv_layers = 0;    ///< published K/V layers: the session's QSA layers, then the draft layer
    uint64_t checkpoints = 0;  ///< the chain's length (kSidecarChain's entry count)
    int64_t layer_lo = 0;      ///< the session carve the image was captured from
    int64_t layer_hi = 0;
    uint32_t cvec = 0;         ///< the steering mode the image was captured with (1: cvec)
    uint32_t reserved32 = 0;
    int64_t geometry[18] = {}; ///< SavedConversation::geometry, field for field
    uint64_t ids_bytes = 0;    ///< the live state's spans, so a reader can check what it finds against the header
    uint64_t gdn_bytes = 0;
    uint64_t ple_bytes = 0;
};

static_assert(sizeof(SharedKvShape) == 96, "a plugin built against another copy of this header must fail here");
static_assert(sizeof(SharedChainEntry) == 32, "a plugin built against another copy of this header must fail here");
static_assert(sizeof(SharedSidecarMember) == 24, "a plugin built against another copy of this header must fail here");
static_assert(sizeof(SharedSidecarHeader) == 240, "a plugin built against another copy of this header must fail here");
static_assert(sizeof(SharedSidecarHeader) % 8 == 0, "the payload follows the header 8-byte aligned");

/// The sidecar as a reader walks it: the header, the member table, and the span they lie in.  Nothing here
/// writes, and every accessor refuses a block that is not this version's layout.
struct SharedSidecarView {
    const SharedSidecarHeader* header = nullptr;
    const SharedSidecarMember* members = nullptr;
    const uint8_t* base = nullptr;   ///< the sidecar's first byte (the span kSegmentSidecar points at)
    uint64_t bytes = 0;              ///< that span's published length

    /// False unless this is one of our sidecars and every member lies inside the span it came in.
    bool valid() const {
        if (header == nullptr || base == nullptr) return false;
        if (header->magic != kSharedSidecarMagic || header->version != kSharedSidecarVersion) return false;
        if (header->bytes == 0 || header->bytes > bytes) return false;
        const uint64_t table = sizeof(SharedSidecarHeader) + (uint64_t) header->members * sizeof(SharedSidecarMember);
        if (table > header->bytes || members == nullptr) return false;
        for (uint32_t i = 0; i < header->members; ++i) {
            const SharedSidecarMember& m = members[i];
            if (m.offset > header->bytes || m.bytes > header->bytes - m.offset) return false;
        }
        return true;
    }
    /// The member, or nullptr when the sidecar does not carry it.
    const SharedSidecarMember* find(uint32_t kind, uint32_t index) const {
        if (!valid()) return nullptr;
        for (uint32_t i = 0; i < header->members; ++i)
            if (members[i].kind == kind && members[i].index == index) return &members[i];
        return nullptr;
    }
    /// One member's bytes; nullptr when it is not there.
    const uint8_t* data(const SharedSidecarMember* member) const {
        return member == nullptr ? nullptr : base + member->offset;
    }
};

/// Read a published span as a sidecar.  False when the span is too small to hold a header.
inline bool shared_sidecar_view(const void* base, uint64_t bytes, SharedSidecarView& view) {
    view = {};
    if (base == nullptr || bytes < sizeof(SharedSidecarHeader)) return false;
    view.base = static_cast<const uint8_t*>(base);
    view.bytes = bytes;
    view.header = reinterpret_cast<const SharedSidecarHeader*>(view.base);
    view.members = reinterpret_cast<const SharedSidecarMember*>(view.base + sizeof(SharedSidecarHeader));
    return true;
}

/// The layout one sidecar publishes, from the members' byte counts alone: the engine sizes and carves the span
/// before its device copies know where they land, and the writer needs no second arithmetic.  The *_at offsets
/// are payload positions from the sidecar's own first byte; the *_bytes members are the logical lengths the
/// member table publishes (the align gaps between members are padding, never part of a member).
struct SharedSidecarLayout {
    uint64_t bytes = 0;          ///< the sidecar's whole length: header, member table and payload
    uint32_t members = 0;        ///< the member-table entries that follow the header
    uint64_t images_bytes = 0;   ///< each member's logical length, as the table publishes it
    uint64_t tails_bytes = 0;
    uint64_t dead_bytes = 0;
    uint64_t block_pos_bytes = 0;
    uint64_t images_at = 0;      ///< each member's payload offset, from the sidecar's first byte
    uint64_t tails_at = 0;
    uint64_t dead_at = 0;
    uint64_t block_pos_at = 0;
    uint64_t shapes_at = 0;
    uint64_t chain_at = 0;
};

/// The members, in the order they are written: the live checkpoint's small ones, one shape per K/V layer, then
/// the chain's directory.  One arithmetic for the size and for the write, so a reader's offsets cannot be the
/// second copy of it.
inline SharedSidecarLayout conversation_share_sidecar_layout(uint64_t images_bytes, uint64_t tails_bytes,
                                                             uint64_t dead_bytes, uint64_t block_pos_bytes,
                                                             uint64_t kv_layers, uint64_t checkpoints) {
    SharedSidecarLayout l;
    l.images_bytes = images_bytes;
    l.tails_bytes = tails_bytes;
    l.dead_bytes = dead_bytes;
    l.block_pos_bytes = block_pos_bytes;
    l.members = 4 + (uint32_t) kv_layers + (checkpoints != 0 ? 1u : 0u);
    uint64_t at = sizeof(SharedSidecarHeader) + (uint64_t) l.members * sizeof(SharedSidecarMember);
    auto place = [&at](uint64_t& slot, uint64_t bytes) {
        at = (at + 7) / 8 * 8;
        slot = at;
        at += bytes;
    };
    place(l.images_at, images_bytes);
    place(l.tails_at, tails_bytes);
    place(l.dead_at, dead_bytes);
    place(l.block_pos_at, block_pos_bytes);
    place(l.shapes_at, kv_layers * sizeof(SharedKvShape));
    place(l.chain_at, checkpoints * sizeof(SharedChainEntry));
    l.bytes = l.chain_at + checkpoints * sizeof(SharedChainEntry);
    return l;
}

/// What the writer puts into the header and the host-side members.  The three small arrays' payloads are the
/// engine's device copies' to write at the layout's offsets - they must be in place when a reader comes, not
/// when this fills.
struct SharedSidecarMeta {
    uint64_t tokens = 0;                 ///< the live ids' count (ids_bytes = tokens * 4)
    uint64_t gdn_bytes = 0;              ///< the live state's logical lengths, for the reader to check its spans
    uint64_t ple_bytes = 0;
    const ConversationImageKey* images = nullptr;
    uint64_t images_count = 0;
    const SharedKvShape* shapes = nullptr;
    uint64_t kv_layers = 0;
    const SharedChainEntry* chain = nullptr;
    uint64_t checkpoints = 0;
    int64_t layer_lo = 0, layer_hi = 0;
    uint32_t cvec = 0;
    std::array<int64_t, 18> geometry{};  ///< SavedConversation::geometry, field for field
};

/// Fill `out` (which needs `layout.bytes` bytes) with the header, the member table and the host-side members:
/// the image keys, one shape per K/V layer, the chain's directory.  The ids' logical length follows from
/// `tokens`; the three small arrays' payloads are left exactly as the device copies wrote them.
inline void conversation_share_sidecar_write(uint8_t* out, const SharedSidecarLayout& layout,
                                             const SharedSidecarMeta& meta) {
    auto* header = reinterpret_cast<SharedSidecarHeader*>(out);
    *header = SharedSidecarHeader{};
    header->members = layout.members;
    header->bytes = layout.bytes;
    header->tokens = meta.tokens;
    header->kv_layers = meta.kv_layers;
    header->checkpoints = meta.checkpoints;
    header->layer_lo = meta.layer_lo;
    header->layer_hi = meta.layer_hi;
    header->cvec = meta.cvec;
    for (size_t i = 0; i < meta.geometry.size() && i < 18; ++i) header->geometry[i] = meta.geometry[i];
    header->ids_bytes = meta.tokens * sizeof(int32_t);
    header->gdn_bytes = meta.gdn_bytes;
    header->ple_bytes = meta.ple_bytes;
    auto* table = reinterpret_cast<SharedSidecarMember*>(out + sizeof(SharedSidecarHeader));
    uint32_t n = 0;
    auto member = [&](uint32_t kind, uint32_t index, uint64_t offset, uint64_t bytes) {
        table[n].kind = kind;
        table[n].index = index;
        table[n].offset = offset;
        table[n].bytes = bytes;
        n += 1;
    };
    member(kSidecarImages, 0, layout.images_at, layout.images_bytes);
    member(kSidecarTails, 0, layout.tails_at, layout.tails_bytes);
    member(kSidecarDead, 0, layout.dead_at, layout.dead_bytes);
    member(kSidecarBlockPos, 0, layout.block_pos_at, layout.block_pos_bytes);
    for (uint64_t i = 0; i < meta.kv_layers; ++i)
        member(kSidecarKvShape, (uint32_t) i, layout.shapes_at + i * sizeof(SharedKvShape), sizeof(SharedKvShape));
    if (meta.checkpoints != 0) member(kSidecarChain, 0, layout.chain_at, meta.checkpoints * sizeof(SharedChainEntry));
    auto put = [&](uint64_t at, const void* src, uint64_t bytes) {
        if (bytes != 0) std::memcpy(out + at, src, (size_t) bytes);
    };
    put(layout.images_at, meta.images, layout.images_bytes);
    for (uint64_t i = 0; i < meta.kv_layers; ++i)
        put(layout.shapes_at + i * sizeof(SharedKvShape), meta.shapes + (size_t) i, sizeof(SharedKvShape));
    put(layout.chain_at, meta.chain, meta.checkpoints * sizeof(SharedChainEntry));
}

/// The chain checkpoint's publication: `entry` is filled with what the checkpoint IS (never fails), and, for
/// every running-state buffer the share holds - the session_active-class ids, gdn and ple - a kind 1/2/3 row is
/// recorded at `row_index` naming where it lies (page-rounded, like every published span).  No bytes are moved:
/// the rows name buffers that already lie in the share.  A buffer outside the share (its carve fell back to a
/// private mapping) gets no row, like the live ids; the tails/dead/block_pos a checkpoint holds are plain heap
/// vectors, not share storage, and are not published here.  False when the arena is not bound or the table has
/// no room for a row; the entry still describes the checkpoint either way.
inline bool conversation_share_chain_publish(const ConversationCheckpoint& c, uint32_t row_index,
                                             SharedChainEntry& entry, std::string& error) {
    entry = SharedChainEntry{};
    entry.tokens = c.ids.size();
    entry.images = c.imgs.size();
    entry.used = c.used;
    entry.pinned = c.pinned ? 1u : 0u;
    error.clear();
    if (shared_arena_control() == nullptr) {
        error = "the share arena is not bound";
        return false;
    }
    auto row = [&](uint32_t kind, const void* p, uint64_t bytes) {
        if (bytes == 0) return true;
        const uint64_t off = shared_arena_offset(p);
        if (off == kNoOffset) return true;   // not share storage: no row names it
        const uint64_t rounded = (bytes + kSharedArenaPageBytes - 1) & ~(kSharedArenaPageBytes - 1);
        return shared_arena_record_bytes(kind, row_index, off, rounded);
    };
    bool ok = true;
    auto fail = [&](const std::string& why) {
        if (ok) error = why;
        ok = false;
        return false;
    };
    if (!row(kSegmentIds, c.ids.data(), c.ids.size() * sizeof(int32_t)))
        return fail("the share's table has no room for the checkpoint's ids row");
    if (!row(kSegmentGdn, c.gdn.data(), c.gdn.size()))
        return fail("the share's table has no room for the checkpoint's GDN row");
    if (!row(kSegmentPle, c.ple.data(), c.ple.size()))
        return fail("the share's table has no room for the checkpoint's PLE row");
    return true;
}

/// The DUMP's kind-8 row for one K/V layer: the pooled indexer keys copied into their own share span at
/// DUMP time (the session build records no row for them - that part stays device-side, layer.cpp's pool
/// recording).  `offset` names the span the device copy lands in (page-aligned, like every carve of the
/// arena), `logical_bytes` the sidecar shape's part_bytes[4] - the length the plugin compares the row
/// against; the row publishes the page-rounded span, which is never shorter.  False when the arena is not
/// bound, the arguments name no span, or the table has no room for the row.
inline bool conversation_share_pooled_publish(uint32_t layer, uint64_t offset, uint64_t logical_bytes) {
    if (logical_bytes == 0 || offset == kNoOffset) return false;
    const uint64_t rounded = (logical_bytes + kSharedArenaPageBytes - 1) & ~(kSharedArenaPageBytes - 1);
    return shared_arena_record_bytes(kSegmentPooled, layer, offset, rounded);
}

}   // namespace strata::core
