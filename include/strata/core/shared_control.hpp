// The conversation cache plugin's share control block (protocol v3, S1 skeleton).  The plugin builds the share
// (a memfd it maps at kPluginShareBase) and owns the numbers it fills in here; the engine maps the same memfd
// at the same base and reads this block first - a wrong magic or version refuses the run (exit 2).  The first
// kControlBytes bytes are this block, then the segment table (kTableCapacity rows of SharedControlSegment),
// then the segments themselves.  All layout constants live here so both sides compile the same numbers.
//
// Host code only (no CUDA).
#pragma once

#include <cstdint>

namespace strata::core {

// The fixed base the plugin announces (hello's share_base) and the engine maps the memfd at.  Free in this
// tree today (no other mapping at it; the engine maps with MAP_FIXED_NOREPLACE and refuses a relocation).
constexpr uint64_t kPluginShareBase = 0x600000000000ull;

// A segment table row.  S1 writes none (no segment moves yet - the skeleton only fixes the layout so the
// table's extent never moves in later steps).
struct SharedControlSegment {
    uint64_t offset;  // where the segment's payload starts, in bytes from the share base
    uint64_t bytes;   // the segment's payload size
    uint32_t kind;    // what the span holds (ids, gdn, ple, K/V part, sidecar)
    uint32_t index;   // which entry of that kind (the layer for K/V)
};

struct SharedControl {
    static constexpr uint64_t kMagic = 0x5354524154410001ull;   // "STRATA" 0x00 0x01
    static constexpr uint32_t kVersion = 3;
    static constexpr uint32_t kStatePlugin = 0;   // the plugin owns the share (the engine waits)
    static constexpr uint32_t kStateEngine = 1;   // the engine owns it (the plugin waits)
    static constexpr uint64_t kControlBytes = 4096;   // this block: the first bytes of the share
    static constexpr uint64_t kTableOffset = 4096;    // the segment table starts right behind the block
    static constexpr uint32_t kTableCapacity = 1024;  // segment rows the table holds (24 bytes each)

    uint64_t magic;
    uint32_t version;
    uint32_t state;             // kStatePlugin while the plugin works in the share, kStateEngine when handed over
    uint64_t epoch;             // the plugin's handover counter (acquire/release pairs with it, later steps)
    uint64_t bytes;             // the share's total size (hello's share_bytes repeats it)
    uint64_t table_offset;      // kTableOffset, written by the plugin so the engine need not assume
    uint32_t table_capacity;    // kTableCapacity, same
    uint32_t segment_count;     // live rows in the table right now
    uint64_t tokens;            // the tokens the share's segments cover, as the plugin counted them
    uint8_t identity[32];       // the model/config identity of the share (S1 leaves it zeroed)
    uint8_t reserved[4008];
};

static_assert(sizeof(SharedControl) == 4096, "the control block is the share's first 4096 bytes");
static_assert(sizeof(SharedControlSegment) == 24, "one segment row, three pointers' worth");
static_assert(SharedControl::kTableOffset + (uint64_t) SharedControl::kTableCapacity * sizeof(SharedControlSegment)
                  == 28672,
              "the segment table ends at 28672; the segments themselves start there");

}   // namespace strata::core
