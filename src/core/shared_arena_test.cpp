// Host-only tests for the plugin share as the session's arena (shared_arena.hpp + vma_alloc.hpp): a memfd
// mapped at the announced base binds the process-wide arena once (a second bind refuses), the page-aligned
// bump hands out publishable slices from behind the segment table, a session_active VmaAllocator carves from
// the share while conv_cache keeps its own named mapping, the recorded rows land in the control block's table
// on the plugin's 24-byte layout with the plugin's kind numbers, and an arena that ran out falls back to the
// kernel mapping - which deallocate still unmaps - without publishing a row.  CPU only: no CUDA, no model, a
// mebibyte of RAM.
#include "strata/core/shared_arena.hpp"
#include "strata/core/vma_alloc.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE   // kernel 4.17+; an older toolchain's headers may lack the name
#define MAP_FIXED_NOREPLACE 0x100000
#endif

using strata::core::SharedControl;
using strata::core::SharedControlSegment;
using strata::core::VmaAllocator;
namespace vma_class = strata::core::vma_class;

namespace {
int checks = 0, fails = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// The [anon:<name>] entry in /proc/self/maps, when there is one.
bool mapped_with_name(const char* name) {
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (f == nullptr) return false;
    char line[1024];
    bool found = false;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::strstr(line, name) != nullptr) { found = true; break; }
    std::fclose(f);
    return found;
}

// The share's mapping is one contiguous range [base, base+bytes): a munmap inside it (a deallocate that
// forgot the arena owns its slices) would split it in two.
bool share_mapping_whole(uintptr_t base, uint64_t bytes) {
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (f == nullptr) return false;
    char line[1024];
    bool ok = false;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        unsigned long long lo = 0, hi = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) != 2) continue;
        if (lo != base) continue;
        ok = (uint64_t) hi == base + bytes;
        break;
    }
    std::fclose(f);
    return ok;
}

// The share a fake plugin would have built: one memfd with a written control block, mapped at the announced
// base with MAP_FIXED_NOREPLACE - exactly what generate.cpp does after the hello.
struct ShareMap {
    int fd = -1;
    void* base = nullptr;
    SharedControl* ctl = nullptr;
    uint64_t bytes = 0;

    bool make(uint64_t share_bytes) {
        bytes = share_bytes;
        fd = ::memfd_create("strata-shared-arena-test", MFD_CLOEXEC);
        if (fd < 0) return false;
        if (::ftruncate(fd, (off_t) bytes) != 0) return false;
        SharedControl c{};
        c.magic = SharedControl::kMagic;
        c.version = SharedControl::kVersion;
        c.state = SharedControl::kStatePlugin;
        c.bytes = bytes;
        c.table_offset = SharedControl::kTableOffset;
        c.table_capacity = SharedControl::kTableCapacity;
        if (::pwrite(fd, &c, sizeof(c), 0) != (ssize_t) sizeof(c)) return false;
        base = ::mmap(reinterpret_cast<void*>(static_cast<uintptr_t>(strata::core::kPluginShareBase)), bytes,
                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
        if (base == MAP_FAILED) return false;
        if (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)) != strata::core::kPluginShareBase) return false;
        ctl = static_cast<SharedControl*>(base);
        return true;
    }
    void drop() {
        if (base != nullptr && base != MAP_FAILED) ::munmap(base, bytes);
        if (fd >= 0) ::close(fd);
        fd = -1;
        base = nullptr;
        ctl = nullptr;
    }
};

}   // namespace

int main() {
    const uint64_t share_bytes = 1024 * 1024;   // 1 MiB: the table plus room for a handful of slices

    // --- the kind numbers are the plugin's, verbatim (its session_encode.hpp fixes them) ---
    CHECK(strata::core::kSegmentIds == 1 && strata::core::kSegmentGdn == 2 && strata::core::kSegmentPle == 3 &&
          strata::core::kSegmentK == 4 && strata::core::kSegmentV == 5 && strata::core::kSegmentKScale == 6 &&
          strata::core::kSegmentVScale == 7 && strata::core::kSegmentPooled == 8 &&
          strata::core::kSegmentSidecar == 9);

    // --- the memfd maps at the announced base and binds the arena once; a second bind refuses ---
    ShareMap share;
    CHECK(share.make(share_bytes));
    std::string err;
    CHECK(strata::core::shared_arena_bind(share.base, share.bytes, share.ctl, err));
    CHECK(err.empty());
    CHECK(strata::core::shared_arena_bytes() == share_bytes);
    CHECK(strata::core::shared_arena_control() == share.ctl);
    CHECK(strata::core::shared_arena_handle() != nullptr);
    std::string err2;
    CHECK(!strata::core::shared_arena_bind(share.base, share.bytes, share.ctl, err2));   // twice: refused
    CHECK(!err2.empty());
    CHECK(strata::core::shared_arena_bytes() == share_bytes);   // the first binding stands

    // --- the bump: page-aligned offsets from behind the segment table, pointers into the share's pages ---
    const uint64_t o1 = strata::core::shared_arena_alloc(100, 4096);
    const uint64_t o2 = strata::core::shared_arena_alloc(100, 16);   // a smaller align is raised to the page
    CHECK(o1 == strata::core::kSharedArenaFirstOffset);              // the first slice starts at 28672
    CHECK((o1 & 4095) == 0 && (o2 & 4095) == 0);                     // page aligned, a publishable row
    CHECK(o2 == o1 + 4096);                                          // ...right behind the first slice's page
    auto* p1 = static_cast<uint8_t*>(strata::core::shared_arena_ptr(o1));
    CHECK(p1 == static_cast<uint8_t*>(share.base) + o1);             // the offset is from the share's base
    p1[0] = 0x5a;
    p1[4095] = 0xa5;
    auto* p1_back = static_cast<const uint8_t*>(strata::core::shared_arena_ptr(o1));
    CHECK(p1_back[0] == 0x5a && p1_back[4095] == 0xa5);   // the bytes are really the share's
    CHECK(strata::core::shared_arena_alloc(share_bytes, 4096) == strata::core::kNoOffset);   // does not fit

    // --- a session_active allocator carves from the share; conv_cache keeps its own named mapping ---
    const uintptr_t lo = reinterpret_cast<uintptr_t>(share.base);
    {
        std::vector<int32_t, VmaAllocator<int32_t>> act{VmaAllocator<int32_t>(vma_class::session_active)};
        act.resize(64, 7);   // 256 bytes: one page-rounded slice of the arena
        CHECK(act.size() == 64 && act[63] == 7);
        const uintptr_t a = reinterpret_cast<uintptr_t>(act.data());
        CHECK(a >= lo + strata::core::kSharedArenaFirstOffset && a < lo + share_bytes);   // inside the share
        CHECK((a & 4095) == 0);
        std::vector<uint8_t, VmaAllocator<uint8_t>> parked{VmaAllocator<uint8_t>(vma_class::conv_cache)};
        parked.resize(8192, 0x33);
        const uintptr_t pk = reinterpret_cast<uintptr_t>(parked.data());
        CHECK(pk < lo || pk >= lo + share_bytes);   // outside the share: an own anonymous mapping
        CHECK(mapped_with_name("[anon:strata/conv_cache]"));
        CHECK(!mapped_with_name("[anon:strata/session_active]"));   // the carve is in the (unnameable) share
    }
    CHECK(share_mapping_whole(lo, share_bytes));   // the arena slice went back as a no-op, the share is whole

    // --- the rows: recorded at carve time, published into the table on the plugin's 24-byte layout ---
    CHECK(strata::core::shared_arena_record(o1, 100, strata::core::kSegmentIds, 0));
    CHECK(strata::core::shared_arena_record(o2, 4096, strata::core::kSegmentK, 3));   // kind 4, layer 3
    std::string perr;
    CHECK(strata::core::shared_arena_publish(11, perr));
    CHECK(perr.empty());
    CHECK(share.ctl->state == SharedControl::kStateEngine);
    CHECK(share.ctl->epoch == 1);   // the plugin wrote 0; the publish counted the handover
    CHECK(share.ctl->segment_count == 2);
    CHECK(share.ctl->bytes == share_bytes && share.ctl->table_offset == SharedControl::kTableOffset &&
          share.ctl->table_capacity == SharedControl::kTableCapacity);
    const auto* table = reinterpret_cast<const SharedControlSegment*>(
        static_cast<const uint8_t*>(share.base) + SharedControl::kTableOffset);
    CHECK(table[0].offset == o1 && table[0].bytes == 100 && table[0].kind == 1 && table[0].index == 0);
    CHECK(table[1].offset == o2 && table[1].bytes == 4096 && table[1].kind == 4 && table[1].index == 3);
    SharedControlSegment raw{};   // the layout, byte for byte: row 1 sits exactly 24 bytes into the table
    std::memcpy(&raw, static_cast<const uint8_t*>(share.base) + SharedControl::kTableOffset + sizeof(raw),
                sizeof(raw));
    CHECK(raw.offset == o2 && raw.bytes == 4096 && raw.kind == 4 && raw.index == 3);


    // --- the inverse of shared_arena_ptr: the offset of an address inside the arena, kNoOffset outside it ---
    CHECK(strata::core::shared_arena_offset(p1) == o1);
    CHECK(strata::core::shared_arena_offset(static_cast<uint8_t*>(share.base) + share_bytes) ==
          strata::core::kNoOffset);   // one past the share's last byte: outside
    CHECK(strata::core::shared_arena_offset(&share_bytes) == strata::core::kNoOffset);   // an unrelated address
    CHECK(strata::core::shared_arena_offset(nullptr) == strata::core::kNoOffset);

    // --- the table append helper: rows land at segment_count, and a full table refuses with the count kept ---
    CHECK(strata::core::shared_control_append_segment(*share.ctl, o1, 4096, strata::core::kSegmentV, 7));
    CHECK(share.ctl->segment_count == 3);
    CHECK(table[2].offset == o1 && table[2].bytes == 4096 && table[2].kind == 5 && table[2].index == 7);
    bool filled = true;
    while (share.ctl->segment_count < SharedControl::kTableCapacity)
        filled = filled && strata::core::shared_control_append_segment(*share.ctl, 0, 4096,
                                                                       strata::core::kSegmentSidecar, 0);
    CHECK(filled);
    CHECK(share.ctl->segment_count == SharedControl::kTableCapacity);
    CHECK(!strata::core::shared_control_append_segment(*share.ctl, 0, 4096, strata::core::kSegmentSidecar, 0));
    CHECK(share.ctl->segment_count == SharedControl::kTableCapacity);   // the refusal left the count unchanged

    // --- the arena's own record refuses the same way once the rows it will publish fill the table ---
    while (strata::core::shared_arena_record(0, 4096, strata::core::kSegmentSidecar, 0)) {}
    const uint64_t capacity_epoch = share.ctl->epoch;
    CHECK(strata::core::shared_arena_publish(11, perr));
    CHECK(share.ctl->segment_count == SharedControl::kTableCapacity);
    CHECK(share.ctl->epoch == capacity_epoch + 1);

    // --- reset: tests can unbind and bind again; production never does ---
    strata::core::shared_arena_reset();
    CHECK(strata::core::shared_arena_bytes() == 0);
    CHECK(strata::core::shared_arena_control() == nullptr);
    CHECK(strata::core::shared_arena_handle() == nullptr);
    CHECK(strata::core::shared_arena_alloc(4096, 4096) == strata::core::kNoOffset);
    CHECK(strata::core::shared_arena_record(0, 4096, strata::core::kSegmentIds, 0) == false);
    CHECK(strata::core::shared_arena_offset(share.base) == strata::core::kNoOffset);   // unbound: nothing is inside
    CHECK(strata::core::shared_arena_bind(share.base, share.bytes, share.ctl, err));
    CHECK(strata::core::shared_arena_alloc(4096, 4096) == strata::core::kSharedArenaFirstOffset);

    // --- record_bytes: the same row record takes, with the arguments in the order a caller states them ---
    CHECK(strata::core::shared_arena_record_bytes(strata::core::kSegmentIds, 0, 4096, 100));
    CHECK(strata::core::shared_arena_publish(11, perr));
    CHECK(share.ctl->segment_count == 1);
    CHECK(table[0].offset == 4096 && table[0].bytes == 100 && table[0].kind == 1 && table[0].index == 0);

    // --- an arena that ran out: the fallback is the kernel's mapping - named, and munmapped by deallocate -
    // and no row is published for it ---
    {
        std::vector<uint8_t, VmaAllocator<uint8_t>> big{VmaAllocator<uint8_t>(vma_class::session_active)};
        big.resize(share_bytes, 1);   // the whole share cannot fit behind the bump
        const uintptr_t bg = reinterpret_cast<uintptr_t>(big.data());
        CHECK(bg < lo || bg >= lo + share_bytes);   // outside the share: an own anonymous mapping
        CHECK(mapped_with_name("[anon:strata/session_active]"));
        // Fail closed: a session with a fallback mapping is not published at all (empty table, tokens 0),
        // so a reader cannot resume from a partial image.
        const uint64_t epoch_before = share.ctl->epoch;
        CHECK(strata::core::shared_arena_publish(11, perr));
        CHECK(share.ctl->segment_count == 0 && share.ctl->tokens == 0 &&
              share.ctl->epoch == epoch_before + 1);
    }
    CHECK(!mapped_with_name("[anon:strata/session_active]"));   // deallocate still munmaps the fallback
    CHECK(share_mapping_whole(lo, share_bytes));                // and never touched the share
    strata::core::shared_arena_clear_incomplete();               // the next case starts a complete session

    share.drop();
    if (fails != 0) std::fprintf(stderr, "shared_arena_test: %d of %d checks FAILED\n", fails, checks);
    else std::fprintf(stderr, "shared_arena_test: %d checks passed\n", checks);
    return fails ? 1 : 0;
}
