// VMA naming (vma_alloc.hpp): a mapping the process owns shows as [anon:<name>] in /proc/self/maps - the one
// identity a RAM reader outside the process can see - a named-allocator vector really IS that mapping (one
// allocation, page aligned, name set before the first touch), vma_rename() flips a live buffer's class and the
// class follows the buffer (a later reallocation is named with the new one), and a failed mapping is a
// std::bad_alloc that leaves the operation's inputs whole (the split/save rollback path, mirroring
// conversation_split_failure_test's operator-new sweep on the heap path).
// The parked K/V storage (ConversationBuffer, the plan's main target) is checked at the end.
// CPU only: no device, no CUDA, a few pages of RAM.
#include "strata/core/conversation_buffer.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/vma_alloc.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::ConversationCheckpoint;
using strata::core::VmaAllocator;
namespace vma_class = strata::core::vma_class;

namespace {
int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// The [anon:<name>] (or [anon_shmem:<name>]) entry in /proc/self/maps, when there is one.
bool mapped_with_name(const char* name, std::string* line_out = nullptr) {
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (f == nullptr) return false;
    char line[1024];
    bool found = false;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        if (std::strstr(line, name) == nullptr) continue;
        found = true;
        if (line_out != nullptr) *line_out = line;
        break;
    }
    std::fclose(f);
    return found;
}

// Every field of a checkpoint, its stage parts recursively: the rollback assertion of the failure sweep.
bool equal_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails ||
        a.dead != b.dead || a.block_pos != b.block_pos || a.used != b.used || a.pinned != b.pinned ||
        a.stage_parts.size() != b.stage_parts.size()) return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!equal_checkpoint(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
} // namespace

int main() {
    // --- a mapping this process owns, named by hand ---
    const size_t bytes = 2 * 4096;
    void* raw = strata::core::vma_detail::map_bytes(bytes);
    if (raw == nullptr) { std::printf("vma_alloc_test: cannot map %zu bytes\n", bytes); return 1; }
    if (!strata::core::vma_name(raw, bytes, "strata/test")) {
        std::printf("vma_alloc_test: this kernel has no PR_SET_VMA_ANON_NAME (or refused it) - skipped\n");
        strata::core::vma_detail::unmap_bytes(raw, bytes);
        return 0;
    }
    CHECK(mapped_with_name("[anon:strata/test]"));
    ((volatile char*) raw)[0] = 1;              // faulted pages keep the name
    ((volatile char*) raw)[bytes - 1] = 1;
    CHECK(mapped_with_name("[anon:strata/test]"));
    strata::core::vma_detail::unmap_bytes(raw, bytes);
    CHECK(!mapped_with_name("[anon:strata/test]"));   // the name goes with the mapping

    // --- the allocator's storage is that mapping: round-trip, page aligned, named at allocation ---
    {
        std::vector<uint8_t, VmaAllocator<uint8_t>> buf{VmaAllocator<uint8_t>(vma_class::conv_cache)};
        buf.resize(3 * 4096, 0xab);
        CHECK(buf.capacity() * sizeof(uint8_t) >= 3 * 4096);
        CHECK(((uintptr_t) buf.data() & 4095) == 0);   // the base the kernel requires of a naming call
        CHECK(mapped_with_name("[anon:strata/conv_cache]"));
        uint64_t sum = 0;
        for (uint8_t v : buf) sum += v;                // the bytes are really ours
        CHECK(sum == 3 * 4096 * 0xab);
    }
    CHECK(!mapped_with_name("[anon:strata/conv_cache]"));   // destructor gives the mapping back

    // --- vma_rename flips a live buffer between classes (the parked -> active transition) ---
    {
        std::vector<uint8_t, VmaAllocator<uint8_t>> buf{VmaAllocator<uint8_t>(vma_class::conv_cache)};
        buf.resize(8192, 7);
        CHECK(mapped_with_name("[anon:strata/conv_cache]"));
        CHECK(strata::core::vma_rename(buf, vma_class::session_active));
        std::string line;
        CHECK(mapped_with_name("[anon:strata/session_active]", &line));
        CHECK(line.find("conv_cache") == std::string::npos);   // renamed, not named twice
        CHECK(strata::core::vma_rename(buf, vma_class::session_active));   // naming to the current class succeeds
        CHECK(strata::core::vma_rename(buf, "strata/other"));              // any static name, not only the class table's
        CHECK(mapped_with_name("[anon:strata/other]"));
    }
    CHECK(!mapped_with_name("[anon:strata/session_active]"));

    // --- after a rename the class FOLLOWS the buffer: a later reallocation is named with the new class, never
    // the one the allocator was constructed with ---
    {
        std::vector<uint8_t, VmaAllocator<uint8_t>> buf{VmaAllocator<uint8_t>("strata/rename-src")};
        buf.resize(4096, 3);
        CHECK(mapped_with_name("[anon:strata/rename-src]"));
        CHECK(strata::core::vma_rename(buf, "strata/rename-dst"));
        CHECK(mapped_with_name("[anon:strata/rename-dst]"));
        buf.reserve(buf.capacity() * 4);   // beyond capacity: a fresh mapping, named by the vector's own allocator
        std::string line;
        CHECK(mapped_with_name("[anon:strata/rename-dst]", &line));
        CHECK(line.find("rename-src") == std::string::npos);   // the new mapping did not take the old class
        CHECK(!mapped_with_name("[anon:strata/rename-src]"));
        CHECK(buf.size() == 4096 && buf[0] == 3 && buf[4095] == 3);   // the payload came across
    }
    CHECK(!mapped_with_name("[anon:strata/rename-dst]"));

    // --- the production rename walks the same cell: the checkpoint chain's own helper, buffers empty at the
    // rename included ---
    {
        ConversationCheckpoint c;   // a parked checkpoint: the default `strata/conv_cache` class
        c.ids.assign({1, 2, 3});
        CHECK(mapped_with_name("[anon:strata/conv_cache]"));
        strata::core::conversation_checkpoint_rename(c, vma_class::session_active);
        CHECK(!mapped_with_name("[anon:strata/conv_cache]"));   // the ids mapping was re-tagged in place
        c.ids.reserve(4096);   // growth: the class followed the rename
        c.gdn.resize(1024);    // empty at the rename: its first mapping still takes the new class
        CHECK(mapped_with_name("[anon:strata/session_active]"));
        CHECK(!mapped_with_name("[anon:strata/conv_cache]"));
    }
    CHECK(!mapped_with_name("[anon:strata/session_active]"));

    // --- the shape a checkpoint's token list uses: an int32 vector that outgrows its first mapping ---
    {
        std::vector<int32_t, VmaAllocator<int32_t>> ids{VmaAllocator<int32_t>("strata/ids")};
        ids.assign({1, 2, 3});
        CHECK(mapped_with_name("[anon:strata/ids]"));
        CHECK(ids.size() == 3 && ids[0] == 1 && ids[2] == 3);
        ids.resize(4096, -5);                    // growth reallocates: a new mapping, named too (the name is the
        CHECK(ids[0] == 1 && ids[4095] == -5);   // allocator's, so it survives every reallocation)
        CHECK(mapped_with_name("[anon:strata/ids]"));
    }
    CHECK(!mapped_with_name("[anon:strata/ids]"));

    // --- the parked K/V storage: a ConversationBuffer segment (the plan's main marking target) ---
    {
        strata::core::ConversationBuffer buf;
        buf.resize(4096, 9);
        CHECK(mapped_with_name("[anon:strata/conv_cache]"));
        CHECK(buf.bytes() >= 4096);
        buf.resize(0);                 // the segment goes back to the kernel: the name goes with it
        CHECK(!mapped_with_name("[anon:strata/conv_cache]"));
    }

    // --- a failed mapping is a std::bad_alloc (the allocator's contract), not a half-built container ---
    {
        std::vector<uint8_t, VmaAllocator<uint8_t>> buf{VmaAllocator<uint8_t>("strata/fail")};
        strata::core::vma_detail::map_fail_after() = 1;   // the next mapping call fails
        bool threw = false;
        try {
            buf.resize(4096, 1);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        strata::core::vma_detail::map_fail_after() = 0;
        CHECK(threw);
        CHECK(buf.capacity() == 0);   // nothing half-allocated
        CHECK(!mapped_with_name("[anon:strata/fail]"));
    }

    // --- the split/save rollback path with the mmap-backed members: arm the k-th mapping call to fail and run
    // conversation_checkpoints_split - either it goes through in full (and merge() gives every input back in
    // order), or the failure leaves every input checkpoint untouched.  The mmap twin of
    // conversation_split_failure_test's operator-new sweep, which still covers the heap members. ---
    {
        int faults = 0, swept = 0;
        bool finished = false;
        for (int k = 1; k < 128 && !finished; ++k) {
            std::vector<ConversationCheckpoint> checks(4);
            for (size_t i = 0; i < checks.size(); ++i) {
                auto& c = checks[i];
                c.ids.assign({1, 2, (int32_t) i + 3});
                c.gdn.assign({1, 2, 3});
                c.ple.assign({4});
                c.used = (uint64_t) i;
                if (i == 2) continue;   // incomplete: no stage parts (it stays in `rest`)
                c.stage_parts.resize(3);
                for (size_t j = 0; j < 3; ++j) {
                    c.stage_parts[j].ids.assign(c.ids.begin(), c.ids.end());
                    c.stage_parts[j].gdn.assign({5, 6, (uint8_t) (7 + j)});
                }
            }
            const auto backup = checks;
            strata::core::vma_detail::map_fail_after() = (uint64_t) k;
            try {
                auto split = strata::core::conversation_checkpoints_split(std::move(checks), 3);
                strata::core::vma_detail::map_fail_after() = 0;   // the merge back may not allocate
                if (!strata::core::conversation_checkpoints_merge(std::move(split), checks)) { CHECK(false); break; }
                finished = true;   // the split went through: every input is back, in order
                swept = k;
                // (each stage part's identity is its parent's by design, so only the state is compared)
                bool state_back = checks.size() == backup.size();
                for (size_t i = 0; state_back && i < checks.size(); ++i) {
                    const auto& x = checks[i];
                    const auto& y = backup[i];
                    state_back = x.ids == y.ids && x.gdn == y.gdn && x.ple == y.ple && x.used == y.used &&
                                 x.stage_parts.size() == y.stage_parts.size();
                    for (size_t j = 0; state_back && j < x.stage_parts.size(); ++j)
                        state_back = x.stage_parts[j].ids == y.stage_parts[j].ids &&
                                     x.stage_parts[j].gdn == y.stage_parts[j].gdn;
                }
                CHECK(state_back);
            } catch (const std::bad_alloc&) {
                strata::core::vma_detail::map_fail_after() = 0;
                ++faults;
                bool intact = checks.size() == backup.size();
                for (size_t i = 0; intact && i < checks.size(); ++i) intact = equal_checkpoint(checks[i], backup[i]);
                if (!intact)
                    std::fprintf(stderr, "vma_alloc_test: mapping failure %d partially moved a checkpoint\n", k);
                CHECK(intact);
            }
        }
        CHECK(finished);      // some k let the split through: the sweep is complete
        CHECK(faults > 0);    // and the mapping failures really landed on the named members
        std::printf("vma_alloc_test: split sweep: %d mapping-failure cases rolled back whole (the split went through at k=%d)\n",
                    faults, swept);
    }

    std::printf("vma_alloc_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
