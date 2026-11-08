// Host-only tests for the plugin share's sidecar (conversation_share.hpp): the format the plugin's parser
// (its segment_sidecar.hpp) is built against, ported from the disk-cache line's conversation_share (commit
// 941e1faf).  The magic and version are pinned against the plugin's own literals, the layout arithmetic puts
// every member 8-byte aligned behind one header and member table, the writer fills the header, the table and
// the host-side members while leaving the three device-copied payloads alone, and a dumped chain publishes
// rows (kinds 1-3, index 1) that name the checkpoint's share-resident ids/gdn/ple at the right offsets and
// sizes - no bytes moved.  CPU only: no CUDA, no model, a mebibyte of RAM.
#include "strata/core/conversation_share.hpp"
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

using strata::core::ConversationCheckpoint;
using strata::core::ConversationImageKey;
using strata::core::SharedChainEntry;
using strata::core::SharedControl;
using strata::core::SharedKvShape;
using strata::core::SharedSidecarLayout;
using strata::core::SharedSidecarMember;
using strata::core::SharedSidecarView;
using strata::core::VmaAllocator;
namespace vma_class = strata::core::vma_class;

namespace {
int checks = 0, fails = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// The share a fake plugin would have built: one memfd with a written control block, mapped at the announced
// base with MAP_FIXED_NOREPLACE - exactly what generate.cpp does after the hello.
struct ShareMap {
    int fd = -1;
    void* base = nullptr;
    SharedControl* ctl = nullptr;
    uint64_t bytes = 0;

    bool make(uint64_t share_bytes) {
        bytes = share_bytes;
        fd = ::memfd_create("strata-conversation-share-test", MFD_CLOEXEC);
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

const SharedSidecarMember* find(const SharedSidecarView& v, uint32_t kind, uint32_t index) { return v.find(kind, index); }

}   // namespace

int main() {
    // --- the format is the one the plugin's parser (its segment_sidecar.hpp) is built against ---
    CHECK(strata::core::kSharedSidecarMagic == 0x5354524153494445ull);   // "STRASIDE", the plugin's kSidecarMagic
    CHECK(strata::core::kSharedSidecarVersion == 1);                     // the plugin's kSidecarVersion
    CHECK(sizeof(SharedKvShape) == 96);                                  // the plugin's SidecarKvLayer
    CHECK(sizeof(SharedChainEntry) == 32);                               // the plugin's SidecarChainEntry
    CHECK(sizeof(strata::core::SharedSidecarHeader) == 240);             // the plugin's SidecarHeader
    CHECK(sizeof(strata::core::SharedSidecarMember) == 24);              // the plugin's SidecarMember
    CHECK(strata::core::kSidecarImages == 1 && strata::core::kSidecarTails == 2 &&
          strata::core::kSidecarDead == 3 && strata::core::kSidecarBlockPos == 4 &&
          strata::core::kSidecarKvShape == 5 && strata::core::kSidecarChain == 6);
    CHECK(sizeof(ConversationImageKey) == 16);   // the plugin's SidecarImage

    // --- the layout: one header, one member table, every member 8-byte aligned, exact byte count ---
    // images 32 (two keys), tails 8, dead 8, block_pos 4 (a 4-mod-8 member start), 1 shape, 1 chain entry:
    // table ends at 240 + 6*24 = 384; images 384..416, tails 416..424, dead 424..432, block_pos 432..436,
    // shapes 440..536, chain 536..568.
    const SharedSidecarLayout l = strata::core::conversation_share_sidecar_layout(32, 8, 8, 4, 1, 1);
    CHECK(l.members == 6);
    CHECK(l.bytes == 568);
    CHECK(l.images_at == 384 && l.tails_at == 416 && l.dead_at == 424 && l.block_pos_at == 432);
    CHECK(l.shapes_at == 440 && l.chain_at == 536);
    CHECK(l.images_bytes == 32 && l.tails_bytes == 8 && l.dead_bytes == 8 && l.block_pos_bytes == 4);
    const SharedSidecarLayout empty = strata::core::conversation_share_sidecar_layout(0, 0, 0, 0, 0, 0);
    CHECK(empty.members == 4 && empty.bytes == sizeof(strata::core::SharedSidecarHeader) + 4 * 24);
    CHECK(empty.images_at == empty.tails_at && empty.tails_at == empty.dead_at);   // all zero-byte members stack

    // --- the writer: header, table and host-side members; the device-copied payloads are left alone ---
    std::vector<uint8_t> buf((size_t) l.bytes, 0);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = 0xa5;   // a sentinel the writer must not disturb
    std::vector<ConversationImageKey> imgs = {{3, 0x1111}, {9, 0x2222}};
    SharedKvShape shape{};
    shape.format = 2;
    shape.cells = 4096;
    shape.heads = 8;
    shape.head_dim = 128;
    shape.page_size = 64;
    shape.pooled_rows = 66;
    shape.idx_dim = 256;
    shape.parts[0] = 100;
    shape.parts[4] = 200;
    SharedChainEntry chain{};
    chain.tokens = 77;
    chain.images = 2;
    chain.used = 5;
    chain.pinned = 1;
    strata::core::SharedSidecarMeta meta;
    meta.tokens = 64;
    meta.gdn_bytes = 1234;
    meta.ple_bytes = 5678;
    meta.images = imgs.data();
    meta.images_count = imgs.size();
    meta.shapes = &shape;
    meta.kv_layers = 1;
    meta.chain = &chain;
    meta.checkpoints = 1;
    meta.layer_lo = 0;
    meta.layer_hi = 48;
    meta.cvec = 1;
    for (size_t i = 0; i < meta.geometry.size(); ++i) meta.geometry[i] = (int64_t) 100 + (int64_t) i;
    strata::core::conversation_share_sidecar_write(buf.data(), l, meta);
    // the device-copied payloads keep the sentinel: the writer never touches the small arrays' bytes (the
    // engine's carve memsets the span first, so the align gaps reach the reader zero)
    CHECK(buf[(size_t) l.tails_at] == 0xa5 && buf[(size_t) l.tails_at + 7] == 0xa5);
    CHECK(buf[(size_t) l.dead_at] == 0xa5 && buf[(size_t) l.block_pos_at + 3] == 0xa5);
    // the align gap behind block_pos is untouched too - the caller's bytes
    CHECK(buf[(size_t) l.block_pos_at + 4] == 0xa5 && buf[(size_t) l.block_pos_at + 7] == 0xa5);

    SharedSidecarView v;
    CHECK(strata::core::shared_sidecar_view(buf.data(), buf.size(), v));
    CHECK(v.valid());
    CHECK(v.header->magic == 0x5354524153494445ull && v.header->version == 1);
    CHECK(v.header->members == 6 && v.header->bytes == l.bytes);
    CHECK(v.header->tokens == 64 && v.header->ids_bytes == 256);
    CHECK(v.header->gdn_bytes == 1234 && v.header->ple_bytes == 5678);
    CHECK(v.header->kv_layers == 1 && v.header->checkpoints == 1);
    CHECK(v.header->layer_lo == 0 && v.header->layer_hi == 48 && v.header->cvec == 1);
    CHECK(v.header->geometry[0] == 100 && v.header->geometry[17] == 117);
    const SharedSidecarMember* m_imgs = find(v, strata::core::kSidecarImages, 0);
    const SharedSidecarMember* m_tails = find(v, strata::core::kSidecarTails, 0);
    const SharedSidecarMember* m_dead = find(v, strata::core::kSidecarDead, 0);
    const SharedSidecarMember* m_bp = find(v, strata::core::kSidecarBlockPos, 0);
    const SharedSidecarMember* m_shape = find(v, strata::core::kSidecarKvShape, 0);
    const SharedSidecarMember* m_chain = find(v, strata::core::kSidecarChain, 0);
    CHECK(m_imgs != nullptr && m_imgs->offset == l.images_at && m_imgs->bytes == 32);
    CHECK(m_tails != nullptr && m_tails->offset == l.tails_at && m_tails->bytes == 8);
    CHECK(m_dead != nullptr && m_dead->offset == l.dead_at && m_dead->bytes == 8);
    CHECK(m_bp != nullptr && m_bp->offset == l.block_pos_at && m_bp->bytes == 4);
    CHECK(m_shape != nullptr && m_shape->offset == l.shapes_at && m_shape->bytes == 96);
    CHECK(m_chain != nullptr && m_chain->offset == l.chain_at && m_chain->bytes == 32);
    ConversationImageKey back[2];
    std::memcpy(back, v.data(m_imgs), sizeof(back));
    CHECK(back[0].start == 3 && back[0].hash == 0x1111);
    CHECK(back[1].start == 9 && back[1].hash == 0x2222);
    SharedKvShape shape_back{};
    std::memcpy(&shape_back, v.data(m_shape), sizeof(shape_back));
    CHECK(shape_back.format == 2 && shape_back.cells == 4096 && shape_back.heads == 8 && shape_back.head_dim == 128);
    CHECK(shape_back.page_size == 64 && shape_back.pooled_rows == 66 && shape_back.idx_dim == 256);
    CHECK(shape_back.parts[0] == 100 && shape_back.parts[4] == 200);
    SharedChainEntry chain_back{};
    std::memcpy(&chain_back, v.data(m_chain), sizeof(chain_back));
    CHECK(chain_back.tokens == 77 && chain_back.images == 2 && chain_back.used == 5 && chain_back.pinned == 1);
    CHECK(v.find(strata::core::kSidecarKvShape, 1) == nullptr);   // one layer announced, one shape carried
    CHECK(v.find(strata::core::kSidecarTails, 3) == nullptr);     // the small arrays are index 0 only

    // --- a block that is not this format's is refused ---
    SharedSidecarView bad;
    CHECK(strata::core::shared_sidecar_view(buf.data(), buf.size(), bad));
    CHECK(bad.valid());
    auto* mutable_header = reinterpret_cast<strata::core::SharedSidecarHeader*>(buf.data());
    mutable_header->magic ^= 0xff;   // the magic's first byte
    CHECK(!bad.valid());
    mutable_header->magic ^= 0xff;
    mutable_header->version = 2;     // a newer layout: refused, not guessed at
    CHECK(!bad.valid());
    mutable_header->version = 1;
    mutable_header->members = 99;    // a table that runs past the block
    CHECK(!bad.valid());

    // --- the chain's rows: a dumped checkpoint's share-resident state, named where it lies ---
    const uint64_t share_bytes = 1024 * 1024;
    ShareMap share;
    CHECK(share.make(share_bytes));
    std::string err;
    CHECK(strata::core::shared_arena_bind(share.base, share.bytes, share.ctl, err));
    CHECK(err.empty());

    // no arena, no rows: (the reset below re-arms this case at the end)
    // a checkpoint the active chain could hold: session_active-class storage, so its ids/gdn/ple carve from the
    // share; tails/dead/block_pos keep the default allocator (plain heap, not share storage)
    ConversationCheckpoint c(vma_class::session_active);
    c.ids.assign(64, 7);
    c.gdn.resize(1000, 0x5a);      // 1000 bytes: one page-rounded slice
    c.ple.resize(4096, 0x3c);      // exactly one page
    c.tails.resize(8, 1);          // plain heap: no row can name it
    c.imgs.push_back(ConversationImageKey{1, 9});
    c.used = 42;
    c.pinned = true;
    SharedChainEntry entry{};
    CHECK(strata::core::conversation_share_chain_publish(c, 1, entry, err));
    CHECK(err.empty());
    CHECK(entry.tokens == 64 && entry.images == 1 && entry.used == 42 && entry.pinned == 1);
    const auto row_of = [&](uint32_t kind, uint32_t index) -> const strata::core::SharedControlSegment* {
        for (const auto& r : strata::core::shared_arena_instance().rows)
            if (r.kind == kind && r.index == index) return &r;
        return nullptr;
    };
    const auto round_page = [](uint64_t n) { return (n + 4095ull) & ~4095ull; };
    const strata::core::SharedControlSegment *r_ids = row_of(strata::core::kSegmentIds, 1),
                                             *r_gdn = row_of(strata::core::kSegmentGdn, 1),
                                             *r_ple = row_of(strata::core::kSegmentPle, 1);
    CHECK(r_ids != nullptr && r_ids->offset == strata::core::shared_arena_offset(c.ids.data()) &&
          r_ids->bytes == round_page(64 * sizeof(int32_t)));
    CHECK(r_gdn != nullptr && r_gdn->offset == strata::core::shared_arena_offset(c.gdn.data()) &&
          r_gdn->bytes == round_page(1000));
    CHECK(r_ple != nullptr && r_ple->offset == strata::core::shared_arena_offset(c.ple.data()) &&
          r_ple->bytes == 4096);
    CHECK(row_of(strata::core::kSegmentSidecar, 0) == nullptr);   // the chain publishes no sidecar row of its own

    // a second checkpoint takes index 2: the rows name buffers that already lie in the share, no bytes move
    const uint64_t c_gdn_off = r_gdn->offset;
    ConversationCheckpoint d(vma_class::session_active);
    d.ids.assign(16, 3);
    d.gdn.resize(128, 0x77);
    SharedChainEntry entry2{};
    CHECK(strata::core::conversation_share_chain_publish(d, 2, entry2, err));
    CHECK(entry2.tokens == 16 && entry2.images == 0 && entry2.used == 0 && entry2.pinned == 0);
    const strata::core::SharedControlSegment* d_gdn = row_of(strata::core::kSegmentGdn, 2);
    CHECK(d_gdn != nullptr && d_gdn->offset == strata::core::shared_arena_offset(d.gdn.data()) &&
          d_gdn->offset != c_gdn_off && d_gdn->bytes == 4096);
    int gdn_rows = 0;
    for (const auto& r : strata::core::shared_arena_instance().rows)
        if (r.kind == strata::core::kSegmentGdn && (r.index == 1 || r.index == 2)) ++gdn_rows;
    CHECK(gdn_rows == 2);

    // a checkpoint outside the share (plain storage) still gets its entry, and no rows name it
    strata::core::shared_arena_reset();
    ConversationCheckpoint plain(vma_class::conv_cache);
    plain.ids.assign(8, 1);
    plain.gdn.resize(64, 2);
    SharedChainEntry entry3{};
    CHECK(!strata::core::conversation_share_chain_publish(plain, 1, entry3, err));   // no arena: refused
    CHECK(entry3.tokens == 8 && entry3.images == 0);                                 // ...but described
    CHECK(strata::core::shared_arena_bind(share.base, share.bytes, share.ctl, err));
    CHECK(strata::core::conversation_share_chain_publish(plain, 1, entry3, err));
    CHECK(err.empty());
    CHECK(entry3.tokens == 8);
    int plain_rows = 0;
    for (const auto& r : strata::core::shared_arena_instance().rows)
        if (r.index == 1) ++plain_rows;
    CHECK(plain_rows == 0);   // conv_cache storage is not share storage: no row names it

    // the table's room is the row limit: fill it and the next chain row refuses, the entry still fills
    while (strata::core::shared_arena_record(0, 4096, strata::core::kSegmentK, 0)) {}
    ConversationCheckpoint full(vma_class::session_active);
    full.ids.assign(4, 5);
    full.gdn.resize(16, 6);
    SharedChainEntry entry4{};
    CHECK(!strata::core::conversation_share_chain_publish(full, 1, entry4, err));
    CHECK(!err.empty());
    CHECK(entry4.tokens == 4);   // described either way

    strata::core::shared_arena_reset();
    share.drop();

    std::fprintf(stderr, "%s: %d checks, %d failed\n", "conversation_share_test", checks, fails);
    return fails == 0 ? 0 : 1;
}
