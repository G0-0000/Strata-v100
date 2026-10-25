// CPU-only tests for the on-disk session format (include/strata/core/conversation_file.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"
#include "strata/core/session_save_reclaim.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <new>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(1000, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        // > 16 MiB so the buffer spans several segments
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// format v1 golden: the file the fixed sample() gives (size and session_hash64 of all bytes, seed 0)
constexpr size_t kGoldenSize = 17878535;
constexpr uint64_t kGoldenHash = 0x70f812b350360d09ull;

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}

// no temporary file (any leftover of a write) in the directory
bool no_temp(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0 && n != "keep.bin.tmp") return false;
    }
    return true;
}

bool rejects(const fs::path& p, const SessionFileIdentity& id, const char* expect = nullptr,
             const SessionReadLimits& limits = {}) {
    SavedConversation out;
    size_t bytes = 0;
    std::string error;
    const bool ok = session_file_read(p.string(), id, out, bytes, error, limits);
    if (!ok && expect && error.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  unexpected error text: %s (wanted %s)\n", error.c_str(), expect);
        return false;
    }
    return !ok && !error.empty();
}

// ---- format v2 (--conversation-cache-disk-state-ref): a checkpoint whose five running-state arrays (gdn, ple, tails,
// dead, block_pos) are all byte-identical to the live state's is stored as REF descriptors of it; ids/imgs/used are
// saved as before.  homo() builds such a checkpoint with its own ids/imgs/used (those never take part in the decision).
uint64_t checkpoint_state_bytes(const ConversationCheckpoint& c) {
    return c.gdn.size() + c.ple.size() + c.tails.size() + c.dead.size() + c.block_pos.size();
}
ConversationCheckpoint homo(const ConversationCheckpoint& live, size_t tokens, uint64_t used, uint64_t imghash) {
    ConversationCheckpoint c;
    c.ids.assign(live.ids.begin(), live.ids.begin() + (std::ptrdiff_t) tokens);
    c.imgs = {{7, imghash}};
    c.gdn = live.gdn; c.ple = live.ple; c.tails = live.tails; c.dead = live.dead; c.block_pos = live.block_pos;
    c.used = used;
    return c;
}
uint32_t file_version_of(const std::vector<char>& d) {
    uint32_t v = 0; std::memcpy(&v, d.data() + 8, 4); return v;
}

// A hand serializer for the frozen v2 grammar (u64 codec, u64 logical, u64 stored, bytes[stored] per array, in the
// order gdn, ple, tails, dead, block_pos).  The writer only emits all-five REF groups, but the reader must accept any
// legal RAW/REF mix, so reader fixtures are built byte by byte here.
struct V2Fixture {
    std::vector<char> file;                            // the whole file: header, payload, trailer
    std::vector<std::array<size_t, 5>> codec;          // file offset of each array's codec word; [0] is live, else i+1
};
void fx_bytes(std::vector<char>& v, const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    v.insert(v.end(), c, c + n);
}
void fx_u64(std::vector<char>& v, uint64_t x) { fx_bytes(v, &x, 8); }
void fx_i64(std::vector<char>& v, int64_t x) { fx_bytes(v, &x, 8); }
void fx_checkpoint(std::vector<char>& p, V2Fixture& fx, size_t who, const ConversationCheckpoint& c,
                   const std::array<bool, 5>& ref) {
    fx_u64(p, c.ids.size());
    fx_bytes(p, c.ids.data(), c.ids.size() * sizeof(int32_t));
    fx_u64(p, c.imgs.size());
    for (const auto& k : c.imgs) { fx_i64(p, k.start); fx_u64(p, k.hash); }
    const auto put = [&](const auto& arr, size_t which) {
        fx.codec[who][which] = p.size() + 64;        // the file offset (the 64-byte header precedes the payload)
        fx_u64(p, ref[which] ? 1 : 0);
        fx_u64(p, (uint64_t) arr.size());
        fx_u64(p, ref[which] ? 0 : (uint64_t) arr.size());
        if (!ref[which]) fx_bytes(p, arr.data(), arr.size());
    };
    put(c.gdn, 0); put(c.ple, 1); put(c.tails, 2); put(c.dead, 3); put(c.block_pos, 4);
    fx_u64(p, c.used);
}
V2Fixture fx_v2(const SavedConversation& s, const std::vector<std::array<bool, 5>>& refs,
                const SessionFileIdentity& id) {
    V2Fixture fx;
    fx.codec.resize(1 + s.checkpoints.size());
    std::vector<char> p;
    for (int64_t g : s.geometry) fx_i64(p, g);
    fx_i64(p, s.layer_lo); fx_i64(p, s.layer_hi); fx_u64(p, s.cvec ? 1 : 0);
    fx_checkpoint(p, fx, 0, s.live, {false, false, false, false, false});
    fx_u64(p, s.checkpoints.size());
    for (size_t i = 0; i < s.checkpoints.size(); ++i) fx_checkpoint(p, fx, i + 1, s.checkpoints[i], refs[i]);
    fx_u64(p, s.kv.size());
    for (const auto& k : s.kv) {
        fx_i64(p, k.format); fx_i64(p, k.cells); fx_i64(p, k.heads); fx_i64(p, k.head_dim);
        fx_i64(p, k.page_size); fx_i64(p, k.pooled_rows); fx_i64(p, k.idx_dim);
        const ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (const ConversationBuffer* b : parts) {
            fx_u64(p, b->size());
            b->visit(0, b->size(), [&](const uint8_t* q, size_t c, size_t) { fx_bytes(p, q, c); return true; });
        }
    }
    std::vector<char> h(64, 0);
    std::memcpy(h.data(), "STRSESS\x01", 8);
    const uint32_t version = 2, hsize = 64;
    const uint64_t payload = p.size();
    std::memcpy(h.data() + 8, &version, 4); std::memcpy(h.data() + 12, &hsize, 4);
    std::memcpy(h.data() + 16, &id.model, 8); std::memcpy(h.data() + 24, &id.config, 8);
    std::memcpy(h.data() + 32, &payload, 8);
    const uint64_t hh = session_hash64(h.data(), 56, 0);
    std::memcpy(h.data() + 56, &hh, 8);
    fx.file = h;
    fx.file.insert(fx.file.end(), p.begin(), p.end());
    const uint64_t ph = session_hash64(p.data(), p.size(), 0);
    fx_u64(fx.file, ph);
    fx_bytes(fx.file, "STRSEND\x01", 8);
    return fx;
}
void fx_patch(std::vector<char>& f, size_t off, uint64_t v) { std::memcpy(f.data() + off, &v, 8); }
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-session-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    const SavedConversation original = sample();
    const fs::path good = dir / "good.bin";

    // round trip
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "write succeeds");
    check(error.empty(), "write leaves no error");

    // golden: format v1 is frozen. The fixed sample must give these exact bytes (little-endian header fields at
    // fixed offsets, and a fixed hash of the whole file); a format change must bump the version and this test.
    {
        const std::vector<char> g = slurp(good);
        auto u32 = [&](size_t o) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        auto u64 = [&](size_t o) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        check(g.size() >= 80 && std::memcmp(g.data(), "STRSESS\x01", 8) == 0, "golden: magic");
        check(u32(8) == 1 && u32(12) == 64, "golden: version 1, header 64 bytes (little-endian)");
        check(u64(16) == id.model && u64(24) == id.config, "golden: fingerprints at offsets 16 and 24");
        check(u64(32) == g.size() - 64 - 16, "golden: payload length at offset 32");
        check(u64(40) == 0 && u64(48) == 0, "golden: reserved fields are zero");
        check(std::memcmp(g.data() + g.size() - 8, "STRSEND\x01", 8) == 0, "golden: end marker");
        const uint64_t h = session_hash64(g.data(), g.size(), 0);
        const char* golden = std::getenv("STRATA_SESSION_GOLDEN_PRINT");
        if (golden) std::printf("golden size %zu hash %016llx\n", g.size(), (unsigned long long)h);
        check(g.size() == kGoldenSize && h == kGoldenHash, "golden: the fixed sample's file bytes are unchanged");
    }
    check(written == fs::file_size(good), "write reports the file size");
    check(no_temp(dir), "no temporary left behind");
    {
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(good.string(), id, back, read, error), "read succeeds");
        check(read == written, "read reports the file size");
        check(same(original, back), "round trip is identical");
    }
    // writing twice produces the same bytes (deterministic format)
    {
        const fs::path again = dir / "again.bin";
        check(session_file_write(again.string(), original, id, written, error), "second write");
        check(slurp(again) == slurp(good), "format is deterministic");
    }
    // saving over an existing file replaces it (MoveFileExW on Windows, rename elsewhere)
    {
        const fs::path over = dir / "over.bin";
        spit(over, std::vector<char>(100, 'o'));
        check(session_file_write(over.string(), original, id, written, error), "write over an existing file");
        check(slurp(over) == slurp(good), "existing file replaced by the session");
        check(no_temp(dir), "no temporary left behind after replace");
    }
    // buffered I/O (the path taken where direct I/O is refused or unavailable) writes and reads the same bytes
    {
        const char* inherited = std::getenv("STRATA_SESSION_BUFFERED");
        const std::string kept = inherited ? inherited : "";
        set_env("STRATA_SESSION_BUFFERED", "1");
        const fs::path p = dir / "buffered.bin";
        check(session_file_write(p.string(), original, id, written, error), "buffered write");
        check(slurp(p) == slurp(good), "buffered write equals the default write");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error) && same(original, back), "buffered read");
        set_env("STRATA_SESSION_BUFFERED", inherited ? kept.c_str() : nullptr);   // the caller's choice again
    }

    const std::vector<char> image = slurp(good);
    // truncation at every structural boundary and in the middle
    for (size_t cut : {size_t(0), size_t(7), size_t(63), size_t(64), size_t(65), size_t(200), image.size() / 2,
                       image.size() - 17, image.size() - 16, image.size() - 8, image.size() - 1}) {
        const fs::path p = dir / "cut.bin";
        spit(p, std::vector<char>(image.begin(), image.begin() + (std::ptrdiff_t) cut));
        check(rejects(p, id), "truncated file rejected");
    }
    // trailing garbage
    {
        auto d = image; d.push_back('x');
        const fs::path p = dir / "tail.bin"; spit(p, d);
        check(rejects(p, id, "size"), "file with trailing bytes rejected");
    }
    // flipped payload byte (inside the big K buffer), flipped header byte, flipped trailer
    for (size_t at : {size_t(64 + 300), image.size() / 2, size_t(20), image.size() - 12, image.size() - 3}) {
        auto d = image; d[at] ^= 0x40;
        const fs::path p = dir / "flip.bin"; spit(p, d);
        check(rejects(p, id), "corrupted byte rejected");
    }
    // other model or other configuration
    check(rejects(good, {id.model ^ 1, id.config}, "model"), "different model fingerprint rejected");
    check(rejects(good, {id.model, id.config ^ 1}, "config"), "different configuration rejected");
    // wrong version: header field patched and header hash recomputed so only the version differs.  (Version 2 is now
    // the state-ref format, so the unsupported-version probe must use a version no reader accepts.)
    {
        auto d = image;
        uint32_t v = 3; std::memcpy(d.data() + 8, &v, 4);
        const uint64_t h = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &h, 8);
        const fs::path p = dir / "ver.bin"; spit(p, d);
        check(rejects(p, id, "version"), "unknown version rejected");
    }
    // a corrupted element count with a recomputed payload hash must fail on bounds, not allocate
    {
        auto d = image;
        // payload starts at 64: geometry 18*8, layer_lo, layer_hi, cvec = 21*8 bytes, then live.ids count
        const size_t at = 64 + 21 * 8;
        uint64_t huge = 0x0fffffffffffffffull; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "count.bin"; spit(p, d);
        check(rejects(p, id, "count"), "oversized count rejected before allocation");
    }
    // missing file
    check(rejects(dir / "absent.bin", id), "missing file rejected");
    // a failed read leaves the destination untouched
    {
        SavedConversation keep = sample();
        size_t b = 0;
        check(!session_file_read((dir / "absent.bin").string(), id, keep, b, error), "missing read fails");
        check(same(keep, original), "failed read does not modify the destination");
    }
    // layer-split images are not written; no file appears
    {
        SavedConversation split = sample();
        split.live.stage_parts.push_back(checkpoint(1, 9));
        const fs::path p = dir / "split.bin";
        check(!session_file_write(p.string(), split, id, written, error), "layer-split image refused");
        check(!fs::exists(p) && no_temp(dir), "refused write leaves no file");
    }
    // a write into a missing directory fails cleanly
    check(!session_file_write((dir / "nope" / "x.bin").string(), original, id, written, error),
          "write into a missing directory fails");
    // the hash is position-sensitive and seed-sensitive
    {
        const char a[] = "abcdefghijklmnopqrstuvwxyz0123456789", b[] = "bacdefghijklmnopqrstuvwxyz0123456789";
        check(session_hash64(a, sizeof a, 0) != session_hash64(b, sizeof b, 0), "hash sees swapped bytes");
        check(session_hash64(a, sizeof a, 0) != session_hash64(a, sizeof a, 1), "hash sees the seed");
        SessionHasher h; h.update(a, 5); h.update(a + 5, sizeof a - 5);
        check(h.digest() == session_hash64(a, sizeof a, 0), "streaming hash equals one-shot hash");
    }
    // model fingerprint: changes with content at the head of a file and with the file list
    {
        const fs::path m1 = dir / "m1.bin", m2 = dir / "m2.bin";
        spit(m1, std::vector<char>(3u << 20, 'a'));
        spit(m2, std::vector<char>(100, 'b'));
        uint64_t f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0, f8 = 0;
        const std::vector<SessionModelFile> both = {{"a", m1.string()}, {"b", m2.string()}};
        check(session_model_fingerprint(both, f1, error), "fingerprint ok");
        const std::vector<SessionModelFile> subset = {{"a", m1.string()}};
        const std::vector<SessionModelFile> roles = {{"b", m1.string()}, {"a", m2.string()}};
        check(session_model_fingerprint(subset, f2, error), "fingerprint subset ok");
        check(session_model_fingerprint(roles, f5, error), "fingerprint roles ok");
        check(f1 != f5, "fingerprint follows the role each file plays");
        // a moved folder: same roles, same bytes, other paths
        const fs::path moved = dir / "moved";
        fs::create_directories(moved);
        fs::copy_file(m1, moved / "m1.bin"); fs::copy_file(m2, moved / "m2.bin");
        const std::vector<SessionModelFile> moved_list = {{"a", (moved / "m1.bin").string()}, {"b", (moved / "m2.bin").string()}};
        check(session_model_fingerprint(moved_list, f6, error) && f6 == f1, "fingerprint does not follow the folder");
        // an optional input that is absent is recorded, not an error; a present one changes the fingerprint
        const std::vector<SessionModelFile> absent = {{"a", m1.string()}, {"b", m2.string()}, {"c", (dir / "nothere").string(), true}};
        const std::vector<SessionModelFile> present = {{"a", m1.string()}, {"b", m2.string()}, {"c", m2.string(), true}};
        check(session_model_fingerprint(absent, f7, error), "absent optional input ok");
        check(session_model_fingerprint(present, f8, error) &&
              f7 != f8 && f7 != f1, "optional input presence enters the fingerprint");
        auto d = slurp(m1); d[10] = 'z'; spit(m1, d);
        check(session_model_fingerprint(both, f3, error), "fingerprint after edit ok");
        check(f1 != f2 && f1 != f3, "fingerprint follows files and content");
        check(!session_model_fingerprint({{"x", (dir / "absent").string()}}, f4, error), "fingerprint of a missing file fails");
    }

    // only the deepest checkpoint is worth a disk write: the next turn resumes from it
    {
        std::vector<ConversationCheckpoint> chain;
        check(session_checkpoints_to_save(chain).empty(), "no checkpoints, none saved");
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.push_back(checkpoint(500, 4));
        const auto kept = session_checkpoints_to_save(chain);
        check(kept.size() == 1 && same_checkpoint(kept[0], chain[1]), "deepest checkpoint kept, alone");
    }
    // K/V streamed from its owner (no host copy): the same bytes as writing the captured image
    {
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : original.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                return parts[part]->read(dst, offset, n);
            };
            sources.push_back(std::move(s));
        }
        const fs::path p = dir / "streamed.bin";
        check(session_file_write(p.string(), meta, sources, id, written, error), "streamed write succeeds");
        check(slurp(p) == image, "streamed write equals the captured-image write");
        // a failing source leaves no file
        auto broken = sources;
        broken[1].read = [](size_t, size_t, void*, size_t) { return false; };
        const fs::path q = dir / "broken.bin";
        check(!session_file_write(q.string(), meta, broken, id, written, error), "failing source fails the write");
        check(!fs::exists(q) && no_temp(dir), "failed streamed write leaves no file");
        // K/V in the image and as sources at once is ambiguous: refused
        check(!session_file_write((dir / "both.bin").string(), original, sources, id, written, error),
              "image K/V plus sources refused");
    }

    // --conversation-cache-disk-delta: a base file plus appended records that carry only the K/V from an offset on.
    // A restore walks the base then every record in order and writes each block at its recorded offset, so a grown
    // conversation is rebuilt exactly; a same-size edit inside a record is refused by its payload hash.
    {
        const fs::path base = dir / "delta-base.bin";
        const fs::path log = dir / "delta.dlog";
        check(session_file_write(base.string(), original, id, written, error), "delta: base write");
        // the grown state: the base's K/V with every part's tail from byte 1024 on rewritten (a page boundary), and
        // two more tokens so it is a longer conversation
        SavedConversation grown = original;
        grown.live.ids.push_back(9001);
        grown.live.ids.push_back(9002);
        grown.checkpoints.back() = checkpoint(1002, 7);
        for (size_t i = 0; i < grown.kv.size(); ++i) {
            const std::array<ConversationBuffer*, 5> parts = {&grown.kv[i].k, &grown.kv[i].v, &grown.kv[i].k_scale,
                                                              &grown.kv[i].v_scale, &grown.kv[i].pooled};
            for (size_t part = 0; part < 5; ++part) {
                if (parts[part]->size() <= 1024) continue;
                ConversationBuffer& b = *parts[part];
                b.visit(1024, b.size() - 1024, [&](uint8_t* p, size_t c, size_t at) {
                    for (size_t j = 0; j < c; ++j) p[j] = uint8_t((at + j) * 37u + i + part);
                    return true;
                });
            }
        }
        auto delta_sources = [&](const SavedConversation& want, bool first_half) {
            std::vector<SessionKvSource> sources;
            for (const auto& k : want.kv) {
                SessionKvSource s;
                s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
                s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
                const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
                for (size_t i = 0; i < 5; ++i) s.first_bytes[i] = first_half && parts[i]->size() > 1024 ? 1024 : 0;
                s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                    return parts[part]->read(dst, offset, n);
                };
                sources.push_back(std::move(s));
            }
            return sources;
        };
        SavedConversation meta = grown;
        meta.kv.clear();
        auto sources = delta_sources(grown, true);
        SessionWriteOptions opt;
        uint64_t wrote = 0;
        // an offset past its part is refused before the file is touched
        {
            auto bad = sources;
            bad[0].first_bytes[0] = bad[0].sizes[0] + 8;
            check(!session_delta_write(log.string(), 0, meta, bad, (int64_t) grown.live.ids.size(), id, wrote, error,
                                       opt),
                  "delta: an offset past its part is refused");
            check(!fs::exists(log), "delta: the refusal writes nothing");
        }
        bool overflow = false;
        const uint64_t estimate = session_delta_record_bytes(meta, sources, &overflow);
        check(!overflow && estimate > 0, "delta: the record's size is counted");
        check(session_delta_write(log.string(), 0, meta, sources, (int64_t) grown.live.ids.size(), id, wrote, error, opt),
              "delta: record write");
        check(wrote == estimate, "delta: the estimate equals the write");
        check((uint64_t) fs::file_size(log) == wrote, "delta: the log holds exactly the record");
        // the read pass: record structure, offsets and sizes
        SessionReadLimits lim;   // open limits
        SessionDeltaRecord rec;
        uint64_t next = 0;
        const bool record_read_ok = session_delta_read(log.string(), 0, wrote, id, lim, rec, next, error);
        if (!record_read_ok) std::fprintf(stderr, "  delta read error: %s\n", error.c_str());
        check(record_read_ok, "delta: record read");
        check(next == wrote && rec.meta.live.ids == grown.live.ids, "delta: the record reads back whole");
        check(rec.kv_offset.size() == grown.kv.size() && rec.kv_offset[0][0] == 1024 && rec.kv_bytes[0][0] > 0,
              "delta: the K/V offsets are recorded");
        check(session_delta_read(log.string(), 0, wrote - 1, id, lim, rec, next, error) == false,
              "delta: a record past the committed length is refused");
        // the apply: base then record, into plain host buffers, equals the grown state byte for byte
        SavedConversation base_meta;
        std::vector<std::array<uint64_t, 5>> base_sizes;
        size_t base_bytes = 0;
        check(session_file_read_streamed(base.string(), id, base_meta, base_bytes, error, lim, &base_sizes),
              "delta: the base reads streamed");
        std::vector<std::array<std::vector<uint8_t>, 5>> rebuilt(grown.kv.size());
        for (size_t l = 0; l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; part < 5; ++part) rebuilt[l][part].resize(parts[part]->size());
        }
        auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= rebuilt.size() || offset > rebuilt[layer][part].size() ||
                n > rebuilt[layer][part].size() - offset) return false;
            std::memcpy(rebuilt[layer][part].data() + offset, data, n);
            return true;
        };
        check(session_file_apply_streamed(base.string(), id, sink, base_sizes, error),
              "delta: the base applies");
        check(session_delta_apply(log.string(), 0, id, rec, sink, error), "delta: the record applies");
        bool rebuilt_same = true;
        for (size_t l = 0; rebuilt_same && l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; rebuilt_same && part < 5; ++part) {
                std::vector<uint8_t> want(parts[part]->size());
                if (!want.empty())
                    parts[part]->visit(0, want.size(), [&](const uint8_t* p, size_t c, size_t at) {
                        std::memcpy(want.data() + at, p, c);
                        return true;
                    });
                rebuilt_same = rebuilt[l][part] == want;
            }
        }
        check(rebuilt_same, "delta: base + record rebuild the grown K/V exactly");
        // a second record appends after the first; the two reapply in order
        uint64_t wrote2 = 0;
        SavedConversation meta2 = grown;
        meta2.kv.clear();
        auto sources2 = delta_sources(original, true);   // the K/V prefix (the rewritten-with-wrong-tail check)
        check(session_delta_write(log.string(), wrote, meta2, sources2, (int64_t) grown.live.ids.size(), id, wrote2,
                                  error, opt), "delta: append the second record");
        SessionDeltaRecord rec2;
        check(session_delta_read(log.string(), wrote, wrote + wrote2, id, lim, rec2, next, error) && next == wrote + wrote2,
              "delta: the second record reads at its offset");
        // The control-path walk releases each prior record's dynamic capacity before reusing its single output object.
        {
            SessionDeltaRecord released;
            std::swap(rec2, released);
        }
        {
            uint64_t at = 0;
            size_t walked = 0;
            size_t first_bytes = 0;
            bool bounded = true;
            while (at < wrote + wrote2) {
                {
                    SessionDeltaRecord released;
                    std::swap(rec, released);
                }
                uint64_t end = 0;
                check(session_delta_read(log.string(), at, wrote + wrote2, id, lim, rec, end, error) && end > at,
                      "delta walk: each committed record reads sequentially");
                const size_t current_bytes = rec.meta.bytes() +
                    rec.kv_offset.capacity() * sizeof(std::array<uint64_t, 5>) +
                    rec.kv_bytes.capacity() * sizeof(std::array<uint64_t, 5>);
                if (walked == 0) first_bytes = current_bytes;
                else bounded = bounded && current_bytes == first_bytes;
                at = end;
                ++walked;
            }
            check(walked == 2 && bounded && at == wrote + wrote2,
                  "delta walk: two records keep one record's bounded metadata, independent of chain length");
        }
        // Admission refusal happens before the first parser-owned allocation; the caller's record and next offset
        // stay unchanged, as does the valid disk record.
        {
            SessionDeltaRecord denied;
            denied.first_token = 777;
            denied.meta.live.ids = {12345};
            const auto before_ids = denied.meta.live.ids;
            SessionReadLimits no_memory = lim;
            uint64_t asked = 0;
            no_memory.admit = [&](uint64_t need, std::string& why) {
                asked = need;
                check(denied.first_token == 777 && denied.meta.live.ids == before_ids,
                      "delta admission: caller output is unchanged when admission runs");
                why = "controlled test refusal";
                return false;
            };
            uint64_t next_before = 0xabcdef;
            check(!session_delta_read(log.string(), 0, wrote + wrote2, id, no_memory, denied, next_before, error) &&
                  asked > 0 && error.find("controlled test refusal") != std::string::npos,
                  "delta admission: a real parser allocation can be refused");
            check(denied.first_token == 777 && denied.meta.live.ids == before_ids && next_before == 0xabcdef,
                  "delta admission: refusal leaves output and next offset untouched");
        }
        // a same-size edit inside the first record is refused by the hash
        const std::vector<char> pristine = slurp(log);
        // Counts are rejected before any vector allocation: both a valid count above the engine bound and a
        // malformed UINT64_MAX count leave the caller's record untouched.
        {
            SessionReadLimits too_few = lim;
            too_few.max_tokens = 2;
            bool admitted = false;
            too_few.admit = [&](uint64_t, std::string&) { admitted = true; return true; };
            SessionDeltaRecord preserved;
            preserved.first_token = 991;
            preserved.meta.live.ids = {7};
            uint64_t at = 123;
            check(!session_delta_read(log.string(), 0, wrote + wrote2, id, too_few, preserved, at, error) &&
                  !admitted && error.find("exceeds this engine's limit") != std::string::npos,
                  "delta bounds: over-limit token count is refused before admission/allocation");
            check(preserved.first_token == 991 && preserved.meta.live.ids.size() == 1 &&
                  preserved.meta.live.ids[0] == 7 && at == 123, "delta bounds: over-limit refusal preserves output");
        }
        {
            std::vector<char> malformed = pristine;
            const uint64_t impossible_count = UINT64_MAX;
            std::memcpy(malformed.data() + 208, &impossible_count, sizeof(impossible_count));
            spit(log, malformed);
            SessionDeltaRecord preserved;
            preserved.first_token = 992;
            preserved.meta.live.ids = {8};
            uint64_t at = 456;
            check(!session_delta_read(log.string(), 0, wrote + wrote2, id, lim, preserved, at, error) &&
                  error.find("element count exceeds the record") != std::string::npos,
                  "delta bounds: malformed overflowing count is rejected against remaining record bytes");
            check(preserved.first_token == 992 && preserved.meta.live.ids.size() == 1 &&
                  preserved.meta.live.ids[0] == 8 && at == 456,
                  "delta bounds: malformed count refusal preserves output and offset");
            spit(log, pristine);
            SessionReadLimits extreme;
            extreme.max_tokens = extreme.max_checkpoints = extreme.max_kv_layers = UINT64_MAX;
            extreme.max_state_bytes.fill(UINT64_MAX);
            check(session_delta_walk_peak_bytes(extreme) == UINT64_MAX,
                  "delta bounds: allocation-size arithmetic saturates rather than wrapping");
        }
        {
            std::vector<char> edited = pristine;
            edited[64] = char(edited[64] ^ 0x40);
            spit(log, edited);
            SessionDeltaRecord bad;
            uint64_t at = 0;
            check(!session_delta_read(log.string(), 0, wrote + wrote2, id, lim, bad, at, error) &&
                  error.find("checksum") != std::string::npos, "delta: an edited record is refused");
            spit(log, pristine);   // put the record back for the checks below
        }
        // a wrong identity is refused
        check(!session_delta_read(log.string(), 0, wrote + wrote2, {id.model ^ 1, id.config}, lim, rec, next, error),
              "delta: a foreign identity is refused");
        // the apply pass refuses a record whose recorded part sizes differ from the read pass's record
        {
            SessionDeltaRecord bad = rec;
            bad.kv_bytes[0][0] += 8;
            check(!session_delta_apply(log.string(), 0, id, bad, sink, error), "delta: a structure change is refused");
        }
        // The read pass admits real metadata and its 16 MiB scratch buffer, not the streamed payload's disk length.
        {
            constexpr uint64_t payload_bytes = 20ull << 20;
            constexpr uint64_t resident_limit = 18ull << 20;
            const fs::path streamed_log = dir / "delta-large-stream.dlog";
            SavedConversation tiny_meta;
            tiny_meta.geometry = original.geometry;
            tiny_meta.layer_lo = 0;
            tiny_meta.layer_hi = 48;
            tiny_meta.cvec = false;
            auto shrink = [](ConversationCheckpoint& c) {
                c.gdn.resize(32); c.ple.resize(16); c.tails.resize(8); c.dead.resize(8); c.block_pos.resize(4);
            };
            tiny_meta.live = checkpoint(16, 11);
            shrink(tiny_meta.live);
            tiny_meta.checkpoints.push_back(checkpoint(8, 12));
            shrink(tiny_meta.checkpoints.back());
            SessionKvSource source;
            source.format = 2; source.cells = 1024; source.heads = 2; source.head_dim = 256;
            source.page_size = 64; source.pooled_rows = 0; source.idx_dim = 128;
            source.sizes[0] = payload_bytes;
            source.read = [](size_t, size_t, void* dst, size_t bytes) { std::memset(dst, 0x5a, bytes); return true; };
            const std::vector<SessionKvSource> streamed_source = {source};
            uint64_t written_stream = 0;
            check(session_delta_write(streamed_log.string(), 0, tiny_meta, streamed_source, 16, id,
                                      written_stream, error), "delta admission: stream-generated record writes");
            SessionReadLimits bounded;
            bounded.max_tokens = 64;
            bounded.max_checkpoints = 1;
            bounded.max_kv_layers = 1;
            bounded.max_state_bytes.fill(64);
            check(session_delta_walk_peak_bytes(bounded) < resident_limit,
                  "delta admission: bounded metadata and scratch fit the resident allowance");
            uint64_t largest_admission = 0;
            bounded.admit = [&](uint64_t need, std::string& why) {
                if (need > largest_admission) largest_admission = need;
                if (need <= resident_limit) return true;
                why = "over resident allowance";
                return false;
            };
            SessionDeltaRecord streamed_record;
            uint64_t streamed_next = 0;
            check(session_delta_read(streamed_log.string(), 0, written_stream, id, bounded,
                                     streamed_record, streamed_next, error) && streamed_next == written_stream,
                  "delta admission: payload larger than resident allowance is accepted by streaming read");
            check(written_stream > resident_limit && largest_admission <= resident_limit &&
                  streamed_record.kv_bytes.size() == 1 && streamed_record.kv_bytes[0][0] == payload_bytes,
                  "delta admission: admitted bytes track bounded allocations, not streamed disk payload");
            fs::remove(streamed_log);
        }
        fs::remove(log);
        fs::remove(base);
    }

    // A+B->A+C: one base, two records, the second overwriting the first's tail from the shared boundary on.  Restoring
    // base, then B, then C yields exactly C - the last record wins, so a branched conversation restores its newest
    // state through the same chain.
    {
        const fs::path base = dir / "branch-base.bin";
        const fs::path log = dir / "branch.dlog";
        check(session_file_write(base.string(), original, id, written, error), "branch: base write");
        auto branched = [&](int64_t extra1, int64_t extra2, uint8_t salt) {
            SavedConversation s = original;
            s.live.ids.push_back(int32_t(extra1));
            s.live.ids.push_back(int32_t(extra2));
            s.checkpoints.back() = checkpoint(1002, salt);
            for (size_t i = 0; i < s.kv.size(); ++i) {
                const std::array<ConversationBuffer*, 5> parts = {&s.kv[i].k, &s.kv[i].v, &s.kv[i].k_scale,
                                                                  &s.kv[i].v_scale, &s.kv[i].pooled};
                for (size_t part = 0; part < 5; ++part) {
                    if (parts[part]->size() <= 1024) continue;
                    ConversationBuffer& b = *parts[part];
                    b.visit(1024, b.size() - 1024, [&](uint8_t* p, size_t c, size_t at) {
                        for (size_t j = 0; j < c; ++j) p[j] = uint8_t((at + j) * 53u + salt + i + part);
                        return true;
                    });
                }
            }
            return s;
        };
        const SavedConversation bstate = branched(9001, 9002, 7);
        const SavedConversation cstate = branched(9101, 9102, 9);
        auto tail_sources = [](const SavedConversation& want) {
            std::vector<SessionKvSource> sources;
            for (const auto& k : want.kv) {
                SessionKvSource s;
                s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
                s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
                const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t i = 0; i < 5; ++i) {
                    s.sizes[i] = parts[i]->size();
                    s.first_bytes[i] = parts[i]->size() > 1024 ? 1024 : 0;
                }
                s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                    return parts[part]->read(dst, offset, n);
                };
                sources.push_back(std::move(s));
            }
            return sources;
        };
        auto meta_only = [](const SavedConversation& want) { SavedConversation m = want; m.kv.clear(); return m; };
        SessionWriteOptions opt;
        uint64_t wb = 0, wc = 0;
        check(session_delta_write(log.string(), 0, meta_only(bstate), tail_sources(bstate),
                                  (int64_t) bstate.live.ids.size(), id, wb, error, opt), "branch: record B");
        check(session_delta_write(log.string(), wb, meta_only(cstate), tail_sources(cstate),
                                  (int64_t) cstate.live.ids.size(), id, wc, error, opt), "branch: record C");
        check((uint64_t) fs::file_size(log) == wb + wc, "branch: the log holds both records");
        // restore: base + B + C into buffers sized to C
        SessionReadLimits lim;
        SavedConversation base_meta;
        std::vector<std::array<uint64_t, 5>> base_sizes;
        size_t base_bytes = 0;
        check(session_file_read_streamed(base.string(), id, base_meta, base_bytes, error, lim, &base_sizes),
              "branch: the base reads streamed");
        std::vector<std::array<std::vector<uint8_t>, 5>> rebuilt(cstate.kv.size());
        for (size_t l = 0; l < cstate.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&cstate.kv[l].k, &cstate.kv[l].v,
                                                                    &cstate.kv[l].k_scale, &cstate.kv[l].v_scale,
                                                                    &cstate.kv[l].pooled};
            for (size_t part = 0; part < 5; ++part) rebuilt[l][part].resize(parts[part]->size());
        }
        auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= rebuilt.size() || offset > rebuilt[layer][part].size() ||
                n > rebuilt[layer][part].size() - offset) return false;
            std::memcpy(rebuilt[layer][part].data() + offset, data, n);
            return true;
        };
        check(session_file_apply_streamed(base.string(), id, sink, base_sizes, error), "branch: the base applies");
        SessionDeltaRecord rec;
        uint64_t next = 0;
        check(session_delta_read(log.string(), 0, wb + wc, id, lim, rec, next, error) && next == wb,
              "branch: record B reads at its offset");
        check(session_delta_apply(log.string(), 0, id, rec, sink, error), "branch: record B applies");
        check(session_delta_read(log.string(), wb, wb + wc, id, lim, rec, next, error) && next == wb + wc,
              "branch: record C reads at its offset");
        check(rec.meta.live.ids == cstate.live.ids, "branch: the last record's metadata is C's");
        check(session_delta_apply(log.string(), wb, id, rec, sink, error), "branch: record C applies");
        bool rebuilt_same = true;
        for (size_t l = 0; rebuilt_same && l < cstate.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&cstate.kv[l].k, &cstate.kv[l].v,
                                                                    &cstate.kv[l].k_scale, &cstate.kv[l].v_scale,
                                                                    &cstate.kv[l].pooled};
            for (size_t part = 0; rebuilt_same && part < 5; ++part) {
                std::vector<uint8_t> want(parts[part]->size());
                if (!want.empty())
                    parts[part]->visit(0, want.size(), [&](const uint8_t* p, size_t c, size_t at) {
                        std::memcpy(want.data() + at, p, c);
                        return true;
                    });
                rebuilt_same = rebuilt[l][part] == want;
            }
        }
        check(rebuilt_same, "branch: base + B + C rebuilds exactly C");
        fs::remove(log);
        fs::remove(base);
    }

    // session_delta_first_bytes: the per-part offset a delta record starts at (the snapshot keep rule, host-testable).
    // Main-layer parts round down to whole cells through the last full page, the pooled part to whole indexer rows;
    // the draft layer (index=false) rewrites every pooled row; an offset past its part is refused.
    {
        SessionKvSource s;
        s.cells = 1024; s.page_size = 64;
        const size_t per_cell = 4;                  // k and v: 4096 bytes = 1024 cells x 4
        const size_t scale_per_cell = 1;            // k_scale and v_scale: 1024 bytes = 1024 cells x 1
        const size_t row = 128 * 4;                 // one indexer row: idx_key_dim 128 floats, 16 tokens a row (idx_block)
        s.sizes = {1024 * per_cell, 1024 * per_cell, 1024 * scale_per_cell, 1024 * scale_per_cell, 9 * row};
        std::string e;
        // 63 tokens: the page the last token sits in is partial, so no cell before it is proven unchanged
        check(session_delta_first_bytes(s, 63, true, 16, 128, e) && s.first_bytes[0] == 0 &&
              s.first_bytes[2] == 0 && s.first_bytes[4] == 3 * row, "first_bytes: below one full page keeps nothing");
        // 64 tokens: exactly one full page; 65 rounds down to the same cells
        check(session_delta_first_bytes(s, 64, true, 16, 128, e) && s.first_bytes[0] == 64 * per_cell &&
              s.first_bytes[1] == 64 * per_cell && s.first_bytes[2] == 64 * scale_per_cell &&
              s.first_bytes[4] == 4 * row, "first_bytes: one full page rounds down to its cells");
        check(session_delta_first_bytes(s, 65, true, 16, 128, e) && s.first_bytes[0] == 64 * per_cell &&
              s.first_bytes[4] == 4 * row, "first_bytes: a partial page keeps the whole page out");
        // the draft layer rewrites every pooled row (and is handed one token less by the snapshot caller)
        check(session_delta_first_bytes(s, 64, false, 16, 128, e) && s.first_bytes[0] == 64 * per_cell &&
              s.first_bytes[4] == 0, "first_bytes: the draft keeps no pooled row");
        // an offset past its part is refused (kept tokens beyond the layer's cells)
        check(!session_delta_first_bytes(s, 2048, true, 16, 128, e), "first_bytes: past its part is refused");
    }

    // the configuration fingerprint: every field counts, doubles by their exact bits
    {
        SessionConfig c;
        c.engine_version = "0.1.38"; c.backend = "cuda"; c.kv = "int8"; c.max_context = 65536; c.kv_resident = 0;
        c.mtp_window = 4; c.rope.type = 0; c.rope.freq_base = 1e7; c.rope.factor = 1.0; c.rope.freq_scale = 1.0;
        c.rope.orig_ctx = 262144; c.rope.attn_factor = 1.0; c.rope.beta_fast = 32; c.rope.beta_slow = 1;
        c.switches = {{"STRATA_FAST_GDN", 0}};
        const uint64_t base = session_config_fingerprint(c);
        check(session_config_fingerprint(c) == base, "config fingerprint is deterministic");
        auto differs = [&](auto edit) { SessionConfig d = c; edit(d); return session_config_fingerprint(d) != base; };
        check(differs([](SessionConfig& d) { d.engine_version = "0.1.39"; }), "engine version enters the config");
        check(differs([](SessionConfig& d) { d.backend = "hip"; }), "backend enters the config");
        check(differs([](SessionConfig& d) { d.kv = "fp16"; }), "kv enters the config");
        check(differs([](SessionConfig& d) { d.max_context = 65537; }), "max context enters the config");
        check(differs([](SessionConfig& d) { d.kv_resident = 1; }), "kv residency enters the config");
        check(differs([](SessionConfig& d) { d.mtp_window = 0; }), "mtp window enters the config");
        check(differs([](SessionConfig& d) { d.kv_rot = true; }), "kv rotation enters the config");
        check(differs([](SessionConfig& d) { d.rope.factor = std::nextafter(1.0, 2.0); }),
              "a one-ulp rope change enters the config");
        check(differs([](SessionConfig& d) { d.rope.type = 2; }), "rope type enters the config");
        check(differs([](SessionConfig& d) { d.rope.beta_slow = 2; }), "rope beta enters the config");
        check(differs([](SessionConfig& d) { d.cvec = 7; }), "control vectors enter the config");
        check(differs([](SessionConfig& d) { d.switches[0].second = 1; }), "a switch value enters the config");
        check(differs([](SessionConfig& d) { d.switches.push_back({"X", 0}); }), "the switch list enters the config");
        // field boundaries: ("ab","c") and ("a","bc") hash differently
        SessionIdentityBuilder x(1), y(1);
        x.str("ab", "c"); y.str("a", "bc");
        check(x.digest() != y.digest(), "identity fields are length-delimited");
        SessionIdentityBuilder i(1), u(1);
        i.i64("n", 5); u.u64("n", 5);
        check(i.digest() != u.digest(), "identity fields are typed");
    }
    // bounds a read checks before it allocates, and the caller's admission
    {
        SessionReadLimits l;
        l.max_tokens = 999;   // the live state holds 1000
        check(rejects(good, id, "limit", l), "token count over the limit refused");
        l = {}; l.max_checkpoints = 1;
        check(rejects(good, id, "limit", l), "checkpoint count over the limit refused");
        l = {}; l.max_kv_layers = 2;
        check(rejects(good, id, "limit", l), "K/V layer count over the limit refused");
        l = {}; l.max_file_bytes = image.size() - 1;
        check(rejects(good, id, "limit", l), "file over the size limit refused");
        l = {};
        uint64_t asked = 0;
        l.admit = [&](uint64_t n, std::string& why) { asked = n; why = "no room for it"; return false; };
        check(rejects(good, id, "no room", l) && asked == session_read_peak_bytes(image.size()) &&
              asked > image.size(), "admission sees the parse's peak (more than the file) and can refuse");
        l = {};
        bool admitted = false;
        l.admit = [&](uint64_t, std::string&) { admitted = true; return true; };
        check(rejects(good, {id.model ^ 1, id.config}, "model", l) && !admitted,
              "admission is not asked for a foreign file");
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, l) && admitted && same(back, original),
              "admitted file reads");
    }
    // free-space reserve: an impossible reserve refuses the write and leaves the existing file untouched
    {
        const fs::path p = dir / "reserve.bin";
        spit(p, std::vector<char>(10, 'r'));
        SessionWriteOptions opt;
        opt.min_free_bytes = UINT64_MAX / 2;
        check(!session_file_write(p.string(), original, id, written, error, opt) &&
              error.find("disk space") != std::string::npos, "write over the free-space reserve refused");
        check(slurp(p) == std::vector<char>(10, 'r') && no_temp(dir), "refused write keeps the old file, no temporary");
        opt.min_free_bytes = 1;
        uint64_t calls = 0;
        opt.progress = [&](uint64_t, uint64_t) { ++calls; };
        check(session_file_write(p.string(), original, id, written, error, opt) && slurp(p) == image,
              "write with a small reserve succeeds");
    }
    // a failed write over an existing session keeps it, and touches no other file (a planted "<name>.tmp", the old
    // fixed temporary pattern included)
    {
        const fs::path p = dir / "keep.bin", planted = dir / "keep.bin.tmp";
        check(session_file_write(p.string(), original, id, written, error), "keep: first write");
        spit(planted, std::vector<char>(5, 'p'));
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> broken(1);
        broken[0].sizes = {100, 0, 0, 0, 0};
        broken[0].read = [](size_t, size_t, void*, size_t) { return false; };
        check(!session_file_write(p.string(), meta, broken, id, written, error), "keep: failing write fails");
        check(slurp(p) == image, "keep: the previous session survives a failed write");
        check(slurp(planted) == std::vector<char>(5, 'p') && no_temp(dir), "keep: no other file touched");
    }
    // R1: the input list carries every expert file the loader resolved, per (layer, role) - a GGUF that only
    // native_experts.txt names changes the fingerprint when it changes; one file in two roles is two entries
    {
        const fs::path g1 = dir / "ext-a.gguf", g2 = dir / "ext-b.gguf";
        spit(g1, std::vector<char>(4096, 'g'));
        spit(g2, std::vector<char>(4096, 'h'));
        SessionInputs in;
        in.native_shards = {g1.string()};
        in.experts = {{"expert blk.0.ffn_gate", g1.string()}, {"expert blk.0.ffn_up", g2.string()},
                      {"expert blk.0.ffn_down", g2.string()}};
        const auto list = session_model_inputs(in);
        check(list.size() == 4, "inputs: shard plus three expert roles, none merged");
        uint64_t a = 0, b = 0, c = 0;
        check(session_model_fingerprint(list, a, error), "inputs: fingerprint");
        auto d = slurp(g2); d[0] = 'X'; spit(g2, d);   // the expert-only file changes; the CLI shard does not
        check(session_model_fingerprint(list, b, error) && a != b, "inputs: an expert-only GGUF enters the fingerprint");
        SessionInputs swapped = in;
        std::swap(swapped.experts[0].second, swapped.experts[1].second);
        check(session_model_fingerprint(session_model_inputs(swapped), c, error) && c != b,
              "inputs: the role each expert file plays enters the fingerprint");
    }
    // R2: runtime limits bound every array before it is allocated, and the file size follows from them
    {
        auto exact = [&] {
            SessionReadLimits l;
            l.max_tokens = 1000; l.max_checkpoints = 2; l.max_kv_layers = 3;
            l.geometry = original.geometry;
            l.layer_range = std::make_pair(original.layer_lo, original.layer_hi);
            l.max_state_bytes = {4099, 77, 301, 64, 8};
            for (const auto& k : original.kv)
                l.max_kv_bytes.push_back({k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()});
            return l;
        };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, exact()) && same(back, original),
              "limits: a file within the exact runtime limits reads");
        check(session_read_max_file_bytes(exact()) >= image.size() &&
              session_read_max_file_bytes(exact()) < UINT64_MAX, "limits: the runtime limits bound the file size");
        check(session_read_max_file_bytes(SessionReadLimits{}) == UINT64_MAX, "limits: open limits leave it open");
        auto l = exact(); l.max_state_bytes[0] = 4098;
        check(rejects(good, id, "limit", l), "limits: a GDN state over the runtime size refused");
        l = exact(); l.max_state_bytes[4] = 7;
        check(rejects(good, id, "limit", l), "limits: block_pos over the runtime size refused");
        l = exact(); l.max_kv_bytes[0][0] = original.kv[0].k.size() - 1;
        check(rejects(good, id, "limit", l), "limits: a K buffer over the runtime size refused");
        l = exact(); l.max_kv_bytes[2][1] = original.kv[2].v.size() - 1;
        check(rejects(good, id, "limit", l), "limits: the draft's V over the runtime size refused");
        l = exact(); (*l.geometry)[3] ^= 1;
        check(rejects(good, id, "geometry", l), "limits: another geometry refused before the state is read");
        l = exact(); l.layer_range = std::make_pair(int64_t(0), int64_t(24));
        check(rejects(good, id, "layer", l), "limits: another layer range refused");
        // a huge count in a state array with a recomputed hash: refused by the bound, nothing allocated
        auto d = image;
        // live: ids count (8) + 1000*4, imgs count (8) + 1*16, then the gdn byte count
        const size_t at = 64 + 21 * 8 + 8 + 1000 * 4 + 8 + 16;
        uint64_t was = 0; std::memcpy(&was, d.data() + at, 8);
        check(was == 4099, "limits: test offset finds the GDN count");
        uint64_t huge = 3ull << 30; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "bigstate.bin"; spit(p, d);
        check(rejects(p, id, nullptr, exact()), "limits: an oversized state count refused before allocation");
        // status kinds of a read
        SessionStatus st;
        check(!session_file_read(p.string(), id, back, n, error, exact(), &st) && st.error == SessionError::invalid,
              "status: a bad file is 'invalid'");
        l = exact();
        l.admit = [](uint64_t, std::string& why) { why = "no RAM"; return false; };
        check(!session_file_read(good.string(), id, back, n, error, l, &st) && st.error == SessionError::memory,
              "status: the RAM preflight is 'memory'");
    }
    // S: the streaming read (metadata and the whole-payload check, no K/V materialized) and the apply pass that
    // hands every K/V block to the caller - what a session larger than RAM restores through (conversation_file.hpp)
    {
        SessionReadLimits lim;
        lim.max_tokens = 1000; lim.max_checkpoints = 2; lim.max_kv_layers = 3;
        lim.max_state_bytes = {4099, 77, 301, 64, 8};
        uint64_t asked = 0;
        lim.admit = [&](uint64_t need, std::string&) { asked = need; return true; };
        SavedConversation meta;
        std::vector<std::array<uint64_t, 5>> sizes;
        size_t n = 0;
        check(session_file_read_streamed(good.string(), id, meta, n, error, lim, &sizes), "streamed: read succeeds");
        check(n == written, "streamed: reports the file size");
        bool sizes_ok = sizes.size() == original.kv.size();
        for (size_t i = 0; sizes_ok && i < sizes.size(); ++i) {
            const ConversationBuffer* bufs[5] = {&original.kv[i].k, &original.kv[i].v, &original.kv[i].k_scale,
                                                 &original.kv[i].v_scale, &original.kv[i].pooled};
            for (size_t p2 = 0; p2 < 5; ++p2) sizes_ok = sizes_ok && sizes[i][p2] == bufs[p2]->size();
        }
        check(sizes_ok, "streamed: the read pass reports every K/V part size");
        check(asked == session_read_stream_peak_bytes(image.size(), lim) &&
              asked < session_read_peak_bytes(image.size()),
              "streamed: the preflight prices the blocks and the metadata, not the snapshot");
        check(meta.kv.size() == original.kv.size(), "streamed: keeps the K/V layer headers");
        bool headers_ok = true, no_bytes = true;
        for (size_t i = 0; i < meta.kv.size(); ++i) {
            const ConversationKv& x = original.kv[i]; const ConversationKv& y = meta.kv[i];
            headers_ok = headers_ok && x.format == y.format && x.cells == y.cells && x.heads == y.heads &&
                         x.head_dim == y.head_dim && x.page_size == y.page_size && x.pooled_rows == y.pooled_rows &&
                         x.idx_dim == y.idx_dim;
            no_bytes = no_bytes && y.k.size() == 0 && y.v.size() == 0 && y.k_scale.size() == 0 &&
                       y.v_scale.size() == 0 && y.pooled.size() == 0;
        }
        check(headers_ok, "streamed: the headers match the image");
        check(no_bytes, "streamed: no K/V byte is materialized");
        check(same_checkpoint(original.live, meta.live) && meta.checkpoints.size() == original.checkpoints.size(),
              "streamed: keeps the running state");
        // the apply pass: in-order blocks of at most 16 MiB, together byte-identical to the image
        std::vector<std::vector<uint8_t>> part(meta.kv.size() * 5);
        size_t biggest = 0;
        check(session_file_apply_streamed(good.string(), id,
              [&](size_t layer, size_t slot, uint64_t offset, const void* data, size_t len) {
                  std::vector<uint8_t>& out = part[layer * 5 + slot];
                  if (offset != out.size()) return false;
                  const uint8_t* p = static_cast<const uint8_t*>(data);
                  out.insert(out.end(), p, p + len);
                  if (len > biggest) biggest = len;
                  return true;
              }, sizes, error), "streamed: apply succeeds");
        check(biggest > 0 && biggest <= (16u << 20), "streamed: blocks are at most 16 MiB");
        bool same_part = true;
        for (size_t i = 0; i < meta.kv.size(); ++i) {
            const ConversationBuffer* bufs[5] = {&original.kv[i].k, &original.kv[i].v, &original.kv[i].k_scale,
                                                 &original.kv[i].v_scale, &original.kv[i].pooled};
            for (size_t p2 = 0; p2 < 5; ++p2) {
                const std::vector<uint8_t>& got = part[i * 5 + p2];
                if (bufs[p2]->size() != got.size()) { same_part = false; continue; }
                bufs[p2]->visit(0, bufs[p2]->size(), [&](const uint8_t* q, size_t len, size_t at) {
                    same_part = same_part && std::memcmp(q, got.data() + at, len) == 0;
                    return same_part;
                });
            }
        }
        check(same_part, "streamed: the applied K/V equals the image byte for byte");
        // a file changed between the passes: the apply pass walks it to the end and refuses (the caller fail-stops)
        {
            const fs::path p = dir / "changed.bin";
            check(session_file_write(p.string(), original, id, written, error), "streamed: changed-file write");
            SavedConversation m2; size_t n2 = 0;
            check(session_file_read_streamed(p.string(), id, m2, n2, error), "streamed: changed-file read");
            auto d = slurp(p);
            d[d.size() / 2] ^= 0x20;   // a K/V payload byte
            spit(p, d);
            SessionStatus st;
            check(!session_file_apply_streamed(p.string(), id,
                  [](size_t, size_t, uint64_t, const void*, size_t) { return true; }, sizes, error, &st) &&
                  error.find("changed") != std::string::npos,
                  "streamed: a file changed between the passes is refused");
        }
        // a truncated file fails the read pass before anything could be applied
        {
            auto d = slurp(good);
            d.resize(d.size() / 2);
            const fs::path p = dir / "short.bin"; spit(p, d);
            SavedConversation m2; size_t n2 = 0;
            check(!session_file_read_streamed(p.string(), id, m2, n2, error),
                  "streamed: a truncated file fails the read pass");
            check(!session_file_apply_streamed(p.string(), id,
                  [](size_t, size_t, uint64_t, const void*, size_t) { return false; }, sizes, error),
                  "streamed: apply on a truncated file fails");
        }
        // the apply pass is bound to the read pass: a file grown between the passes (more K/V layers) or one with a
        // smaller part is refused before any block reaches the sink, instead of being applied against other targets
        {
            const fs::path p = dir / "grown.bin";
            SavedConversation bigger = original;
            bigger.kv.push_back(bigger.kv.back());
            check(session_file_write(p.string(), bigger, id, written, error), "streamed: grown-file write");
            size_t calls = 0;
            check(!session_file_apply_streamed(p.string(), id,
                  [&](size_t, size_t, uint64_t, const void*, size_t) { ++calls; return true; }, sizes, error) &&
                  calls == 0 && error.find("layer count") != std::string::npos,
                  "streamed: a file with more K/V layers than the read pass saw is refused before any block");
        }
        {
            const fs::path p = dir / "smallpart.bin";
            SavedConversation smaller = original;
            smaller.kv[0].k = pattern(smaller.kv[0].k.size() - 4096, 9);
            check(session_file_write(p.string(), smaller, id, written, error), "streamed: smaller-part write");
            size_t calls = 0;
            check(!session_file_apply_streamed(p.string(), id,
                  [&](size_t, size_t, uint64_t, const void*, size_t) { ++calls; return true; }, sizes, error) &&
                  calls == 0 && error.find("part size") != std::string::npos,
                  "streamed: a shrunken K/V part is refused before any block");
        }
        // the read pass rejects the same foreign or corrupted files the materializing read does
        check(!session_file_read_streamed(good.string(), {id.model ^ 1, id.config}, meta, n, error),
              "streamed: another model refused");
        auto d = image; d[image.size() / 2] ^= 0x40;
        const fs::path p = dir / "flip.bin"; spit(p, d);
        check(!session_file_read_streamed(p.string(), id, meta, n, error), "streamed: a corrupted byte refused");
    }
    // R3/R5: injected failures at each write step - kind, publication and the old file
    {
        const fs::path p = dir / "fault.bin";
        const std::vector<char> old(10, 'q');
        struct Case { const char* step; int err; SessionError kind; bool published; };
        for (const Case c : {Case{"write", ENOSPC, SessionError::storage, false},
#ifdef EDQUOT
                             Case{"write", EDQUOT, SessionError::storage, false},
#endif
                             Case{"write", EIO, SessionError::io, false},
                             Case{"file_flush", EIO, SessionError::io, false},
                             Case{"rename", EACCES, SessionError::io, false},
                             Case{"dir_flush", EIO, SessionError::io, true}}) {
            spit(p, old);
            SessionWriteOptions opt;
            const std::string want = c.step;
            const int code = c.err;
            opt.fault = [want, code](const char* step) { return want == step ? code : 0; };
            SessionStatus st;
            const bool ok = session_file_write(p.string(), original, id, written, error, opt, &st);
            check(!ok && st.error == c.kind, "fault: the failure has its kind");
            check(st.published == c.published, "fault: publication reported exactly");
            if (c.published) {
                check(slurp(p) == image && error.find("replaced") != std::string::npos,
                      "fault: after a published failure the new file is there and the error says so");
            } else {
                check(slurp(p) == old, "fault: a failure before the rename keeps the old file");
            }
            check(no_temp(dir), "fault: no temporary left behind");
        }
#ifndef _WIN32
        // a filesystem that cannot flush a folder: saved, and said
        spit(p, old);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::string(step) == "dir_flush" ? EINVAL : 0; };
        SessionStatus st;
        check(session_file_write(p.string(), original, id, written, error, opt, &st) && st.dir_flush_unsupported &&
              slurp(p) == image, "fault: EINVAL on the folder flush is not a failure, and is reported");
#endif
        // the free-space preflight is 'storage'
        SessionWriteOptions big;
        big.min_free_bytes = UINT64_MAX / 2;
        SessionStatus st2;
        check(!session_file_write(p.string(), original, id, written, error, big, &st2) &&
              st2.error == SessionError::storage, "status: the free-space preflight is 'storage'");
        check(std::string(session_error_name(SessionError::storage)) == "storage" &&
              std::string(session_error_name(SessionError::memory)) == "memory" &&
              std::string(session_error_name(SessionError::invalid)) == "invalid" &&
              std::string(session_error_name(SessionError::io)) == "io", "status: protocol names");
    }
    // B: progress after every block, the last partial one included (a file below 16 MiB reports too), and the
    // blocking steps announced before they start
    {
        const fs::path p = dir / "progress.bin";
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        std::vector<std::string> phases;
        SessionWriteOptions opt;
        opt.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        opt.phase = [&](const char* ph, uint64_t b) { phases.push_back(std::string(ph) + ":" + std::to_string(b)); };
        // a file below one block (the sample without its K/V layers) reports too: at the start and its one block
        SavedConversation small = original;
        small.kv.clear();
        check(session_file_write(p.string(), small, id, written, error, opt), "progress: small write");
        check(written > 0 && written < (16u << 20), "progress: the small file is below one block");
        check(seen.size() == 2 && seen[0] == std::make_pair(uint64_t(0), uint64_t(written)) &&
              seen[1] == std::make_pair(uint64_t(written), uint64_t(written)), "progress: start and the partial block");
        check(phases.size() == 2 && phases[0] == "flush:" + std::to_string(written) &&
              phases[1] == "publish:" + std::to_string(written), "progress: flush and publish announced, in order");
        opt.durable = false;
        phases.clear();
        check(session_file_write(p.string(), small, id, written, error, opt) && phases.size() == 1 &&
              phases[0].rfind("publish:", 0) == 0, "progress: no flush step announced without a flush");
        // a read reports every block, the first (the header's) included
        seen.clear();
        SessionReadLimits l;
        l.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 1 &&
              seen[0] == std::make_pair(uint64_t(n), uint64_t(n)), "progress: a small read reports its one block");
        // the fixed sample: one full block and a partial one, both reported on write and on read
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), original, id, written, error, opt) && written > (16u << 20) &&
              written < (32u << 20), "progress: sample write");
        check(seen.size() == 3 && seen[1].first == (16u << 20) && seen[2].first == written,
              "progress: the sample's full block and its partial last block");
        seen.clear();
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 2 &&
              seen[0].first == (16u << 20) && seen[1].first == n, "progress: the sample's read reports both blocks");
        // a multi-block streamed write: 16 MiB blocks then the partial one, never more than a block apart
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> big(1);
        big[0].sizes = {(40u << 20) + 123, 0, 0, 0, 0};
        big[0].read = [](size_t, size_t, void* d, size_t c) { std::memset(d, 0x5a, c); return true; };
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), meta, big, id, written, error, opt), "progress: 40 MiB write");
        bool steps = seen.size() == 4 && seen.front().first == 0 && seen.back().first == written;
        for (size_t i = 1; i < seen.size(); ++i)
            steps = steps && seen[i].first > seen[i - 1].first && seen[i].first - seen[i - 1].first <= (16u << 20);
        check(steps, "progress: every block reported, at most 16 MiB apart, the last one at the end");
        // the allowance of a blocking step: bounded, growing with its size
        check(session_phase_limit_s(0) == 60 && session_phase_limit_s(1198691396) == 60 + (1198691396 >> 22) &&
              session_phase_limit_s(UINT64_MAX) == 3600, "phase limit: 60 s + 1 s per 4 MiB, at most an hour");
    }
    // A: the SAVE's RAM preflight comes BEFORE the checkpoint is copied
    {
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.back().gdn.assign(8u << 20, 0x11);   // the deepest: an 8 MiB running state
        const auto before = chain;
        SessionSaveLive live;
        live.state_bytes = 5000; live.tokens = 1000; live.images = 1; live.kv_layers = 13;
        check(session_deepest_checkpoint(chain) == &chain[1], "save: deepest chosen by reference");
        {   // pin=N: the pinned checkpoint is the one kept, even when a deeper periodic one exists
            auto pinned_chain = chain;
            pinned_chain[0].pinned = true;
            check(session_deepest_checkpoint(pinned_chain) == &pinned_chain[0], "save: a pinned shared prefix beats a deeper checkpoint");
            check(session_deepest_checkpoint(chain) == &chain[1], "save: no pin, the deepest as before");
        }
        const uint64_t need = session_save_peak_bytes(&chain[1], live);
        check(need >= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (16u << 20) &&
              need <= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (18u << 20),
              "save: the peak counts two checkpoint copies, the live state, tokens and the buffer");
        SessionSaveLive huge = live;
        huge.state_bytes = UINT64_MAX;
        check(session_save_peak_bytes(&chain[1], huge) == UINT64_MAX, "save: the peak saturates");
        std::vector<ConversationCheckpoint> out(1);
        std::string why;
        uint64_t asked = 0;
        bool empty_when_asked = false;
        const auto refuse = [&](uint64_t n, std::string& w) { asked = n; empty_when_asked = out.empty(); w = "low RAM"; return false; };
        check(!session_save_checkpoints(chain, live, refuse, out, why) && why == "low RAM" && asked == need,
              "save: the preflight is asked for the peak and can refuse");
        check(empty_when_asked && out.empty(), "save: refused before any copy (nothing copied)");
        bool same_chain = chain.size() == before.size();
        for (size_t i = 0; same_chain && i < chain.size(); ++i) same_chain = same_checkpoint(chain[i], before[i]);
        check(same_chain, "save: a refusal leaves the live checkpoints untouched");
        const auto agree = [&](uint64_t, std::string&) { return true; };
        check(session_save_checkpoints(chain, live, agree, out, why) && out.size() == 1 &&
              same_checkpoint(out[0], chain[1]), "save: admitted, the deepest checkpoint is copied");
        check(session_save_checkpoints({}, live, agree, out, why) && out.empty(), "save: no checkpoints, none copied");
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__) && !defined(STRATA_NO_RLIMIT_TEST)
        // under a real address-space limit that cannot hold one more copy: the refusal comes first, no bad_alloc;
        // and the same limit with an admitting preflight does fail at the copy (the test can see the difference)
        std::vector<ConversationCheckpoint> fat(1);
        fat[0].ids = {1, 2, 3};
        fat[0].gdn.assign(256u << 20, 0x22);
        const fs::path old_file = dir / "old-session.bin";
        spit(old_file, std::vector<char>(9, 'o'));
        std::fflush(nullptr);
        const pid_t pid = ::fork();
        if (pid == 0) {
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) pages = 0; std::fclose(f); }
            struct rlimit rl {};
            rl.rlim_cur = rl.rlim_max = (rlim_t) pages * (rlim_t) ::sysconf(_SC_PAGESIZE) + (64u << 20);
            if (pages <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) ::_exit(10);
            std::vector<ConversationCheckpoint> o2;
            std::string w;
            const auto low = [](uint64_t n, std::string& m) { m = "need " + std::to_string(n); return false; };
            bool refused = false;
            try { refused = !session_save_checkpoints(fat, live, low, o2, w) && o2.empty(); } catch (...) { ::_exit(11); }
            if (!refused) ::_exit(12);
            try { session_save_checkpoints(fat, live, agree, o2, w); } catch (const std::bad_alloc&) { ::_exit(0); }
            ::_exit(13);   // the copy fitted: the limit did not bite, the test proves nothing
        }
        int status = 0;
        check(pid > 0 && ::waitpid(pid, &status, 0) == pid, "save: low-memory child ran");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "save: under a low address-space limit the preflight refuses before the copy (an admitted copy fails)");
        check(fat[0].gdn.size() == (256u << 20) && fat[0].gdn[12345] == 0x22 &&
              slurp(old_file) == std::vector<char>(9, 'o'), "save: live state and the old file intact");
#endif
    }
    // Opt-in pressure reclamation does not change the selected file bytes, even with pin=N.
    for (bool pin : {false, true}) {
        auto source = sample();
        source.checkpoints[0].pinned = pin;
        const auto live_before = source.live;
        auto expected = sample();
        expected.checkpoints = session_checkpoints_to_save(source.checkpoints);
        const fs::path before_path = dir / "reclaim-before.bin", after_path = dir / "reclaim-after.bin";
        check(session_file_write(before_path.string(), expected, id, written, error), "reclaim: baseline file written");
        ConversationCache parked(1u << 20, 4);
        SessionSaveLive sl; sl.state_bytes = source.live.bytes(); sl.tokens = source.live.ids.size();
        const auto freed = session_save_reclaim(source.checkpoints, parked, sl, 0,
                                                [] { return std::optional<uint64_t>(0); });
        check(freed.checkpoints == 1, "reclaim: a non-selected checkpoint released");
        check(same_checkpoint(source.live, live_before), "reclaim: live state remains intact");
        check(session_file_write(after_path.string(), source, id, written, error), "reclaim: file written after pressure");
        check(slurp(before_path) == slurp(after_path), "reclaim: file byte-identical to original selection");
        const auto old = slurp(after_path);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::strcmp(step, "write") == 0 ? ENOSPC : 0; };
        SessionStatus status;
        check(!session_file_write(after_path.string(), source, id, written, error, opt, &status) &&
              status.error == SessionError::storage && !status.published, "reclaim: later write failure reported");
        check(slurp(after_path) == old && same_checkpoint(source.live, live_before) && no_temp(dir),
              "reclaim: later failure preserves old file and live state, no temporary left");
    }
    // R6: the folder the free-space query is asked about
    {
        check(session_free_space_dir("\\\\server\\share\\sessions\\chat.bin", true) == "\\\\server\\share\\sessions\\",
              "free space: UNC subfolder keeps its trailing backslash");
        check(session_free_space_dir("\\\\server\\share\\chat.bin", true) == "\\\\server\\share\\",
              "free space: UNC share root keeps its trailing backslash");
        check(session_free_space_dir("//server/share/s/chat.bin", true) == "\\\\server\\share\\s\\",
              "free space: forward slashes become backslashes");
        check(session_free_space_dir("C:\\chat.bin", true) == "C:\\", "free space: drive root");
        check(session_free_space_dir("C:chat.bin", true) == "C:\\", "free space: drive-relative name");
        check(session_free_space_dir("chat.bin", true).empty(), "free space: bare name is the current folder");
        check(session_free_space_dir("/a/b/chat.bin", false) == "/a/b", "free space: POSIX folder");
        check(session_free_space_dir("/chat.bin", false) == "/", "free space: POSIX root");
        check(session_free_space_dir("chat.bin", false) == ".", "free space: POSIX bare name");
    }
#ifndef _WIN32
    // what restore will not open: a symbolic link, a hard-linked file, a FIFO, a directory
    {
        const fs::path link = dir / "link.bin";
        fs::create_symlink(good, link);
        check(rejects(link, id, "symbolic link"), "a symbolic link is not followed");
        const fs::path hard = dir / "hard.bin";
        fs::create_hard_link(dir / "again.bin", hard);
        check(rejects(hard, id, "hard link"), "a hard-linked file is refused");
        fs::remove(hard);
        check(!rejects(dir / "again.bin", id), "the same file with one name again reads");
        const fs::path fifo = dir / "fifo.bin";
        check(::mkfifo(fifo.c_str(), 0600) == 0, "mkfifo");
        check(rejects(fifo, id, "regular"), "a FIFO is refused without blocking");
        check(rejects(dir / "moved", id, "regular"), "a directory is refused");
    }
    // saving over a symbolic link replaces the link, never writes through it; the new file is the owner's only
    {
        const fs::path target = dir / "target.bin", link = dir / "out-link.bin";
        spit(target, std::vector<char>(7, 't'));
        fs::create_symlink(target, link);
        check(session_file_write(link.string(), original, id, written, error), "write at a link name");
        check(!fs::is_symlink(link) && slurp(link) == image, "the link was replaced by the file");
        check(slurp(target) == std::vector<char>(7, 't'), "the link's target is untouched");
        struct stat st {};
        check(::stat(link.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "the session file is mode 0600");
    }
#endif

    // ---- format v2 (--conversation-cache-disk-state-ref, PLAN §6): prefix saves whose deepest checkpoint's five
    // running-state arrays are all byte-identical to the live state's store one copy on disk and reference it.
    {
        SavedConversation h = sample();
        h.checkpoints = {homo(h.live, 10, 999, 0xbeefull)};
        const fs::path v1p = dir / "ref-v1.bin", v2p = dir / "ref-v2.bin";
        check(session_file_write(v1p.string(), h, id, written, error), "v2ref: the same image writes v1 without the flag");
        const size_t v1_size = written;
        SessionWriteOptions sro;
        sro.state_ref = true;
        check(session_file_write(v2p.string(), h, id, written, error, sro), "v2ref: a homomorphic prefix writes with the flag");
        const std::vector<char> v2file = slurp(v2p);
        // a REF hit emits version 2; a miss (or the flag off) stays byte-identical v1
        check(file_version_of(v2file) == 2, "v2ref: a REF hit emits version 2");
        check(file_version_of(slurp(v1p)) == 1, "v2ref: without the flag the file is v1");
        // exact whole-file savings: one live + one REF checkpoint saves state_bytes - 160 (the five v1 length words
        // become 24-byte descriptors for the live and the checkpoint); the returned bytes close with the file size
        const uint64_t state_bytes = checkpoint_state_bytes(h.checkpoints[0]);
        check(state_bytes == checkpoint_state_bytes(h.live), "v2ref: the fixture's checkpoint arrays equal the live state's");
        check(v1_size - v2file.size() == state_bytes - 160, "v2ref: exact whole-file savings state_bytes - 160");
        check(written == v2file.size() && written == fs::file_size(v2p), "v2ref: returned bytes are the file's size");
        // write-once sizing: the file is written at its final size in one pass (progress reaches the total, nothing
        // is written and truncated after)
        {
            const fs::path pp = dir / "ref-once.bin";
            SessionWriteOptions once;
            once.state_ref = true;
            std::vector<std::pair<uint64_t, uint64_t>> seen;
            once.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
            check(session_file_write(pp.string(), h, id, written, error, once), "v2ref: write-once pass");
            check(seen.size() >= 2 && seen.back() == std::make_pair(uint64_t(written), uint64_t(written)),
                  "v2ref: the single pass reports to the final size");
        }
        // the hand serializer must reproduce the writer's bytes exactly, or the mutation fixtures below prove nothing
        {
            V2Fixture fx = fx_v2(h, {{true, true, true, true, true}}, id);
            check(fx.file == v2file, "v2ref: the hand v2 serializer reproduces the writer's bytes");
        }
        // round trip: the checkpoint's arrays come back byte-identical; its own ids/imgs/used are preserved; the
        // restored checkpoint does not alias the live state (no read-side sharing)
        {
            SavedConversation back;
            size_t n = 0;
            check(session_file_read(v2p.string(), id, back, n, error) && same(back, h), "v2ref: round trip is identical");
            check(back.checkpoints[0].ids == h.checkpoints[0].ids && back.checkpoints[0].imgs == h.checkpoints[0].imgs &&
                  back.checkpoints[0].used == h.checkpoints[0].used, "v2ref: ids/imgs/used kept their stored values");
            back.checkpoints[0].gdn[0] ^= 0xff;
            check(back.checkpoints[0].gdn != back.live.gdn && back.live.gdn == h.live.gdn,
                  "v2ref: the restored checkpoint is an independent copy, not a reference into live");
            back.live.ple[0] ^= 0xff;
            check(back.live.ple != back.checkpoints[0].ple && back.checkpoints[0].ple == h.checkpoints[0].ple,
                  "v2ref: editing the live state does not touch the restored checkpoint");
        }
        // all-five discipline: one byte different in ANY of the five arrays, or any length different, falls back to
        // the byte-identical v1 file for the whole image
        for (int which = 0; which < 5; ++which) {
            for (int mode = 0; mode < 2; ++mode) {   // 0: one byte changed, 1: one byte longer
                SavedConversation s = sample();
                ConversationCheckpoint cp = homo(s.live, 10, 77, 0xabc);
                const auto one_byte = [](auto& v) { v[0] ^= 0xff; };
                const auto one_more = [](auto& v) { v.push_back(1); };
                switch (which) {
                    case 0: mode ? one_more(cp.gdn) : one_byte(cp.gdn); break;
                    case 1: mode ? one_more(cp.ple) : one_byte(cp.ple); break;
                    case 2: mode ? one_more(cp.tails) : one_byte(cp.tails); break;
                    case 3: mode ? one_more(cp.dead) : one_byte(cp.dead); break;
                    default: mode ? one_more(cp.block_pos) : one_byte(cp.block_pos); break;
                }
                s.checkpoints = {cp};
                const fs::path a = dir / "ref-mut-a.bin", b = dir / "ref-mut-b.bin";
                check(session_file_write(a.string(), s, id, written, error), "v2ref: mutation baseline writes");
                check(session_file_write(b.string(), s, id, written, error, sro), "v2ref: mutated state writes with the flag");
                check(slurp(a) == slurp(b) && file_version_of(slurp(b)) == 1,
                      mode ? "v2ref: a length change in one array keeps the whole file v1"
                           : "v2ref: a one-byte change in one array keeps the whole file v1");
            }
        }
        // partial groups: only the homomorphic checkpoints are referenced; the formula stays exact
        {
            SavedConversation s = sample();
            s.checkpoints = {homo(s.live, 10, 1, 0xa1), checkpoint(500, 3), homo(s.live, 500, 2, 0xb2)};
            const uint64_t sb = checkpoint_state_bytes(s.live);
            const fs::path a = dir / "ref-part-a.bin", b = dir / "ref-part-b.bin";
            check(session_file_write(a.string(), s, id, written, error), "v2ref: partial baseline writes");
            check(session_file_write(b.string(), s, id, written, error, sro), "v2ref: partial groups write with the flag");
            check(file_version_of(slurp(b)) == 2, "v2ref: a partial hit still emits version 2");
            check(slurp(a).size() - slurp(b).size() == 2 * sb - 80 * 4,
                  "v2ref: two REF checkpoints save 2*state_bytes - 80*(N+1) exactly");
            SavedConversation back; size_t n = 0;
            check(session_file_read(b.string(), id, back, n, error) && same(back, s),
                  "v2ref: partial groups round trip byte-identically");
        }
        // no checkpoint at all: the flag has nothing to reference and the file stays v1
        {
            SavedConversation s = sample();
            s.checkpoints.clear();
            const fs::path b = dir / "ref-none.bin";
            check(session_file_write(b.string(), s, id, written, error, sro) && file_version_of(slurp(b)) == 1,
                  "v2ref: no checkpoint, the flag stays v1");
        }
        // empty running-state arrays: the hit still works and the file is exactly 160 bytes LARGER (a tiny state has
        // no savings; the test asserts the formula, not a sign)
        {
            SavedConversation e = sample();
            e.live.ids = {9, 8, 7};
            e.live.imgs.clear();
            e.live.gdn.clear(); e.live.ple.clear(); e.live.tails.clear(); e.live.dead.clear();
            e.live.block_pos.clear();
            ConversationCheckpoint cp = homo(e.live, 1, 5, 3);
            cp.ids = {9};
            e.checkpoints = {cp};
            e.kv.clear();
            const fs::path a = dir / "ref-empty-a.bin", b = dir / "ref-empty-b.bin";
            check(session_file_write(a.string(), e, id, written, error), "v2ref: empty baseline writes");
            const size_t empty_v1 = written;
            check(session_file_write(b.string(), e, id, written, error, sro) && file_version_of(slurp(b)) == 2,
                  "v2ref: empty arrays still hit REF");
            check(written == empty_v1 + 160, "v2ref: an empty state is exactly 160 bytes larger (descriptor overhead)");
            SavedConversation back; size_t n = 0;
            check(session_file_read(b.string(), id, back, n, error) && same(back, e),
                  "v2ref: empty arrays round trip");
        }
        // huge/overflow boundaries: a REF descriptor whose logical length is not the live array's is refused at the
        // semantic check; a RAW stored count past the remaining payload is refused before any allocation; a v2 count
        // above the engine's limit is refused exactly like v1
        {
            V2Fixture fx = fx_v2(h, {{true, true, true, true, true}}, id);
            std::vector<char> d = fx.file;
            fx_patch(d, fx.codec[1][0] + 8, h.live.gdn.size() + (uint64_t(1) << 40));
            const fs::path p = dir / "ref-huge.bin"; spit(p, d);
            SavedConversation keep; size_t n = 0;
            check(!session_file_read(p.string(), id, keep, n, error) &&
                  error.find("differs from the live state's") != std::string::npos,
                  "v2ref: a REF logical length past the live array's is refused (no huge allocation)");
        }
        {
            V2Fixture fx = fx_v2(h, {{true, true, true, true, true}}, id);
            std::vector<char> d = fx.file;
            fx_patch(d, fx.codec[0][0] + 8, uint64_t(3) << 40);      // the live gdn is RAW: logical
            fx_patch(d, fx.codec[0][0] + 16, uint64_t(3) << 40);     // stored: would consume bytes that are not there
            const fs::path p = dir / "ref-overflow.bin"; spit(p, d);
            check(rejects(p, id, "element count exceeds the payload"),
                  "v2ref: a stored count past the remaining payload is refused before allocation");
        }
        {
            SessionReadLimits l;
            l.max_state_bytes = {16, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
            check(rejects(v2p, id, "exceeds this engine's limit", l),
                  "v2ref: an over-limit state array is refused with the v1 limit wording");
        }
    }

    // header-hash / version / reserved (PLAN 6.3, corrected): a patched version or reserved field with a RECOMPUTED
    // header hash reaches the unsupported-version / invalid-header checks; without the recompute the header checksum
    // fires first
    {
        const fs::path p = dir / "hdr.bin";
        {   // version 3, hash recomputed: unsupported version (v2 is a valid version now, so probe with 3)
            auto d = image;
            const uint32_t v = 3; std::memcpy(d.data() + 8, &v, 4);
            const uint64_t hh = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &hh, 8);
            spit(p, d);
            check(rejects(p, id, "unsupported version"), "v2hdr: a recomputed version 3 is an unsupported version");
        }
        {   // the same patch without the recompute: the header checksum is what refuses it
            auto d = image;
            const uint32_t v = 3; std::memcpy(d.data() + 8, &v, 4);
            spit(p, d);
            check(rejects(p, id, "header checksum"), "v2hdr: an unrecomputed version patch fails the header checksum");
        }
        {   // a reserved field set, hash recomputed: invalid header fields
            auto d = image;
            const uint64_t r = 1; std::memcpy(d.data() + 40, &r, 8);
            const uint64_t hh = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &hh, 8);
            spit(p, d);
            check(rejects(p, id, "invalid header"), "v2hdr: a set reserved field with a recomputed hash is an invalid header");
        }
        {   // a v1-labelled file holding a v2 payload is refused during the parse, before any output is replaced.
            // The rejection lands inside the v1 grammar's count check (the v2 descriptor words misread as v1 counts),
            // so the error text is an implementation detail - asserted to pin the failure point, not promised by PLAN
            SavedConversation h2 = sample();
            h2.checkpoints = {homo(h2.live, 10, 5, 9)};
            V2Fixture fx = fx_v2(h2, {{true, true, true, true, true}}, id);
            auto d = fx.file;
            const uint32_t v = 1; std::memcpy(d.data() + 8, &v, 4);
            const uint64_t hh = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &hh, 8);
            spit(p, d);
            SavedConversation keep = original; size_t n = 0;
            SessionStatus st;
            const bool ok = session_file_read(p.string(), id, keep, n, error, {}, &st);
            if (ok || error.find("element count exceeds the payload") == std::string::npos)
                std::fprintf(stderr, "  v1-labelled v2hdr: error was: %s\n", error.c_str());
            check(!ok && st.error == SessionError::invalid && st.file_version == 1 && n == 0 &&
                  same(keep, original) && error.find("element count exceeds the payload") != std::string::npos,
                  "v2hdr: a v1-labelled v2 payload is refused as invalid, the output untouched");
        }
        {   // the mirror: a v2-labelled file holding a v1 payload is refused the same way - the v2 reader sees the
            // v1 state lengths as codec words
            auto d = image;
            const uint32_t v = 2; std::memcpy(d.data() + 8, &v, 4);
            const uint64_t hh = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &hh, 8);
            spit(p, d);
            SavedConversation keep = original; size_t n = 0;
            SessionStatus st;
            const bool ok = session_file_read(p.string(), id, keep, n, error, {}, &st);
            if (ok || error.find("unknown state array codec") == std::string::npos)
                std::fprintf(stderr, "  v2-labelled v1hdr: error was: %s\n", error.c_str());
            check(!ok && st.error == SessionError::invalid && st.file_version == 2 && n == 0 &&
                  same(keep, original) && error.find("unknown state array codec") != std::string::npos,
                  "v2hdr: a v2-labelled v1 payload is refused as invalid, the output untouched");
        }
        {   // the same v1-labelled v2 payload through the streamed read pass and the apply pass: both dispatch on the
            // header's version, so both refuse before any output (the metadata, the byte count, the sink) is touched
            SavedConversation h2 = sample();
            h2.checkpoints = {homo(h2.live, 10, 5, 9)};
            V2Fixture fx = fx_v2(h2, {{true, true, true, true, true}}, id);
            auto d = fx.file;
            const uint32_t v = 1; std::memcpy(d.data() + 8, &v, 4);
            const uint64_t hh = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &hh, 8);
            const fs::path s = dir / "hdr-str.bin"; spit(s, d);
            SavedConversation meta = original; size_t n = 0;
            SessionStatus st;
            const bool ok = session_file_read_streamed(s.string(), id, meta, n, error, {}, nullptr, &st);
            if (ok || error.find("element count exceeds the payload") == std::string::npos)
                std::fprintf(stderr, "  streamed v1-labelled v2hdr: error was: %s\n", error.c_str());
            check(!ok && st.error == SessionError::invalid && st.file_version == 1 && n == 0 &&
                  same(meta, original) && error.find("element count exceeds the payload") != std::string::npos,
                  "v2hdr: the streamed read pass refuses a v1-labelled v2 payload, the metadata untouched");
            size_t sink_calls = 0;
            const bool aok = session_file_apply_streamed(s.string(), id,
                  [&](size_t, size_t, uint64_t, const void*, size_t) { ++sink_calls; return true; },
                  {}, error, &st);
            if (aok || error.find("element count exceeds the payload") == std::string::npos)
                std::fprintf(stderr, "  apply v1-labelled v2hdr: error was: %s\n", error.c_str());
            check(!aok && sink_calls == 0 && st.error == SessionError::invalid && st.file_version == 1 &&
                  error.find("element count exceeds the payload") != std::string::npos,
                  "v2hdr: the apply pass refuses a v1-labelled v2 payload before the first sink byte");
        }
    }

    // descriptor hash coverage (PLAN 6.3): the trailer's payload hash is recomputed independently over the whole
    // payload, descriptors included
    {
        SavedConversation h2 = sample();
        h2.checkpoints = {homo(h2.live, 10, 999, 0xbeefull)};
        SessionWriteOptions sro; sro.state_ref = true;
        const fs::path p = dir / "ref-hash.bin";
        check(session_file_write(p.string(), h2, id, written, error, sro), "v2hash: write");
        const std::vector<char> d = slurp(p);
        const size_t payload = d.size() - 64 - 16;
        uint64_t stored = 0; std::memcpy(&stored, d.data() + d.size() - 16, 8);
        check(session_hash64(d.data() + 64, payload, 0) == stored,
              "v2hash: the payload hash covers the descriptors (independent recompute matches the trailer)");
    }

    // mixed RAW/REF reader fixture (PLAN 6.2): the writer never emits a partial reference, but the grammar allows any
    // legal mix; a checkpoint may hold REF arrays beside RAW ones, and RAW checkpoints may coexist with REF ones
    {
        SavedConversation m = sample();
        m.checkpoints = {homo(m.live, 10, 11, 0xc1), checkpoint(500, 6)};
        V2Fixture fx = fx_v2(m, {{true, true, false, false, false}, {false, false, false, false, false}}, id);
        const fs::path p = dir / "ref-mixed.bin";
        spit(p, fx.file);
        SavedConversation back; size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error) && same(back, m),
              "v2mix: a hand-built RAW/REF-per-array fixture reads back byte-identically");
        check(file_version_of(fx.file) == 2 && back.checkpoints[1].gdn == m.checkpoints[1].gdn,
              "v2mix: the RAW checkpoint's arrays are its own bytes");
    }

    // semantic codec/descriptor refusals (PLAN 6.3): each illegal field is rejected at the semantic check with the
    // file's version known (2) and the caller's output object untouched
    {
        SavedConversation h = sample();
        h.checkpoints = {homo(h.live, 10, 999, 0xbeefull), checkpoint(500, 6)};
        const auto expect_bad = [&](const char* label, std::vector<std::pair<size_t, uint64_t>> patches,
                                    const char* want) {
            V2Fixture fx = fx_v2(h, {{true, true, true, true, true}, {false, false, false, false, false}}, id);
            for (const auto& [off, value] : patches) fx_patch(fx.file, off, value);
            const fs::path p = dir / "ref-bad.bin";
            spit(p, fx.file);
            SavedConversation keep = sample();
            const SavedConversation before = keep;
            size_t n = 0;
            SessionStatus st;
            const bool ok = session_file_read(p.string(), id, keep, n, error, {}, &st);
            if (ok || error.find(want) == std::string::npos)
                std::fprintf(stderr, "  %s: error was: %s\n", label, error.c_str());
            check(!ok && error.find(want) != std::string::npos, label);
            check(st.error == SessionError::invalid && st.file_version == 2, "  (bad file, version known)");
            check(same(keep, before), "  (the caller's output is untouched)");
        };
        V2Fixture base = fx_v2(h, {{true, true, true, true, true}, {false, false, false, false, false}}, id);
        expect_bad("v2bad: an unknown codec is rejected at the semantic check", {{base.codec[1][0], 7}},
                   "unknown state array codec");
        expect_bad("v2bad: the live state cannot reference itself", {{base.codec[0][0], 1}},
                   "the live state cannot reference itself");
        expect_bad("v2bad: a REF array stores no bytes", {{base.codec[1][1] + 16, 8}},
                   "a REF state array stores no bytes");
        expect_bad("v2bad: a REF length must equal the live array's", {{base.codec[1][2] + 8, h.live.tails.size() + 8}},
                   "a REF state array's length differs from the live state's");
        expect_bad("v2bad: a RAW stored size must equal its logical size",
                   {{base.codec[2][3] + 16, h.checkpoints[1].dead.size() + 8}},
                   "a RAW state array's stored size differs from its logical size");
        expect_bad("v2bad: a RAW logical size must equal its stored size",
                   {{base.codec[2][4] + 8, h.checkpoints[1].block_pos.size() + 1}},
                   "a RAW state array's stored size differs");
        expect_bad("v2bad: a RAW count past the payload is refused before allocation",
                   {{base.codec[2][0] + 8, uint64_t(1) << 40}, {base.codec[2][0] + 16, uint64_t(1) << 40}},
                   "element count exceeds the payload");
    }

    // truncation at the actual first rejection point (PLAN 6.3): any cut shortens the file, so the header's size
    // field mismatches the file before the payload is parsed; the failure is 'invalid' and the output is untouched
    {
        SavedConversation h = sample();
        h.checkpoints = {homo(h.live, 10, 999, 0xbeefull)};
        SessionWriteOptions sro; sro.state_ref = true;
        const fs::path p = dir / "ref-cut.bin";
        check(session_file_write(p.string(), h, id, written, error, sro), "v2cut: write");
        const std::vector<char> whole = slurp(p);
        for (size_t cut : {whole.size() - 1, whole.size() - 8, whole.size() - 16, whole.size() - 17, whole.size() / 2}) {
            const fs::path c = dir / "ref-cut-t.bin";
            spit(c, std::vector<char>(whole.begin(), whole.begin() + (std::ptrdiff_t) cut));
            SavedConversation keep; size_t n = 0;
            SessionStatus st;
            check(!session_file_read(c.string(), id, keep, n, error, {}, &st) &&
                  error.find("does not match the header") != std::string::npos && st.error == SessionError::invalid,
                  "v2cut: a truncated v2 file is refused at the header size mismatch");
        }
        // a payload byte flipped inside the K/V region parses and is caught by the payload hash
        auto d = whole;
        d[d.size() / 2] ^= 0x40;
        const fs::path f = dir / "ref-cut-f.bin";
        spit(f, d);
        check(rejects(f, id, "payload checksum"), "v2cut: a flipped payload byte fails the payload hash");
    }

    // v2 memory admission (PLAN 6.4): the read keeps the v1 upfront screen, then checks EVERY real allocation point
    // (ids/imgs, the live RAW arrays, each checkpoint's REF/RAW arrays, the checkpoint and K/V directories, the K/V
    // buffers).  Refusing at each point in turn must give a memory failure with the file's version known (2), never
    // touching the caller's output; the per-point asks price single allocations, not a running total.
    {
        SavedConversation h = sample();
        h.checkpoints = {homo(h.live, 10, 999, 0xbeefull)};
        SessionWriteOptions sro; sro.state_ref = true;
        const fs::path p = dir / "ref-adm.bin";
        check(session_file_write(p.string(), h, id, written, error, sro), "v2adm: write");
        std::vector<uint64_t> asks;
        SessionReadLimits all;
        all.admit = [&](uint64_t need, std::string&) { asks.push_back(need); return true; };
        SavedConversation back; size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error, all) && same(back, h), "v2adm: an admitting read succeeds");
        check(asks.size() > 1 && asks[0] == session_read_peak_bytes(fs::file_size(p)),
              "v2adm: the upfront screen is the v1 ask, then per-allocation checks follow");
        bool per_allocation = true;
        for (size_t i = 1; i < asks.size(); ++i) per_allocation = per_allocation && asks[i] <= asks[0];
        check(per_allocation, "v2adm: per-allocation asks price single allocations, not a cumulative total");
        for (size_t allow = 0; allow + 1 < asks.size(); ++allow) {
            SessionReadLimits l;
            size_t calls = 0;
            l.admit = [&](uint64_t, std::string& w) { ++calls; w = "controlled v2 refusal"; return calls <= allow + 1; };
            SavedConversation keep; size_t n2 = 0;
            SessionStatus st;
            check(!session_file_read(p.string(), id, keep, n2, error, l, &st) &&
                  st.error == SessionError::memory && st.file_version == 2 &&
                  error.find("controlled v2 refusal") != std::string::npos,
                  "v2adm: every allocation point can refuse with a memory failure (v2 known)");
        }
        {   // refusing the upfront screen itself is a memory failure too, with the version already verified
            SessionReadLimits l;
            l.admit = [](uint64_t, std::string& w) { w = "controlled v2 refusal"; return false; };
            SavedConversation keep; size_t n2 = 0;
            SessionStatus st;
            check(!session_file_read(p.string(), id, keep, n2, error, l, &st) &&
                  st.error == SessionError::memory && st.file_version == 2,
                  "v2adm: the upfront screen refusal is memory, the v2 file distinguished from a bad file");
        }
    }

    // the v2 physical size gate (PLAN 3.4/6.4): an all-RAW v2 file (a legal layout the writer never emits) can be
    // larger than the v1-derived bound - only just, so this uses a tiny conversation where the bound's slack is
    // smaller than the descriptor overhead.  The v2 bound (the v1 bound plus 80*(1+max_checkpoints)) must admit it;
    // a v1 file past the v1-derived bound is refused exactly as before, and the absolute file cap binds v2 too
    {
        SavedConversation t = sample();
        t.live.ids = {1, 2, 3};
        t.live.imgs.clear();
        t.checkpoints = {homo(t.live, 3, 9, 5)};
        t.kv.clear();
        const fs::path hv1 = dir / "gate-v1.bin";
        check(session_file_write(hv1.string(), t, id, written, error), "v2gate: v1 write");
        auto exact = [&] {
            SessionReadLimits l;
            l.max_tokens = 3; l.max_checkpoints = 1; l.max_kv_layers = 0;
            l.geometry = t.geometry;
            l.layer_range = std::make_pair(t.layer_lo, t.layer_hi);
            l.max_state_bytes = {4099, 77, 301, 64, 8};
            return l;
        };
        check(session_read_max_file_bytes(exact()) >= fs::file_size(hv1) &&
              session_read_max_file_bytes(exact()) < fs::file_size(hv1) + 160,
              "v2gate: the v1-derived bound keeps its v1 slack (smaller than the v2 descriptor overhead here)");
        V2Fixture raw = fx_v2(t, {{false, false, false, false, false}}, id);
        check(raw.file.size() == fs::file_size(hv1) + 160,
              "v2gate: all-RAW v2 is exactly 80*(1+N) bytes over the v1 encoding");
        check(raw.file.size() > session_read_max_file_bytes(exact()),
              "v2gate: the all-RAW v2 file is past the v1-derived bound");
        const fs::path hv2 = dir / "gate-v2raw.bin";
        spit(hv2, raw.file);
        SavedConversation back; size_t n = 0;
        check(session_file_read(hv2.string(), id, back, n, error, exact()) && same(back, t),
              "v2gate: the v2 physical bound admits a file the v1-derived bound would not");
        {
            SessionReadLimits tight = exact();
            tight.max_state_bytes[0] = 4039;
            const uint64_t v1bound = session_read_max_file_bytes(tight);
            check(v1bound < fs::file_size(hv1), "v2gate: the tightened bound is below the v1 file's size");
            check(rejects(hv1, id, "exceeds this engine's limit", tight),
                  "v2gate: a v1 file past the v1-derived bound is still refused");
            check(raw.file.size() > v1bound + 160 && rejects(hv2, id, "exceeds this engine's limit", tight),
                  "v2gate: v2 admits only the descriptor slack, not more");
        }
        {
            SessionReadLimits cap;
            cap.max_file_bytes = raw.file.size() - 1;
            check(rejects(hv2, id, "exceeds this engine's limit", cap),
                  "v2gate: the absolute file cap binds a v2 file too");
        }
    }

    // v2 streamed read + apply (PLAN 6.4): the read pass admits its blocks and metadata per allocation; the apply
    // pass, with the read pass's metadata still resident, is bounded by the read pass and asks one additional peak
    // before the first sink byte - a refusal there applies nothing and changes nothing
    {
        SavedConversation h = sample();
        h.checkpoints = {homo(h.live, 10, 999, 0xbeefull)};
        SessionWriteOptions sro; sro.state_ref = true;
        const fs::path p = dir / "ref-str.bin";
        check(session_file_write(p.string(), h, id, written, error, sro), "v2str: write");
        std::vector<uint64_t> asks;
        SessionReadLimits lim;
        lim.admit = [&](uint64_t need, std::string&) { asks.push_back(need); return true; };
        SavedConversation meta;
        std::vector<std::array<uint64_t, 5>> sizes;
        size_t n = 0;
        check(session_file_read_streamed(p.string(), id, meta, n, error, lim, &sizes), "v2str: the read pass succeeds");
        check(asks.size() > 1 && asks[0] == session_read_stream_peak_bytes(fs::file_size(p), lim),
              "v2str: the read pass screens upfront, then checks each allocation (the I/O block included)");
        for (size_t allow = 0; allow + 1 < asks.size(); ++allow) {
            SessionReadLimits l;
            size_t calls = 0;
            l.admit = [&](uint64_t, std::string& w) { ++calls; w = "controlled v2 refusal"; return calls <= allow + 1; };
            SavedConversation m2; size_t n2 = 0;
            SessionStatus st;
            check(!session_file_read_streamed(p.string(), id, m2, n2, error, l, nullptr, &st) &&
                  st.error == SessionError::memory && st.file_version == 2,
                  "v2str: every read-pass allocation point can refuse with a memory failure (v2 known)");
        }
        check(same_checkpoint(meta.live, h.live) && same_checkpoint(meta.checkpoints[0], h.checkpoints[0]),
              "v2str: the read pass restores the referenced checkpoint's state");
        // the apply pass's finite bounds come from the read pass's metadata, not from open limits
        SessionReadLimits bounds;
        session_read_bounds_from_meta(meta, bounds);
        check(bounds.max_checkpoints == meta.checkpoints.size() && bounds.max_kv_layers == meta.kv.size() &&
              bounds.geometry && *bounds.geometry == meta.geometry &&
              bounds.max_state_bytes[0] == h.live.gdn.size() && bounds.max_state_bytes[0] == meta.checkpoints[0].gdn.size(),
              "v2str: the apply bounds derive from the read pass's metadata");
        // an admitted apply delivers the K/V byte-identically and asks exactly one additional peak
        {
            SessionReadLimits b2 = bounds;
            size_t ask_calls = 0;
            uint64_t asked = 0;
            b2.admit = [&](uint64_t need, std::string&) { ++ask_calls; asked = need; return true; };
            std::vector<std::vector<uint8_t>> parts(meta.kv.size() * 5);
            size_t biggest = 0;
            check(session_file_apply_streamed(p.string(), id,
                  [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t len) {
                      std::vector<uint8_t>& out = parts[layer * 5 + part];
                      if (offset != out.size()) return false;
                      const uint8_t* q = static_cast<const uint8_t*>(data);
                      out.insert(out.end(), q, q + len);
                      if (len > biggest) biggest = len;
                      return true;
                  }, sizes, error, nullptr, &b2), "v2str: an admitted apply succeeds");
            check(ask_calls == 1 && asked == session_stream_apply_peak_bytes(bounds),
                  "v2str: the apply pass asks exactly its additional coexisting peak");
            check(biggest > 0 && biggest <= (16u << 20), "v2str: blocks are at most 16 MiB");
            bool same_part = true;
            for (size_t i = 0; same_part && i < meta.kv.size(); ++i) {
                const ConversationBuffer* bufs[5] = {&h.kv[i].k, &h.kv[i].v, &h.kv[i].k_scale, &h.kv[i].v_scale,
                                                     &h.kv[i].pooled};
                for (size_t p2 = 0; p2 < 5; ++p2) {
                    std::vector<uint8_t> want;
                    bufs[p2]->visit(0, bufs[p2]->size(), [&](const uint8_t* q, size_t c, size_t) {
                        want.insert(want.end(), q, q + c);
                        return true;
                    });
                    same_part = same_part && parts[i * 5 + p2] == want;
                }
            }
            check(same_part, "v2str: the applied K/V equals the image byte for byte");
        }
        // a refusal before the first sink byte: nothing is applied, the file is byte-identical, the failure is memory
        {
            SessionReadLimits low;
            session_read_bounds_from_meta(meta, low);
            size_t ask_calls = 0, sink_calls = 0;
            low.admit = [&](uint64_t, std::string& w) { ++ask_calls; w = "controlled apply refusal"; return false; };
            const std::vector<char> before = slurp(p);
            SessionStatus st;
            check(!session_file_apply_streamed(p.string(), id,
                  [&](size_t, size_t, uint64_t, const void*, size_t) { ++sink_calls; return true; },
                  sizes, error, &st, &low) &&
                  sink_calls == 0 && ask_calls == 1 &&
                  st.error == SessionError::memory && st.file_version == 2 &&
                  error.find("controlled apply refusal") != std::string::npos,
                  "v2str: the apply pass refuses before the first sink byte when the coexisting peak is denied");
            check(slurp(p) == before, "v2str: a refused apply leaves the file byte-identical");
        }
        // the pre-feature call form (no bounds) still applies a v2 file
        check(session_file_apply_streamed(p.string(), id,
              [](size_t, size_t, uint64_t, const void*, size_t) { return true; }, sizes, error),
              "v2str: the legacy apply form (no bounds) still serves a v2 file");
    }

    // the delta writer never takes the flag: a record for a homomorphic state is still a v1 record, sized and read
    // exactly as before (PLAN 6.5: delta encoding and size goldens unchanged)
    {
        SavedConversation grown = sample();
        grown.live.ids.push_back(4242);
        grown.checkpoints = {homo(grown.live, 10, 77, 7)};
        SavedConversation meta = grown;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : grown.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t c2) {
                return parts[part]->read(dst, offset, c2);
            };
            sources.push_back(std::move(s));
        }
        const fs::path log = dir / "ref-delta.dlog";
        SessionWriteOptions sro; sro.state_ref = true;   // ignored by the delta writer
        uint64_t rec = 0;
        bool overflow = false;
        const uint64_t estimate = session_delta_record_bytes(meta, sources, &overflow);
        check(!overflow && estimate > 0 &&
              session_delta_write(log.string(), 0, meta, sources, (int64_t) grown.live.ids.size(), id, rec, error, sro) &&
              rec == estimate && (uint64_t) fs::file_size(log) == rec,
              "v2delta: the delta writer ignores the flag and sizes the record exactly as before");
        SessionDeltaRecord read_back;
        uint64_t next = 0;
        check(session_delta_read(log.string(), 0, rec, id, {}, read_back, next, error) && next == rec &&
              read_back.meta.live.ids == grown.live.ids,
              "v2delta: the record is still a v1 record and reads back");
        fs::remove(log);
    }

    fs::remove_all(dir);
    std::printf("conversation_file_test: %d checks passed\n", checks);
    return 0;
}
