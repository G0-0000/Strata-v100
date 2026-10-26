// Host benchmark for the session file/delta writers' stage timings (SessionWriteTimings), at the production
// geometry of the 2026-10-25 RHT .15 deployment: one live state plus the deepest checkpoint, each GDN state
// logical 117,669,888 B (RHT stored 61,931,520 B), synthetic finite f32, K/V from preallocated host memory.
// It writes base and delta, RAW and RHT, durable=false, several interleaved rounds, and reports per-stage
// medians/ranges.  Pure host: no GPU, no engine, output goes to a scratch directory that it removes.
//
// Timing-accounting contract (nested counters, do NOT sum all fields):
//   state_ns     covers the meta/state section and INCLUDES rht_ns (the encode happens inside it);
//   kv_ns        covers the K/V section framing and INCLUDES kv_source_ns (the source reads happen inside it);
//   os_write_ns  accumulates only the time the producing thread spends inside OS write() calls (drains + the
//                padded tail block).  Drains fire synchronously from commit() DURING the state/kv/finish
//                windows: this is serial interleaving on one thread, not parallelism, and os_write_ns is a
//                NESTED SUBSET of those windows (the final block write is inside finish_ns too) - never add
//                it to any stage;
//   finish_ns    truncate + flush( skipped, durable=false) + close of the temporary file;
//   publish_ns   rename + folder flush (base only; the delta appender never renames).
// Serial writer time == state_ns + kv_ns + finish_ns + publish_ns.

#include "strata/core/conversation_file.hpp"
#include "strata/core/rht_state_codec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

static int fail(const std::string& what) {
    std::fprintf(stderr, "bench setup failed: %s\n", what.c_str());
    return 1;
}

// The production geometry, exactly as rht_state_codec_test's production case: 48 layers, QSA every 4th
// (36 GDN layers), rows 6144 x dim 128 recurrent plus the conv tail.
static std::array<int64_t, 18> production_geometry() {
    return {2560, 48, 4, 128, 16, 48, 4, 10240, 6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};
}

// deterministic finite filler (LCG), values in [-1, 1] with a few larger outliers
static void fill_f32(std::vector<float>& v, uint64_t& seed) {
    for (auto& f : v) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        f = (float) ((int32_t) (seed >> 40) & 0xFFFFF) * (2.0f / 0xFFFFF) - 1.0f;
    }
    for (size_t i = 0; i < v.size(); i += 4096) v[i] *= 8.0f;
}

static void set_ids(ConversationCheckpoint& c, int64_t tokens, uint64_t& seed) {
    c.ids.clear();
    c.ids.reserve((size_t) tokens);
    for (int64_t i = 0; i < tokens; ++i) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        c.ids.push_back((int32_t) (seed >> 32));
    }
}

static void set_bytes(std::vector<uint8_t, VmaAllocator<uint8_t>>& dst, const std::vector<uint8_t>& src) {
    dst.resize(src.size());
    if (!src.empty()) std::memcpy(dst.data(), src.data(), src.size());
}

// one synthetic running state: finite f32 GDN at the production logical size, ple/tails/dead/block_pos small
static ConversationCheckpoint make_state(int64_t tokens, uint64_t gdn_bytes, uint64_t& seed) {
    ConversationCheckpoint c;
    set_ids(c, tokens, seed);
    std::vector<float> gdn(gdn_bytes / sizeof(float));
    fill_f32(gdn, seed);
    set_bytes(c.gdn, std::vector<uint8_t>((const uint8_t*) gdn.data(), (const uint8_t*) gdn.data() + gdn_bytes));
    std::vector<uint8_t> ple(360u << 10), tails(4096), dead(2048), block_pos(2048);
    uint64_t s2 = seed ^ 0x5bd1e995ull;
    for (auto* v : {&ple, &tails, &dead, &block_pos})
        for (auto& b : *v) {
            s2 = s2 * 6364136223846793005ull + 1442695040888963407ull;
            b = (uint8_t) (s2 >> 56);
        }
    seed = s2;
    set_bytes(c.ple, ple);
    c.tails = tails;
    c.dead = dead;
    c.block_pos = block_pos;
    c.used = 1;
    return c;
}

// preallocated host K/V for one layer; `read` copies straight out of it like a resident host pool would
struct Part {
    std::vector<uint8_t> bytes;
};
struct LayerHost {
    std::array<Part, 5> part;
};

static SessionKvSource make_source(const LayerHost& h, std::array<size_t, 5> first, int64_t cells, bool draft) {
    SessionKvSource s;
    s.format = 0;
    s.cells = cells;
    s.heads = draft ? 1 : 2;
    s.head_dim = 128;
    s.page_size = 256;
    s.pooled_rows = draft ? cells : (cells + 15) / 16;
    s.idx_dim = 64;
    for (size_t p = 0; p < 5; ++p) {
        s.sizes[p] = h.part[p].bytes.size();
        s.first_bytes[p] = std::min(first[p], s.sizes[p]);
    }
    s.read = [&h](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset + n > h.part[part].bytes.size()) return false;
        std::memcpy(dst, h.part[part].bytes.data() + offset, n);
        return true;
    };
    return s;
}

struct Sample {
    SessionWriteTimings t;
    uint64_t wall_ns = 0;
    size_t bytes = 0;
    uint32_t file_version = 0;
};

struct Stats {
    uint64_t median = 0, lo = 0, hi = 0;
};

static Stats stats_of(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.median = v[v.size() / 2];
    s.lo = v.front();
    s.hi = v.back();
    return s;
}

static double ms(uint64_t ns) { return (double) ns / 1.0e6; }

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 3;
    if (rounds < 1) return fail("rounds");
    const fs::path out_dir = argc > 2 ? fs::path(argv[2]) : fs::path("build-test/conversation_file_bench_tmp");
    fs::remove_all(out_dir);
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) return fail("cannot create " + out_dir.string() + ": " + ec.message());

    rht::Layout layout{};
    if (!rht::layout_from(production_geometry(), 0, 48, layout)) return fail("production geometry rejected");
    if (layout.logical != 117669888ull || layout.stored != 61931520ull)
        return fail("unexpected layout sizes");
    std::printf("geometry: GDN logical=%" PRIu64 " B (%.1f MiB), RHT stored=%" PRIu64 " B (%.1f MiB), "
                "layers=%" PRIu64 " rows=%" PRIu64 " dim=%" PRIu64 " conv_floats=%" PRIu64 "\n",
                layout.logical, layout.logical / 1048576.0, layout.stored, layout.stored / 1048576.0,
                layout.layers, layout.rows, layout.dim, layout.conv_floats);

    uint64_t seed = 0xC0FFEE1234567ull;
    SavedConversation image;
    image.geometry = production_geometry();
    image.layer_lo = 0;
    image.layer_hi = 48;
    image.cvec = true;
    image.live = make_state(6166, layout.logical, seed);          // production A: 6166 prompt tokens
    image.checkpoints.push_back(make_state(6000, layout.logical, seed));

    // K/V host pools.  Part sizes are chosen so the RHT base file lands on the production 218,849,020 B
    // (two RHT states ~118.125 MiB + ~89.9 MiB K/V); the sizes print below next to what the writer reports.
    std::vector<LayerHost> layers(13);                            // 12 main + 1 draft
    for (size_t l = 0; l < layers.size(); ++l) {
        const bool draft = l == layers.size() - 1;
        const size_t k = 3542016, v = 3542016, ks = 17408, vs = 17408;
        const size_t pooled = draft ? 1243612 : 32768;
        const std::array<size_t, 5> sizes = {k, v, ks, vs, pooled};
        for (size_t p = 0; p < 5; ++p) {
            layers[l].part[p].bytes.resize(sizes[p]);
            for (auto& b : layers[l].part[p].bytes) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                b = (uint8_t) (seed >> 56);
            }
        }
    }
    uint64_t kv_bytes = 0;
    for (const auto& l : layers)
        for (const auto& p : l.part) kv_bytes += p.bytes.size();
    std::printf("state: live+1 checkpoint, ids 6166/6000 tokens; K/V host pool: %zu layers, %" PRIu64
                " B (%.1f MiB)\n", layers.size(), kv_bytes, kv_bytes / 1048576.0);

    // delta: the 6198-token follow-up; only part tails rewrite (production delta record ~125,250,924 B,
    // K/V tails < 1 MiB, the rest two RHT states again)
    SavedConversation grown = image;
    set_ids(grown.live, 6198, seed);
    set_ids(grown.checkpoints[0], 6100, seed);
    const std::array<size_t, 5> main_tail = {16384, 16384, 0, 0, 4096};
    const std::array<size_t, 5> draft_tail = {16384, 16384, 0, 0, 108644};

    auto base_sources = [&]() {
        std::vector<SessionKvSource> out;
        for (size_t l = 0; l < layers.size(); ++l)
            out.push_back(make_source(layers[l], {0, 0, 0, 0, 0}, 6166, l == layers.size() - 1));
        return out;
    };
    auto delta_sources = [&]() {
        std::vector<SessionKvSource> out;
        for (size_t l = 0; l < layers.size(); ++l) {
            const auto& tail = l == layers.size() - 1 ? draft_tail : main_tail;
            std::array<size_t, 5> first{};
            for (size_t p = 0; p < 5; ++p)
                first[p] = tail[p] ? layers[l].part[p].bytes.size() - tail[p] : layers[l].part[p].bytes.size();
            out.push_back(make_source(layers[l], first, 6198, l == layers.size() - 1));
        }
        return out;
    };
    uint64_t delta_kv = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& tail = l == layers.size() - 1 ? draft_tail : main_tail;
        for (size_t p = 0; p < 5; ++p) delta_kv += tail[p];
    }
    std::printf("delta: 6198 tokens, K/V tails %" PRIu64 " B (%.1f MiB)\n\n", delta_kv, delta_kv / 1048576.0);

    const SessionFileIdentity id{0xD00D, 0xFEED};
    enum { C_BASE_RAW, C_BASE_RHT, C_DELTA_RAW, C_DELTA_RHT, C_COUNT };
    const char* names[C_COUNT] = {"base/RAW", "base/RHT", "delta/RAW", "delta/RHT"};
    std::vector<Sample> samples[C_COUNT];

    auto run_one = [&](int cfg, int round) -> bool {
        SessionWriteOptions opt;
        SessionWriteTimings t{};
        opt.timings = &t;
        opt.durable = false;                       // no fsync: measure the writer, not the disk's sync latency
        std::string error;
        Sample s;
        s.t = t;
        const auto wall0 = std::chrono::steady_clock::now();
        bool ok = false;
        if (cfg == C_BASE_RAW || cfg == C_BASE_RHT) {
            opt.state_codec_rht = cfg == C_BASE_RHT;
            size_t bytes = 0;
            SessionStatus status{};
            const fs::path p = out_dir / ("base-" + std::to_string(cfg) + "-" + std::to_string(round) + ".sess");
            ok = session_file_write(p.string(), image, base_sources(), id, bytes, error, opt, &status);
            s.bytes = bytes;
            s.file_version = status.file_version;
        } else {
            opt.state_codec_rht = cfg == C_DELTA_RHT;
            uint64_t bytes = 0;
            const fs::path p = out_dir / ("delta-" + std::to_string(cfg) + "-" + std::to_string(round) + ".dlog");
            ok = session_delta_write(p.string(), 0, grown, delta_sources(), 6166, id, bytes, error, opt);
            s.bytes = bytes;
        }
        s.wall_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wall0).count();
        s.t = t;
        if (!ok) {
            std::fprintf(stderr, "%s round %d FAILED: %s\n", names[cfg], round, error.c_str());
            return false;
        }
        samples[cfg].push_back(s);
        return true;
    };

    // interleaved rounds; the config order alternates each round to cancel drift
    for (int r = 0; r < rounds; ++r) {
        const bool forward = r % 2 == 0;
        for (int k = 0; k < C_COUNT; ++k) {
            const int cfg = forward ? k : C_COUNT - 1 - k;
            if (!run_one(cfg, r)) return 1;
        }
        std::fprintf(stderr, "round %d done\n", r + 1);
    }

    std::printf("%-9s %4s | %10s %10s %10s %10s %10s %10s %10s | %9s %12s\n", "config", "rnd", "state_ms",
                "kv_ms", "kvsrc_ms", "rht_ms", "osw_ms", "finish_ms", "pub_ms", "wall_ms", "bytes");
    for (int cfg = 0; cfg < C_COUNT; ++cfg) {
        for (size_t r = 0; r < samples[cfg].size(); ++r) {
            const Sample& s = samples[cfg][r];
            std::printf("%-9s %4zu | %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f | %9.2f %12zu\n",
                        names[cfg], r + 1, ms(s.t.state_ns), ms(s.t.kv_ns), ms(s.t.kv_source_ns),
                        ms(s.t.rht_ns), ms(s.t.os_write_ns), ms(s.t.finish_ns), ms(s.t.publish_ns),
                        ms(s.wall_ns), s.bytes);
        }
        (void)0;
    }

    std::printf("\n== per-stage median (min..max) over %d rounds, ms ==\n", rounds);
    std::printf("%-9s | %11s %11s %11s %11s %11s %11s %11s | %11s\n", "config", "state", "kv", "kv_source",
                "rht", "os_write", "finish", "publish", "wall");
    for (int cfg = 0; cfg < C_COUNT; ++cfg) {
        auto col = [&](uint64_t SessionWriteTimings::*m) {
            std::vector<uint64_t> v;
            for (const auto& s : samples[cfg]) v.push_back(s.t.*m);
            return stats_of(std::move(v));
        };
        auto wall = stats_of([&] {
            std::vector<uint64_t> v;
            for (const auto& s : samples[cfg]) v.push_back(s.wall_ns);
            return v;
        }());
        const Stats st = col(&SessionWriteTimings::state_ns), kv = col(&SessionWriteTimings::kv_ns),
                    kvs = col(&SessionWriteTimings::kv_source_ns),
                    rht = col(&SessionWriteTimings::rht_ns), osw = col(&SessionWriteTimings::os_write_ns),
                    fin = col(&SessionWriteTimings::finish_ns), pub = col(&SessionWriteTimings::publish_ns);
        std::printf("%-9s |", names[cfg]);
        for (const Stats* s : {&st, &kv, &kvs, &rht, &osw, &fin, &pub})
            std::printf(" %6.2f(%5.2f..%5.2f)", ms(s->median), ms(s->lo), ms(s->hi));
        std::printf(" | %6.2f(%5.2f..%5.2f)\n", ms(wall.median), ms(wall.lo), ms(wall.hi));
    }

    std::printf("\n== derived (medians; nested counters are NOT summed) ==\n");
    for (int cfg = 0; cfg < C_COUNT; ++cfg) {
        auto med = [&](uint64_t SessionWriteTimings::*m) {
            std::vector<uint64_t> v;
            for (const auto& s : samples[cfg]) v.push_back(s.t.*m);
            return stats_of(std::move(v)).median;
        };
        const uint64_t st = med(&SessionWriteTimings::state_ns), kv = med(&SessionWriteTimings::kv_ns),
                       kvs = med(&SessionWriteTimings::kv_source_ns), rht = med(&SessionWriteTimings::rht_ns),
                       osw = med(&SessionWriteTimings::os_write_ns), fin = med(&SessionWriteTimings::finish_ns),
                       pub = med(&SessionWriteTimings::publish_ns);
        const uint64_t serial = st + kv + fin + pub;   // the writer's own serial stages only
        const uint64_t bytes = samples[cfg].front().bytes;
        std::printf("%-9s: serial state+kv+finish+publish = %6.2f ms (wall check above);", names[cfg], ms(serial));
        std::printf(" state share %.1f%% of serial;", 100.0 * st / serial);
        if (rht) std::printf(" rht = %5.2f ms = %4.1f%% of state_ns;", ms(rht), 100.0 * rht / st);
        std::printf(" kv_source = %5.2f ms = %4.1f%% of kv_ns;", ms(kvs), 100.0 * kvs / kv);
        std::printf(" os_write = %5.2f ms (%.0f MB/s apparent);", ms(osw), ms(osw) > 0 ? bytes / 1048576.0 / (ms(osw) / 1e3) : 0.0);
        if (rht) std::printf(" rht/os_write = %.2f;", ms(rht) / ms(osw));
        std::printf(" file v%u, %zu B (%.1f MiB)\n", samples[cfg].front().file_version, bytes, bytes / 1048576.0);
    }

    std::printf("\nnotes:\n");
    std::printf(" - state_ns includes rht_ns; kv_ns includes kv_source_ns; os_write_ns is the time the producing\n");
    std::printf("   thread sits in write() (drains + final block), NESTED inside the state/kv/finish windows on\n");
    std::printf("   that same thread - serial interleaving, not overlap; never add it to any stage.\n");
    std::printf(" - durable=false: finish_ns has no fsync; publish_ns is the base's rename+dir-flush only.\n");
    std::printf(" - K/V sources are preallocated host memory (memcpy reads), not GPU/pinned pools, and the\n");
    std::printf("   checkpoint state is a fresh synthetic copy (never REF): RAW writes 2 x logical GDN.\n");

    fs::remove_all(out_dir);
    std::printf("\nscratch dir %s removed\n", out_dir.string().c_str());
    return 0;
}
