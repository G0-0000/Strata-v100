// A durable disk tier for --serve's parked conversations: when the RAM cache evicts a conversation, it is written
// as an ordinary session file (conversation_file.hpp, the same format and model/config identity as the slot save/
// restore API) and can be restored after a restart. A small sidecar beside each file holds only the token/image
// metadata, so a new request finds the best disk match without reading the conversation's K/V.
//
// This is host code only (no CUDA): it calls the session-file writer and reader, which are host code too.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationSpillMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;
    bool live = false;
    // --conversation-cache-disk-delta: with a chain, the delta log's committed length (`path` is its base) and the
    // token count the chain's state holds (the last record's live ids).
    uint64_t delta_committed = 0;
    int64_t live_tokens = 0;
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

class ConversationSpillCache {
public:
    // Opens (creating if needed) the spill directory and indexes conversations already in it. Left-over write
    // temporaries and incomplete owned groups are cleaned when possible; a partial cleanup is retained only as an
    // ineligible accounting entry. Files whose sidecar names another model/config identity are KEPT: never deleted,
    // not indexed, outside the disk budget; open() logs a warning with their count and bytes - remove them manually.
    // delta_chains (--conversation-cache-disk-delta) selects what NEW files look like: with it, saves stamp their
    // sidecar v2 (last-use stamp) and the budget evicts least-recently-used; without it (the default) new sidecars
    // stay baseline v1 and first-spilled eviction runs. Both modes read v1/v2 sidecars; a valid chain is retained and
    // accounted in base-only mode, but only delta mode offers it to the streaming restore.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              std::string& error, bool delta_chains = false);
    bool enabled() const { return enabled_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t stale_files_wiped() const { return stale_files_wiped_; }
    size_t disk_evictions() const { return disk_evictions_; }

    // The best resume this directory offers for the prompt, from the sidecars only (no K/V read). similarity and
    // n_min filter weak hits exactly as the RAM cache's best() does.
    template<class Token, class TokenAllocator, class ImageAllocator = std::allocator<ConversationImageKey>>
    ConversationSpillMatch best(const std::vector<Token, TokenAllocator>& prompt,
                                const std::vector<ConversationImageKey, ImageAllocator>& images, bool cvec,
                                double similarity, int64_t n_min) const {
        ConversationSpillMatch best_match;
        int64_t best_tokens = 0;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& entry = entries_[i];
            // A partial removal, a missing dependency, or a chain opened without the streaming restore mode is
            // never an eligible hit. Check regular-file types as well as path presence: a directory or symlink at a
            // required path is not a session/manifest/log.
            if (!entry.complete || (entry.has_chain && !delta_) || !entry_files_valid(entry)) continue;
            const ConversationMatch match = conversation_metadata_match(entry.live_meta,
                    entry.checkpoint_lengths, prompt, images, cvec, entry.cvec, similarity, n_min);
            if (match.tokens > best_tokens)
                best_tokens = match.tokens,
                best_match = {entry.session_path(), entry.file_bytes, match.tokens, match.live, entry.delta_bytes,
                              (int64_t) entry.live_meta.ids.size()};
        }
        return best_match;
    }

    // Reads a spilled conversation back into image (its K/V included), with the engine's read limits and identity.
    // `status` (optional) reports the verified file format: on a RAM refusal of a verified format-v2 file the caller
    // keeps the file for a later attempt (a format-v1 refusal, as before, discards it).
    bool load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
              std::string& error, SessionStatus* status = nullptr) const;
    // Writes a parked conversation (its K/V in RAM) as a session file plus its sidecar, and indexes it.
    bool spill(const SavedConversation& image, std::string& error);
    // --conversation-cache-disk-only: the same file and sidecar, written straight from the live session - `meta`
    // carries everything but the K/V (meta.kv empty), `kv` streams each layer from its pool into the file - so no
    // host image of the conversation is ever held.  `bytes` is the file's size.  durable=false skips the flushes: a
    // file torn by a crash fails the payload hash when it is read and is discarded, so a cache need not pay for them.
    // prefix=true writes a PREFIX entry (strata-conv-prefix-N): a conversation's root - its system prompt and tools -
    // that any new conversation opening the same way resumes from.  Prefix entries persist: drop_superseded never
    // removes them and the disk budget evicts every ordinary conversation before one of them.
    bool spill_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv, size_t& bytes,
                        std::string& error, bool durable = false, bool prefix = false, bool state_ref = false,
                        bool state_codec_rht = false);
    // --conversation-cache-disk-delta: appends one delta record for an indexed entry instead of rewriting its file.
    // `meta` + `kv` describe the conversation's current state; kv[i].sizes are full part sizes and first_bytes the
    // offset each part's data starts at.  The record is written to the entry's `<stem>.dlog`, its committed length is
    // published in `<stem>.dchain` (atomically) and then in the sidecar, so a crash between any two leaves the chain
    // at its last committed record.  `bytes` is the record's length.  On success the entry's sidecar describes the
    // new state. A prefix, disabled directory or incomplete/unusable entry refuses.
    bool spill_delta(size_t index, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                     int64_t first_token, size_t& bytes, std::string& error, bool durable = false,
                     bool state_codec_rht = false);
    // The index of the entry whose session file is `path`, or size() when none.
    size_t find(const std::string& path) const;
    // What a delta save needs to know about an entry (see generate.cpp): its token count, its deepest checkpoint,
    // its file sizes and whether its live ids are a prefix of this conversation's tokens and images.  False for a
    // prefix entry or an unknown index.
    bool entry_info(size_t index, int64_t& tokens, int64_t& deepest, uint64_t& base_bytes, uint64_t& delta_bytes,
                    bool& prefix, std::string& path) const;
    // For a delta save: the entry's saved token ids (and images) match this state's first `n` tokens - the prefix
    // the chain's K/V covers.  Only that prefix matters: a restore rewrites the tail, so the state may be shorter
    // than the entry and still be a valid append target.
    bool entry_matches_prefix(size_t index, const ConversationIds& ids, size_t n,
                              const std::vector<ConversationImageKey>& images) const;
    // The publication-gap check (the sidecar is written after the chain manifest, so a crash can leave it one save
    // behind): the chain's last committed record must describe exactly what the sidecar advertises - the same live
    // token ids and images and the same checkpoint lengths - or restoring would mix two saves' state.  False for a
    // prefix entry or an unknown index.
    bool entry_chain_tail_matches(size_t index, const ConversationIds& tail_ids,
                                  const std::vector<ConversationImageKey>& tail_imgs,
                                  const std::vector<size_t>& tail_checkpoint_lengths) const;
    // A successful restore is an access: stamps the entry as the most recently used (persisted in the sidecar, so
    // the budget's eviction order - least recently used first - survives a restart).  Failures are not.
    void touch(const std::string& path);
    // The entry's persisted last-use stamp in milliseconds since the epoch (0: a v1 sidecar, unknown age).
    uint64_t entry_last_used(size_t index) const;
    // A prefix entry holding exactly these tokens (and images) is on disk already.
    bool has_prefix(const ConversationIds& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;
    size_t prefix_count() const;
    bool erase(const std::string& path, std::string& error);
    // A conversation being restored is held out of eviction until it is put back in RAM or dropped.
    void pin(const std::string& path);
    void unpin(const std::string& path);
    // Removes the disk copies of this conversation a turn back, the same rule the RAM cache applies before evicting.
    // `keep` (a session path) is never dropped: a disk-only save writes the new copy first and then drops the old ones.
    size_t drop_superseded(const ConversationIds& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                           const std::string& keep = {});
    // The session path of the newest file (the one the last spill wrote), or empty.
    std::string newest_path() const { return entries_.empty() ? std::string() : entries_.back().session_path(); }

private:
    struct Entry {
        std::filesystem::path stem;          // "<dir>/strata-conv-<serial>" (entry artifacts share it)
        uint64_t file_bytes = 0;             // accounted bytes of the regular .sess base
        uint64_t delta_bytes = 0;            // committed length of "<stem>.dlog" (restore/append boundary)
        uint64_t delta_accounted_bytes = 0;   // bytes of the log actually charged to this entry
        uint64_t meta_bytes = 0;              // zero for a legacy v1 sidecar; otherwise its charged bytes
        uint64_t chain_file_bytes = 0;        // charged bytes of "<stem>.dchain"
        uint64_t last_used = 0;               // milliseconds since the epoch of the last save or successful restore
        bool cvec = true;
        ConversationCheckpoint live_meta;    // ids + imgs only; no running state
        std::vector<size_t> checkpoint_lengths;
        bool prefix = false;                 // a persistent root entry (named strata-conv-prefix-N)
        bool has_chain = false;              // a log/manifest exists or this entry has published a chain
        bool complete = true;                // false while owned files are partial/unusable; retained for accounting
        std::string session_path() const { return (stem.string() + ".sess"); }
        std::string meta_path() const { return (stem.string() + ".meta"); }
        std::string delta_path() const { return (stem.string() + ".dlog"); }
        std::string chain_path() const { return (stem.string() + ".dchain"); }
        uint64_t total_bytes() const { return file_bytes + delta_accounted_bytes + meta_bytes + chain_file_bytes; }
    };
    bool entry_files_valid(const Entry& entry) const;
    void enforce_budget();
    void adjust_bytes(uint64_t before, uint64_t after);
    // Removes only this entry's owned artifacts. `removed` is the bytes debited from this entry's existing ledger;
    // a false result retains an incomplete, ineligible entry with exact remaining ownership charges for retry.
    bool remove_group(Entry& entry, uint64_t& removed, std::vector<std::string>& left);
    std::filesystem::path next_stem(bool prefix = false);
    bool index_written(const std::filesystem::path& stem, uint64_t file_bytes, const SavedConversation& meta,
                       std::string& error, bool prefix = false);
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                      bool& stamped /* out: the v2 sidecar carries a persisted last-use stamp */, std::string& error) const;
    // with_last_used (--conversation-cache-disk-delta) writes the v2 sidecar carrying Entry::last_used; without it
    // the baseline v1 bytes are written, so a directory a pre-delta build reads is never migrated by a delta-off run
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error,
                       bool with_last_used) const;
    // The chain manifest beside <stem>.dlog: base size (must match the session file) and the committed length.
    bool read_chain(const std::filesystem::path& chain, uint64_t base_size, const std::filesystem::path& dlog,
                    uint64_t& committed, std::string& error) const;
    bool write_chain(const std::filesystem::path& chain, uint64_t base_size, uint64_t committed,
                     std::string& error) const;

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    size_t stale_files_wiped_ = 0, disk_evictions_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // index order only; the budget evicts by Entry::last_used (least recent first)
                                   // when delta_chains is on, else first-spilled first (the baseline policy)
    bool delta_ = false;           // --conversation-cache-disk-delta: v2 sidecars, LRU eviction, chain appends
    std::string pinned_path_;
};

} // namespace strata::core
